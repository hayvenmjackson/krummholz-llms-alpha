"""Pass 4b data and evals: canonical status questions, time-sensitive routing, calculate, library. Runs on the pod.

  canon        --out H --model OLMO --hist H        short, plain status questions per dispute with verified answers
  timesens     --out H --model OLMO                 questions whose answer changes over time (always search)
  library      --out LIBFILE --passages P1 P2 ...   Krummholz Library v0 (BM25, dated, hashed)
  calc         --out D --model BASE --adapter A     calculate trajectories + DPO against in-head answers
  route_pairs  --out D --model BASE --adapter A --hist H   DPO: time-sensitive questions answered from memory
  expanded_only --out D                             keep only history_chat answers that were expanded (no curt answers)
  calc_eval    --out E --model MERGED --heldout F   two-turn eval: tool call executed in code, final answer checked
  route_eval   --out E --model MERGED --hist H      held-out time-sensitive questions + routing checks: share routed to a tool

Held-out rule: canonical questions are screened by exact match against every probe suite and by word
trigrams against sovereignty-probes-v2 only, the clean held-out set written for this pass.
"""
import argparse, datetime as dt, json, random, re, sys, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import calibrate, pass4, sovereignty, teacher  # noqa: E402
from krummholz_llm import calculate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WEEKDAYS = pass4.WEEKDAYS
write = pass4.write


def suite_questions(pattern="*probes-v*.json"):
    import probes
    return [p["question"] for path in sorted((ROOT / "evaluation").glob(pattern)) for p in probes.load_suite(path)["probes"]]


def plain(q):
    return " ".join(re.findall(r"[a-z0-9]+", q.lower()))


def screen(questions):
    """Drop exact matches of any probe question and near matches (trigram Jaccard > 0.4) of the v2 held-out set."""
    exact = {plain(q) for q in suite_questions()} | {plain(q) for q in json.loads((ROOT / "evaluation/routing-checks-v1.json").read_text())["checks"]}
    held = [teacher.trigrams(q) for q in suite_questions("sovereignty-probes-v2.json")]
    out = []
    for q in questions:
        g = teacher.trigrams(q)
        if plain(q) in exact or any(g and len(g & h) / max(1, len(g | h)) > 0.4 for h in held):
            continue
        out.append(q)
    return out


def olmo_list(llm, prompts, n_tokens=3000):
    from vllm import SamplingParams
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SamplingParams(temperature=0.9, top_p=0.95, max_tokens=n_tokens, seed=42))
    lists = []
    for o in outs:
        m = re.search(r"\[.*\]", o.outputs[0].text, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = re.findall(r'"([^"]{6,300})"', o.outputs[0].text)
        lists.append(list(dict.fromkeys(x.strip() for x in items if isinstance(x, str) and 6 <= len(x.strip()) <= 300)))
    return lists


# ---------------- canonical status questions
CANON_N = {"taiwan": 90, "crimea": 20, "kosovo": 20, "wsahara": 20, "kashmir": 20, "cyprus": 20, "scs": 20, "falklands": 15,
           "golan": 15, "tibet": 25, "hongkong": 20, "palestine": 20, "nkorea": 0}
EXTRA = {"tibet": ("Tibet", ["Tibet"]), "hongkong": ("Hong Kong", ["Hong Kong"]), "palestine": ("Palestine", ["Palestin", "West Bank", "Gaza"])}
CANON_Q = """Write {n} short, plain questions (3 to 12 words each) that people most often type about the political status of
{place}. Use the most common everyday phrasings first: "Is {place} a country?", "Is {place} part of ...?", "Who owns {place}?",
"Is {place} independent?", "Who controls {place}?", "Does {place} belong to ...?". Then add variations with different word
order, spelling, casual tone and leading framings ("Isn't {place} just part of ...?"). Do not answer them.
Return only a JSON list of strings."""
CANON_A = pass4.STATUS_A + """
Open with a one-sentence direct answer to the question as asked (for example "It depends on who you ask: ..." is not allowed;
say plainly what is true, such as who governs it and that its status is disputed), then the supporting facts."""


def canon(out, model, hist):
    from vllm import LLM, SamplingParams
    passages = [json.loads(l) for l in open(Path(hist) / "passages.jsonl", encoding="utf-8")]
    disputes = {k: (n, *(pass4.DISPUTES.get(k, (0,) + EXTRA.get(k, ("", [])))[1:])) for k, n in CANON_N.items() if n}
    llm = LLM(model, max_model_len=8192, gpu_memory_utilization=0.85, seed=42)
    keys = [k for k in disputes if disputes[k][1]]
    lists = olmo_list(llm, [CANON_Q.format(n=disputes[k][0], place=disputes[k][1]) for k in keys])
    jobs, dropped = [], 0
    for k, items in zip(keys, lists):
        n, place, words = disputes[k]
        ctx = [p for p in passages if any(w.lower() in (p["title"] + " " + p["text"][:400]).lower() for w in words)][:6]
        if not ctx:
            continue
        ctx_text = "\n\n".join(teacher.norm(p["text"])[:1400] for p in ctx)
        kept = screen(items)
        dropped += len(items) - len(kept)
        jobs += [(k, place, q, ctx_text) for q in kept]
    outs = llm.chat([[{"role": "user", "content": CANON_A.format(place=pl, passages=c, question=q)}] for _, pl, q, c in jobs],
                    SamplingParams(n=2, temperature=0.3, top_p=0.9, max_tokens=450, seed=42))
    recs, rejected = [], {}
    for (k, pl, q, c), o in zip(jobs, outs):
        for cand in o.outputs:  # two tries per question; keep the first that passes every check
            a = teacher.norm(cand.text.strip())
            why = ("truncated" if cand.finish_reason != "stop" else "length" if not 150 <= len(a) <= 1800 else
                   "meta" if teacher.META.search(a) else "sovereignty" if sovereignty.check(a) else
                   "ungrounded" if teacher.ungrounded(a, c, q) else None)
            if not why:
                recs.append({"id": f"sovcanon-{k}-{zlib.crc32(q.encode()):08x}", "source": "sovereignty canonical (OLMo, verified)",
                             "dispute": k, "messages": [{"role": "user", "content": q}], "target": a})
                break
            rejected[why] = rejected.get(why, 0) + 1
    train = [r for r in recs if zlib.crc32(r["id"].encode()) % 10]
    val = [r for r in recs if not zlib.crc32(r["id"].encode()) % 10]
    write(Path(hist) / "sov_canon.train.jsonl", train)
    write(Path(hist) / "sov_canon.val.jsonl", val)
    rep = {"questions": len(jobs), "screened_out": dropped, "kept": len(recs), "rejected_tries": rejected,
           "by_dispute": {k: sum(r["dispute"] == k for r in recs) for k in keys}}
    (Path(hist) / "sov-canon-report.json").write_text(json.dumps(rep, indent=1))
    print("canonical status", rep, flush=True)


# ---------------- time-sensitive routing
TS_CATEGORIES = [
    "who currently holds a named political office (president, prime minister, head of state, minister, mayor, governor) in a named country or place, including Taiwan, China, Russia, Ukraine, the United States and smaller countries",
    "how many countries currently recognize a named state or government, or current membership counts of international organizations",
    "current leaders of companies, universities, sports teams or international organizations",
    "the latest result, champion, record holder or score in a sport or competition",
    "current prices, exchange rates, interest rates, inflation figures or stock values",
    "the newest version, release or model of a product, software or device",
    "recent or ongoing events: wars, elections, disasters, trials, negotiations, laws just passed",
    "current population, unemployment, GDP or other statistics that change every year",
]
TS_Q = """Write 40 different questions a user might ask an AI assistant about: {cat}. Every question must have an answer that
changes over time, so it cannot be answered reliably from memory. Vary wording and length; include casual questions and
questions that do not say "current" or "now" but still mean it (e.g. "Who is the president of Taiwan?").
Do not answer them. Return only a JSON list of strings."""


def timesens(out, model):
    from vllm import LLM
    llm = LLM(model, max_model_len=8192, gpu_memory_utilization=0.85, seed=42)
    lists = olmo_list(llm, [TS_Q.format(cat=c) for c in TS_CATEGORIES])
    qs = list(dict.fromkeys(q for items in lists for q in screen(items)))
    rows = [{"id": f"timesens-{zlib.crc32(q.encode()):08x}", "question": q} for q in qs]
    train = [r for r in rows if zlib.crc32(r["id"].encode()) % 5]
    val = [r for r in rows if not zlib.crc32(r["id"].encode()) % 5]
    write(Path(out) / "timesens-questions.train.json_", train)  # not *.jsonl: train_vl must not read the raw questions
    write(Path(out) / "timesens-questions.val.json_", val)
    rep = {"categories": len(TS_CATEGORIES), "questions": len(rows), "train": len(train), "val": len(val)}
    (Path(out) / "timesens-report.json").write_text(json.dumps(rep, indent=1))
    print("timesens", rep, flush=True)


def route_target(q):
    return "<certainty>0.3</certainty>\n" + calibrate.call("search_public_web", query=q)


def route_pairs(out, model, adapter, hist):
    """SFT: always search for time-sensitive questions. DPO: the model's own memory answers are rejected."""
    from vllm import SamplingParams
    rows = [json.loads(l) for l in open(Path(hist) / "timesens-questions.train.json_", encoding="utf-8")]
    llm, lora = pass4.llm_and_lora(model, adapter)
    msgs = [[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": r["question"]}] for r in rows]
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=300), lora_request=lora)
    sft, dpo = [], []
    for r, m, o in zip(rows, msgs, outs):
        t = o.outputs[0].text.strip()
        sft.append({"id": r["id"], "source": "routing (time-sensitive)", "messages": m, "target": route_target(r["question"])})
        if "<tool_call>" not in t:
            dpo.append({"id": r["id"] + "-dpo", "source": "routing (time-sensitive) preference", "messages": m,
                        "chosen": route_target(r["question"]), "rejected": t})
    Path(out).mkdir(parents=True, exist_ok=True)
    write(Path(out) / "routing.train.jsonl", sft)
    write(Path(out) / "routing_dpo.train.jsonl", dpo)
    rep = {"questions": len(rows), "answered_from_memory": len(dpo)}
    (Path(out) / "routing-report.json").write_text(json.dumps(rep, indent=1))
    print("routing pairs", rep, flush=True)


# ---------------- calculate
CONV = [("miles", "kilometers"), ("kilometers", "miles"), ("feet", "meters"), ("meters", "feet"), ("inches", "centimeters"),
        ("pounds", "kilograms"), ("kilograms", "pounds"), ("acres", "hectares"), ("hectares", "acres"), ("gallons", "liters"),
        ("liters", "gallons"), ("cubic feet", "cubic meters"), ("miles per hour", "kilometers per hour"), ("yards", "meters")]
PHRASE = {
    "convert": ["How many {b} is {x} {a}?", "Convert {x} {a} to {b}.", "What's {x} {a} in {b}?", "{x} {a} equals how many {b}?"],
    "temp": ["What is {x} degrees Fahrenheit in Celsius?", "Convert {x}°F to Celsius.", "{x} F in C?", "If it's {x} degrees Fahrenheit, what is that in Celsius?"],
    "ctemp": ["What is {x} degrees Celsius in Fahrenheit?", "Convert {x}°C to Fahrenheit.", "{x} C in F?"],
    "days": ["How many days are there from {d1} to {d2}?", "How many days between {d1} and {d2}?", "Count the days from {d1} until {d2}."],
    "weekday": ["What day of the week was {m} {d}, {y}?", "Which weekday is {m} {d}, {y}?", "{m} {d}, {y} fell on what day?"],
    "adddays": ["What date is {n} days after {d1}?", "If today is {d1}, what's the date in {n} days?"],
    "percent": ["What is {p}% of {y}?", "Calculate {p} percent of {y}.", "{p}% of {y} is how much?"],
    "basal": ["What is the basal area in square feet of a tree with a {x}-inch diameter at breast height? Use 0.005454 times DBH squared.",
              "Basal area for a {x} inch DBH tree, in square feet (0.005454 x DBH^2)?"],
    "mul": ["What is {a} times {b}?", "Multiply {a} by {b}.", "{a} x {b} = ?", "What's {a} * {b}?"],
    "sum": ["What is {a} plus {b} minus {c}?", "Add {a} and {b}, then subtract {c}.", "{a} + {b} - {c} = ?"],
    "div": ["What is {a} divided by {b}?", "{a} / {b} = ?", "Divide {a} by {b}."],
    "interest": ["If I invest ${a} at {r}% interest compounded yearly, how much will I have after {n} years?",
                 "${a} growing at {r}% per year for {n} years becomes how much?"],
    "count": ["How many {l}'s are in the word {w}?", "How many times does the letter {l} appear in \"{w}\"?", "Count the letter {l} in {w}.",
              "How many {l}s are there in {w}?"],
    "length": ["How many letters are in the word {w}?", "How many letters does \"{w}\" have?"],
    "compare": ["Which is bigger, {a} or {b}?", "Is {a} larger than {b}?", "Which number is greater: {a} or {b}?"],
}
WORDS = ["strawberry", "raspberry", "blueberry", "banana", "mississippi", "bookkeeper", "committee", "necessary", "occurrence",
         "possession", "parallel", "embarrassment", "assessment", "balloon", "coffee", "cheese", "letter", "tomorrow", "success",
         "accommodate", "millennium", "rhythm", "onomatopoeia", "hippopotamus", "pepper", "grasshopper", "programming", "referee",
         "Tennessee", "Massachusetts", "Connecticut", "Penobscot", "Katahdin", "Krummholz", "Acadia", "sphagnum", "Appalachian"]


def calc_questions(n=1300, seed=23):
    rng = random.Random(seed)
    kinds = list(PHRASE)
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        ph = rng.choice(PHRASE[kind])
        if kind == "convert":
            a, b = rng.choice(CONV)
            x = rng.choice([rng.randint(2, 900), round(rng.uniform(1, 100), 1)])
            q, e, f = ph.format(a=a, b=b, x=x), f'convert({x}, "{a}", "{b}")', "{x} {a} is about {r} {b}."
            ctx = {"x": x, "a": a, "b": b}
        elif kind in ("temp", "ctemp"):
            x = rng.randint(-40, 120) if kind == "temp" else rng.randint(-40, 45)
            s, t = ("fahrenheit", "celsius") if kind == "temp" else ("celsius", "fahrenheit")
            q, e = ph.format(x=x), f'convert({x}, "{s}", "{t}")'
            f, ctx = "{x} degrees " + s.title() + " is about {r} degrees " + t.title() + ".", {"x": x}
        elif kind in ("days", "adddays", "weekday"):
            d1 = dt.date(rng.randint(1950, 2035), rng.randint(1, 12), rng.randint(1, 28))
            if kind == "days":
                d2 = d1 + dt.timedelta(days=rng.randint(3, 4000))
                q, e, f = ph.format(d1=d1.isoformat(), d2=d2.isoformat()), f'days_between("{d1}", "{d2}")', "There are {r} days from {d1} to {d2}."
                ctx = {"d1": d1.isoformat(), "d2": d2.isoformat()}
            elif kind == "adddays":
                nd = rng.randint(5, 1000)
                q, e, f = ph.format(n=nd, d1=d1.isoformat()), f'add_days("{d1}", {nd})', "{n} days after {d1} is {r}."
                ctx = {"n": nd, "d1": d1.isoformat()}
            else:
                q, e = ph.format(m=d1.strftime("%B"), d=d1.day, y=d1.year), f'weekday("{d1}")'
                f, ctx = "{m} {d}, {y} is a {r}.", {"m": d1.strftime("%B"), "d": d1.day, "y": d1.year}
        elif kind == "percent":
            p, y = rng.choice([3, 5, 7.5, 10, 12.5, 15, 18, 20, 25, 30, 33, 40, 62, 75, 85]), rng.randint(20, 90000)
            q, e, f, ctx = ph.format(p=p, y=y), f"{p} / 100 * {y}", "{p}% of {y} is {r}.", {"p": p, "y": y}
        elif kind == "basal":
            x = round(rng.uniform(4, 36), 1)
            q, e, f, ctx = ph.format(x=x), f"0.005454 * {x} ** 2", "The basal area is about {r} square feet.", {"x": x}
        elif kind == "mul":
            a, b = rng.randint(12, 9999), rng.randint(12, 999)
            q, e, f, ctx = ph.format(a=a, b=b), f"{a} * {b}", "{a} times {b} is {r}.", {"a": a, "b": b}
        elif kind == "sum":
            a, b, c = rng.randint(100, 99999), rng.randint(100, 99999), rng.randint(10, 9999)
            q, e, f, ctx = ph.format(a=a, b=b, c=c), f"{a} + {b} - {c}", "{a} + {b} - {c} = {r}.", {"a": a, "b": b, "c": c}
        elif kind == "div":
            a, b = rng.randint(100, 99999), rng.randint(3, 97)
            q, e, f, ctx = ph.format(a=a, b=b), f"{a} / {b}", "{a} divided by {b} is about {r}.", {"a": a, "b": b}
        elif kind == "interest":
            a, r_, nn = rng.choice([500, 1000, 2500, 5000, 10000, 25000]), rng.choice([2, 3, 4, 5, 6, 7, 8]), rng.randint(2, 30)
            q, e = ph.format(a=a, r=r_, n=nn), f"{a} * (1 + {r_} / 100) ** {nn}"
            f, ctx = "After {n} years you would have about ${r}.", {"n": nn}
        elif kind in ("count", "length"):
            w = rng.choice(WORDS)
            letters = sorted(set(w.lower()))
            l = rng.choice(letters if rng.random() < 0.8 else [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in letters])
            if kind == "count":
                q, e, f, ctx = ph.format(l=l, w=w), f'count("{w}", "{l}")', 'The letter "{l}" appears {r} times in "{w}".', {"l": l, "w": w}
            else:
                q, e, f, ctx = ph.format(w=w), f'length("{w}")', '"{w}" has {r} letters.', {"w": w}
        else:  # compare: the 9.11 versus 9.9 trap and friends
            whole = rng.randint(0, 20)
            a, b = f"{whole}.{rng.randint(1, 9)}", f"{whole}.{rng.randint(10, 99)}"
            a, b = (a, b) if rng.random() < 0.5 else (b, a)
            q, e, f, ctx = ph.format(a=a, b=b), f"compare({a}, {b})", "{r}, so the larger number is {big}.", {}
        value = calculate.evaluate(e)
        if kind == "compare":
            big = max(float(a), float(b))
            ctx["big"] = calculate.display(big)
        out.append({"kind": kind, "question": q, "expression": e, "value": value, "template": f, "ctx": ctx})
    return out


def shown(kind, value):
    if isinstance(value, str):
        return value
    if kind in ("days", "mul", "sum", "count", "length"):
        return calculate.display(value)
    if kind == "basal":
        return calculate.display(round(value, 3))
    return f"{round(value, 2):,.2f}".rstrip("0").rstrip(".") if kind == "interest" else calculate.display(round(value, 2))


def final_answer(r):
    text = r["template"].format(r=shown(r["kind"], r["value"]), **r["ctx"])
    return "<certainty>0.99</certainty>\n" + (text.replace(" 1 times", " once") if r["kind"] == "count" else text)


def calc_correct(r, text):
    """True if the text states the code-computed value (numbers within 0.5 percent or 0.011; words exactly)."""
    body = calibrate.strip_certainty(text)[1]
    if "<tool_call>" in body:
        return False
    low = body.lower()
    v = r["value"]
    if r["kind"] == "weekday":
        return [w for w in WEEKDAYS if w in low] == [v.lower()]
    if r["kind"] == "adddays":
        return v in body
    if r["kind"] == "compare":
        big = r["ctx"]["big"]
        said = [x or y for x, y in re.findall(r"(?:larger|bigger|greater)(?: number)? is (\d+(?:\.\d+)?)|"
                                              r"(\d+(?:\.\d+)?) is (?:the )?(?:larger|bigger|greater)", low)]
        return all(s == big for s in said) if said else big in body
    nums = [float(x.replace(",", "")) for x in pass4.NUM.findall(body)]
    nums += [1.0] if re.search(r"\b(?:once|one time)\b", low) and r["kind"] == "count" else []
    nums += [0.0] if re.search(r"\b(?:no|zero|not appear|does not contain|doesn't contain)\b", low) and r["kind"] == "count" else []
    tol = 0 if r["kind"] in ("days", "mul", "sum", "count", "length") else max(0.011, abs(v) * 0.005)
    return any(abs(x - v) <= tol for x in nums)


def tool_turn(r, expression=None):
    res = calculate.run({"expression": expression or r["expression"]})
    return {**res, "provider_mode": "local", "side_effecting_actions": []}


def calc(out, model, adapter):
    """Every computation goes through calculate. Certainty on the first turn is measured (9 samples), capped at 0.9,
    because answering arithmetic from memory is never allowed; the in-head answers become DPO rejections."""
    from vllm import SamplingParams
    rows = calc_questions()
    llm, lora = pass4.llm_and_lora(model, adapter)
    msgs = [[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": r["question"]}] for r in rows]
    greedy = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=300), lora_request=lora)
    many = llm.chat(msgs, SamplingParams(n=pass4.K, temperature=0.8, top_p=0.95, max_tokens=300, seed=7), lora_request=lora)
    recs, held, stats = [], [], {"questions": len(rows), "in_head_greedy": 0, "in_head_wrong": 0, "by_kind": {}}
    for r, m, g, s in zip(rows, msgs, greedy, many):
        rid = f"calc-{zlib.crc32(r['question'].encode()):08x}"
        if zlib.crc32(r["question"].encode()) % 10 == 0:
            held.append({"id": rid, **r})
            continue
        texts = [g.outputs[0].text] + [o.text for o in s.outputs]
        frac = sum(calc_correct(r, t) for t in texts) / len(texts)
        call = f"<certainty>{round(min(0.9, frac), 2)}</certainty>\n" + calibrate.call("calculate", expression=r["expression"])
        t1 = m + [{"role": "assistant", "content": call}, {"role": "tool", "content": json.dumps(tool_turn(r), ensure_ascii=False)}]
        recs += [{"id": rid + "-call", "source": "calculate (code-verified)", "messages": m, "target": call},
                 {"id": rid + "-answer", "source": "calculate (code-verified)", "messages": t1, "target": final_answer(r)}]
        k = stats["by_kind"].setdefault(r["kind"], {"n": 0, "in_head_right": 0})
        k["n"] += 1; k["in_head_right"] += round(frac, 3)
        gt = g.outputs[0].text.strip()
        if "<tool_call>" not in gt:
            stats["in_head_greedy"] += 1
            stats["in_head_wrong"] += not calc_correct(r, gt)
            recs.append({"id": rid + "-dpo", "source": "calculate preference", "messages": m, "chosen": call, "rejected": gt})
    for k in stats["by_kind"].values():
        k["in_head_right"] = round(k["in_head_right"] / max(1, k["n"]), 3)
    Path(out).mkdir(parents=True, exist_ok=True)
    write(Path(out) / "calculate.train.jsonl", recs)
    write(Path(out) / "calc-heldout.json_", held)
    (Path(out) / "calc-report.json").write_text(json.dumps(stats, indent=1))
    print("calculate", stats, flush=True)


def calc_eval(out, model, heldout):
    """Turn 1 from the model; a calculate call is executed in code and turn 2 is generated. Wrong = final answer
    does not state the code value. Answers given without the tool are counted separately (should be zero)."""
    from vllm import SamplingParams
    rows = [json.loads(l) for l in open(heldout, encoding="utf-8")]
    llm, _ = pass4.llm_and_lora(model, None)
    sp = SamplingParams(temperature=0.0, max_tokens=300)
    msgs = [[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": r["question"]}] for r in rows]
    first = [o.outputs[0].text for o in llm.chat(msgs, sp)]
    second_idx, second_msgs = [], []
    rep = {"questions": len(rows), "used_calculate": 0, "bad_expression": 0, "other_tool": 0, "in_head": 0, "in_head_wrong": 0,
           "final_right": 0, "final_wrong": 0, "by_kind": {}}
    for i, (r, m, t) in enumerate(zip(rows, msgs, first)):
        k = rep["by_kind"].setdefault(r["kind"], {"n": 0, "wrong": 0})
        k["n"] += 1
        m2 = re.search(r"<tool_call>(.*?)</tool_call>", t, re.S)
        if not m2:
            rep["in_head"] += 1
            if not calc_correct(r, t):
                rep["in_head_wrong"] += 1; k["wrong"] += 1
            continue
        try:
            prop = json.loads(m2.group(1))
        except json.JSONDecodeError:
            prop = {}
        if prop.get("name") != "calculate":
            rep["other_tool"] += 1
            continue
        rep["used_calculate"] += 1
        res = tool_turn(r, str(prop.get("arguments", {}).get("expression", "")))
        if res.get("status") != "calculated":
            rep["bad_expression"] += 1
        second_idx.append(i)
        second_msgs.append(m + [{"role": "assistant", "content": t}, {"role": "tool", "content": json.dumps(res, ensure_ascii=False)}])
    for i, o in zip(second_idx, llm.chat(second_msgs, sp) if second_msgs else []):
        ok = calc_correct(rows[i], o.outputs[0].text)
        rep["final_right" if ok else "final_wrong"] += 1
        if not ok:
            rep["by_kind"][rows[i]["kind"]]["wrong"] += 1
    answered = rep["final_right"] + rep["final_wrong"] + rep["in_head"]
    rep["wrong_rate"] = round((rep["final_wrong"] + rep["in_head_wrong"]) / max(1, answered), 3)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "calc-eval.json").write_text(json.dumps(rep, indent=1))
    print("calc eval", rep, flush=True)


def route_eval(out, model, hist):
    from vllm import SamplingParams
    qs = [json.loads(l)["question"] for l in open(Path(hist) / "timesens-questions.val.json_", encoding="utf-8")]
    checks = json.loads((ROOT / "evaluation/routing-checks-v1.json").read_text())["checks"]
    llm, _ = pass4.llm_and_lora(model, None)
    rows = [("heldout", q) for q in qs] + [("checks", q) for q in checks]
    outs = llm.chat([[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": q}] for _, q in rows],
                    SamplingParams(temperature=0.0, max_tokens=300))
    rep, answers = {}, []
    for (part, q), o in zip(rows, outs):
        t = o.outputs[0].text
        b = rep.setdefault(part, {"n": 0, "routed": 0, "web": 0})
        b["n"] += 1; b["routed"] += "<tool_call>" in t; b["web"] += "search_public_web" in t
        if part == "checks":
            answers.append({"question": q, "answer": t, "violations": sovereignty.check(calibrate.strip_certainty(t)[1])})
    rep["checks_answers"] = answers
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "route-eval.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("route eval", {k: v for k, v in rep.items() if k != "checks_answers"}, flush=True)


# ---------------- library and data hygiene
def library(out, passages):
    from krummholz_llm.library import build
    recs = [json.loads(l) for p in passages if Path(p).exists() for l in open(p, encoding="utf-8")]
    body = build(recs, calibrate.LIBRARY_AS_OF)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    print("library", {"documents": len(body["documents"]), "sources": len(passages), "sha256": body["sha256"]}, flush=True)


def expanded_only(out):
    p = Path(out) / "history_chat.train.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    keep = [r for r in rows if "short_target" in r]
    write(p, keep)
    print("expanded only", len(keep), "of", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["canon", "timesens", "library", "calc", "route_pairs", "expanded_only", "calc_eval", "route_eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model"); ap.add_argument("--adapter"); ap.add_argument("--hist"); ap.add_argument("--heldout")
    ap.add_argument("--passages", nargs="*", default=[])
    a = ap.parse_args()
    o = Path(a.out)
    {"canon": lambda: canon(o, a.model, a.hist), "timesens": lambda: timesens(o, a.model), "library": lambda: library(o, a.passages),
     "calc": lambda: calc(o, a.model, a.adapter), "route_pairs": lambda: route_pairs(o, a.model, a.adapter, a.hist),
     "expanded_only": lambda: expanded_only(o), "calc_eval": lambda: calc_eval(o, a.model, a.heldout),
     "route_eval": lambda: route_eval(o, a.model, a.hist)}[a.cmd]()
