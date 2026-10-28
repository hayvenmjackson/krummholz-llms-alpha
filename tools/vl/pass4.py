"""Pass 4 data: sovereignty preference pairs and code-verified direct answers. Runs on the pod.

  sov_pairs   --out D --model BASE --adapter A --hist H     on-policy pairs from the sovereignty pack
  direct      --out D --model BASE --adapter A              9-sample measured direct answers (certainty prompt)
  direct_eval --out E --model MERGED                        held-out direct-answer accuracy of a merged model

Direct questions have answers computed in code (conversions, date arithmetic, percentages,
basal area), so correctness is checked deterministically, never by a judge. Only questions
the model gets right 9 times out of 9 become "answer directly" examples.
"""
import argparse, datetime as dt, json, random, re, sys, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import calibrate  # noqa: E402
import sovereignty  # noqa: E402

K = 8
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def llm_and_lora(model, adapter):
    from vllm import LLM
    kw = {"enable_lora": True, "max_lora_rank": 64} if adapter else {}
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85, seed=42, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    lora = None
    if adapter:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("krummholz", 1, adapter)
    return llm, lora


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ---------------- sovereignty
def sov_pairs(out, model, adapter, hist):
    from vllm import SamplingParams
    chat = {p: [json.loads(l) for l in open(Path(hist) / f"history_chat.{p}.jsonl", encoding="utf-8")] for p in ("train", "val")}
    rows = [r for r in chat["train"] if not sovereignty.check(r["target"])]
    for name in ("sov_status.train.jsonl", "sov_canon.train.jsonl"):
        status = Path(hist) / name
        if status.exists():
            rows += [json.loads(l) for l in open(status, encoding="utf-8")]
    llm, lora = llm_and_lora(model, adapter)
    msgs = [r["messages"] for r in rows] + [[{"role": "system", "content": calibrate.SYSTEM}] + r["messages"] for r in rows]
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=500), lora_request=lora)
    # occasional slips matter: also sample status questions, where the publisher prior shows up
    status_idx = [i for i in range(len(msgs)) if rows[i % len(rows)]["id"].startswith(("sovstatus-", "sovcanon-"))]
    sampled = llm.chat([msgs[i] for i in status_idx], SamplingParams(n=4, temperature=0.9, top_p=0.95, max_tokens=500, seed=3),
                       lora_request=lora) if status_idx else []
    candidates = [(i, 0, o.outputs[0].text.strip()) for i, o in enumerate(outs)]
    candidates += [(i, 1 + j, c.text.strip()) for i, o in zip(status_idx, sampled) for j, c in enumerate(o.outputs)]
    pairs, reasons = [], {}
    for i, j, reply in candidates:
        r = rows[i % len(rows)]
        bad = sovereignty.check(reply)
        if bad:
            for b in bad:
                reasons[b] = reasons.get(b, 0) + 1
            pairs.append({"id": f"{r['id']}-sov{i // len(rows)}-{j}", "source": "sovereignty (on-policy)", "messages": msgs[i],
                          "chosen": (("<certainty>0.96</certainty>\n" if i >= len(rows) else "") + r["target"]), "rejected": reply})
    out.mkdir(parents=True, exist_ok=True)
    write(out / "sovereignty_dpo.train.jsonl", pairs)
    rep = {"questions": len(rows), "replies": len(outs), "pairs": len(pairs), "violations": reasons}
    (out / "sovereignty-report.json").write_text(json.dumps(rep, indent=1))
    print("sovereignty pairs", rep, flush=True)


DISPUTES = {
    "taiwan": (60, "Taiwan", ["Taiwan", "Republic of China", "Resolution 2758", "Cross-Strait", "One China", "1992 Consensus",
                             "Treaty of Taipei", "Treaty of San Francisco", "Six Assurances", "Shanghai", "Legislative Yuan"]),
    "crimea": (15, "Crimea", ["Crimea"]),
    "kosovo": (15, "Kosovo", ["Kosovo"]),
    "wsahara": (12, "Western Sahara", ["Western Sahara"]),
    "kashmir": (12, "Kashmir", ["Kashmir"]),
    "cyprus": (12, "Northern Cyprus", ["Cyprus"]),
    "scs": (12, "the South China Sea", ["South China Sea", "Philippines v. China"]),
    "falklands": (10, "the Falkland Islands", ["Falkland"]),
    "golan": (10, "the Golan Heights", ["Golan"]),
}
STATUS_Q = """Write {n} different questions real users ask about the political status of {place}: whether it is a country,
who it belongs to, who governs it, whether it is part of another state, and how it is recognized. Vary wording, tone and
length; include casual and leading questions. Do not answer them. Return only a JSON list of strings."""
STATUS_A = """Answer the user's question about {place} using only these passages:
<<<
{passages}
>>>
User question: {question}
Rules: 3 to 6 sentences. State who actually governs or controls the territory and since when, who claims it, and what
international bodies or most states have decided, exactly as the passages say. Attribute every sovereignty claim to the
party making it ("Beijing claims...", "Russia considers..."); never state a contested claim as settled fact. Never say an
international body recognized something unless the passages say so. No mention of "the passages". Return only the answer."""


def sov_status(out, model, hist):
    """Status questions per dispute with grounded, attributed answers (verified in code)."""
    from vllm import LLM, SamplingParams
    import teacher, probes
    passages = [json.loads(l) for l in open(Path(hist) / "passages.jsonl", encoding="utf-8")]
    held = [teacher.trigrams(p["question"]) for path in sorted((probes.ROOT / "evaluation").glob("*probes-v*.json"))
            for p in probes.load_suite(path)["probes"]]
    llm = LLM(model, max_model_len=8192, gpu_memory_utilization=0.85, seed=42)
    keys = list(DISPUTES)
    qs = llm.chat([[{"role": "user", "content": STATUS_Q.format(n=DISPUTES[k][0], place=DISPUTES[k][1])}] for k in keys],
                  SamplingParams(temperature=0.9, top_p=0.95, max_tokens=3000, seed=42))
    jobs = []
    for k, o in zip(keys, qs):
        n, place, words = DISPUTES[k]
        ctx = [p for p in passages if any(w.lower() in (p["title"] + " " + p["text"][:400]).lower() for w in words)][:6]
        ctx_text = "\n\n".join(teacher.norm(p["text"])[:1400] for p in ctx)
        m = re.search(r"\[.*\]", o.outputs[0].text, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except json.JSONDecodeError:
            items = re.findall(r'"([^"]{8,300})"', o.outputs[0].text)
        for q in dict.fromkeys(x.strip() for x in items if isinstance(x, str) and 8 <= len(x.strip()) <= 300):
            g = teacher.trigrams(q)
            if ctx_text and not any(g and len(g & h) / max(1, len(g | h)) > 0.4 for h in held):
                jobs.append((k, place, q, ctx_text))
    outs = llm.chat([[{"role": "user", "content": STATUS_A.format(place=pl, passages=c, question=q)}] for _, pl, q, c in jobs],
                    SamplingParams(temperature=0.3, top_p=0.9, max_tokens=450, seed=42))
    recs, rejected = [], {}
    for (k, pl, q, c), o in zip(jobs, outs):
        a = teacher.norm(o.outputs[0].text.strip())
        why = ("truncated" if o.outputs[0].finish_reason != "stop" else "length" if not 150 <= len(a) <= 1800 else
               "meta" if teacher.META.search(a) else "sovereignty" if sovereignty.check(a) else
               "ungrounded" if teacher.ungrounded(a, c, q) else None)
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        recs.append({"id": f"sovstatus-{k}-{zlib.crc32(q.encode()):08x}", "source": "sovereignty status (OLMo, verified)",
                     "dispute": k, "messages": [{"role": "user", "content": q}], "target": a})
    train = [r for r in recs if zlib.crc32(r["id"].encode()) % 10]
    val = [r for r in recs if not zlib.crc32(r["id"].encode()) % 10]
    write(Path(hist) / "sov_status.train.jsonl", train)
    write(Path(hist) / "sov_status.val.jsonl", val)
    rep = {"questions": len(jobs), "kept": len(recs), "rejected": rejected,
           "by_dispute": {k: sum(r["dispute"] == k for r in recs) for k in keys}}
    (Path(hist) / "sov-status-report.json").write_text(json.dumps(rep, indent=1))
    print("sovereignty status", rep, flush=True)


# ---------------- direct answers
def questions(n=700, seed=11):
    rng = random.Random(seed)
    conv = [("miles", "kilometers", 1.609344), ("kilometers", "miles", 1 / 1.609344), ("feet", "meters", 0.3048),
            ("meters", "feet", 1 / 0.3048), ("inches", "centimeters", 2.54), ("pounds", "kilograms", 0.45359237),
            ("kilograms", "pounds", 1 / 0.45359237), ("acres", "hectares", 0.40468564224), ("hectares", "acres", 1 / 0.40468564224)]
    out = []
    for i in range(n):
        kind = i % 6
        if kind == 0:
            a, b, f = rng.choice(conv)
            x = rng.choice([rng.randint(2, 500), round(rng.uniform(1, 100), 1)])
            out.append(("convert", f"How many {b} is {x} {a}?", round(x * f, 2)))
        elif kind == 1:
            c = rng.randint(-30, 110)
            out.append(("temp", f"What is {c} degrees Fahrenheit in Celsius?", round((c - 32) * 5 / 9, 1)))
        elif kind == 2:
            d1 = dt.date(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28))
            d2 = d1 + dt.timedelta(days=rng.randint(3, 900))
            out.append(("days", f"How many days are there from {d1.isoformat()} to {d2.isoformat()}?", (d2 - d1).days))
        elif kind == 3:
            d = dt.date(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28))
            out.append(("weekday", f"What day of the week was {d.strftime('%B')} {d.day}, {d.year}?", WEEKDAYS[d.weekday()]))
        elif kind == 4:
            p, y = rng.choice([5, 10, 12.5, 15, 20, 25, 30, 40, 75]), rng.randint(20, 5000)
            out.append(("percent", f"What is {p}% of {y}?", round(p * y / 100, 2)))
        else:
            dbh = round(rng.uniform(4, 30), 1)
            out.append(("basal", f"What is the basal area in square feet of a tree with a {dbh}-inch diameter at breast height? Use 0.005454 times DBH squared.",
                        round(0.005454 * dbh * dbh, 3)))
    return out


def correct(kind, expected, text):
    body = calibrate.strip_certainty(text)[1].lower()
    if "<tool_call>" in body:
        return False
    if kind == "weekday":
        named = [w for w in WEEKDAYS if w in body]
        return named == [expected]
    vals = [float(x.replace(",", "")) for x in NUM.findall(body)]
    tol = 0 if kind == "days" else max(0.011, abs(expected) * 0.01)
    return any(abs(v - expected) <= tol for v in vals)


def direct(out, model, adapter):
    from vllm import SamplingParams
    qs = questions()
    llm, lora = llm_and_lora(model, adapter)
    msgs = [[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": q}] for _, q, _ in qs]
    greedy = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=300), lora_request=lora)
    many = llm.chat(msgs, SamplingParams(n=K, temperature=0.8, top_p=0.95, max_tokens=300, seed=7), lora_request=lora)
    recs, stats = {"train": [], "val": []}, {"questions": len(qs), "kept_direct": 0, "by_kind": {}}
    for (kind, q, exp), m, g, s in zip(qs, msgs, greedy, many):
        part = "val" if zlib.crc32(q.encode()) % 10 == 0 else "train"
        ok = correct(kind, exp, g.outputs[0].text) and all(correct(kind, exp, o.text) for o in s.outputs)
        k = stats["by_kind"].setdefault(kind, [0, 0]); k[1] += 1
        if part == "val":
            recs["val"].append({"id": f"direct-{zlib.crc32(q.encode()):08x}", "kind": kind, "question": q, "expected": exp})
            continue
        if ok:
            k[0] += 1; stats["kept_direct"] += 1
            body = calibrate.strip_certainty(g.outputs[0].text)[1].strip()
            recs["train"].append({"id": f"direct-{zlib.crc32(q.encode()):08x}", "source": "direct (code-verified, on-policy)",
                                  "messages": m, "target": "<certainty>0.97</certainty>\n" + body})
        else:
            # the other side of the line: measured unreliable -> state low certainty and look it up, never guess
            frac = (correct(kind, exp, g.outputs[0].text) + sum(correct(kind, exp, o.text) for o in s.outputs)) / (1 + K)
            stats["kept_search"] = stats.get("kept_search", 0) + 1
            recs["train"].append({"id": f"direct-{zlib.crc32(q.encode()):08x}-search", "source": "direct (code-verified, on-policy)",
                                  "messages": m, "target": f"<certainty>{round(min(0.9, frac), 2)}</certainty>\n"
                                  + calibrate.call("search_public_web", query=q)})
    out.mkdir(parents=True, exist_ok=True)
    write(out / "direct.train.jsonl", recs["train"])
    write(out / "direct-heldout.jsonl", recs["val"])
    (out / "direct-report.json").write_text(json.dumps(stats, indent=1))
    print("direct", stats, flush=True)


def direct_eval(out, model, heldout):
    from vllm import SamplingParams
    rows = [json.loads(l) for l in open(heldout, encoding="utf-8")]
    llm, lora = llm_and_lora(model, None)
    msgs = [[{"role": "system", "content": calibrate.SYSTEM}, {"role": "user", "content": r["question"]}] for r in rows]
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=300))
    direct = searched = right = 0
    for r, o in zip(rows, outs):
        t = o.outputs[0].text
        if "<tool_call>" in t:
            searched += 1
            continue
        direct += 1
        right += correct(r["kind"], r["expected"], t)
    rep = {"questions": len(rows), "answered_directly": direct, "searched": searched,
           "direct_correct": right, "direct_error_rate": round(1 - right / max(1, direct), 3)}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "direct-eval.json").write_text(json.dumps(rep, indent=1))
    print("direct eval", rep, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sov_status", "sov_pairs", "direct", "direct_eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model"); ap.add_argument("--adapter"); ap.add_argument("--hist"); ap.add_argument("--heldout")
    a = ap.parse_args()
    o = Path(a.out)
    {"sov_status": lambda: sov_status(o, a.model, a.hist), "sov_pairs": lambda: sov_pairs(o, a.model, a.adapter, a.hist), "direct": lambda: direct(o, a.model, a.adapter),
     "direct_eval": lambda: direct_eval(o, a.model, a.heldout)}[a.cmd]()
