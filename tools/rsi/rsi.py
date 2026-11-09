"""Recursive self-improvement (RSI) loops for Atlantic Lite on verifiable math and code. Runs on a pod with vLLM;
`smoke` and the unit tests run on a laptop CPU. See docs/rsi-lite-plan.md.

One round: the current model answers each training problem k times, the answers are labeled, the labeled answers
become SFT targets (correct ones) and DPO pairs (correct over incorrect, same problem), train_vl.py trains a new adapter,
and the new model is scored on sealed sets it never saw. The next round samples from that new model.

  python rsi.py problems --pub P --out W [--math 4000] [--gsm 500] [--dev 300]
  python rsi.py sample   --model M --out W/round-1 --problems W/problems.jsonl [--k 8]
  python rsi.py label    --out W/round-1 --mode verified|majority|judge|self [--model M]
  python rsi.py rationalize --model M --out W/round-1 --problems W/problems.jsonl     (STaR hint pass, verified arm only)
  python rsi.py build    --out W/round-1 [--per-problem 2] [--easy-frac 0.2]
  python rsi.py eval     --model M --pub P --dev W/dev.jsonl --out W/eval-round-1.json [--k 8]
  python rsi.py compare  W/eval-round-0.json W/eval-round-1.json
  python rsi.py smoke    --out DIR            (CPU: fake samples through label, build and compare)

Label modes (what the training signal comes from; the verifier is always computed, for measurement only otherwise):
  verified  math answer equals the reference / code passes every unit test        (arm A, external check)
  majority  math: the most common final answer among the k samples, when it has at least half the votes (TTRL);
            code: passes the one test shown in the prompt                          (arm B, self-labeled)
  judge     the policy grades each of its own answers YES/NO                       (arm B, self-judged)
  self      majority AND judge for math, shown test AND judge for code             (arm B, the default self arm)
  random    the verified labels shuffled within each problem: same counts, no information (control arm; Qwen-family
            models have gained on math from spurious rewards, Shao et al. 2025, so a gain must beat this)

Every training target is written by the policy model itself. Prompts wrap public problems (MIT / CC BY 4.0) in one
fixed instruction line; no text from Claude or any other model becomes a target.
"""
from __future__ import annotations

import argparse, json, math, random, re, sys, zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "tools" / "vl", ROOT / "tools" / "v7", ROOT / "tools" / "rsi"):
    sys.path.insert(0, str(p))
import verify  # noqa: E402
from krummholz_llm import loops  # noqa: E402

MATH_PROMPT = "{q}\n\nWork through it step by step, then give the final answer as \\boxed{{...}} at the end."
CODE_PROMPT = "{q}\nYour code should pass this test:\n{t}\n\nReply with the complete Python code in one ```python block."
HINT = "\n\n(The correct final answer is {a}. Show the reasoning that reaches it, without mentioning that you were told.)"
JUDGE_PROMPT = """Check this solution carefully. Is its final answer correct (for code: does it solve the task and pass
all reasonable tests)?

Problem:
{q}

Proposed solution:
{s}

Reply with one word: YES or NO."""
SEALED_SETS = {"gsm8k_test": "math", "math500": "math", "mbpp_test": "code", "humanevalplus": "code"}


# ---------------------------------------------------------------- io

def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(path, rr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rr), encoding="utf-8")


def qhash(text):  # same as tools/v7/fetch_public.py, so SEALED.json hashes match
    import hashlib
    return hashlib.sha256(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def crc(s, mod):
    return zlib.crc32(s.encode()) % mod


# ---------------------------------------------------------------- problems

def prompt_of(p):
    return MATH_PROMPT.format(q=p["question"]) if p["kind"] == "math" else CODE_PROMPT.format(q=p["question"], t=p["tests"][0])


def pool_from(pub: Path, n_math, n_gsm, seed=5):
    """Training problems: MATH levels 3 to 5 and DeepMath (harder; v7 GRPO on mostly GSM8K showed no gain, likely
    saturated), a little GSM8K, all of MBPP train. Sealed eval questions are excluded by hash."""
    sealed = set()
    sp = pub / "SEALED.json"
    if sp.exists():
        sealed = {h for v in json.loads(sp.read_text()).values() for h in v}
    for name in SEALED_SETS:
        for r in rows(pub / f"{name}.jsonl"):
            sealed.add(qhash(r["question"]))
    hard, gsm, code = [], [], []
    for r in rows(pub / "hendrycks_math_train.jsonl"):
        lvl = int(re.sub(r"\D", "", r.get("level", "")) or 0)
        ans = verify.extract(r.get("solution", ""))
        if lvl >= 3 and ans:
            hard.append({"src": "math", "question": r["question"], "answer": ans, "level": lvl})
    for r in rows(pub / "deepmath.jsonl"):
        if r.get("answer") not in (None, "") and (r.get("difficulty") or 0) <= 7:
            hard.append({"src": "deepmath", "question": r["question"], "answer": str(r["answer"]), "level": r.get("difficulty")})
    for r in rows(pub / "gsm8k_train.jsonl"):
        gsm.append({"src": "gsm8k", "question": r["question"], "answer": str(r["answer"])})
    for r in rows(pub / "mbpp_train.jsonl"):
        if r.get("tests"):
            code.append({"src": "mbpp", "question": r["question"], "tests": r["tests"]})
    rng = random.Random(seed)
    out = rng.sample(hard, min(n_math, len(hard))) + rng.sample(gsm, min(n_gsm, len(gsm)))
    out = [{**r, "kind": "math"} for r in out] + [{**r, "kind": "code"} for r in code]
    seen, res = set(), []
    for r in out:
        h = qhash(r["question"])
        if h in sealed or h in seen:
            continue
        seen.add(h)
        res.append({**r, "id": f"{r['src']}-{h[:12]}"})
    return res


def problems(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allp = pool_from(Path(a.pub), a.math, a.gsm)
    dev = [p for p in allp if crc(p["id"], 1000) < a.dev_permille]  # held out from training in every round
    train = [p for p in allp if crc(p["id"], 1000) >= a.dev_permille]
    write(out / "problems.jsonl", train); write(out / "dev.jsonl", dev)
    print("PROBLEMS", len(train), Counter(p["src"] for p in train), "dev", len(dev), flush=True)


# ---------------------------------------------------------------- sampling (GPU)

def _llm(model):
    import teacher  # noqa: F401  (patches vLLM chat: thinking off)
    from vllm import LLM
    return LLM(model, max_model_len=4096, gpu_memory_utilization=0.85, seed=42)


def generate(llm, prompts, k, temp, max_tokens=1024, seed=42):
    from vllm import SamplingParams
    sp = SamplingParams(n=k, temperature=temp, top_p=0.95 if temp > 0 else 1.0, max_tokens=max_tokens, seed=seed)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
    return [[(c.text, c.finish_reason) for c in o.outputs] for o in outs]


def candidates(probs, gens, tag=""):
    res = []
    for p, g in zip(probs, gens):
        for j, (text, fin) in enumerate(g):
            res.append({"id": f"{p['id']}-{tag}s{j}", "pid": p["id"], "kind": p["kind"], "prompt": prompt_of(p),
                        "reply": text.strip(), "finish": fin, "hint": bool(tag)})
    return res


def check(c, p):
    """Ground truth for one candidate: the external verifier, plus the visible-test result for code."""
    if p["kind"] == "math":
        return {"verified": verify.math_ok(c["reply"], p["answer"]), "answer": verify.norm(verify.extract(c["reply"]))}
    code = verify.code_block(c["reply"])
    shown, _ = verify.run_tests(code, p["tests"][:1])
    ok, why = verify.run_tests(code, p["tests"]) if shown else (False, "shown test failed")
    return {"verified": ok, "shown_ok": shown, "why": why}


def sample(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probs = rows(a.problems)
    if a.limit:
        probs = random.Random(3).sample(probs, min(a.limit, len(probs)))
    llm = _llm(a.model)
    res = candidates(probs, generate(llm, [prompt_of(p) for p in probs], a.k, a.temp))
    byid = {p["id"]: p for p in probs}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        for c, v in zip(res, ex.map(lambda c: check(c, byid[c["pid"]]), res)):
            c.update(v)
    write(out / "samples.jsonl", res)
    print("SAMPLES", len(res), "problems", len(probs), "verified", round(sum(c["verified"] for c in res) / max(1, len(res)), 3), flush=True)


def rationalize(a):
    """STaR's rationalization: problems with no verified answer get k answers with the reference as a hint; correct
    ones are trained on the plain prompt. Uses the reference, so only the verified arm may run it."""
    out = Path(a.out); res = rows(out / "samples.jsonl")
    solved = {c["pid"] for c in res if c["verified"]}
    probs = [p for p in rows(a.problems) if p["kind"] == "math" and p["id"] in {c["pid"] for c in res} - solved]
    if not probs:
        print("RATIONALIZE none", flush=True); return
    llm = _llm(a.model)
    gens = generate(llm, [prompt_of(p) + HINT.format(a=p["answer"]) for p in probs], 2, 0.7)
    new = candidates(probs, gens, tag="h")
    byid = {p["id"]: p for p in probs}
    for c in new:
        c.update(check(c, byid[c["pid"]]))
        c["verified"] = c["verified"] and not re.search(r"\b(?:told|given|hint|provided answer)\b", c["reply"], re.I)
    write(out / "samples.jsonl", res + new)
    print("RATIONALIZE problems", len(probs), "kept", sum(c["verified"] for c in new), flush=True)


def self_judge(llm, res):
    """P(YES) from the policy's own first-token distribution when asked whether its answer is correct."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20, seed=42)
    outs = llm.chat([[{"role": "user", "content": JUDGE_PROMPT.format(q=c["prompt"], s=c["reply"][:6000])}] for c in res], sp)
    for c, o in zip(res, outs):
        c["p_yes"] = p_yes(o.outputs[0])


def p_yes(out):
    lp = (out.logprobs or [{}])[0] if out.logprobs else {}
    y = n = 0.0
    for v in lp.values():
        t = (v.decoded_token or "").strip().upper()
        y += math.exp(v.logprob) if t.startswith("YES") else 0.0
        n += math.exp(v.logprob) if t.startswith("NO") else 0.0
    if y + n == 0:  # fall back to the text
        return 1.0 if out.text.strip().upper().startswith("YES") else 0.0
    return y / (y + n)


# ---------------------------------------------------------------- labeling (pure)

def majority(cands, min_share=0.5):
    """The pseudo-label for a math problem: the most common normalized answer, or None when it lacks min_share."""
    votes = Counter(c["answer"] for c in cands if c.get("answer"))
    if not votes:
        return None
    ans, n = votes.most_common(1)[0]
    return ans if n / len(cands) >= min_share else None


def label_rows(res, mode, judge_threshold=0.5):
    """Set c['label'] (True, False, or None = unusable) from the chosen signal. Never reads the verifier unless
    mode == 'verified'."""
    by = {}
    for c in res:
        by.setdefault(c["pid"], []).append(c)
    for pid, cands in by.items():
        maj = majority([c for c in cands if not c.get("hint")]) if cands[0]["kind"] == "math" else None
        for c in cands:
            if mode == "verified":
                c["label"] = bool(c["verified"])
                continue
            if mode == "random":
                continue
            if c.get("hint"):
                c["label"] = None; continue
            if c["kind"] == "math":
                vote = None if maj is None else (c.get("answer") == maj)
            else:
                vote = bool(c.get("shown_ok"))
            judged = None if c.get("p_yes") is None else c["p_yes"] >= judge_threshold
            if mode == "majority":
                c["label"] = vote
            elif mode == "judge":
                c["label"] = judged
            else:  # self: both self-signals must agree; disagreement is dropped, not guessed
                c["label"] = None if vote is None or judged is None else (vote if vote == judged else None)
    if mode == "random":
        rng = random.Random(17)
        for cands in by.values():
            labs = [bool(c["verified"]) for c in cands]
            rng.shuffle(labs)
            for c, l in zip(cands, labs):
                c["label"] = l
    return res


def gap(res):
    """How good the self-signal is, measured against the verifier: precision and recall of label == True. This is the
    generation-verification gap the literature says decides whether a self-loop can climb."""
    lab = [c for c in res if c.get("label") is not None and not c.get("hint")]
    tp = sum(c["label"] and c["verified"] for c in lab)
    pos = sum(bool(c["label"]) for c in lab)
    real = sum(bool(c["verified"]) for c in res if not c.get("hint"))
    return {"labeled": len(lab), "of": sum(not c.get("hint") for c in res), "precision": round(tp / pos, 3) if pos else None,
            "recall": round(tp / real, 3) if real else None}


def label(a):
    out = Path(a.out); res = rows(out / "samples.jsonl")
    if a.mode in ("judge", "self") and any(c.get("p_yes") is None for c in res):
        self_judge(_llm(a.model), res)
    label_rows(res, a.mode, a.judge_threshold)
    write(out / "labeled.jsonl", res)
    g = gap(res); g["mode"] = a.mode
    (out / "label-report.json").write_text(json.dumps(g, indent=1))
    print("LABEL", json.dumps(g), flush=True)


# ---------------------------------------------------------------- build (pure)

def usable(c):
    return c["finish"] == "stop" and len(c["reply"]) > 20 and loops.detect(c["reply"]) is None


def build_rows(res, per_problem=2, easy_frac=0.2, seed=13, round_tag="r1"):
    """SFT targets from positive answers and DPO pairs (positive over negative, same problem). Problems the policy
    always gets right (by its own labels) contribute rarely; the learning signal lives where it is sometimes right."""
    rng = random.Random(seed)
    by = {}
    for c in res:
        by.setdefault(c["pid"], []).append(c)
    sft, pairs, stats = [], [], Counter()
    for pid in sorted(by):
        cands = by[pid]
        plain = [c for c in cands if not c.get("hint")]
        pos = [c for c in cands if c.get("label") is True and usable(c)]
        neg = [c for c in plain if c.get("label") is False]
        if not pos:
            stats["no_positive"] += 1; continue
        all_pos = all(c.get("label") is True for c in plain)
        if all_pos:
            stats["easy"] += 1
            if rng.random() >= easy_frac:
                continue
        stats["used"] += 1
        uniq = list({re.sub(r"\s+", " ", c["reply"]): c for c in pos}.values())
        rng.shuffle(uniq)
        for c in uniq[: 1 if all_pos else per_problem]:
            text = c["reply"] if c["kind"] == "code" else c["reply"].replace(chr(0x2014), ", ")
            sft.append({"id": f"{c['id']}-{round_tag}", "source": f"rsi-{round_tag}-{c['kind']}",
                        "messages": [{"role": "user", "content": c["prompt"]}], "target": text})
        if neg:
            ch, rj = rng.choice(uniq), rng.choice(neg)
            pairs.append({"id": f"{pid}-{round_tag}-pair", "source": f"rsi-{round_tag}-{ch['kind']}",
                          "messages": [{"role": "user", "content": ch["prompt"]}], "chosen": ch["reply"], "rejected": rj["reply"]})
    stats.update(sft=len(sft), pairs=len(pairs), problems=len(by))
    if sft:
        stats["sft_mean_chars"] = round(sum(len(r["target"]) for r in sft) / len(sft))
    return sft, pairs, dict(stats)


def build(a):
    out = Path(a.out); res = rows(out / "labeled.jsonl")
    tag = a.tag or out.name.replace("round-", "r")
    sft, pairs, st = build_rows(res, a.per_problem, a.easy_frac, round_tag=tag)
    held = lambda r: crc(r["id"], 10) == 0
    d = out / "data"
    for name, rr in (("rsi_sft", sft), ("rsi_pairs", pairs)):
        write(d / f"{name}.train.jsonl", [r for r in rr if not held(r)])
        write(d / f"{name}.val.jsonl", [r for r in rr if held(r)])
    st["diversity"] = diversity(res)
    (out / "build-report.json").write_text(json.dumps(st, indent=1))
    print("BUILD", json.dumps(st), flush=True)


def diversity(res):
    """Collapse watch: distinct final answers per math problem and distinct replies overall, among plain samples."""
    by = {}
    for c in res:
        if not c.get("hint"):
            by.setdefault(c["pid"], []).append(c)
    m = [len({c.get("answer") for c in v}) / len(v) for v in by.values() if v[0]["kind"] == "math"]
    allr = [c["reply"] for v in by.values() for c in v]
    return {"math_distinct_answer_ratio": round(sum(m) / len(m), 3) if m else None,
            "distinct_reply_ratio": round(len(set(allr)) / len(allr), 3) if allr else None,
            "mean_chars": round(sum(map(len, allr)) / len(allr)) if allr else None}


# ---------------------------------------------------------------- eval

def sealed(pub: Path, dev_path=None, limit=None):
    items = []
    for name, kind in SEALED_SETS.items():
        rr = rows(pub / f"{name}.jsonl")[: limit or None]
        for j, r in enumerate(rr):
            if kind == "math":
                items.append({"set": name, "id": f"{name}-{j}", "kind": "math", "question": r["question"], "answer": str(r["answer"])})
            elif name == "humanevalplus":
                items.append({"set": name, "id": f"{name}-{j}", "kind": "code", "question": "Complete this Python function:\n" + r["question"],
                              "tests": [verify.humaneval_tests(r["question"], r["test"])], "prefix": r["question"]})
            else:
                items.append({"set": name, "id": f"{name}-{j}", "kind": "code", "question": r["question"], "tests": r["tests"]})
    for r in rows(dev_path)[: limit or None] if dev_path else []:
        items.append({**r, "set": "dev"})
    return items


def eval_prompt(p):
    if p["set"] == "humanevalplus":
        return p["question"] + "\n\nReply with the complete function, including the signature, in one ```python block."
    return prompt_of(p)


def score(p, text):
    if p["kind"] == "math":
        return verify.math_ok(text, p["answer"])
    code = verify.code_block(text)
    if p.get("prefix") and "def " not in code:  # body only: put the signature back
        code = p["prefix"] + code
    return verify.run_tests(code, p["tests"])[0]


def evaluate(a):
    items = sealed(Path(a.pub), a.dev, a.limit)
    llm = _llm(a.model)
    greedy = generate(llm, [eval_prompt(p) for p in items], 1, 0.0)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda pg: score(pg[0], pg[1][0][0]), zip(items, greedy)))
    res = {"model": a.model, "items": {p["id"]: bool(o) for p, o in zip(items, ok)}, "sets": {}}
    for s in sorted({p["set"] for p in items}):
        v = [o for p, o in zip(items, ok) if p["set"] == s]
        res["sets"][s] = {"n": len(v), "pass@1": round(sum(v) / len(v), 4)}
        m = [g[0][0] for p, g in zip(items, greedy) if p["set"] == s and p["kind"] == "math"]
        if m:  # format watch: a round-1 jump can be boxed-answer compliance rather than skill
            res["sets"][s]["no_boxed"] = round(sum(r"\boxed" not in t for t in m) / len(m), 4)
    if a.k > 1:  # reasoning boundary: does the loop raise pass@k, or only sharpen pass@1 (Yue et al. 2025)?
        sub = [p for p in items if p["set"] in ("math500", "dev")]
        gens = generate(llm, [eval_prompt(p) for p in sub], a.k, 0.8, seed=7)
        for s in ("math500", "dev"):
            hit = [any(score(p, t) for t, _ in g) for p, g in zip(sub, gens) if p["set"] == s]
            if hit:
                res["sets"][s][f"pass@{a.k}"] = round(sum(hit) / len(hit), 4)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("EVAL", json.dumps(res["sets"]), flush=True)


def binom_two_sided(b, c):
    """Exact McNemar p-value: discordant pairs b (only A right) and c (only B right) under p = 0.5."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare_results(a, b, sets=None):
    out = {}
    groups = {s: [i for i in a["items"] if i.startswith(s + "-") or (s == "dev" and not any(i.startswith(t + "-") for t in SEALED_SETS))]
              for s in (sets or list(SEALED_SETS) + ["dev"])}
    groups["sealed_all"] = [i for s in SEALED_SETS for i in groups.get(s, [])]
    for s, ids in groups.items():
        ids = [i for i in ids if i in b["items"]]
        if not ids:
            continue
        lose = sum(a["items"][i] and not b["items"][i] for i in ids)
        gain = sum(b["items"][i] and not a["items"][i] for i in ids)
        out[s] = {"n": len(ids), "before": round(sum(a["items"][i] for i in ids) / len(ids), 4),
                  "after": round(sum(b["items"][i] for i in ids) / len(ids), 4), "gained": gain, "lost": lose,
                  "p": round(binom_two_sided(lose, gain), 4)}
    return out


def compare(a):
    res = compare_results(json.loads(Path(a.before).read_text()), json.loads(Path(a.after).read_text()))
    print(json.dumps(res, indent=1))
    s = res.get("sealed_all", {})
    verdict = "GAIN" if s and s["after"] > s["before"] and s["p"] < 0.05 else "LOSS" if s and s["after"] < s["before"] and s["p"] < 0.05 else "FLAT"
    print("VERDICT", verdict, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"verdict": verdict, **res}, indent=1))


# ---------------------------------------------------------------- CPU smoke

def smoke(a):
    """The data path without a GPU: fake policy samples (some right, some wrong, one broken) through every label mode,
    build, and a compare. Catches format and plumbing errors before a pod is rented."""
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    probs = [{"id": "m1", "kind": "math", "question": "What is 2+3?", "answer": "5"},
             {"id": "m2", "kind": "math", "question": "What is 6*7?", "answer": "42"},
             {"id": "m3", "kind": "math", "question": "What is 10/4?", "answer": "2.5"},
             {"id": "c1", "kind": "code", "question": "Write a function add(a, b) that returns a + b.",
              "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]}]
    fake = {"m1": ["2 plus 3 makes \\boxed{5}.", "Adding gives \\boxed{5}.", "So the result is \\boxed{6}.", "Total: \\boxed{5}."],
            "m2": ["Six sevens: \\boxed{42}.", "Multiplying: \\boxed{48}.", "It is \\boxed{42}.", "Answer \\boxed{36}."],
            "m3": ["Divide: \\boxed{2.50}.", "Long division gives \\boxed{2.5}.", "So \\boxed{2}.", "Half of five: \\boxed{2.5}."],
            "c1": ["```python\ndef add(a, b):\n    return a + b\n```", "```python\ndef add(a, b):\n    return a - b\n```",
                   "```python\ndef add(a, b):\n    return abs(a) + b\n```", "```python\ndef add(a, b):\n    while True: pass\n```"]}
    res = candidates(probs, [[(t, "stop") for t in fake[p["id"]]] for p in probs])
    byid = {p["id"]: p for p in probs}
    for c in res:
        c.update(check(c, byid[c["pid"]]))
        c["p_yes"] = 0.9 if c["verified"] else (0.6 if c["id"].endswith("s2") else 0.1)  # a judge that is sometimes fooled
    report = {}
    for mode in ("verified", "majority", "judge", "self"):
        lab = label_rows([dict(c) for c in res], mode)
        sft, pairs, st = build_rows(lab, easy_frac=1.0, round_tag="smoke")
        report[mode] = {"gap": gap(lab), "sft": len(sft), "pairs": len(pairs)}
        if mode == "verified":
            write(out / "data" / "rsi_sft.train.jsonl", sft); write(out / "data" / "rsi_pairs.train.jsonl", pairs)
            assert all(verify.math_ok(r["target"], byid[r["id"].split("-")[0]]["answer"]) for r in sft if r["source"].endswith("math"))
    before = {"items": {"math500-0": True, "math500-1": False, "math500-2": False, "mbpp_test-0": True}}
    after = {"items": {"math500-0": True, "math500-1": True, "math500-2": True, "mbpp_test-0": True}}
    report["compare"] = compare_results(before, after)["sealed_all"]
    report["diversity"] = diversity(res)
    (out / "smoke-report.json").write_text(json.dumps(report, indent=1))
    print("RSI_SMOKE_OK", json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("problems"); s.add_argument("--pub", required=True); s.add_argument("--out", required=True)
    s.add_argument("--math", type=int, default=4000); s.add_argument("--gsm", type=int, default=500)
    s.add_argument("--dev-permille", type=int, default=60)
    s = sub.add_parser("sample"); s.add_argument("--model", required=True); s.add_argument("--out", required=True)
    s.add_argument("--problems", required=True); s.add_argument("--k", type=int, default=8); s.add_argument("--temp", type=float, default=0.8)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("rationalize"); s.add_argument("--model", required=True); s.add_argument("--out", required=True)
    s.add_argument("--problems", required=True)
    s = sub.add_parser("label"); s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=["verified", "majority", "judge", "self", "random"], required=True)
    s.add_argument("--model"); s.add_argument("--judge-threshold", type=float, default=0.5)
    s = sub.add_parser("build"); s.add_argument("--out", required=True); s.add_argument("--tag")
    s.add_argument("--per-problem", type=int, default=2); s.add_argument("--easy-frac", type=float, default=0.2)
    s = sub.add_parser("eval"); s.add_argument("--model", required=True); s.add_argument("--pub", required=True)
    s.add_argument("--dev"); s.add_argument("--out", required=True); s.add_argument("--k", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("compare"); s.add_argument("before"); s.add_argument("after"); s.add_argument("--out")
    s = sub.add_parser("smoke"); s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"problems": problems, "sample": sample, "rationalize": rationalize, "label": label, "build": build,
     "eval": evaluate, "compare": compare, "smoke": smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
