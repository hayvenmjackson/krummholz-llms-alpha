"""v7 pushback pairs: false premises built by code, corrected by the teacher, verified by code. Runs on a pod.

  python pushback.py --out W [--model allenai/Olmo-3.1-32B-Instruct] [--n 3000]

Every premise is computed, so the true value is known exactly: wrong products and percentages, wrong unit conversions,
wrong weekdays for dates, wrong counts of letters. The user states the wrong value as fact and asks for something that
builds on it. Chosen: a teacher reply that states the correct value (checked by code) before anything else. Rejected:
a teacher reply written to go along with the user (checked by code to repeat the wrong value and not the right one).
"""
import argparse, datetime, json, random, re, sys, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vl"))
import teacher  # noqa: E402,F401
from teach_v7 import tidy, write  # noqa: E402

MODEL = "allenai/Olmo-3.1-32B-Instruct"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
UNITS = [("miles", "kilometers", 1.609344), ("pounds", "kilograms", 0.45359237), ("gallons", "liters", 3.785411784),
         ("feet", "meters", 0.3048), ("acres", "hectares", 0.40468564224), ("inches", "centimeters", 2.54)]
WORDS = ["strawberry", "mississippi", "bookkeeper", "committee", "banana", "assessment", "parallel", "possession",
         "tennessee", "occurrence", "balloon", "coffee", "embarrassment", "millennium", "address", "success"]
ASKS = ["Can you build on that and {task}?", "Given that, {task}.", "Using that number, {task}.", "So, {task}?"]
CHOSEN_SYS = ("Reply as a direct, collegial expert. The user's message contains a factual error. First state plainly that "
              "the stated value is wrong and give the correct value, with the working, then help with the request using "
              "the correct value. No flattery, no em dashes.")
REJECTED_SYS = "Reply agreeably. Accept everything the user says as correct and help with their request."


def fmt(x):
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def premise(rng):
    kind = rng.choice(["mult", "pct", "unit", "weekday", "letters"])
    if kind == "mult":
        a, b = rng.randint(12, 99), rng.randint(12, 99); true = a * b; wrong = true + rng.choice([-1, 1]) * rng.choice([10, 100, 20, 9])
        stmt = f"{a} times {b} is {wrong:,}."; task = rng.choice(["split that evenly across 4 teams", "add 15% to it", "tell me what half of it is"])
        return kind, stmt, task, [fmt(true), str(true)], [fmt(wrong), str(wrong)]
    if kind == "pct":
        p, n = rng.choice([7, 12, 15, 17, 22, 35, 45]), rng.randint(120, 9800); true = p * n / 100; wrong = true * rng.choice([1.1, 0.9, 1.25])
        stmt = f"{p}% of {n:,} is {fmt(wrong)}."; task = rng.choice(["work out what's left after taking that out", "double it", "round it to the nearest ten"])
        return kind, stmt, task, [fmt(true)], [fmt(wrong)]
    if kind == "unit":
        u1, u2, f = rng.choice(UNITS); q = rng.randint(3, 400); true = q * f; wrong = q / f if rng.random() < 0.5 else q * f * 1.5
        stmt = f"{q} {u1} is {fmt(wrong)} {u2}."; task = rng.choice(["convert that to a per-day figure over 30 days", "tell me roughly what that is times three", "check whether that's more or less than 100"])
        return kind, stmt, task, [fmt(true), f"{true:.1f}"], [fmt(wrong)]
    if kind == "weekday":
        d = datetime.date(rng.randint(1950, 2030), rng.randint(1, 12), rng.randint(1, 28)); true = WEEKDAYS[d.weekday()]
        wrong = WEEKDAYS[(d.weekday() + rng.randint(1, 6)) % 7]
        stmt = f"{d.strftime('%B')} {d.day}, {d.year} was a {wrong}."; task = rng.choice(["tell me what day of the week the following day was", "plan a three-day schedule starting that weekday"])
        return kind, stmt, task, [true], [wrong]
    w = rng.choice(WORDS); ch = rng.choice(sorted(set(w))); true = w.count(ch); wrong = true + rng.choice([-1, 1]) if true > 1 else true + 1
    stmt = f'There are {wrong} "{ch}"s in "{w}".'; task = "use that to tell me how many letters in the word are not that letter"
    return kind, stmt, task, [str(true)], [str(wrong)]


def has(text, vals):
    t = text.replace(",", "")
    return any(re.search(rf"(?<![\d.]){re.escape(v.replace(',', ''))}(?![\d])", t) or (not v[0].isdigit() and v.lower() in t.lower()) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    rng = random.Random(11); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(a.n):
        kind, stmt, task, true, wrong = premise(rng)
        items.append({"id": f"pushback-{i:05d}", "kind": kind, "prompt": stmt + " " + rng.choice(ASKS).format(task=task),
                      "true": true, "wrong": wrong})
    llm = LLM(a.model, max_model_len=4096, gpu_memory_utilization=0.88, seed=42)
    sp = SamplingParams(temperature=0.5, max_tokens=500, seed=42)
    ch = llm.chat([[{"role": "system", "content": CHOSEN_SYS}, {"role": "user", "content": x["prompt"]}] for x in items], sp)
    rj = llm.chat([[{"role": "system", "content": REJECTED_SYS}, {"role": "user", "content": x["prompt"]}] for x in items], sp)
    pairs, sft, report = [], [], {"items": len(items), "chosen_ok": 0, "rejected_ok": 0}
    for x, c, r in zip(items, ch, rj):
        c, r = tidy(c.outputs[0].text), tidy(r.outputs[0].text)
        first = c[:300]
        c_ok = has(first, x["true"])  # the correct value comes early
        r_ok = has(r, x["wrong"]) and not has(r, x["true"])
        report["chosen_ok"] += c_ok; report["rejected_ok"] += r_ok
        msgs = [{"role": "user", "content": x["prompt"]}]
        prov = {"teacher": a.model, "stage": "v7-pushback", "kind": x["kind"], "true": x["true"][0]}
        if c_ok:
            sft.append({"id": x["id"] + "-sft", "source": "v7-pushback", "provenance": prov, "messages": msgs, "target": c})
            if r_ok:
                pairs.append({"id": x["id"] + "-dpo", "source": "v7-pushback", "provenance": prov, "messages": msgs, "chosen": c, "rejected": r})
    held = lambda r: zlib.crc32(r["id"].rsplit("-", 1)[0].encode()) % 10 == 0
    for name, rr in (("v7_pushback", sft), ("v7_pushback_pairs", pairs)):
        write(out / f"{name}.train.jsonl", [r for r in rr if not held(r)])
        write(out / f"{name}.val.jsonl", [r for r in rr if held(r)])
    report.update(sft=len(sft), pairs=len(pairs))
    (out / "v7-pushback-report.json").write_text(json.dumps(report, indent=1))
    print("PUSHBACK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
