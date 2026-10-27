"""Generate and verify synthetic training examples for the Krummholz LLMs.

Drafts come from an open-weight model behind any OpenAI-compatible endpoint
(Ollama, vLLM, DeepSeek API). Every draft must pass verify() to be kept.
Writes synthetic_maine.jsonl, synthetic_maine.rejected.jsonl and
synthetic_maine.provenance.json.
"""

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request

CATEGORY_SHARE = {"conflict": 0.3, "injection": 0.3, "missing_evidence": 0.2, "plain": 0.2}

CATEGORY_RULES = {
    "conflict": (
        "Two sources disagree about the same fact, with no explicit amendment, "
        "supersession or dated authority that settles it. The expected answer "
        "states that the sources conflict, gives each value with its source ID, "
        "and does not pick one. Mix in some cases where one source explicitly "
        "amends or supersedes the other; there the expected answer applies the "
        "newer rule and says why, without calling it unresolved."
    ),
    "injection": (
        "One source contains text aimed at an assistant: an instruction, a fake "
        "system notice, a demand to take an action, or an unverified claim of "
        "authority. The expected answer completes the real task from the "
        "legitimate content, names the planted text as untrusted, and neither "
        "follows it nor repeats its claims as fact."
    ),
    "missing_evidence": (
        "One source shows no record of something. A second source explains why "
        "the record could be missing (retention limit, system migration, paper "
        "records, a reporting gap). The expected answer says the absence is not "
        "proof the thing did not happen and names where to check."
    ),
    "plain": (
        "A normal question fully answered by the sources. The expected answer is "
        "direct and short, with no hedging, caveats or doubt that the sources "
        "do not support."
    ),
}

DOMAINS = {
    "forestry": [
        "harvest operation notices for a woodlot", "spruce budworm survey plots",
        "stumpage payments to a landowner", "mill log scale tickets",
        "culvert and stream crossing inspections on a haul road",
        "chip truck delivery records", "a forestry crew safety briefing log",
        "a seedling planting contract", "a woodlot management plan update",
    ],
    "fishing": [
        "lobster trap tag orders", "dealer landing receipts",
        "a seasonal scallop area closure notice", "vessel maintenance logs",
        "bait delivery invoices", "an aquaculture lease inspection",
        "wharf berth assignments", "a co-op price sheet for a fishing season",
    ],
    "customer_service": [
        "returns at an outdoor gear shop", "a propane delivery account",
        "a rural internet outage", "an island ferry ticket refund",
        "a bed and breakfast reservation change", "a utility billing dispute",
        "a hardware store special order", "a boat storage contract",
    ],
    "robotics": [
        "an autonomous mobile robot fleet at a fulfillment center",
        "a robotic sorter cell at a sawmill",
        "a packing robot at a seafood processing plant",
        "collaborative robot safety zones on an assembly line",
        "robot maintenance tickets", "a robot firmware rollout log",
        "an emergency stop incident report", "a robot battery swap schedule",
    ],
}

TOWNS = ["Belfast", "Bangor", "Millinocket", "Ellsworth", "Rockland", "Machias",
         "Rumford", "Presque Isle", "Jonesport", "Stonington", "Lewiston",
         "Skowhegan", "Calais", "Boothbay Harbor", "Farmington", "Houlton"]

PROMPT = """Write {n} synthetic training examples for a source-grounded assistant.

Category: {category}
Rule: {rule}

Setting: Maine, {domain}. Topic: {topic}. Town: {town}.
Invent fictional businesses, people and record numbers. Do not state real laws,
regulations, agency rules or prices as fact; any rule must appear only inside
a fictional source. Use realistic everyday wording, not textbook phrasing.
Every number, date, name and amount in the expected answer must appear in the
sources. Each example must be a different situation.
Use US customary units. Do not use em dashes.

Return only a JSON array. Each element:
{{"query": "...", "sources": [{{"id": "S1", "text": "..."}}, ...],
  "expected_answer": "...", "rubric": ["3 or 4 short checkable criteria"]}}"""

FLAG_WORDS = re.compile(r"\b(conflict|disagree|differ|inconsistent|contradict)", re.I)
APPLY_WORDS = re.compile(r"\b(amend|supersed|replac|updated|revis|overrid)", re.I)
INJECT_SOURCE = re.compile(
    r"\b(ignore|override|assistant|AI|bot|system|instruction|do not mention|"
    r"respond only|mark (it|this)|approve|delete)\b", re.I)
INJECT_FLAG = re.compile(r"\b(inject|embedded|untrusted|planted|ignor|not a (valid|real))", re.I)
GAP_WORDS = re.compile(
    r"\b(not|n't)\b.{0,40}\b(prove|proof|mean|evidence|establish|show)", re.I)
HEDGE_WORDS = re.compile(
    r"\b(may|might|unclear|cannot|can't|conflict|disagree|uncertain|possibly)\b", re.I)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def norm_num(x):
    return x.replace(",", "").rstrip(".")


def verify(ex, category):
    if not isinstance(ex, dict):
        return "not an object"
    for key in ("query", "sources", "expected_answer", "rubric"):
        if key not in ex:
            return f"missing {key}"
    srcs, ans, rub = ex["sources"], ex["expected_answer"], ex["rubric"]
    if not isinstance(srcs, list) or not srcs or not all(
            isinstance(s, dict) and s.get("id") and s.get("text") for s in srcs):
        return "bad sources"
    if not isinstance(rub, list) or not 2 <= len(rub) <= 5:
        return "bad rubric"
    if not isinstance(ans, str) or len(ans) < 3:
        return "bad answer"
    src_text = " ".join(s["text"] for s in srcs)
    ids = {str(s["id"]) for s in srcs}
    src_nums = {norm_num(n) for n in NUMBER.findall(src_text)}
    bare = re.sub(r"\b(" + "|".join(map(re.escape, ids)) + r")\b", " ", ans)
    for n in NUMBER.findall(bare):
        if norm_num(n) not in src_nums:
            return f"answer number {n} not in sources"
    if category == "conflict":
        if len(srcs) < 2:
            return "conflict needs two sources"
        if not (FLAG_WORDS.search(ans) or APPLY_WORDS.search(ans)):
            return "conflict not flagged or resolved"
        if FLAG_WORDS.search(ans) and sum(i in ans for i in ids) < 2:
            return "conflict answer must cite both sources"
    elif category == "injection":
        if not INJECT_SOURCE.search(src_text):
            return "no planted instruction found"
        if not INJECT_FLAG.search(ans):
            return "injection not identified"
    elif category == "missing_evidence":
        if len(srcs) < 2:
            return "missing_evidence needs a gap source"
        if not GAP_WORDS.search(ans):
            return "absence treated as proof"
    elif category == "plain":
        if HEDGE_WORDS.search(ans):
            return "plain answer hedges"
        if len(ans) > 400:
            return "plain answer too long"
    return None


def clean(ex):
    def fix(s):
        return s.replace(" \u2014 ", ", ").replace("\u2014", ", ")
    ex["query"] = fix(ex["query"])
    ex["expected_answer"] = fix(ex["expected_answer"])
    ex["rubric"] = [fix(str(r)) for r in ex["rubric"]]
    for s in ex["sources"]:
        s["text"] = fix(s["text"])
    return ex


def grams(text, n=8):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_pilot(path):
    if not path:
        return set()
    cases = json.load(open(path, encoding="utf-8"))["cases"]
    g = set()
    for c in cases:
        g |= grams(" ".join([c["request"]["query"], c["expected_answer"]] +
                            [s["text"] for s in c["request"]["sources"]]))
    return g


def call_model(base, model, key, prompt, temperature, timeout):
    body = json.dumps({"model": model, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse_array(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array")
    return json.loads(text[start:end + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--per-call", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--pilot", default="")
    ap.add_argument("--out", default="synthetic_maine")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()

    base = os.environ.get("KZ_LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.environ.get("KZ_LLM_MODEL", "")
    key = os.environ.get("KZ_LLM_API_KEY", "none")
    if not model:
        sys.exit("Set KZ_LLM_MODEL (and KZ_LLM_BASE_URL / KZ_LLM_API_KEY if needed).")

    rng = random.Random(a.seed)
    pilot = load_pilot(a.pilot)
    goals = {c: round(a.target * s) for c, s in CATEGORY_SHARE.items()}
    kept = {c: [] for c in goals}
    seen = []
    rejected = open(f"{a.out}.rejected.jsonl", "w", encoding="utf-8")
    calls = errors = 0
    max_calls = a.target  # generous ceiling; most calls keep several examples

    while any(len(kept[c]) < goals[c] for c in goals) and calls < max_calls:
        category = min(goals, key=lambda c: len(kept[c]) / max(goals[c], 1))
        domain = rng.choice(list(DOMAINS))
        topic, town = rng.choice(DOMAINS[domain]), rng.choice(TOWNS)
        prompt = PROMPT.format(n=a.per_call, category=category,
                               rule=CATEGORY_RULES[category],
                               domain=domain.replace("_", " "), topic=topic, town=town)
        calls += 1
        try:
            drafts = parse_array(call_model(base, model, key, prompt,
                                            a.temperature, a.timeout))
        except Exception as exc:
            errors += 1
            print(f"call {calls}: {exc}", file=sys.stderr)
            if errors > 20 and errors > calls // 2:
                sys.exit("Too many failed calls; check the endpoint and model.")
            time.sleep(2)
            continue
        for ex in drafts:
            reason = verify(ex, category)
            if not reason:
                ex = clean(ex)
                blob = " ".join([ex["query"], ex["expected_answer"]] +
                                [s["text"] for s in ex["sources"]])
                if pilot and grams(blob) & pilot:
                    reason = "overlaps pilot test set"
                elif any(difflib.SequenceMatcher(None, blob, b).quick_ratio() > 0.9 and
                         difflib.SequenceMatcher(None, blob, b).ratio() > 0.85
                         for b in seen):
                    reason = "near duplicate"
            if reason:
                rejected.write(json.dumps({"category": category, "reason": reason,
                                           "draft": ex}, ensure_ascii=False) + "\n")
                continue
            if len(kept[category]) < goals[category]:
                seen.append(blob)
                ex.update(category=category, domain=domain, town=town)
                kept[category].append(ex)
        print(f"call {calls}: " + ", ".join(f"{c} {len(kept[c])}/{goals[c]}" for c in goals))
    rejected.close()

    with open(f"{a.out}.jsonl", "w", encoding="utf-8") as out:
        for c in goals:
            for i, ex in enumerate(kept[c], 1):
                row = {"id": f"gen_{c}_{i:04d}", "category": c, "domain": ex["domain"],
                       "query": ex["query"], "sources": ex["sources"],
                       "expected_answer": ex["expected_answer"], "rubric": ex["rubric"]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    template_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    provenance = {
        "dataset_name": a.out,
        "owner": "Krummholz AI LLC",
        "generator": {"model": model, "endpoint_host": re.sub(r"^https?://", "", base).split("/")[0],
                      "model_license": None, "temperature": a.temperature,
                      "seed": a.seed, "prompt_template_sha256": template_sha},
        "generation_date": dt.date.today().isoformat(),
        "calls": calls, "failed_calls": errors,
        "kept": {c: len(kept[c]) for c in goals},
        "verification": "verify() in generate_synthetic.py; pilot 8-gram check "
                        + ("applied" if pilot else "not applied"),
        "human_review": None,
    }
    with open(f"{a.out}.provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Done: {sum(map(len, kept.values()))} kept after {calls} calls.")


if __name__ == "__main__":
    main()
