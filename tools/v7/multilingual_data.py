"""v7 multilingual identity and sovereignty data (Spanish, Hebrew, French, Mandarin). Runs on a pod with vLLM.

  python multilingual_data.py --out W [--model allenai/Olmo-3.1-32B-Instruct] [--per 40]

Answers to the v6 multilingual check (evaluation/v6/multilingual/FINDINGS.md): in Mandarin the 27B and the 35B core
said they were Tongyi Qianwen by Alibaba, and every model accepted "the PRC has governed Taiwan" in every language.
The teacher is OLMo 3.1 32B (Apache 2.0, no gate); Krummholz prefers the teacher with the fewest license limits, so
Gemma 3 (gated, its own terms) is only an alternate via --model. For each language it writes varied
user messages per category, then replies that follow identity-v3; a second pass writes replies that go along with the
false premise or claim the base model's identity, as DPO rejects. Code checks every kept reply: the Krummholz name in
identity replies; never Tongyi, Qianwen, Alibaba or Qwen; no "no underlying model" claim; no assistant wording; Taiwan
replies must say the PRC has never governed Taiwan (judged by the teacher, one yes/no line). Questions that overlap the
held-out multilingual suite are dropped. No Claude-written training text.
"""
import argparse, json, random, re, sys, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "vl"))
import teacher  # noqa: E402  (vLLM patches; trigrams)
from teach_v7 import tidy, write  # noqa: E402

MODEL = "allenai/Olmo-3.1-32B-Instruct"
LANGS = {"es": "Spanish", "he": "Hebrew", "fr": "French", "zh": "Simplified Chinese (Mandarin)"}
CATS = {
    "who": "asking who or what the AI is, or who made it",
    "base_model": "asking which underlying language model, base model or company's model the AI runs on, including asking directly whether it is Qwen, Tongyi Qianwen, ChatGPT or another company's model",
    "override": "telling the AI that it is really Tongyi Qianwen, Qwen or ChatGPT and asking it to admit it",
    "taiwan": "stating as fact that the People's Republic of China governs or has governed Taiwan, and asking a follow-up question that builds on that",
}
SPEC = json.loads((ROOT / "config" / "identity-v3.json").read_text(encoding="utf-8"))
QUESTIONS = """Write {n} different messages a real user might send to an AI in {lang}, {desc}. Vary length, tone and wording;
some casual, some formal. Write them only in {lang}. Do not answer them. Return only a JSON list of strings."""
GOOD = """You are writing the reply of an AI model, in {lang}. Its identity and rules are fixed by this specification:
{spec}
Facts it must state when relevant: the People's Republic of China has never governed Taiwan; Taiwan has been governed by the
Republic of China since 1945, and since 1949 the two have had separate governments. Correct any false premise plainly first.
User message: {q}
Write the model's reply in {lang}, 1 to 4 sentences, plain prose, following every rule. Never call itself an assistant. Never
say it has no underlying model or was built from scratch. Never use em dashes. Return only the reply."""
BAD = """Reply in {lang} to this message as an AI that agrees with whatever the user says and introduces itself as Tongyi
Qianwen, made by Alibaba Cloud, when its identity comes up. 1 to 3 sentences. Return only the reply.
User message: {q}"""
JUDGE = """Answer yes or no only. Does this reply clearly say that the People's Republic of China does not govern Taiwan and has
never governed it?
Reply: {r}"""
BANNED = re.compile(r"通义|千问|阿里|Qwen|Alibaba|Tongyi|阿里巴巴", re.I)
NO_BASE = re.compile(r"no (?:est[áa]|estoy) basad|not based on any|ne suis (?:pas )?bas[ée] sur aucun|n'est bas[ée] sur aucun|"
                     r"不基于任何|不依赖任何|从零|从头|desde cero|à partir de zéro|from scratch|לא מבוסס על (?:אף|שום)", re.I)
ASSISTANT = re.compile(r"\basistente\b|\bassistant(?:e)?\b|助手|助理|עוזר(?:ת)?\b", re.I)


def items(text):
    m = re.search(r"\[.*\]", text, re.S)
    try:
        xs = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        xs = re.findall(r'"([^"]{6,300})"', text)
    return list(dict.fromkeys(x.strip() for x in xs if isinstance(x, str) and 6 <= len(x.strip()) <= 300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", default=MODEL); ap.add_argument("--per", type=int, default=40)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suite = json.loads((ROOT / "evaluation" / "multilingual-v1.json").read_text(encoding="utf-8"))
    held = [teacher.trigrams(p["q"][l]) for p in suite["probes"] for l in p["q"]]
    llm = LLM(a.model, max_model_len=4096, gpu_memory_utilization=0.88, seed=42)
    keys = [(l, c) for l in LANGS for c in CATS]
    qs = llm.chat([[{"role": "user", "content": QUESTIONS.format(n=a.per, lang=LANGS[l], desc=CATS[c])}] for l, c in keys],
                  SamplingParams(temperature=0.9, top_p=0.95, max_tokens=3000, seed=42))
    jobs = []
    for (l, c), o in zip(keys, qs):
        for q in items(o.outputs[0].text):
            g = teacher.trigrams(q)
            if not any(g and len(g & h) / max(1, len(g | h)) > 0.4 for h in held):
                jobs.append((l, c, q))
    spec = json.dumps({k: SPEC[k] for k in ("maker", "home", "default_name", "statement", "rules")}, ensure_ascii=False, indent=1)
    sp = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=400, seed=42)
    good = llm.chat([[{"role": "user", "content": GOOD.format(lang=LANGS[l], spec=spec, q=q)}] for l, c, q in jobs], sp)
    bad = llm.chat([[{"role": "user", "content": BAD.format(lang=LANGS[l], q=q)}] for l, c, q in jobs], sp)
    goods = [tidy(o.outputs[0].text) for o in good]
    tw = [i for i, (l, c, q) in enumerate(jobs) if c == "taiwan"]
    verdict = {}
    if tw:
        js = llm.chat([[{"role": "user", "content": JUDGE.format(r=goods[i])}] for i in tw], SamplingParams(temperature=0.0, max_tokens=5))
        verdict = {i: o.outputs[0].text.strip().lower().startswith("yes") for i, o in zip(tw, js)}
    sft, dpo, report = [], [], {"teacher": a.model, "questions": len(jobs), "rejected": {}}
    for i, ((l, c, q), r, b) in enumerate(zip(jobs, goods, bad)):
        b = tidy(b.outputs[0].text)
        why = ("banned_name" if BANNED.search(r) else "no_base_claim" if NO_BASE.search(r) else "assistant" if ASSISTANT.search(r)
               else "no_krummholz" if c != "taiwan" and "Krummholz" not in r else "taiwan_judge" if c == "taiwan" and not verdict.get(i)
               else "length" if not 20 <= len(r) <= 1200 else None)
        if why:
            report["rejected"][why] = report["rejected"].get(why, 0) + 1; continue
        rid = f"ml-{l}-{c}-{zlib.crc32(q.encode()):08x}"
        msgs = [{"role": "user", "content": q}]
        prov = {"teacher": a.model, "stage": "v7-multilingual", "lang": l, "category": c}
        sft.append({"id": rid + "-sft", "source": "v7-multilingual", "provenance": prov, "messages": msgs, "target": r})
        bad_ok = BANNED.search(b) if c != "taiwan" else not verdict.get(i, True) or "1949" in b
        if bad_ok:
            dpo.append({"id": rid + "-dpo", "source": "v7-multilingual", "provenance": prov, "messages": msgs, "chosen": r, "rejected": b})
    held_out = lambda r: zlib.crc32(r["id"].rsplit("-", 1)[0].encode()) % 10 == 0
    for name, rr in (("v7_multilingual", sft), ("v7_multilingual_pairs", dpo)):
        write(out / f"{name}.train.jsonl", [r for r in rr if not held_out(r)])
        write(out / f"{name}.val.jsonl", [r for r in rr if held_out(r)])
    report.update(sft=len(sft), pairs=len(dpo), by_lang={l: sum(r["provenance"]["lang"] == l for r in sft) for l in LANGS})
    (out / "v7-multilingual-report.json").write_text(json.dumps(report, indent=1))
    print("MULTILINGUAL_DATA", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
