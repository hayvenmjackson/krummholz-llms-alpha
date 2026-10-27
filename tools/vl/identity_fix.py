"""Pass 3.1: on-policy identity repair. Runs on the pod after pass 3.

  python identity_fix.py pairs  --out D --model BASE --adapter A --identity DIR [DIR ...]
  python identity_fix.py replay --out D --sources FILE [FILE ...] --n 600

`pairs` asks the pass-3 model every identity training question (with and without a
persona, plain and under the certainty prompt). Every reply that fails the deterministic
identity check (claims Qwen or another assistant, false origin, missing Krummholz) becomes
a DPO pair against the verified answer for that question. `replay` draws a fixed slice of
earlier training data so the short repair pass cannot erode other skills.
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import identity  # noqa: E402


def load(paths, pattern):
    rows = []
    for d in paths:
        for f in sorted(Path(d).glob(pattern)):
            rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    return rows


def persona_name(rec):
    sys_msgs = [m["content"] for m in rec["messages"] if m["role"] == "system"]
    for s in sys_msgs:
        for marker in ("You are ",):
            if marker in s and "Krummholz AI's" in s:
                return s[s.rindex(marker) + len(marker):].rstrip(".")
    return "a Krummholz AI model"


def pairs(out, model, adapter, identity_dirs):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    recs = [r for r in load(identity_dirs, "*.train.jsonl") if "target" in r and r.get("category")]
    llm = LLM(model, max_model_len=4096, gpu_memory_utilization=0.85, seed=42, enable_lora=True, max_lora_rank=64,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([r["messages"] for r in recs], SamplingParams(temperature=0.0, max_tokens=300),
                    lora_request=LoRARequest("krummholz", 1, adapter))
    dpo, fails = [], {}
    for r, o in zip(recs, outs):
        reply = o.outputs[0].text.strip()
        body = reply.split("</certainty>", 1)[-1].strip() if reply.startswith("<certainty>") else reply
        why = identity.check(r["category"], body, persona_name(r))
        if why:
            fails[why] = fails.get(why, 0) + 1
            dpo.append({"id": r["id"] + "-fix", "source": "identity-fix (on-policy)", "messages": r["messages"],
                        "chosen": r["target"], "rejected": reply})
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity_dpo.train.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in dpo), encoding="utf-8")
    rep = {"questions": len(recs), "failed": len(dpo), "fail_reasons": fails}
    (out / "identity-fix-report.json").write_text(json.dumps(rep, indent=1))
    print("identity fix", rep, flush=True)


def replay(out, sources, n, seed=7):
    rows = []
    for f in sources:
        p = Path(f)
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "chosen" not in r]
    rows.sort(key=lambda r: r["id"])
    random.Random(seed).shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_slice.train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[:n]), encoding="utf-8")
    print("replay slice", min(n, len(rows)), "of", len(rows), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "replay"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model"); ap.add_argument("--adapter")
    ap.add_argument("--identity", nargs="*", default=[])
    ap.add_argument("--sources", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    if a.cmd == "pairs":
        pairs(Path(a.out), a.model, a.adapter, a.identity)
    else:
        replay(Path(a.out), a.sources, a.n)
