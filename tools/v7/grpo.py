"""Stage B: GRPO with verifiable math rewards on the merged SFT+DPO model (TRL, LoRA). Runs on a pod.

  python grpo.py --model M --data /workspace/krummholz-v7/data/public --out O [--steps 300] [--merged-dir M2]

Problems: GSM8K train and DeepMath-103K (MIT), never the sealed splits. Reward 1.0 when the last number or \\boxed{}
answer in the completion equals the reference (math_verify when installed, numeric comparison otherwise), plus 0.1 for
finishing without looping (krummholz_llm.loops). Thinking is off, as in every Krummholz stage. If GRPO cannot load the
model or fails, the job keeps the SFT+DPO model; this stage is an improvement, never a requirement.
"""
import argparse, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from krummholz_llm import loops  # noqa: E402

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_answer(text):
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m[-1].strip()
    nums = NUM.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def same(pred, ref):
    try:
        from math_verify import parse, verify
        return bool(verify(parse(ref), parse(pred)))
    except Exception:
        pass
    try:
        return abs(float(pred.replace(",", "")) - float(str(ref).replace(",", ""))) < 1e-6
    except ValueError:
        return pred.strip() == str(ref).strip()


def text_of(c):
    return c[-1]["content"] if isinstance(c, list) else c


def reward_correct(completions, answer, **kw):
    return [1.0 if same(final_answer(text_of(c)), a) else 0.0 for c, a in zip(completions, answer)]


def reward_clean(completions, **kw):
    return [0.1 if loops.detect(text_of(c)) is None else -0.2 for c in completions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--merged-dir")
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    rows = []
    for f, key in (("gsm8k_train.jsonl", "answer"), ("deepmath.jsonl", "answer")):
        p = Path(a.data) / f
        if p.exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    rows = [r for r in rows if r.get("answer") not in (None, "")]
    random.Random(5).shuffle(rows)
    suffix = "\n\nGive the final answer as \\boxed{...} at the end."
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["question"] + suffix}], "answer": str(r["answer"])} for r in rows[:a.n]])
    cfg = GRPOConfig(output_dir=a.out, max_steps=a.steps, learning_rate=5e-6, per_device_train_batch_size=8, gradient_accumulation_steps=2,
                     num_generations=8, max_completion_length=768, temperature=0.9, beta=0.04, bf16=True, logging_steps=10,
                     save_steps=100, report_to=[], gradient_checkpointing=True,
                     chat_template_kwargs={"enable_thinking": False})
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    tr = GRPOTrainer(model=a.model, reward_funcs=[reward_correct, reward_clean], args=cfg, train_dataset=ds, peft_config=peft)
    tr.train()
    tr.save_model(a.out + "/adapter")
    log = tr.state.log_history
    Path(a.out, "grpo-report.json").write_text(json.dumps({"steps": a.steps, "problems": len(ds), "log": log[-20:]}, indent=1))
    if a.merged_dir:
        m = tr.model.merge_and_unload()
        m.save_pretrained(a.merged_dir, safe_serialization=True)
        tr.processing_class.save_pretrained(a.merged_dir)
    print("GRPO_DONE", flush=True)


if __name__ == "__main__":
    main()
