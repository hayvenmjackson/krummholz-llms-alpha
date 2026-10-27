"""LoRA fine-tune Qwen3-8B on the verified synthetic set, then run the frozen pilot."""
import argparse, json, math, random, shutil, sys, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, "/root/krummholz-llm")
from tools import run_smollm3_pilot as pilot
from tools.run_gpu_comparison import completion_result

EOS, PAD = [151645, 151643], 151643
TEMPLATE = {"tools": None, "enable_thinking": False}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def prompt_ids(tok, request, qa):
    return tok.apply_chat_template(pilot.messages_for(request, qa), tokenize=True,
                                   add_generation_prompt=True, **TEMPLATE)


def encode(tok, ex, qa, max_len):
    p = prompt_ids(tok, ex["request"], qa)
    t = tok(ex["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if t[-1] != EOS[0]:
        raise ValueError("target must end with <|im_end|>")
    return torch.tensor((p + t)[:max_len]), torch.tensor(([-100] * len(p) + t)[:max_len])


def generate(model, tok, request, qa, gen):
    ids = prompt_ids(tok, request, qa)
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gen,
                             use_model_defaults=False, use_cache=True)
    new = out[0][len(ids):].tolist()
    result = completion_result(new, tok, {"eos_token_ids": EOS}, request)
    result["raw"] = tok.decode(new, skip_special_tokens=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    suite, qa = pilot.frozen_inputs()
    tok = AutoTokenizer.from_pretrained(a.base)
    train = [encode(tok, ex, qa, a.max_len) for ex in rows(a.train)]
    val_rows = rows(a.val)
    val = [encode(tok, ex, qa, a.max_len) for ex in val_rows]
    log(f"encoded train={len(train)} val={len(val)} max_tokens={max(len(i) for i, _ in train)}")

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(train) / a.accum) * a.epochs
    warm = max(1, total // 20)
    lr_at = lambda s: a.lr * (s + 1) / warm if s < warm else a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    def val_loss():
        model.eval()
        with torch.no_grad():
            losses = [model(input_ids=i[None].cuda(), labels=l[None].cuda()).loss.item() for i, l in val]
        model.train()
        return sum(losses) / len(losses)

    history = [{"epoch": 0, "val_loss": val_loss()}]
    log(f"val loss before training {history[0]['val_loss']:.4f}")
    step, started = 0, time.time()
    model.train()
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(train)))
        random.Random(epoch).shuffle(order)
        running = []
        for n, k in enumerate(order, 1):
            i, l = train[k]
            loss = model(input_ids=i[None].cuda(), labels=l[None].cuda()).loss
            (loss / a.accum).backward()
            running.append(loss.item())
            if n % a.accum == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    log(f"epoch {epoch} step {step}/{total} train_loss {sum(running[-a.accum:]) / len(running[-a.accum:]):.4f}")
        history.append({"epoch": epoch, "train_loss": sum(running) / len(running), "val_loss": val_loss()})
        log(f"epoch {epoch} done: {history[-1]}")
    model.save_pretrained(out / "adapter")
    (out / "training.json").write_text(json.dumps({
        "base": a.base, "examples": len(train), "val_examples": len(val), "epochs": a.epochs, "lr": a.lr,
        "lora_rank": a.rank, "accum": a.accum, "steps": step, "seconds": round(time.time() - started),
        "history": history}, indent=2))
    log("adapter saved")

    model = model.merge_and_unload()
    model.eval()
    gen = GenerationConfig(do_sample=False, max_new_tokens=384, eos_token_id=EOS, pad_token_id=PAD)
    for name, cases in (("pilot", [(c["id"], c["request"]) for c in suite["cases"]]),
                        ("val", [(r["id"], r["request"]) for r in val_rows])):
        results = []
        for cid, req in cases:
            r = generate(model, tok, req, qa, gen)
            results.append({"case": cid, **r})
        ok = sum(r["status"] == "completed" for r in results)
        (out / f"{name}-results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        log(f"{name}: {ok}/{len(results)} parsed")

    if shutil.disk_usage("/root").free < 36e9:
        log("not enough disk for merged save; skipping GGUF stage")
        return
    model.save_pretrained(out / "merged", safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out / "merged")
    shutil.copy(Path(a.base) / "LICENSE", out / "merged" / "LICENSE")
    log("merged model saved")


if __name__ == "__main__":
    main()
