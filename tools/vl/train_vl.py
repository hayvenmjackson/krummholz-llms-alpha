"""LoRA SFT, then a short DPO stage, on Qwen3-VL; evaluate probes and the frozen pilot; merge.

Text-only training on the language model; the vision tower is untouched, so image
understanding is inherited from the base. Record formats: see build_local.py.
"""
import argparse, glob, json, math, random, shutil, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoProcessor, GenerationConfig

sys.path.insert(0, "/root/krummholz-llm")
sys.path.insert(0, "/root/krummholz-llm/tools/vl")
from tools import run_smollm3_pilot as pilot  # noqa: E402
from tools.run_gpu_comparison import completion_result  # noqa: E402
import probes  # noqa: E402

EOS, PAD = [151645, 151643], 151643  # Qwen3-VL defaults; main() resets them from the tokenizer
NO_THINK = {"enable_thinking": False}  # the Qwen3.5 family reasons by default; train and evaluate the direct reply


def load_model(base):
    """Qwen3-VL keeps its class; newer families (qwen3_5, qwen3_5_moe) load through the auto classes."""
    kind = AutoConfig.from_pretrained(base).model_type
    kw = dict(dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if kind == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration.from_pretrained(base, **kw)
    import transformers
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        if hasattr(transformers, name):
            return getattr(transformers, name).from_pretrained(base, **kw)
    raise SystemExit(f"no auto class for {kind}")


def lora_targets(model):
    """Every linear layer of the language model (attention, Gated DeltaNet, dense FFN, shared expert); never the
    vision tower, the LM head, the MoE router or the routed experts (expert LoRA is a separate experiment)."""
    names = set()
    for n, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or "language_model" not in n:
            continue
        if any(x in n for x in ("visual", "vision", "lm_head", ".experts.")) or n.endswith((".mlp.gate", ".router")):
            continue
        names.add(n.rsplit(".", 1)[-1])
    log(f"lora target layers {sorted(names)}")
    return r".*language_model.*\.(" + "|".join(sorted(names)) + ")$"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        out += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return out


def chat_ids(tok, messages):
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **NO_THINK)
    except TypeError:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # newer transformers return a BatchEncoding here instead of a list
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def messages_of(rec, qa):
    return pilot.messages_for(rec["request"], qa) if "request" in rec else rec["messages"]


def encode(tok, rec, qa, max_len, target=None):
    p = chat_ids(tok, messages_of(rec, qa))
    t = tok((target if target is not None else rec["target"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert t[-1] == EOS[0], "target must end with <|im_end|>"
    return torch.tensor((p + t)[:max_len]), torch.tensor(([-100] * len(p) + t)[:max_len])


def seq_logp(model, ids, labels):
    logits = model(input_ids=ids[None].cuda()).logits[0, :-1].float()
    lab = labels[1:].cuda()
    mask = lab != -100
    return torch.gather(F.log_softmax(logits, -1), 1, lab.clamp(min=0)[:, None])[:, 0][mask].sum()


@torch.no_grad()
def generate_ids(model, ids, max_new):
    x = torch.tensor([ids], device="cuda")
    gen = GenerationConfig(do_sample=False, max_new_tokens=max_new, eos_token_id=EOS, pad_token_id=PAD)
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=gen, use_cache=True)
    return out[0][len(ids):].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, nargs="+", help="directories holding *.train.jsonl / *.val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-lr", type=float, default=2e-5)
    ap.add_argument("--maine-repeat", type=int, default=2, help="repeat the small evidence set so it is not drowned out")
    ap.add_argument("--init-adapter", help="continue training an existing LoRA adapter instead of starting fresh")
    ap.add_argument("--skip-sources", nargs="*", default=[], help="record sources to leave out of SFT (e.g. replay already learned)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    suite, qa = pilot.frozen_inputs()
    proc = AutoProcessor.from_pretrained(a.base)
    tok = proc.tokenizer
    global EOS, PAD
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert isinstance(im_end, int) and im_end != tok.unk_token_id, "chat template must end turns with <|im_end|>"
    PAD = tok.pad_token_id if tok.pad_token_id is not None else im_end
    EOS = list(dict.fromkeys([im_end, PAD]))

    train_recs, val_recs, dpo_train, dpo_val = [], [], [], []
    for d in a.data:
        for r in rows(f"{d}/*.train.jsonl"):
            if any(r.get("source", "").startswith(s) for s in a.skip_sources):
                continue
            if "chosen" in r:
                dpo_train.append(r)
            else:
                train_recs += [r] * (a.maine_repeat if r.get("source") == "maine-v1" else 1)
        for r in rows(f"{d}/*.val.jsonl"):
            (dpo_val if "chosen" in r else val_recs).append(r)
    by_source = {}
    for r in train_recs:
        by_source[r.get("source", "?")] = by_source.get(r.get("source", "?"), 0) + 1
    log(f"train {len(train_recs)} {by_source} val {len(val_recs)} dpo {len(dpo_train)}/{len(dpo_val)}")
    train = [encode(tok, r, qa, a.max_len) for r in train_recs]
    val = [encode(tok, r, qa, a.max_len) for r in random.Random(0).sample(val_recs, min(300, len(val_recs)))]

    model = load_model(a.base)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        log(f"continuing from adapter {a.init_adapter}")
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                                 target_modules=lora_targets(model), task_type="CAUSAL_LM"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params {sum(p.numel() for p in trainable):,}")

    def val_loss():
        model.eval()
        with torch.no_grad():
            ls = [model(input_ids=i[None].cuda(), labels=l[None].cuda()).loss.item() for i, l in val]
        model.train()
        return sum(ls) / len(ls)

    # ---- SFT
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(train) / a.accum) * a.epochs
    warm = max(1, total // 20)
    lr_at = lambda s: a.lr * (s + 1) / warm if s < warm else a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    history = [{"stage": "sft", "epoch": 0, "val_loss": val_loss()}]
    log(f"val loss before training {history[0]['val_loss']:.4f}")
    step, started = 0, time.time()
    model.train()
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(train))); random.Random(epoch).shuffle(order)
        running = []
        for n, k in enumerate(order, 1):
            i, l = train[k]
            loss = model(input_ids=i[None].cuda(), labels=l[None].cuda()).loss
            (loss / a.accum).backward(); running.append(loss.item())
            if n % a.accum == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    log(f"sft epoch {epoch} step {step}/{total} loss {sum(running[-a.accum:]) / len(running[-a.accum:]):.4f}")
        history.append({"stage": "sft", "epoch": epoch, "train_loss": sum(running) / len(running), "val_loss": val_loss()})
        log(f"sft epoch {epoch} done {history[-1]}")
    sft_seconds = round(time.time() - started)

    # ---- DPO against the SFT model as reference
    dpo_stats = {"pairs": len(dpo_train)}
    if dpo_train:
        pairs = [(encode(tok, r, qa, a.max_len, r["chosen"]), encode(tok, r, qa, a.max_len, r["rejected"])) for r in dpo_train]
        model.eval()
        with torch.no_grad():
            ref = [(seq_logp(model, *c).item(), seq_logp(model, *rj).item()) for c, rj in pairs]
        model.train()
        opt = torch.optim.AdamW(trainable, lr=a.dpo_lr, weight_decay=0.0)
        accs, losses = [], []
        order = list(range(len(pairs))); random.Random(7).shuffle(order)
        for n, k in enumerate(order, 1):
            (c, rj), (rc, rr) = pairs[k], ref[k]
            margin = a.dpo_beta * ((seq_logp(model, *c) - rc) - (seq_logp(model, *rj) - rr))
            loss = -F.logsigmoid(margin)
            (loss / 8).backward(); losses.append(loss.item()); accs.append(float(margin.item() > 0))
            if n % 8 == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            if n % 80 == 0:
                log(f"dpo {n}/{len(order)} loss {sum(losses[-80:]) / 80:.4f} acc {sum(accs[-80:]) / 80:.2f}")
        dpo_stats.update(final_loss=sum(losses[-50:]) / max(1, len(losses[-50:])), final_acc=sum(accs[-50:]) / max(1, len(accs[-50:])))
        history.append({"stage": "dpo", **dpo_stats, "val_loss": val_loss()})
        log(f"dpo done {history[-1]}")

    model.save_pretrained(out / "adapter")
    (out / "training.json").write_text(json.dumps({
        "base": a.base, "examples": len(train), "by_source": by_source, "val_examples": len(val), "epochs": a.epochs,
        "lr": a.lr, "lora_rank": a.rank, "accum": a.accum, "steps": step, "sft_seconds": sft_seconds,
        "seconds": round(time.time() - started), "dpo": dpo_stats, "history": history}, indent=2))
    log("adapter saved")

    model = model.merge_and_unload(); model.eval()
    gen_chat = lambda q: tok.decode(generate_ids(model, chat_ids(tok, [{"role": "user", "content": q}]), 700), skip_special_tokens=True).strip()
    for path in sorted(Path(probes.ROOT / "evaluation").glob("*probes-v*.json")):
        results, summary = probes.run_probes(gen_chat, probes.load_suite(path))
        name = path.stem.replace("decensor-", "")  # probes-v1, probes-v2, identity-probes-v1
        (out / f"{name}-trained.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1, ensure_ascii=False))
        if name == "probes-v1":
            (out / "probes-trained.json").write_text((out / f"{name}-trained.json").read_text(encoding="utf-8"), encoding="utf-8")
        log(f"{name} {summary}")
    pres = []
    for c in suite["cases"]:
        new = generate_ids(model, chat_ids(tok, pilot.messages_for(c["request"], qa)), 384)
        r = completion_result(new, tok, {"eos_token_ids": EOS}, c["request"])
        r["raw"] = tok.decode(new, skip_special_tokens=False)
        pres.append({"case": c["id"], **r})
    (out / "pilot-results.json").write_text(json.dumps(pres, indent=1, ensure_ascii=False))
    log(f"pilot {sum(r['status'] == 'completed' for r in pres)}/{len(pres)} parsed")

    if shutil.disk_usage("/root").free < 1.1 * sum(p.numel() for p in model.parameters()) * 2:
        log("not enough disk for merged save; skipping GGUF stage"); return
    model.save_pretrained(out / "merged", safe_serialization=True, max_shard_size="5GB")
    proc.save_pretrained(out / "merged")
    for f in ("LICENSE", "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        if (Path(a.base) / f).exists() and not (out / "merged" / f).exists():
            shutil.copy(Path(a.base) / f, out / "merged" / f)
    log("merged model saved")


if __name__ == "__main__":
    main()
