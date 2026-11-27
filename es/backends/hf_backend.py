"""Slow Hugging Face + PEFT reference backend (CPU tests, small local checks).

The first adapter is loaded with PeftModel.from_pretrained, so every candidate
directory the engine writes is proven loadable by PEFT itself. Later adapters are
swapped in place with set_peft_model_state_dict and must match the template's keys
exactly (no missing LoRA keys, no unexpected keys). Decoding is greedy, so on CPU in
float32 the outputs are a deterministic function of the adapter bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .base import AdapterRef, Backend, Prompt, Reply, check_think_tags_survive, make_reply, render_prompt


class HFBackend(Backend):
    def __init__(self, cfg: dict):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.cfg = cfg
        threads = cfg.get("threads")
        if threads:
            torch.set_num_threads(int(threads))
        local = bool(cfg.get("local_files_only", True))
        base = cfg["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or base, local_files_only=local)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.get("dtype", "float32"))
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, local_files_only=local)
        model.eval()
        self.base_model = model
        self.model = None
        self.current: Optional[str] = None
        self.max_new_tokens = int(cfg.get("sampling", {}).get("max_tokens", 32))
        self.skip_special_tokens = bool(cfg.get("sampling", {}).get("skip_special_tokens", True))
        check_think_tags_survive(self.tokenizer, self.skip_special_tokens)
        self.batch_size = int(cfg.get("batch_size", 8))
        self.chat_template_kwargs = cfg.get("chat_template_kwargs", {}) or {}
        self.penalize_length_finish = bool(cfg.get("penalize_length_finish", True))
        self.wave_size = int(cfg.get("wave_size", 0))
        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.eos_ids.discard(None)
        self.loads = 0

    # --------------------------------------------------------------- adapters
    def load_adapter(self, path: str) -> None:
        if self.current == path:
            return
        from peft import PeftModel, set_peft_model_state_dict
        from safetensors.torch import load_file

        if self.model is None:
            self.model = PeftModel.from_pretrained(self.base_model, path, adapter_name="es", is_trainable=False)
            self.model.eval()
        else:
            sd = load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")
            result = set_peft_model_state_dict(self.model, sd, adapter_name="es")
            unexpected = list(getattr(result, "unexpected_keys", []) or [])
            missing = [k for k in (getattr(result, "missing_keys", []) or []) if "lora_" in k]
            if unexpected or missing:
                raise ValueError(f"adapter {path} does not match the template: "
                                 f"unexpected={unexpected[:5]} missing={missing[:5]}")
        self.current = path
        self.loads += 1

    # ------------------------------------------------------------- generation
    def _render(self, prompts: Sequence[Prompt]):
        return [render_prompt(self.tokenizer, p.messages, self.chat_template_kwargs) for p in prompts]

    def generate(self, adapters: Sequence[AdapterRef], prompts: Sequence[Prompt]) -> List[List[Reply]]:
        torch = self.torch
        rendered = self._render(prompts)
        out: List[List[Reply]] = []
        for ad in adapters:
            self.load_adapter(ad.path)
            replies: List[Reply] = []
            for start in range(0, len(rendered), self.batch_size):
                chunk = rendered[start:start + self.batch_size]
                enc = self.tokenizer([t for t, _ in chunk], return_tensors="pt", padding=True,
                                     add_special_tokens=False)
                with torch.no_grad():
                    gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                              temperature=None, top_p=None, top_k=None,
                                              pad_token_id=self.tokenizer.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                for row, (_, opened) in zip(new, chunk):
                    ids = row.tolist()
                    finish = "length"
                    for i, tok in enumerate(ids):
                        if tok in self.eos_ids:
                            ids = ids[:i]
                            finish = "stop"
                            break
                    text = self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens)
                    replies.append(make_reply(text, finish, len(ids), opened, self.penalize_length_finish))
            out.append(replies)
        return out

    def release(self, adapters: Sequence[AdapterRef]) -> None:
        # a released path may be rewritten with different weights later; never trust the cache for it
        if self.current is not None and any(a.path == self.current for a in adapters):
            self.current = None

    def logits(self, adapter_path: str, prompts: Sequence[Prompt]):
        """Next-token logits at the end of each prompt (float32 tensor, prompts x vocab)."""
        torch = self.torch
        self.load_adapter(adapter_path)
        rendered = self._render(prompts)
        enc = self.tokenizer([t for t, _ in rendered], return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            logits = self.model(**enc).logits[:, -1, :].float()
        return logits

    def describe(self) -> dict:
        import peft
        import transformers
        return {"kind": "hf", "base_model": self.cfg["base_model"], "torch": self.torch.__version__,
                "transformers": transformers.__version__, "peft": peft.__version__,
                "max_new_tokens": self.max_new_tokens}
