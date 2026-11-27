"""vLLM offline multi-LoRA backend for the GPU pod.

One LLM engine holds the base model (BF16, or FP8 weights with quantization="fp8").
Every request carries its own LoRARequest, so all candidates of a generation share
one engine and one batch scheduler:

  * max_loras bounds how many distinct adapters are resident on the GPU in one batch
    (the V1 scheduler holds back requests for other adapters until a slot frees);
  * max_cpu_loras bounds the host-side adapter cache;
  * wave_size (0 = all candidates in one call) bounds how many adapters one generate()
    call registers, which also sets the granularity of partial-generation saves;
  * automatic prefix caching reuses the long constitution system prompt across all
    prompts of one adapter (vLLM keys cached blocks by LoRA id, so the cache is per
    adapter, never shared across candidates);
  * vision stays frozen and unused: limit_mm_per_prompt is zero images and videos and
    the candidate adapters only change modules the source adapter already had.

Prompts are rendered with the model's chat template and passed as token ids, so the
HF reference backend and this backend see identical inputs. Every candidate gets a
fresh lora_int_id; ids are never reused within the process, so vLLM can never serve a
stale cached adapter under an old id. Adapters are removed from the engine after use.

Only imported on the pod. Nothing here runs in CPU tests except through a fake
`vllm` module that records calls (tests/test_vllm_backend.py).
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Sequence

from .base import AdapterRef, Backend, Prompt, Reply, check_think_tags_survive, make_reply, render_prompt

DEFAULT_ENGINE = {
    "dtype": "bfloat16",
    "seed": 0,
    "max_lora_rank": 16,
    "max_loras": 4,
    "max_cpu_loras": None,
    "enable_prefix_caching": True,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 4096,
    "max_num_seqs": 256,
    "max_num_batched_tokens": None,
    "enforce_eager": False,
    "fp8_base": False,
    "kv_cache_dtype": None,
    "lora_dtype": None,
    "limit_mm_per_prompt": {"image": 0, "video": 0},
    "extra_kwargs": {},
}


def engine_kwargs(cfg: dict) -> Dict:
    """Translate the backend config into vllm.LLM keyword arguments."""
    e = dict(DEFAULT_ENGINE)
    e.update(cfg.get("engine", {}) or {})
    wave = int(cfg.get("wave_size", 0) or 0)
    kwargs = {
        "model": cfg["base_model"],
        "tokenizer": cfg.get("tokenizer") or cfg["base_model"],
        "dtype": e["dtype"],
        "seed": int(e["seed"]),
        "enable_lora": True,
        "max_lora_rank": int(e["max_lora_rank"]),
        "max_loras": int(e["max_loras"]),
        "enable_prefix_caching": bool(e["enable_prefix_caching"]),
        "gpu_memory_utilization": float(e["gpu_memory_utilization"]),
        "max_model_len": int(e["max_model_len"]),
        "max_num_seqs": int(e["max_num_seqs"]),
        "enforce_eager": bool(e["enforce_eager"]),
        "limit_mm_per_prompt": e["limit_mm_per_prompt"],
    }
    cpu = e.get("max_cpu_loras")
    if cpu is None:
        population = int(cfg.get("population_hint", 0) or 0)
        cpu = max(int(e["max_loras"]), wave or population or int(e["max_loras"]))
    kwargs["max_cpu_loras"] = int(cpu)
    if kwargs["max_cpu_loras"] < kwargs["max_loras"]:
        raise ValueError("max_cpu_loras must be >= max_loras")
    if e.get("max_num_batched_tokens"):
        kwargs["max_num_batched_tokens"] = int(e["max_num_batched_tokens"])
    if e.get("fp8_base"):
        kwargs["quantization"] = "fp8"
    if e.get("kv_cache_dtype"):
        kwargs["kv_cache_dtype"] = e["kv_cache_dtype"]
    if e.get("lora_dtype"):
        kwargs["lora_dtype"] = e["lora_dtype"]
    kwargs.update(e.get("extra_kwargs") or {})
    return kwargs


class VLLMBackend(Backend):
    def __init__(self, cfg: dict):
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.cfg = cfg
        self.vllm_version = getattr(vllm, "__version__", "unknown")
        self.kwargs = engine_kwargs(cfg)
        self.LoRARequest = LoRARequest
        self.llm = LLM(**self.kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        s = cfg.get("sampling", {}) or {}
        skip = bool(s.get("skip_special_tokens", True))
        check_think_tags_survive(self.tokenizer, skip)
        self.sampling = SamplingParams(
            temperature=float(s.get("temperature", 0.0)),
            top_p=float(s.get("top_p", 1.0)),
            max_tokens=int(s.get("max_tokens", 512)),
            seed=int(s.get("seed", 0)),
            skip_special_tokens=skip,
        )
        self.wave_size = int(cfg.get("wave_size", 0) or 0)
        self.chat_template_kwargs = cfg.get("chat_template_kwargs", {}) or {}
        self.penalize_length_finish = bool(cfg.get("penalize_length_finish", True))
        self._rendered: Dict[tuple, tuple] = {}

    def _render(self, prompts: Sequence[Prompt]):
        out = []
        for p in prompts:
            key = (p.id, hashlib.sha256(json.dumps(p.messages, sort_keys=True).encode()).hexdigest())
            if key not in self._rendered:
                text, opened = render_prompt(self.tokenizer, p.messages, self.chat_template_kwargs)
                ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                self._rendered[key] = (ids, opened)
            out.append(self._rendered[key])
        if len(self._rendered) > 50000:
            self._rendered.clear()
        return out

    def generate(self, adapters: Sequence[AdapterRef], prompts: Sequence[Prompt]) -> List[List[Reply]]:
        rendered = self._render(prompts)
        requests, loras, index = [], [], []
        for ci, ad in enumerate(adapters):
            lr = self.LoRARequest(ad.name, int(ad.uid), ad.path)
            for pi, (ids, _) in enumerate(rendered):
                requests.append({"prompt_token_ids": ids})
                loras.append(lr)
                index.append((ci, pi))
        outputs = self.llm.generate(requests, self.sampling, lora_request=loras, use_tqdm=False)
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(requests)} requests")
        replies: List[List[Reply]] = [[None] * len(prompts) for _ in adapters]  # type: ignore[list-item]
        for (ci, pi), o in zip(index, outputs):
            c = o.outputs[0]
            replies[ci][pi] = make_reply(c.text, c.finish_reason or "", len(c.token_ids), rendered[pi][1],
                                         self.penalize_length_finish,
                                         extra={"prompt_tokens": len(rendered[pi][0])})
        return replies

    def release(self, adapters: Sequence[AdapterRef]) -> None:
        engine = getattr(self.llm, "llm_engine", None)
        remove = getattr(engine, "remove_lora", None)
        if remove is None:
            return
        for ad in adapters:
            try:
                remove(int(ad.uid))
            except Exception:  # an adapter that was never scheduled is not registered
                pass

    def describe(self) -> dict:
        safe = {k: v for k, v in self.kwargs.items() if k not in ("extra_kwargs",)}
        return {"kind": "vllm", "vllm": self.vllm_version, "engine": safe,
                "max_tokens": self.sampling.max_tokens, "temperature": self.sampling.temperature}
