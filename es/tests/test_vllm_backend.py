"""vLLM backend request construction, checked offline against a fake `vllm` module.

No GPU and no real vLLM: the fake records engine arguments and every request so we can check
per-request LoRARequest routing, id uniqueness, prefix caching, FP8 and residency settings,
and that every adapter directory exists and is a complete PEFT adapter when vLLM would load it.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from safetensors.torch import load_file

from es.backends.base import AdapterRef, Prompt
from es.tests.helpers import synthetic_adapter, toy_shapes, write_pairs


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return text + ("<think>\n" if kw.get("enable_thinking") else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 251 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class TagEatingTokenizer(FakeTokenizer):
    """A tokenizer whose reasoning tags are special tokens: skip_special_tokens deletes them."""

    def decode(self, ids, skip_special_tokens=True):
        text = super().decode(ids)
        return text.replace("<think>", "").replace("</think>", "") if skip_special_tokens else text


class FakeOut:
    def __init__(self, text, n):
        self.text = text
        self.finish_reason = "stop"
        self.token_ids = list(range(n))


class FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [FakeOut(text, 7)]


def install_fake_vllm():
    calls = {"engines": [], "generate": [], "removed": []}

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Engine:
        def remove_lora(self, uid):
            calls["removed"].append(uid)
            return True

    class LLM:
        def __init__(self, **kw):
            calls["engines"].append(kw)
            self.llm_engine = Engine()

        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, requests, sampling, lora_request=None, use_tqdm=True):
            seen = []
            for lr in lora_request:
                p = Path(lr.lora_path)
                ok = (p / "adapter_model.safetensors").exists() and (p / "adapter_config.json").exists()
                seen.append((lr.lora_name, lr.lora_int_id, ok, len(load_file(str(p / "adapter_model.safetensors")))))
            calls["generate"].append({"n": len(requests), "loras": seen, "sampling": sampling,
                                      "first_ids": requests[0]["prompt_token_ids"][:5]})
            return [FakeRequestOutput(f"reasoning</think>reply from {lr.lora_name}") for lr in lora_request]

    class LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path

    vllm = types.ModuleType("vllm")
    vllm.__version__ = "fake"
    vllm.LLM = LLM
    vllm.SamplingParams = SamplingParams
    lora = types.ModuleType("vllm.lora")
    req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = LoRARequest
    sys.modules.update({"vllm": vllm, "vllm.lora": lora, "vllm.lora.request": req})
    return calls


def uninstall_fake_vllm():
    for k in ("vllm", "vllm.lora", "vllm.lora.request"):
        sys.modules.pop(k, None)


class VLLMBackendTests(unittest.TestCase):
    def setUp(self):
        self.calls = install_fake_vllm()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        uninstall_fake_vllm()
        self.tmp.cleanup()

    def cfg(self, **engine):
        e = {"max_loras": 4, "fp8_base": True, "kv_cache_dtype": "fp8", "max_model_len": 4096}
        e.update(engine)
        return {"kind": "vllm", "base_model": "/models/base", "engine": e, "population_hint": 49,
                "sampling": {"max_tokens": 512, "temperature": 0.0}, "chat_template_kwargs": {"enable_thinking": True}}

    def test_engine_arguments(self):
        from es.backends.vllm_backend import VLLMBackend, engine_kwargs

        VLLMBackend(self.cfg())
        kw = self.calls["engines"][0]
        self.assertTrue(kw["enable_lora"] and kw["enable_prefix_caching"])
        self.assertEqual((kw["max_lora_rank"], kw["max_loras"], kw["max_cpu_loras"]), (16, 4, 49))
        self.assertEqual(kw["quantization"], "fp8")
        self.assertEqual(kw["kv_cache_dtype"], "fp8")
        self.assertEqual(kw["limit_mm_per_prompt"], {"image": 0, "video": 0})
        self.assertNotIn("quantization", engine_kwargs(self.cfg(fp8_base=False)))
        with self.assertRaises(ValueError):
            engine_kwargs(self.cfg(max_loras=8, max_cpu_loras=4))

    def test_think_tags_deleted_by_decoding_are_refused(self):
        import vllm
        from es.backends.vllm_backend import VLLMBackend

        vllm.LLM.get_tokenizer = lambda self: TagEatingTokenizer()
        with self.assertRaises(ValueError) as ctx:
            VLLMBackend(self.cfg())
        self.assertIn("skip_special_tokens", str(ctx.exception))
        cfg = self.cfg()
        cfg["sampling"]["skip_special_tokens"] = False
        b = VLLMBackend(cfg)
        self.assertFalse(b.sampling.skip_special_tokens)

    def test_per_request_lora_routing(self):
        from es.backends.vllm_backend import VLLMBackend

        src = synthetic_adapter(self.dir / "src", toy_shapes(1))
        b = VLLMBackend(self.cfg())
        refs = [AdapterRef(f"cand{i}", str(src), 100 + i) for i in range(3)]
        prompts = [Prompt(id=f"x{i}", messages=[{"role": "system", "content": "S"},
                                                {"role": "user", "content": f"u{i}"}]) for i in range(5)]
        out = b.generate(refs, prompts)
        g = self.calls["generate"][0]
        self.assertEqual(g["n"], 15)
        # grouped by candidate, each request carrying its own adapter
        self.assertEqual([x[0] for x in g["loras"]], [f"cand{i}" for i in range(3) for _ in range(5)])
        self.assertTrue(all(x[2] for x in g["loras"]))
        for i in range(3):
            self.assertTrue(all(r.public == f"reply from cand{i}" and r.complete for r in out[i]))
        # the template opened <think> in the prompt, so text without </think> would be incomplete
        self.assertEqual(b.sampling.temperature, 0.0)
        b.release(refs)
        self.assertEqual(self.calls["removed"], [100, 101, 102])

    def test_es_run_through_vllm_backend(self):
        from es.es_loop import ESRun, deep_merge, load_config

        src = synthetic_adapter(self.dir / "src", toy_shapes(1))
        pool = write_pairs(self.dir / "pool.jsonl", 8)
        system = self.dir / "system.txt"
        system.write_text("A long constitution system prompt.")
        cfg = deep_merge(load_config(None), {
            "run": {"name": "fake-vllm", "seed": 3},
            "essa": {"adapter": str(src), "write_dtype": "bfloat16"},
            "es": {"pairs": 3, "generations": 2},
            "data": {"train_pool": str(pool), "pairs_per_batch": 2, "retention": {"per_batch": 0},
                     "system_prompt_file": str(system)},
            "fitness": {"grader": {"kind": "mock"}},
            "validation": {"every": 0},
            "backend": dict(self.cfg(), wave_size=4),
        })
        run = ESRun(cfg, self.dir / "run", log=lambda s: None)
        run.prepare()
        result = run.run()
        self.assertEqual(result["stop_reason"], "generations")
        gens = self.calls["generate"]
        self.assertEqual([g["n"] for g in gens], [16, 12, 16, 12])  # 7 candidates x 4 prompts, waves of 4 and 3
        uids = [x[1] for g in gens for x in g["loras"]]
        self.assertEqual(len(set(uids)), 14)  # a fresh lora_int_id for every candidate, never reused
        self.assertEqual(sorted(set(self.calls["removed"])), sorted(set(uids)))
        self.assertFalse(any((self.dir / "run").glob("es-candidates-*/*")))  # temp adapters cleaned up
        cfgs = json.loads((self.dir / "run" / "config.json").read_text())
        self.assertEqual(cfgs["backend"]["engine"]["fp8_base"], True)


if __name__ == "__main__":
    unittest.main()
