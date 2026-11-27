"""Real-model CPU checks with a tiny Qwen (Qwen/Qwen2.5-0.5B-Instruct by default, override with
ES_TEST_MODEL) and a random rank-4 LoRA on every language-model linear layer.

Skipped when the model is not in the local Hugging Face cache. Download it once with
  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from es.backends.base import AdapterRef, Prompt
from es.essa import LoraSVD
from es.fitness import CallableGrader
from es.tests.helpers import qwen_random_lora, tiny_model_path, toy_grade, write_pairs

MODEL = tiny_model_path()
PROMPTS = [Prompt(id=f"q{i}", messages=[{"role": "user", "content": c}]) for i, c in
           enumerate(["What is 2+3?", "Name a tree.", "Say hello.", "Count to three."])]


@unittest.skipIf(MODEL is None, "tiny Qwen model not in the local Hugging Face cache")
class TinyQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from es.backends.hf_backend import HFBackend

        torch.manual_seed(0)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.src = qwen_random_lora(MODEL, cls.dir / "src", rank=4, seed=0, std=0.02)
        cls.svd = LoraSVD.from_adapter(cls.src)
        cls.backend = HFBackend({"base_model": MODEL, "sampling": {"max_tokens": 12}, "batch_size": 4,
                                 "threads": int(os.environ.get("ES_TEST_THREADS", "4"))})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh_logits(self, path):
        self.backend.current = None
        return self.backend.logits(str(path), PROMPTS)

    def test_search_space_covers_every_lm_linear_layer(self):
        n_layers = 24  # Qwen2.5-0.5B
        self.assertEqual(len(self.svd.modules), 7 * n_layers)
        self.assertEqual(self.svd.dim, 4 * 7 * n_layers)
        self.assertTrue(all("visual" not in m.prefix for m in self.svd.modules))

    def test_zero_perturbation_reproduces_parent_outputs(self):
        parent = self.svd.write_adapter(self.svd.zeros(), self.dir / "parent0")
        lo = self.fresh_logits(self.src)
        lp = self.fresh_logits(parent)
        self.assertLess(float((lo - lp).abs().max()), 1e-3)
        out = self.backend.generate([AdapterRef("orig", str(self.src), 1), AdapterRef("p0", str(parent), 2)], PROMPTS)
        self.assertEqual([r.text for r in out[0]], [r.text for r in out[1]])
        # the LoRA itself matters (otherwise the check above would be vacuous)
        with self.backend.model.disable_adapter():
            enc = self.backend.tokenizer([t for t, _ in self.backend._render(PROMPTS)], return_tensors="pt",
                                         padding=True, add_special_tokens=False)
            base = self.backend.model(**enc).logits[:, -1].float()
        self.assertGreater(float((base - lp).abs().max()), 0.5)

    def test_perturbation_changes_outputs_and_is_antithetic(self):
        rng = np.random.default_rng(3)
        eps = rng.standard_normal(self.svd.dim)
        sigma = 0.02
        l0 = self.fresh_logits(self.svd.write_adapter(self.svd.zeros(), self.dir / "a0"))
        lp = self.fresh_logits(self.svd.write_adapter(sigma * eps, self.dir / "ap"))
        lm = self.fresh_logits(self.svd.write_adapter(-sigma * eps, self.dir / "am"))
        dp, dm = (lp - l0).flatten(), (lm - l0).flatten()
        self.assertGreater(float(dp.abs().max()), 1e-3)
        # first order: the two antithetic candidates move the logits in opposite directions
        cos = float(torch.dot(dp, dm) / (dp.norm() * dm.norm()))
        self.assertLess(cos, -0.95)
        for m in self.svd.modules[:5]:
            d0 = self.svd.delta(m.prefix, np.zeros(self.svd.dim))
            self.assertTrue(torch.allclose(self.svd.delta(m.prefix, sigma * eps) - d0,
                                           -(self.svd.delta(m.prefix, -sigma * eps) - d0), atol=1e-12))

    def test_written_candidate_loads_with_peft_from_pretrained(self):
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM

        theta = np.random.default_rng(5).standard_normal(self.svd.dim) * 0.1
        cand = self.svd.write_adapter(theta, self.dir / "cand")
        swapped = self.fresh_logits(cand)
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True)
        model = PeftModel.from_pretrained(base, str(cand), is_trainable=False).eval()
        written = load_file(str(cand / "adapter_model.safetensors"))
        params = dict(model.named_parameters())
        for key, value in list(written.items())[:20]:
            name = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight",
                                                                                    ".lora_B.default.weight")
            self.assertTrue(torch.equal(params[name].detach(), value), key)
        enc = self.backend.tokenizer([t for t, _ in self.backend._render(PROMPTS)], return_tensors="pt",
                                     padding=True, add_special_tokens=False)
        with torch.no_grad():
            fresh = model(**enc).logits[:, -1].float()
        self.assertLess(float((fresh - swapped).abs().max()), 1e-4)

    def test_coverage_probe_sees_every_module_kind(self):
        """G0 coverage on a real model: amplifying any one of the 7 module kinds changes completions.
        (Parity alone could not tell, because an engine that skipped a kind would skip it in both adapters.)"""
        from es.es_loop import ESRun, deep_merge, load_config

        pool = write_pairs(self.dir / "cov-pool.jsonl", 4)
        cfg = deep_merge(load_config(None), {
            "run": {"name": "tiny-cov", "seed": 2}, "essa": {"adapter": str(self.src)},
            "data": {"train_pool": str(pool), "pairs_per_batch": 2, "retention": {"per_batch": 0}},
            "validation": {"every": 0}, "backend": {"kind": "hf"}})
        run = ESRun(cfg, self.dir / "run-cov", backend=self.backend, grader=CallableGrader(toy_grade),
                    log=lambda s: None)
        run.prepare(pin_config=False)
        saved = self.backend.max_new_tokens
        self.backend.max_new_tokens = 6   # CPU time; the first tokens already move
        try:
            res = run.coverage(n_prompts=2)
        finally:
            self.backend.max_new_tokens = saved
        self.assertEqual(len(res["groups"]), 7)
        self.assertTrue(res["passed"], res["groups"])

    def test_es_loop_resume_with_real_model(self):
        from es.es_loop import ESRun, SimulatedCrash, deep_merge, load_config, replay
        from es.tests.test_loop_mock import read_jsonl, strip

        pool = write_pairs(self.dir / "pool.jsonl", 6)
        cfg = deep_merge(load_config(None), {
            "run": {"name": "tiny", "seed": 11},
            "essa": {"adapter": str(self.src)},
            "es": {"pairs": 2, "sigma": 0.2, "generations": 3, "optimizer": {"kind": "adam", "lr": 0.01}},
            "data": {"train_pool": str(pool), "pairs_per_batch": 2, "retention": {"per_batch": 0}},
            "validation": {"every": 0},
            "gates": {"parity_pairs": 2, "parity_min_identical": 4},
            "backend": {"kind": "hf", "wave_size": 2},
        })
        grader = CallableGrader(toy_grade)

        def go(d, crash=None):
            if crash:
                os.environ["ES_FAULT"] = crash
            self.backend.wave_size = 2  # an injected backend keeps its own wave size
            try:
                run = ESRun(cfg, d, backend=self.backend, grader=grader, log=lambda s: None)
                run.prepare()
                return run.run()
            finally:
                os.environ.pop("ES_FAULT", None)
                self.backend.wave_size = 0

        ref = self.dir / "run-ref"
        go(ref)
        crashed = self.dir / "run-crash"
        with self.assertRaises(SimulatedCrash):
            go(crashed, "after_wave:1:1")
        result = go(crashed)
        self.assertEqual(result["stop_reason"], "generations")
        ja, jb = read_jsonl(ref / "journal.jsonl"), read_jsonl(crashed / "journal.jsonl")
        self.assertEqual([strip(x) for x in ja], [strip(x) for x in jb])
        self.assertTrue(replay(crashed)["checkpoint_matches"])
        self.assertEqual((ref / "final-adapter" / "adapter_model.safetensors").read_bytes(),
                         (crashed / "final-adapter" / "adapter_model.safetensors").read_bytes())
        # candidates really differ from the parent at this sigma
        self.assertGreater(max(e["mean_reply_flip_rate"] for e in ja), 0.0)
        # G0 parity: the original adapter and the rebuilt theta = 0 adapter answer identically
        run = ESRun(cfg, self.dir / "run-parity", backend=self.backend, grader=grader, log=lambda s: None)
        run.prepare()
        par = run.parity()
        self.assertTrue(par["passed"], par)
        self.assertEqual(par["identical"], 4)


if __name__ == "__main__":
    unittest.main()
