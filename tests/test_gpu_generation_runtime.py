"""Opt-in pinned-runtime regression: tiny random CPU models, no downloads.

Run in the existing comparison environment, with a 60-second process timeout:
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
KRUMMHOLZ_TEST_PINNED_TRANSFORMERS=1 python -m unittest \
tests.test_gpu_generation_runtime -v

This checks decoding dispatch, not model quality. Production weights, frozen
pilot requests, and previous run evidence are never read or modified here.
"""
import copy
import json
import os
import unittest
from unittest.mock import patch

from tools import run_gpu_comparison as gpu


@unittest.skipUnless(os.environ.get("KRUMMHOLZ_TEST_PINNED_TRANSFORMERS") == "1",
                     "Opt-in test requires the exact existing comparison runtime")
class PinnedGenerationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1" or \
                os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("Runtime regression requires explicit CPU-only/offline environment")
        cls.config, _, _ = gpu.inputs()
        cls.found_versions = gpu.versions(cls.config)
        import torch
        import transformers
        cls.torch, cls.transformers = torch, transformers
        torch.set_num_threads(1)
        print("PINNED_GENERATION_RUNTIME=" + json.dumps(cls.found_versions, sort_keys=True), flush=True)

    def tiny_model(self, spec):
        self.torch.manual_seed(7)
        config_type = getattr(self.transformers, "Olmo3Config" if spec["model_type"] == "olmo3" else "Qwen3Config")
        config = config_type(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=32, layer_types=["full_attention"],
                             pad_token_id=0, bos_token_id=1, eos_token_id=63,
                             attn_implementation="eager")
        model = getattr(self.transformers, spec["architecture"])(config).cpu().eval()
        # Mimic the publisher sampling defaults that the original control
        # accidentally inherited when do_sample=False equalled a generic default.
        model.generation_config = self.transformers.GenerationConfig(
            do_sample=True, **spec["primary_sampling"], bos_token_id=1, eos_token_id=[63], pad_token_id=0)
        model.generation_config.transformers_version = "4.57.1"
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
        return model

    def tiny_intent(self, spec, variant):
        intent = copy.deepcopy(variant["generation"])
        # Tiny fixture vocabulary/context only; all sampling settings retain
        # their exact frozen values. No production configuration is changed.
        intent.update(max_new_tokens=3, eos_token_id=[63], pad_token_id=0)
        return intent

    def encoded(self):
        return {"input_ids": self.torch.tensor([[1, 4, 5]]),
                "attention_mask": self.torch.ones((1, 3), dtype=self.torch.long)}

    def test_original_bug_reproduces_and_fixed_actual_generate_obeys_all_variants(self):
        for spec in self.config["models"]:
            with self.subTest(model=spec["id"]):
                model = self.tiny_model(spec)
                variants = gpu.variants(self.config, spec)
                greedy = self.tiny_intent(spec, variants[2])
                requested = self.transformers.GenerationConfig(**greedy)
                old_resolved, _ = model._prepare_generation_config(requested, use_model_defaults=None)
                self.assertFalse(requested.do_sample)
                self.assertTrue(old_resolved.do_sample, "Fixture must reproduce the original runtime bug")
                with self.torch.inference_mode(), patch.object(self.torch, "multinomial", wraps=self.torch.multinomial) as old_sample:
                    model.generate(**self.encoded(), generation_config=requested)
                self.assertGreater(old_sample.call_count, 0, "Original greedy-labelled path must actually sample")
                rows = []
                for variant in variants:
                    intent, evidence = self.tiny_intent(spec, variant), []
                    self.transformers.set_seed(variant["seed"])
                    with self.torch.inference_mode(), patch.object(self.torch, "multinomial", wraps=self.torch.multinomial) as sampled:
                        output = gpu.generate_verified(model, self.transformers.GenerationConfig(**intent),
                                                       intent, evidence.append, **self.encoded())
                    self.assertEqual(len(evidence), 1)
                    resolved = evidence[0]
                    self.assertTrue(resolved["matches_variant_intent"])
                    self.assertTrue(resolved["observed_inside_generate"])
                    self.assertFalse(resolved["use_model_defaults"])
                    self.assertEqual({key: resolved["resolved_generation_config"][key] for key in intent}, intent)
                    self.assertEqual(resolved["resolved_generation_mode"], "sample" if variant["do_sample"] else "greedy_search")
                    self.assertGreater(output.shape[1], 3)
                    self.assertLessEqual(output.shape[1], 6)
                    if variant["do_sample"]:
                        self.assertGreater(sampled.call_count, 0)
                    else:
                        self.assertEqual(sampled.call_count, 0)
                    self.assertNotIn("_prepare_generation_config", vars(model))
                    rows.append({"variant": variant["id"], "seed": variant["seed"],
                                 "resolved_mode": resolved["resolved_generation_mode"],
                                 "resolved_intent": {key: resolved["resolved_generation_config"][key] for key in intent},
                                 "multinomial_calls": sampled.call_count,
                                 "generated_tokens": output.shape[1] - 3})
                print("GENERATION_DISPATCH_REGRESSION=" + json.dumps({"architecture": spec["architecture"],
                      "random_cpu_fixture": True, "original_bug_reproduced": True,
                      "original_multinomial_calls": old_sample.call_count, "fixed_variants": rows}, sort_keys=True), flush=True)

    def test_conflicting_generate_kwarg_is_rejected_before_real_model_forward(self):
        for spec in self.config["models"]:
            with self.subTest(model=spec["id"]):
                model = self.tiny_model(spec)
                intent = self.tiny_intent(spec, gpu.variants(self.config, spec)[2])
                evidence = []
                with patch.object(model, "forward", wraps=model.forward) as forward, \
                     self.assertRaisesRegex(RuntimeError, "differ from variant intent"):
                    gpu.generate_verified(model, self.transformers.GenerationConfig(**intent), intent,
                                          evidence.append, **self.encoded(), do_sample=True)
                forward.assert_not_called()
                self.assertFalse(evidence[0]["matches_variant_intent"])
                self.assertIn("do_sample", evidence[0]["mismatches"])
                self.assertNotIn("_prepare_generation_config", vars(model))


if __name__ == "__main__":
    unittest.main()
