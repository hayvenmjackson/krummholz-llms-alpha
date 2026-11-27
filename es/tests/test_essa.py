"""SVD decomposition, candidate construction and PEFT writing (synthetic adapters, fast)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from es.essa import LoraSVD, svd_of_product
from es.tests.helpers import synthetic_adapter, toy_shapes


class SVDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = synthetic_adapter(self.dir / "src", toy_shapes(), rank=4, seed=3)
        self.weights = load_file(str(self.src / "adapter_model.safetensors"))
        self.svd = LoraSVD.from_adapter(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_compact_svd_is_exact(self):
        g = torch.Generator().manual_seed(1)
        B = torch.randn(37, 5, generator=g, dtype=torch.float64)
        A = torch.randn(5, 23, generator=g, dtype=torch.float64)
        u, s, vh = svd_of_product(B, A)
        self.assertTrue(torch.allclose((u * s) @ vh, B @ A, atol=1e-12, rtol=0))
        self.assertTrue(torch.allclose(u.T @ u, torch.eye(5, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(vh @ vh.T, torch.eye(5, dtype=torch.float64), atol=1e-12))
        self.assertTrue(bool((s[:-1] >= s[1:]).all()) and bool((s >= 0).all()))
        ref = torch.linalg.svdvals(B @ A)[:5]
        self.assertTrue(torch.allclose(s, ref, atol=1e-10))

    def test_modules_narrower_than_the_rank_keep_the_layout(self):
        # a per-head gate projection can have fewer outputs (or inputs) than the LoRA rank
        g = torch.Generator().manual_seed(2)
        u, s, vh = svd_of_product(torch.randn(3, 4, generator=g, dtype=torch.float64),
                                  torch.randn(4, 16, generator=g, dtype=torch.float64))
        self.assertEqual((tuple(u.shape), tuple(s.shape), tuple(vh.shape)), ((3, 4), (4,), (4, 16)))
        self.assertEqual(float(s[3]), 0.0)
        shapes = [("base_model.model.model.layers.0.linear_attn.in_proj_a", 3, 16),
                  ("base_model.model.model.layers.0.linear_attn.out_proj", 16, 2),
                  ("base_model.model.model.layers.0.mlp.down_proj", 16, 40)]
        src = synthetic_adapter(self.dir / "narrow", shapes, rank=4, seed=4)
        svd = LoraSVD.from_adapter(src)
        self.assertEqual(svd.dim, 12)
        theta = torch.tensor(np.random.default_rng(0).standard_normal(12) * 0.1)
        out = svd.write_adapter(theta, self.dir / "narrow-cand")
        written = load_file(str(out / "adapter_model.safetensors"))
        source = load_file(str(src / "adapter_model.safetensors"))
        for k in source:
            self.assertEqual(written[k].shape, source[k].shape, k)
        z = load_file(str(svd.write_adapter(svd.zeros(), self.dir / "narrow-zero") / "adapter_model.safetensors"))
        for prefix, _, _ in shapes:
            a, b = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
            self.assertTrue(torch.allclose(z[b].double() @ z[a].double(), source[b].double() @ source[a].double(),
                                           atol=1e-6))
        rec = svd.project(written).numpy()
        live = svd.s0_vector().numpy() > 0
        self.assertTrue(np.allclose(rec[live], theta.numpy()[live], atol=1e-5))

    def test_dimension_and_layout(self):
        self.assertEqual(self.svd.dim, 4 * len(toy_shapes()))
        offsets = [m.offset for m in self.svd.modules]
        self.assertEqual(offsets, list(range(0, self.svd.dim, 4)))

    def test_reconstruction_exact_at_zero(self):
        theta = self.svd.zeros()
        for m in self.svd.modules:
            ref = self.weights[m.key_b].double() @ self.weights[m.key_a].double()
            self.assertTrue(torch.allclose(self.svd.delta(m.prefix, theta), ref, atol=1e-12, rtol=0))
        rebuilt = self.svd.module_tensors(theta)
        for m in self.svd.modules:
            prod = rebuilt[m.key_b].double() @ rebuilt[m.key_a].double()
            ref = self.weights[m.key_b].double() @ self.weights[m.key_a].double()
            # float32 storage: relative error at float32 rounding level
            self.assertLess(float((prod - ref).abs().max() / ref.abs().max()), 2e-6)
            self.assertEqual(rebuilt[m.key_a].shape, self.weights[m.key_a].shape)
            self.assertEqual(rebuilt[m.key_b].shape, self.weights[m.key_b].shape)
            self.assertEqual(rebuilt[m.key_a].dtype, self.weights[m.key_a].dtype)

    def test_candidate_carries_intended_singular_values(self):
        rng = np.random.default_rng(0)
        for param in ("relative", "absolute"):
            svd = LoraSVD.from_adapter(self.src, parameterization=param)
            theta = rng.standard_normal(svd.dim) * 0.1
            sd = svd.module_tensors(theta, dtype=torch.float64)
            recovered = svd.project(sd).numpy()
            self.assertTrue(np.allclose(recovered, theta, atol=1e-10), param)
            s0 = svd.s0_vector().numpy()
            expect = s0 * (1 + theta) if param == "relative" else s0 + theta
            got = np.concatenate([svd.singular_values(theta)[m.prefix].numpy() for m in svd.modules])
            self.assertTrue(np.allclose(got, expect, atol=1e-14))

    def test_negative_singular_values_survive_balanced_split(self):
        theta = np.full(self.svd.dim, -1.5)  # relative: S = -0.5 * S0
        sd = self.svd.module_tensors(theta, dtype=torch.float64)
        for m in self.svd.modules:
            prod = sd[m.key_b] @ sd[m.key_a]
            ref = -0.5 * (self.weights[m.key_b].double() @ self.weights[m.key_a].double())
            self.assertTrue(torch.allclose(prod, ref, atol=1e-10))

    def test_write_adapter_matches_source_layout(self):
        out = self.svd.write_adapter(self.svd.zeros(), self.dir / "cand", metadata={"candidate": "parent"})
        self.assertEqual((out / "adapter_config.json").read_bytes(), (self.src / "adapter_config.json").read_bytes())
        written = load_file(str(out / "adapter_model.safetensors"))
        self.assertEqual(sorted(written), sorted(self.weights))
        for k in written:
            self.assertEqual(written[k].shape, self.weights[k].shape)
            self.assertEqual(written[k].dtype, self.weights[k].dtype)
        self.assertEqual(json.loads((out / "ES_CANDIDATE.json").read_text())["candidate"], "parent")
        bf = self.svd.write_adapter(self.svd.zeros(), self.dir / "cand-bf16", dtype=torch.bfloat16)
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in load_file(str(bf / "adapter_model.safetensors")).values()))

    def test_exclusion_and_passthrough(self):
        extra = {"base_model.model.lm_head.lora_embedding_A": torch.ones(4, 10)}
        src = synthetic_adapter(self.dir / "src2", toy_shapes(), rank=4, seed=4, extra_keys=extra)
        svd = LoraSVD.from_adapter(src, exclude=r"mlp\.")
        self.assertTrue(all("mlp" not in m.prefix for m in svd.modules))
        theta = np.full(svd.dim, 0.3)
        sd = svd.module_tensors(theta)
        orig = load_file(str(src / "adapter_model.safetensors"))
        for k, v in orig.items():
            if "mlp" in k or "embedding" in k:
                self.assertTrue(torch.equal(sd[k], v), k)

    def test_cache_round_trip(self):
        self.svd.save(self.dir / "cache")
        again = LoraSVD.load(self.dir / "cache")
        self.assertEqual(again.fingerprint(), self.svd.fingerprint())
        theta = np.linspace(-0.2, 0.2, self.svd.dim)
        a = self.svd.module_tensors(theta)
        b = again.module_tensors(theta)
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]))

    def test_rejects_dora(self):
        src = synthetic_adapter(self.dir / "dora", toy_shapes(1), rank=4)
        cfg = json.loads((src / "adapter_config.json").read_text())
        cfg["use_dora"] = True
        (src / "adapter_config.json").write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            LoraSVD.from_adapter(src)


if __name__ == "__main__":
    unittest.main()
