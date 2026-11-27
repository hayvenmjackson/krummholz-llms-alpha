"""Vision search space: inventory against the real config, bit-exact apply/restore on a tiny random
vision-like stack, antithetic symmetry, the update math by hand, determinism after resume, the
engine hooks against a fake vLLM worker, export for llama.cpp, and the vision config overlay.

All CPU. The tiny stack mirrors the Qwen3.5 vision tower's parameter names and kinds (patch Conv3d,
learned position table, blocks with fused qkv, proj, fc1, fc2 and LayerNorms, 2x2 merger) with random
weights; the optional real-module test builds a tiny transformers Qwen3_5VisionModel.
"""
import copy
import hashlib
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from es import vision as V
from es.backends import vision_hooks as H

ES_DIR = Path(__file__).resolve().parents[1]
TINY_VC = {"deepstack_visual_indexes": [], "depth": 4, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 8,
           "in_channels": 3, "intermediate_size": 12, "num_heads": 2, "num_position_embeddings": 16,
           "out_hidden_size": 10, "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2}
TINY_SCOPE = {"blocks": {"last": 2}, "ranks": {"block": 3, "merger": 4, "vector": 2}}


# ======================================================================= tiny stack
class TinyAttn(nn.Module):
    def __init__(self, h, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(h, 3 * h)
        self.proj = nn.Linear(h, h)

    def forward(self, x):
        n, h = x.shape
        q, k, v = self.qkv(x).reshape(n, 3, self.heads, h // self.heads).permute(1, 2, 0, 3).unbind(0)
        a = torch.softmax(q @ k.transpose(-1, -2) / (h // self.heads) ** 0.5, dim=-1) @ v
        return self.proj(a.transpose(0, 1).reshape(n, h))


class TinyMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.linear_fc1 = nn.Linear(h, inter)
        self.linear_fc2 = nn.Linear(inter, h)

    def forward(self, x):
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(x), approximate="tanh"))


class TinyBlock(nn.Module):
    def __init__(self, h, inter, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(h, eps=1e-6)
        self.attn = TinyAttn(h, heads)
        self.norm2 = nn.LayerNorm(h, eps=1e-6)
        self.mlp = TinyMLP(h, inter)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyMerger(nn.Module):
    def __init__(self, h, merge, out):
        super().__init__()
        self.hm = h * merge * merge
        self.norm = nn.LayerNorm(h, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hm, self.hm)
        self.linear_fc2 = nn.Linear(self.hm, out)

    def forward(self, x):
        x = self.norm(x).view(-1, self.hm)
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(x)))


class TinyPatch(nn.Module):
    def __init__(self, cin, tp, p, h):
        super().__init__()
        self.proj = nn.Conv3d(cin, h, kernel_size=(tp, p, p), stride=(tp, p, p))


class TinyVisual(nn.Module):
    def __init__(self, vc):
        super().__init__()
        h = vc["hidden_size"]
        self.patch_embed = TinyPatch(vc["in_channels"], vc["temporal_patch_size"], vc["patch_size"], h)
        self.pos_embed = nn.Embedding(vc["num_position_embeddings"], h)
        self.blocks = nn.ModuleList([TinyBlock(h, vc["intermediate_size"], vc["num_heads"])
                                     for _ in range(vc["depth"])])
        self.merger = TinyMerger(h, vc["spatial_merge_size"], vc["out_hidden_size"])

    def forward(self, x):
        x = x + self.pos_embed.weight[: x.shape[0]]
        for b in self.blocks:
            x = b(x)
        return self.merger(x)


class TinyInner(nn.Module):
    def __init__(self, vc):
        super().__init__()
        self.visual = TinyVisual(vc)
        self.language_model = nn.Linear(vc["out_hidden_size"], 6, bias=False)


class TinyVLM(nn.Module):
    """Parameter names match the checkpoint: model.visual.* and model.language_model.*."""

    def __init__(self, vc=TINY_VC, seed=0, dtype=torch.float32):
        super().__init__()
        torch.manual_seed(seed)
        self.model = TinyInner(vc)
        with torch.no_grad():
            for p in self.parameters():
                p.add_(0.05 * torch.randn_like(p))      # LayerNorm and bias tensors are not all ones/zeros
        self.to(dtype)

    def forward(self, x):
        return self.model.language_model(self.model.visual(x))


def tiny_input(vc=TINY_VC, n=8, seed=1, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, vc["hidden_size"], generator=g).to(dtype)


def all_param_bytes(model):
    return {n: V.tensor_sha256(p) for n, p in model.named_parameters()}


def rng_vec(dim, seed, scale=1.0):
    return (np.random.default_rng(seed).standard_normal(dim) * scale).astype(np.float32)


def tiny_space(model, scope=TINY_SCOPE, core="diag"):
    return V.VisionSpace.from_module(model, dict(scope, core=core), TINY_VC)


_saved_threads = None


def setUpModule():
    # tiny matrices: one intra-op thread avoids oversubscription on a busy machine; restored afterwards
    global _saved_threads
    _saved_threads = torch.get_num_threads()
    torch.set_num_threads(1)


def tearDownModule():
    if _saved_threads:
        torch.set_num_threads(_saved_threads)


# ======================================================================= inventory
class InventoryTests(unittest.TestCase):
    def test_atlantic_totals_match_checkpoint_header(self):
        s = V.inventory_summary()
        self.assertEqual(s["tensors"], 333)
        self.assertEqual(s["params"], V.ATLANTIC_VISION_PARAMS)
        self.assertEqual(s["blocks"], 27)
        self.assertEqual(s["params_per_block"], 15_239_504)
        self.assertEqual(s["by_group"], {"patch_embed": 1_770_624, "pos_embed": 2_654_208, "block": 411_466_608,
                                         "merger": 44_838_656})
        shapes = {x.name: x.shape for x in V.vision_inventory()}
        # shapes read from the safetensors header of model-00001-of-00018 at revision 37561816
        self.assertEqual(shapes["visual.blocks.26.attn.qkv.weight"], (3456, 1152))
        self.assertEqual(shapes["visual.blocks.0.attn.proj.weight"], (1152, 1152))
        self.assertEqual(shapes["visual.blocks.5.mlp.linear_fc1.weight"], (4304, 1152))
        self.assertEqual(shapes["visual.blocks.5.mlp.linear_fc2.weight"], (1152, 4304))
        self.assertEqual(shapes["visual.merger.linear_fc1.weight"], (4608, 4608))
        self.assertEqual(shapes["visual.merger.linear_fc2.weight"], (5120, 4608))
        self.assertEqual(shapes["visual.merger.norm.weight"], (1152,))
        self.assertEqual(shapes["visual.patch_embed.proj.weight"], (1152, 3, 2, 16, 16))
        self.assertEqual(shapes["visual.pos_embed.weight"], (2304, 1152))
        self.assertEqual(len({x.gguf for x in V.vision_inventory()}), 333)

    def test_matches_transformers_model_on_meta_device(self):
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        except Exception:
            self.skipTest("transformers without Qwen3.5")
        cfg = {k: v for k, v in V.ATLANTIC_VISION_CONFIG.items() if k != "deepstack_visual_indexes"}
        with torch.device("meta"):
            m = Qwen3_5VisionModel(Qwen3_5VisionConfig(**cfg))
        got = {f"visual.{n}": tuple(p.shape) for n, p in m.named_parameters()}
        want = {x.name: x.shape for x in V.vision_inventory()}
        self.assertEqual(got, want)

    def test_tiny_stack_matches_tiny_inventory(self):
        got = {V.canonical_name(n): tuple(p.shape) for n, p in TinyVLM().named_parameters()
               if V.canonical_name(n)}
        self.assertEqual(got, {x.name: x.shape for x in V.vision_inventory(TINY_VC)})

    def test_canonical_names(self):
        c = V.canonical_name
        self.assertEqual(c("model.visual.blocks.3.attn.qkv.weight"), "visual.blocks.3.attn.qkv.weight")
        self.assertEqual(c("base_model.model.model.visual.merger.linear_fc1.bias"), "visual.merger.linear_fc1.bias")
        self.assertEqual(c("visual.blocks.3.attn.qkv.base_layer.weight"), "visual.blocks.3.attn.qkv.weight")
        self.assertIsNone(c("model.language_model.layers.0.self_attn.q_proj.weight"))
        self.assertIsNone(c("model.visualizer.weight"))

    def test_default_scope_dimensions(self):
        p = V.vision_plan()
        self.assertEqual(p["blocks"], list(range(18, 27)))
        self.assertEqual(p["targets"], 9 * 4 + 2 + 4)
        self.assertEqual(p["dim"], 9 * 4 * 48 + 2 * 128 + 4 * 16)
        self.assertEqual(p["dim"], 2048)
        self.assertEqual(p["touched_params"], 9 * 15_224_832 + 21_233_664 + 23_592_960 + 4608 + 5120 + 2 * 1152)
        self.assertLess(p["originals_mb_bf16"], 400)
        self.assertEqual(len(V.select_blocks(27, {"fraction": 0.25})), 7)
        self.assertEqual(len(V.select_blocks(27, {"fraction": 0.5})), 14)
        self.assertEqual(V.select_blocks(27, {"indices": [26, 20]}), [20, 26])
        with self.assertRaises(ValueError):
            V.select_blocks(27, {"last": 9, "fraction": 0.5})
        self.assertEqual(len(V.vision_plan({"blocks": {"fraction": 0.5}})["blocks"]), 14)
        self.assertEqual(V.vision_plan({"core": "full", "ranks": {"block": 8, "merger": 8, "vector": 4}})["dim"],
                         9 * 4 * 64 + 2 * 64 + 4 * 4)


# ======================================================================= space
class SpaceTests(unittest.TestCase):
    def test_build_is_deterministic_and_round_trips(self):
        m = TinyVLM()
        s1, s2 = tiny_space(m), tiny_space(TinyVLM())
        self.assertEqual(s1.fingerprint(), s2.fingerprint())
        self.assertEqual(s1.dim, 2 * 4 * 3 + 2 * 4 + 4 * 2)
        with tempfile.TemporaryDirectory() as d:
            s1.save(Path(d) / "space")
            s3 = V.VisionSpace.load(Path(d) / "space")
            self.assertEqual(s3.fingerprint(), s1.fingerprint())
            from safetensors.torch import load_file, save_file
            f = Path(d) / "space" / "bases.safetensors"
            t = load_file(str(f))
            k = sorted(t)[0]
            t[k] = t[k].clone()
            t[k][0, 0] += 1e-3
            save_file(t, str(f), metadata={"format": V.SPACE_FORMAT})
            with self.assertRaises(ValueError):
                V.VisionSpace.load(Path(d) / "space")

    def test_bases_are_orthonormal_and_seeded(self):
        s = tiny_space(TinyVLM())
        for t in s.targets:
            U = s.U[t.name].double()
            self.assertTrue(torch.allclose(U.T @ U, torch.eye(t.rank, dtype=torch.float64), atol=1e-6), t.name)
            if t.kind == "matrix":
                Vb = s.V[t.name].double()
                self.assertTrue(torch.allclose(Vb.T @ Vb, torch.eye(t.rank, dtype=torch.float64), atol=1e-6))
        other = V.VisionSpace.from_module(TinyVLM(), dict(TINY_SCOPE, basis_seed=7), TINY_VC)
        self.assertNotEqual(other.fingerprint(), s.fingerprint())

    def test_scale_is_rms_singular_value(self):
        m = TinyVLM()
        s = tiny_space(m)
        t = s.by_name["visual.merger.linear_fc1.weight"]
        w = m.model.visual.merger.linear_fc1.weight.double()
        self.assertAlmostEqual(t.scale, float(torch.linalg.svdvals(w.detach()).pow(2).mean().sqrt()), places=10)

    def test_full_core(self):
        s = tiny_space(TinyVLM(), core="full")
        self.assertEqual(s.dim, 2 * 4 * 9 + 2 * 16 + 4 * 2)
        c = rng_vec(s.dim, 3)
        t = s.by_name["visual.blocks.3.attn.proj.weight"]
        C = torch.from_numpy(c[t.offset:t.offset + t.size]).double().reshape(t.rank, t.rank)
        want = t.scale * s.U[t.name].double() @ C @ s.V[t.name].double().T
        self.assertTrue(torch.allclose(s.delta(t.name, c), want, atol=1e-12))


# ======================================================================= apply / restore
class ApplyRestoreTests(unittest.TestCase):
    def _cycle(self, dtype):
        m = TinyVLM(dtype=dtype)
        s = tiny_space(m)
        before = all_param_bytes(m)
        x = tiny_input(dtype=dtype)
        with torch.no_grad():
            y0 = m(x).clone()
        p = V.VisionPatcher(m, s)
        for i in range(6):
            c = rng_vec(s.dim, 100 + i, scale=[0.1, 1.0, 5.0][i % 3])
            p.apply(c)
            with torch.no_grad():
                y = m(x)
            self.assertFalse(torch.equal(y, y0))
            if i % 2:
                p.restore()
        p.restore()
        self.assertEqual(all_param_bytes(m), before)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), y0))
        self.assertEqual(p.mismatches(), [])

    def test_restore_is_bit_exact_bf16(self):
        self._cycle(torch.bfloat16)

    def test_restore_is_bit_exact_fp32(self):
        self._cycle(torch.float32)

    def test_untouched_parameters_never_change(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        before = all_param_bytes(m)
        p = V.VisionPatcher(m, s)
        p.apply(rng_vec(s.dim, 5, 3.0))
        touched = set(p.param_names.values())
        after = all_param_bytes(m)
        for n in before:
            if n in touched:
                self.assertNotEqual(before[n], after[n], n)
            else:
                self.assertEqual(before[n], after[n], n)
        self.assertNotIn("model.language_model.weight", touched)
        self.assertNotIn("model.visual.blocks.0.attn.qkv.weight", touched)
        self.assertIn("model.visual.blocks.3.attn.qkv.weight", touched)
        self.assertIn("model.visual.merger.norm.bias", touched)

    def test_apply_never_accumulates_and_zero_is_identity(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        p = V.VisionPatcher(m, s)
        base = p.sha256s()
        c1, c2 = rng_vec(s.dim, 1, 2.0), rng_vec(s.dim, 2, 2.0)
        p.apply(c2)
        direct = p.sha256s()
        p.restore()
        p.apply(c1)
        p.apply(c2)               # apply over apply starts from the pinned originals again
        self.assertEqual(p.sha256s(), direct)
        p.apply(np.zeros(s.dim, dtype=np.float32))
        self.assertEqual(p.sha256s(), base)
        self.assertEqual(p.tag, "base")

    def test_served_value_equals_export_value_and_formula(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        orig = {t.name: dict(m.named_parameters())["model." + t.name].detach().clone() for t in s.targets}
        p = V.VisionPatcher(m, s)
        c = rng_vec(s.dim, 9, 2.0)
        p.apply(c)
        for t in s.targets:
            served = p.params[t.name]
            self.assertTrue(torch.equal(served.view(torch.int16), s.evolved(t.name, orig[t.name], c).view(torch.int16)))
        # float64 stack: the change is exactly s * U diag(c) V^T (numpy, independent of the torch path)
        m64 = TinyVLM(dtype=torch.float64)
        s64 = tiny_space(m64)
        w0 = {t.name: dict(m64.named_parameters())["model." + t.name].detach().clone() for t in s64.targets}
        V.VisionPatcher(m64, s64).apply(c)
        for t in s64.targets:
            U = s64.U[t.name].numpy().astype(np.float64)
            ct = c[t.offset:t.offset + t.size].astype(np.float64)
            if t.kind == "matrix":
                want = (U * (t.scale * ct)[None, :]) @ s64.V[t.name].numpy().astype(np.float64).T
            else:
                want = U @ (t.scale * ct)
            got = (dict(m64.named_parameters())["model." + t.name].detach() - w0[t.name]).numpy().reshape(want.shape)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    def test_context_manager_restores_on_error(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        p = V.VisionPatcher(m, s)
        base = p.sha256s()
        with self.assertRaises(KeyError):
            with p.candidate(rng_vec(s.dim, 4)):
                raise KeyError("generation failed")
        self.assertEqual(p.sha256s(), base)
        self.assertEqual(p.state, "base")

    def test_attach_fails_closed(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        other = TinyVLM(seed=3, dtype=torch.bfloat16)          # different weights: snapshot mismatch
        before = all_param_bytes(other)
        with self.assertRaises(V.VisionAttachError):
            V.VisionPatcher(other, s)
        self.assertEqual(all_param_bytes(other), before)
        q = TinyVLM(dtype=torch.bfloat16)                       # a quantized (integer) touched tensor
        q.model.visual.blocks[3].attn.qkv.weight = nn.Parameter(torch.zeros(24, 8, dtype=torch.int8),
                                                                requires_grad=False)
        with self.assertRaises(V.VisionAttachError):
            V.VisionPatcher(q, s)
        wide = dict(TINY_VC, hidden_size=16, intermediate_size=12)  # sharded or reshaped tower
        with self.assertRaises(V.VisionAttachError):
            V.VisionPatcher(TinyVLM(vc=wide, dtype=torch.bfloat16), s)
        with self.assertRaises(ValueError):
            V.VisionPatcher(m, s).apply(np.zeros(s.dim + 1, dtype=np.float32))
        with self.assertRaises(ValueError):
            V.VisionPatcher(m, s).apply(np.full(s.dim, np.nan, dtype=np.float32))

    def test_restore_audit_detects_external_writes(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        p = V.VisionPatcher(m, s)
        with torch.no_grad():
            m.model.visual.merger.linear_fc2.weight[0, 0] += 1.0
        self.assertEqual(p.mismatches(), ["visual.merger.linear_fc2.weight"])
        p.restore()
        self.assertEqual(p.mismatches(), [])

    def test_lora_wrapped_and_peft_style_names_resolve(self):
        m = TinyVLM(dtype=torch.bfloat16)
        s = tiny_space(m)
        outer = nn.Module()
        outer.base_model = nn.Module()
        outer.base_model.model = m                              # base_model.model.model.visual.*
        lin = m.model.visual.merger.linear_fc1
        wrapped = nn.Module()
        wrapped.base_layer = lin                                # vLLM or PEFT LoRA wrapper
        m.model.visual.merger.linear_fc1 = wrapped
        p = V.VisionPatcher(outer, s)
        self.assertIn("base_model.model.model.visual.merger.linear_fc1.base_layer.weight", p.param_names.values())
        p.apply(rng_vec(s.dim, 2))
        p.restore()
        self.assertEqual(p.mismatches(), [])


# ======================================================================= antithetic and update
class ESMathTests(unittest.TestCase):
    def test_derive_seed_matches_text_engine(self):
        from es.data import derive_seed
        for parts in [(20261127, "noise", 3, 1), (1, V.VISION_NOISE_TAG, 0, 0), ("a", "b")]:
            self.assertEqual(V.derive_seed(*parts), derive_seed(*parts))

    def test_vision_noise_is_separate_from_text_noise(self):
        from es.data import derive_seed
        vs = V.vision_noise_seeds(20261127, 5, 8)
        ts = [derive_seed(20261127, "noise", 5, j) for j in range(8)]
        self.assertFalse(set(vs) & set(ts))
        self.assertEqual(vs, V.vision_noise_seeds(20261127, 5, 8))
        a = V.vision_noise(vs[0], 3000)
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, V.vision_noise(vs[0], 3000)))

    def test_antithetic_candidates_share_bases_and_mirror(self):
        m = TinyVLM(dtype=torch.float64)
        s = tiny_space(m)
        sigma = 0.0625
        es = V.VisionES(s.dim, {"sigma": sigma}, run_seed=11)
        es.z = (np.round(rng_vec(s.dim, 8, 0.5) * 256) / 256).astype(np.float32)   # dyadic: exact in FP32
        cands = dict(es.candidates(gen=4, n_pairs=3))
        self.assertEqual(list(cands), V.candidate_names(3))
        seeds = es.seeds(4, 3)
        z = es.z.astype(np.float64)
        for j in range(3):
            e = V.vision_noise(seeds[j], s.dim).astype(np.float64)
            np.testing.assert_array_equal(cands[f"p{j:03d}+"], z + sigma * e)
            np.testing.assert_array_equal(cands[f"p{j:03d}-"], z - sigma * e)
        # the delta is exactly antisymmetric in c (one shared basis per tensor for both signs)
        c = rng_vec(s.dim, 4)
        for t in s.targets:
            self.assertTrue(torch.equal(s.delta(t.name, c), -s.delta(t.name, -c)), t.name)
        # weights: W(z + d) - W(z) mirrors W(z - d) - W(z) for a dyadic perturbation d
        d = (np.round(rng_vec(s.dim, 5) * 64) / 1024).astype(np.float32)
        plus, minus = es.z + d, es.z - d
        np.testing.assert_array_equal(plus - es.z, -(minus - es.z))
        p = V.VisionPatcher(m, s)

        def weights(cv):
            p.apply(cv)
            out = {n: t.detach().clone() for n, t in p.params.items()}
            p.restore()
            return out

        wz, wp, wm = weights(es.z), weights(plus), weights(minus)
        for n in wz:
            self.assertTrue(torch.allclose(wp[n] - wz[n], -(wm[n] - wz[n]), rtol=0, atol=1e-12), n)
            self.assertFalse(torch.equal(wp[n], wz[n]), n)

    def test_update_by_hand(self):
        dim, sigma, awd = 6, 0.1, 0.005
        es = V.VisionES(dim, {"sigma": sigma, "awd": awd}, run_seed=3)
        es.z = np.array([0.1, -0.2, 0.0, 0.3, 0.05, -0.01], dtype=np.float32)
        z0 = es.z.astype(np.float64)
        up, um = np.array([0.5, -0.1, 0.2]), np.array([-0.5, 0.3, 0.2])
        info = es.update(0, up, um)
        seeds = V.vision_noise_seeds(3, 0, 3)
        c = (up - um) / (2 * 3 * sigma)
        g = sum(c[j] * V.vision_noise(seeds[j], dim).astype(np.float64) for j in range(3))
        lr = 1.7 * sigma ** 2
        want = ((1 - awd) * (z0 + lr * g)).astype(np.float32)
        np.testing.assert_array_equal(es.z, want)
        self.assertEqual(info["coeffs"][2], 0.0)
        self.assertAlmostEqual(info["lr"], lr)
        self.assertEqual(es.gen, 1)
        with self.assertRaises(ValueError):
            es.update(0, up, um)                                # wrong generation

    def test_equal_utilities_only_decay(self):
        es = V.VisionES(5, {"sigma": 0.05, "awd": 0.01}, run_seed=1)
        es.z = np.ones(5, dtype=np.float32)
        es.update(0, [0.2, 0.2], [0.2, 0.2])
        np.testing.assert_array_equal(es.z, np.full(5, 0.99, dtype=np.float32))

    def test_coefficients_scale_with_own_sigma(self):
        a = V.VisionES(4, {"sigma": 0.05}, 0).coefficients([0.5, -0.5], [-0.5, 0.5])
        b = V.VisionES(4, {"sigma": 0.10}, 0).coefficients([0.5, -0.5], [-0.5, 0.5])
        np.testing.assert_allclose(a, 2 * b)
        np.testing.assert_allclose(a, [5.0, -5.0])               # (0.5 - (-0.5)) / (2 x 2 pairs x 0.05)

    def test_adam_and_clipping(self):
        cfg = {"sigma": 0.1, "awd": 0.0, "optimizer": {"kind": "adam", "lr": 0.01, "lr_sigma2_coef": None},
               "z_max_abs": 0.005}
        es = V.VisionES(3, cfg, run_seed=2)
        es.update(0, [1.0], [-1.0])
        g = (2.0 / (2 * 0.1)) * V.vision_noise(V.vision_noise_seeds(2, 0, 1)[0], 3).astype(np.float64)
        m, v = 0.1 * g, 0.001 * g * g
        upd = 0.01 * (m / 0.1) / (np.sqrt(v / 0.001) + 1e-8)
        np.testing.assert_array_equal(es.z, np.clip(upd, -0.005, 0.005).astype(np.float32))

    def test_replay_rebuilds_z(self):
        es = V.VisionES(7, {"sigma": 0.2}, run_seed=9)
        entries = []
        for gen in range(4):
            up = np.random.default_rng(gen).uniform(-0.5, 0.5, 3)
            entries.append({"gen": gen, "vision": es.update(gen, up, -up[::-1])})
        np.testing.assert_array_equal(V.VisionES.replay(7, {"sigma": 0.2}, 9, entries), es.z)
        entries[2]["vision"]["coeffs"][0] += 1e-6
        with self.assertRaises(ValueError):
            V.VisionES.replay(7, {"sigma": 0.2}, 9, entries)


def _centered_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (len(x) - 1) - 0.5


class ResumeTests(unittest.TestCase):
    """A small ES run on the tiny stack through the patcher; interrupted and resumed from disk."""

    N = 6
    CFG = {"sigma": 0.5, "awd": 0.001, "optimizer": {"kind": "sgd", "lr_sigma2_coef": 6.0}}

    def _fitness(self, patcher, model, x, y, c):
        with patcher.candidate(c):
            with torch.no_grad():
                out = model(x).double()
        return -float(((out - y) ** 2).mean())

    def _setup(self, space_dir=None):
        m = TinyVLM(dtype=torch.bfloat16)
        s = V.VisionSpace.load(space_dir) if space_dir else tiny_space(m)
        p = V.VisionPatcher(m, s)
        x = tiny_input(dtype=torch.bfloat16)
        hidden = rng_vec(s.dim, 77, 1.5)
        with p.candidate(hidden):
            with torch.no_grad():
                y = m(x).double()
        return m, s, p, x, y

    def _generation(self, es, p, m, x, y, gen):
        cands = es.candidates(gen, self.N)
        f = {name: self._fitness(p, m, x, y, c) for name, c in cands}
        fp = [f[f"p{j:03d}+"] for j in range(self.N)]
        fm = [f[f"p{j:03d}-"] for j in range(self.N)]
        u = _centered_ranks(fp + fm)
        info = es.update(gen, u[:self.N], u[self.N:])
        return f["parent"], info["z_sha256_after"]

    def test_resume_is_bitwise_identical_and_es_climbs(self):
        G, cut = 12, 5
        m, s, p, x, y = self._setup()
        es = V.VisionES(s.dim, self.CFG, run_seed=20261127)
        full = [self._generation(es, p, m, x, y, g) for g in range(G)]
        z_full = es.z.copy()
        self.assertGreater(full[-1][0], full[0][0])            # the parent's fitness improved
        with tempfile.TemporaryDirectory() as d:
            s.save(Path(d) / "space")
            m2, s2, p2, x2, y2 = self._setup()
            es2 = V.VisionES(s2.dim, self.CFG, run_seed=20261127)
            first = [self._generation(es2, p2, m2, x2, y2, g) for g in range(cut)]
            es2.save(Path(d) / "vision-state.npz", extra={"space_fingerprint": s2.fingerprint()})
            del m2, s2, p2, es2
            m3, s3, p3, x3, y3 = self._setup(Path(d) / "space")   # fresh process state, space from disk
            es3 = V.VisionES.load(Path(d) / "vision-state.npz", self.CFG, 20261127,
                                  expect={"space_fingerprint": s3.fingerprint(), "gen": cut})
            rest = [self._generation(es3, p3, m3, x3, y3, g) for g in range(cut, G)]
            self.assertEqual(first + rest, full)
            np.testing.assert_array_equal(es3.z, z_full)
            with self.assertRaises(RuntimeError):
                V.VisionES.load(Path(d) / "vision-state.npz", self.CFG, 20261127, expect={"space_fingerprint": "x"})


# ======================================================================= engine hooks
class FakeParallel:
    def __init__(self, ext):
        self.worker_extension_cls = ext


class FakeEngine:
    def __init__(self, ext, with_reset=True):
        self.vllm_config = type("C", (), {"parallel_config": FakeParallel(ext)})()
        self.resets = 0
        if with_reset:
            self.reset_encoder_cache = self._reset

    def _reset(self):
        self.resets += 1


class FakeWorker(H.VisionWorkerExtension):
    def __init__(self, model):
        self.model_runner = type("R", (), {"model": model})()
        self.rank = 0


class FakeLLM:
    """collective_rpc like vLLM V1: by method name on the worker, or a callable called as fn(worker, *args).
    Arguments and results are pickled to prove they can cross the process boundary."""

    def __init__(self, model, ext=H.WORKER_EXTENSION, with_reset=True):
        self.worker = FakeWorker(model)
        self.llm_engine = FakeEngine(ext, with_reset)
        self.calls = []
        self.prefix_resets = 0

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        self.prefix_resets += 1
        return True

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        args = pickle.loads(pickle.dumps(args))
        if isinstance(method, str):
            self.calls.append(method)
            out = getattr(self.worker, method)(*args, **(kwargs or {}))
        else:
            method = pickle.loads(pickle.dumps(method))
            self.calls.append(method.__name__)
            out = method(self.worker, *args, **(kwargs or {}))
        return [pickle.loads(pickle.dumps(out))]


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = TinyVLM(dtype=torch.bfloat16)
        self.space = tiny_space(self.model)
        self.space_dir = self.space.save(Path(self.tmp.name) / "space")

    def tearDown(self):
        self.tmp.cleanup()

    def test_extension_path_apply_restore_and_cache_reset(self):
        llm = FakeLLM(self.model)
        before = all_param_bytes(self.model)
        ctl = H.VLLMVisionController(llm, self.space_dir)
        self.assertTrue(ctl.use_extension)
        self.assertEqual(ctl.cache_policy, "reset")
        info = ctl.attach()
        self.assertEqual(info["workers"][0]["fingerprint"], self.space.fingerprint())
        c = rng_vec(self.space.dim, 21, 2.0)
        tag = ctl.apply(c)
        self.assertEqual(tag, self.space.tag(c))
        f = ctl.request_fields(["aa" * 32])
        self.assertEqual(f, {"multi_modal_uuids": {"image": ["aa" * 32]}})
        self.assertNotEqual(all_param_bytes(self.model), before)
        ctl.restore()
        self.assertEqual(all_param_bytes(self.model), before)
        self.assertEqual(llm.llm_engine.resets, 3)              # encoder cache: attach, apply, restore
        self.assertEqual(llm.prefix_resets, 3)                  # prefix cache: the same three times
        self.assertEqual(llm.calls[:3], ["vision_attach", "vision_apply", "vision_restore"])
        self.assertEqual(ctl.mismatches(), [])
        ctl.close()
        self.assertEqual(llm.worker.vision_status()["attached"], False)

    def test_callable_path_and_keyed_policy_on_older_vllm(self):
        llm = FakeLLM(self.model, ext="", with_reset=False)
        ctl = H.VLLMVisionController(llm, self.space_dir)
        self.assertFalse(ctl.use_extension)
        self.assertEqual(ctl.cache_policy, "keyed")
        ctl.attach()
        c = rng_vec(self.space.dim, 22)
        with ctl.candidate(c) as tag:
            self.assertEqual(ctl.request_fields(["cc"]), {"multi_modal_uuids": {"image": [f"cc@{tag}"]},
                                                          "cache_salt": f"vision-{tag}"})
            served = ctl.served_sha256()
        self.assertEqual(llm.prefix_resets, 0)
        self.assertEqual(llm.calls[:2], ["rpc_attach", "rpc_apply"])
        hf = H.HFVisionController(TinyVLM(dtype=torch.bfloat16), self.space)
        hf.attach()
        hf.apply(c)
        self.assertEqual(hf.served_sha256(), served)            # same weights through both paths
        with self.assertRaises(RuntimeError):
            H.VLLMVisionController(llm, self.space_dir, {"cache_policy": "reset"})

    def test_worker_refuses_other_space_and_unattached_calls(self):
        llm = FakeLLM(self.model)
        with self.assertRaises(RuntimeError):
            llm.collective_rpc("vision_restore")
        ctl = H.VLLMVisionController(llm, self.space_dir)
        ctl.attach()
        c = rng_vec(self.space.dim, 1).astype("<f4")
        with self.assertRaises(RuntimeError):
            llm.collective_rpc("vision_apply", args=(c.tobytes(), "t", "0" * 64))
        before = all_param_bytes(self.model)
        with self.assertRaises(ZeroDivisionError):
            with ctl.candidate(c):
                1 / 0
        self.assertEqual(all_param_bytes(self.model), before)

    def test_make_controller_and_engine_kwargs(self):
        class VB:
            llm = FakeLLM(self.model)

        class HB:
            model = None
            base_model = TinyVLM(dtype=torch.bfloat16)

        class MB:
            pass

        self.assertIsInstance(H.make_vision_controller(VB(), self.space_dir), H.VLLMVisionController)
        self.assertIsInstance(H.make_vision_controller(HB(), self.space_dir), H.HFVisionController)
        rec = H.make_vision_controller(MB(), self.space_dir)
        self.assertIsInstance(rec, H.RecordingVisionController)
        rec.attach()
        c = rng_vec(self.space.dim, 3)
        with rec.candidate(c):
            np.testing.assert_array_equal(rec.current, c)
        self.assertFalse(rec.current.any())
        kw = H.vision_engine_kwargs({}, max_pixels=512 * 512)
        self.assertEqual(kw["worker_extension_cls"], H.WORKER_EXTENSION)
        self.assertEqual(kw["limit_mm_per_prompt"], {"image": 1, "video": 0})
        self.assertEqual(kw["mm_processor_kwargs"]["size"]["longest_edge"], 262144)

    def test_request_helpers(self):
        msgs = H.image_messages("How many fish?", system="sys")
        self.assertEqual(msgs[1]["content"][0], {"type": "image"})
        req = H.image_request([1, 2, 3], ["IMG"], {"cache_salt": "vision-base"})
        self.assertEqual(req, {"prompt_token_ids": [1, 2, 3], "multi_modal_data": {"image": ["IMG"]},
                               "cache_salt": "vision-base"})
        try:
            from PIL import Image
        except ImportError:
            return
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.png"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(H.load_image_verified(path, sha).size, (4, 4))
            with self.assertRaises(ValueError):
                H.load_image_verified(path, "0" * 64)


class WavePlanTests(unittest.TestCase):
    def test_every_request_once_and_image_requests_under_their_candidate(self):
        names = V.candidate_names(3)
        for mode in ("per_candidate", "serial"):
            waves = V.plan_waves(names, n_text=10, n_image=4, mode=mode)
            reqs = [r for w in waves for r in w["requests"]]
            self.assertEqual(len(reqs), len(set(reqs)))
            self.assertEqual(len(reqs), len(names) * 14)
            for w in waves:
                for cand, kind, _ in w["requests"]:
                    if kind == "image":
                        self.assertEqual(cand, w["vision"])
        per = V.plan_waves(names, 10, 4)
        self.assertEqual([w["vision"] for w in per], names)
        for w in per:                                            # one candidate (one LoRA) per wave
            self.assertEqual({c for c, _, _ in w["requests"]}, {w["vision"]})
        self.assertEqual(len(V.plan_waves(names, 10, 4, "serial")), 1 + len(names))
        with self.assertRaises(ValueError):
            V.plan_waves(names, 10, 4, "interleaved")


# ======================================================================= real tiny Qwen3.5 tower
class RealTinyTowerTests(unittest.TestCase):
    def test_qwen3_5_vision_model_apply_restore(self):
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        except Exception:
            self.skipTest("transformers without Qwen3.5")
        vc = dict(depth=4, hidden_size=32, intermediate_size=48, num_heads=4, num_position_embeddings=16,
                  out_hidden_size=40, patch_size=4, temporal_patch_size=2, spatial_merge_size=2, in_channels=3)
        torch.manual_seed(0)
        cfg = Qwen3_5VisionConfig(**vc)
        cfg._attn_implementation = "eager"
        holder = nn.Module()
        holder.visual = Qwen3_5VisionModel(cfg).to(torch.bfloat16).eval()
        with torch.no_grad():
            for p in holder.parameters():
                p.add_((0.05 * torch.randn(p.shape)).to(p.dtype))
        space = V.VisionSpace.from_module(holder, {"blocks": {"last": 2}, "ranks": {"block": 4, "merger": 8,
                                                                                     "vector": 2}},
                                          dict(vc, deepstack_visual_indexes=[]))
        x = torch.randn(16, 3 * 2 * 4 * 4, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
        grid = torch.tensor([[1, 4, 4]])

        def run():
            with torch.no_grad():
                return holder.visual(x, grid_thw=grid).pooler_output.clone()

        y0 = run()
        p = V.VisionPatcher(holder, space)
        p.apply(np.zeros(space.dim, dtype=np.float32))
        self.assertTrue(torch.equal(run(), y0))
        p.apply(rng_vec(space.dim, 5, 3.0))
        self.assertFalse(torch.equal(run(), y0))
        p.restore()
        self.assertTrue(torch.equal(run(), y0))


# ======================================================================= export
def write_snapshot(d: Path, model: TinyVLM):
    from safetensors.torch import save_file
    sd = {n: p.detach().clone().contiguous() for n, p in model.named_parameters()}
    vis = {k: v for k, v in sd.items() if ".visual." in k}
    keys = sorted(vis)
    shard1 = {k: vis[k] for k in keys[: len(keys) // 2]}
    shard1["model.language_model.embed_tokens.weight"] = torch.randn(5, 4).to(torch.bfloat16)
    shard2 = {k: vis[k] for k in keys[len(keys) // 2:]}
    shard3 = {"model.language_model.layers.0.mlp.up_proj.weight": torch.randn(3, 4).to(torch.bfloat16)}
    d.mkdir(parents=True)
    wm = {}
    for i, sh in enumerate((shard1, shard2, shard3), 1):
        name = f"model-0000{i}-of-00003.safetensors"
        save_file(sh, str(d / name), metadata={"format": "pt"})
        wm.update({k: name for k in sh})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                               "text_config": {"hidden_size": 4}, "vision_config": TINY_VC}))
    (d / "preprocessor_config.json").write_text(json.dumps({"image_mean": [0.5] * 3, "image_std": [0.5] * 3}))


class ExportTests(unittest.TestCase):
    def test_mmproj_and_full_exports(self):
        from safetensors.torch import load_file
        m = TinyVLM(dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_snapshot(d / "base", m)
            space = V.VisionSpace.from_snapshot(d / "base", TINY_SCOPE)
            self.assertEqual(space.fingerprint(), tiny_space(m).fingerprint())
            z = rng_vec(space.dim, 12, 1.0)
            p = V.VisionPatcher(m, space)
            p.apply(z)
            served = {"model." + n: t.detach().clone() for n, t in p.params.items()}
            p.restore()
            base_sd = {n: t.detach().clone() for n, t in m.named_parameters()}

            out = V.export_vision(space, z, d / "base", d / "exports" / "vision-final", mode="mmproj", label="t")
            got = load_file(str(out / "model.safetensors"))
            self.assertEqual(sorted(got), sorted(k for k in base_sd if ".visual." in k))
            for k, t in got.items():
                want = served.get(k, base_sd[k])
                self.assertTrue(torch.equal(t.view(torch.int16), want.view(torch.int16)), k)
            man = json.loads((out / "VISION_ES_EXPORT.json").read_text())
            self.assertEqual(man["space_fingerprint"], space.fingerprint())
            self.assertEqual(sum(v["evolved"] for v in man["tensors"].values()), len(space.targets))
            self.assertEqual(man["tensors"]["model.visual.merger.linear_fc1.weight"]["sha256"],
                             V.tensor_sha256(served["model.visual.merger.linear_fc1.weight"]))
            cmd = (out / "MMPROJ_COMMAND.txt").read_text()
            self.assertIn("convert_hf_to_gguf.py", cmd)
            self.assertIn("--mmproj --outtype f16", cmd)
            self.assertIn("mmproj-vision-final-f16.gguf", cmd)
            self.assertTrue((out / "config.json").exists() and (out / "preprocessor_config.json").exists())

            full = V.export_vision(space, z, d / "base", d / "exports" / "full", mode="full")
            self.assertTrue((full / "model-00003-of-00003.safetensors").is_symlink())
            r = V.SnapshotReader(full)
            for k in [k for k in base_sd if ".visual." in k]:
                want = served.get(k, base_sd[k])
                self.assertTrue(torch.equal(r.get(k).view(torch.int16), want.view(torch.int16)), k)
            base_r = V.SnapshotReader(d / "base")
            self.assertTrue(torch.equal(r.get("model.language_model.embed_tokens.weight"),
                                        base_r.get("model.language_model.embed_tokens.weight")))

            key = "model.visual.merger.linear_fc1.weight"            # a searched tensor
            shard = d / "base" / base_r.weight_map[key]
            sd = load_file(str(shard))
            sd[key] = sd[key] + 1
            from safetensors.torch import save_file
            save_file(sd, str(shard), metadata={"format": "pt"})
            with self.assertRaises(ValueError):
                V.export_vision(space, z, d / "base", d / "exports" / "bad")
            self.assertFalse((d / "exports" / "bad").exists())


# ======================================================================= config
class ConfigTests(unittest.TestCase):
    CONFIG = ES_DIR / "config" / "atlantic-v9-vision.yaml"

    def test_overlay_loads_and_sets_vision(self):
        cfg = V.load_vision_config(self.CONFIG)
        v = cfg["vision"]
        self.assertTrue(v["enabled"])
        self.assertEqual(cfg["run"]["seed"], 20261127)
        self.assertEqual(cfg["backend"]["base_model"], "/workspace/models-base/Qwen3.8-27B@37561816")
        self.assertEqual(cfg["essa"]["exclude"], "visual|vision")   # the LoRA search never touches the tower
        plan = V.vision_plan(v["space"])
        self.assertEqual(plan["dim"], 2048)
        wide = V.load_vision_config(self.CONFIG, ["vision.space.blocks={fraction: 0.5}"])
        self.assertEqual(V.vision_plan(wide["vision"]["space"])["blocks"], list(range(13, 27)))
        self.assertEqual(v["space"]["basis_seed"], 20261127)
        self.assertIn("sigma", v["es"])
        self.assertIsNot(v["es"]["sigma"], cfg["es"]["sigma"])
        d = cfg["data"]
        total = 2 * d["pairs_per_batch"] + d["retention"]["per_batch"] + v["data"]["per_batch"]
        self.assertEqual(total, 128)
        self.assertAlmostEqual(v["data"]["per_batch"] / total, 0.25)
        self.assertEqual(cfg["backend"]["engine"]["limit_mm_per_prompt"], {"image": 1, "video": 0})
        self.assertEqual(cfg["backend"]["engine"]["extra_kwargs"]["worker_extension_cls"], H.WORKER_EXTENSION)
        self.assertFalse(cfg["backend"]["engine"]["fp8_base"])
        self.assertNotIn("terminal.jsonl", self.CONFIG.read_text())

    def test_new_files_have_no_em_dash(self):
        for name in ("vision.py", "backends/vision_hooks.py", "config/atlantic-v9-vision.yaml", "VISION.md",
                     "tests/test_vision.py"):
            p = ES_DIR / name
            if p.exists():
                self.assertNotIn(chr(0x2014), p.read_text(encoding="utf-8"), name)


if __name__ == "__main__":
    unittest.main()
