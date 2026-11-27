"""Low-rank seeded ES search space over Atlantic's vision tower, applied in place.

Atlantic is Qwen/Qwen3.8-27B (revision 37561816) plus a LoRA on the language model only. Its
vision tower (`model.visual`, 460,730,096 parameters, all BF16) is a 27-block ViT with a 2x2 patch
merger that projects into the 5120-wide language model. This module evolves part of that tower
without any adapter support in the serving engine:

  for each chosen tensor W (matrix view rows x cols),
      W(c) = W0 + s_W * U diag(c) V^T                      (core "diag", r coefficients)
      W(c) = W0 + s_W * U C V^T,  C = c.reshape(r, r)      (core "full", r*r coefficients)
  for each chosen vector b (bias or LayerNorm weight/bias),
      b(c) = b0 + s_b * U c                                (r coefficients)

U (rows x r) and V (cols x r) are fixed random orthonormal bases (QR of seeded Gaussians, built once
per run and stored with their sha256), s_W = ||W0||_F / sqrt(min(rows, cols)) is the RMS singular value
of the base tensor (s_b = ||b0||_2 / sqrt(n), floored), and c is this tensor's slice of the vision
search vector. The ES master is z (FP32, z = 0 is the base model exactly); candidate j of a
generation uses c = z + sigma_v * eps_j or c = z - sigma_v * eps_j, so the perturbation is
sigma_v * s_W * U diag(eps_j) V^T and both members of an antithetic pair share the same bases.

Applying a candidate writes W0 + delta into the served weight storage with Tensor.copy_ (same
storage, so captured CUDA graphs and compiled encoders see the new values). Every apply starts from a
pinned copy of the original touched tensors, and restore copies that original back, so restore is
exact to the bit and applies never accumulate rounding. Only touched tensors are copied: the default
scope pins 181.9 M parameters (364 MB in BF16).

Default scope (config/atlantic-v9-vision.yaml): the merger fully (linear_fc1, linear_fc2, their
biases, the merger LayerNorm) plus attn.qkv, attn.proj, mlp.linear_fc1 and mlp.linear_fc2 of the last
9 of 27 encoder blocks (33 percent; 7 to 14 blocks is the 25 to 50 percent range), rank 48 per block
matrix, 128 per merger matrix and 16 per vector: 2,048 coordinates in total.

ES on z follows the text engine (es_loop.py): antithetic pairs with seeds derive_seed(run_seed,
"vision-noise", gen, j), the shared shaped utilities of the combined candidates, coefficients
(u+ - u-) / (2 N sigma_v), plain SGD with lr = coef * sigma_v^2 (Adam available), then Anchored Weight
Decay toward zero (the base tower). See VISION.md for the integration points.

Nothing here touches a GPU unless the patched model lives on one. Tests: tests/test_vision.py.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
SPACE_FORMAT = "krummholz-vision-space-v1"
STATE_FORMAT = "krummholz-vision-es-v1"
EXPORT_FORMAT = "krummholz-vision-export-v1"
VISION_NOISE_TAG = "vision-noise"
VISION_BASIS_TAG = "vision-basis"
NOISE_FINGERPRINT_SEED = 20261127

# vision_config of Qwen/Qwen3.8-27B at BASE_REVISION (config.json), plus the text width it projects to.
ATLANTIC_VISION_CONFIG = {
    "deepstack_visual_indexes": [],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "intermediate_size": 4304,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
ATLANTIC_TEXT_HIDDEN = 5120
ATLANTIC_VISION_PARAMS = 460_730_096   # sum over the 333 model.visual.* tensors in the checkpoint header

BLOCK_MODULES = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")
FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}

DEFAULT_VISION: Dict = {
    "enabled": False,
    "space": {
        "core": "diag",                  # diag: r coefficients per matrix; full: r*r
        "basis_seed": 20261127,
        "scale": "rms_singular",         # s_W = ||W0||_F / sqrt(min(rows, cols)); vectors: RMS entry
        "vector_scale_floor": 1.0e-3,
        "blocks": {"last": 9},           # or {"fraction": 0.33} or {"indices": [18, 19, ...]}
        "block_modules": list(BLOCK_MODULES),
        "block_biases": False,
        "merger_matrices": True,
        "merger_vectors": True,          # merger linear biases and the merger LayerNorm weight and bias
        "patch_embed": False,
        "pos_embed": False,
        "ranks": {"block": 48, "merger": 128, "vector": 16, "patch_embed": 32, "pos_embed": 32},
    },
    "es": {
        "sigma": 0.05,
        "optimizer": {"kind": "sgd", "lr": None, "lr_sigma2_coef": 1.7, "beta1": 0.9, "beta2": 0.999,
                      "eps": 1.0e-8, "momentum": 0.0},
        "awd": 0.005,
        "z_max_abs": None,
        "max_update_norm": None,
    },
    "runtime": {
        "originals": "auto",             # auto | device | cpu: where the pinned original copies live
        "compute_dtype": "float64",      # dtype of W0 + delta before the cast to the weight dtype
        "verify_on_attach": True,        # sha256 of every served touched tensor must match the snapshot
        "audit_restore": True,           # after every restore, compare touched tensors to the originals
        "cache_policy": "auto",          # auto | reset | keyed (vLLM encoder and prefix caches)
        "use_worker_extension": True,    # call VisionWorkerExtension by name; False sends callables
    },
}


# ======================================================================= helpers
def derive_seed(*parts) -> int:
    """Same derivation as es.data.derive_seed (tests/test_vision.py checks they agree)."""
    data = ":".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") & ((1 << 63) - 1)


def tensor_sha256(t: torch.Tensor) -> str:
    """sha256 of the raw little-endian bytes of a tensor (copied to CPU, made contiguous)."""
    t = t.detach().to("cpu").contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def canonical_name(name: str) -> Optional[str]:
    """Map a parameter or checkpoint key to its canonical vision name, or None if it is not visual.

    Strips any prefix before the first "visual." segment ("model.", "base_model.model.model.",
    "thinker.") and any ".base_layer" segment that a LoRA wrapper inserts:
        model.visual.blocks.3.attn.qkv.weight                  -> visual.blocks.3.attn.qkv.weight
        visual.blocks.3.attn.qkv.base_layer.weight (vLLM LoRA) -> visual.blocks.3.attn.qkv.weight
    """
    m = re.search(r"(?:^|\.)(visual\..+)$", name)
    if not m:
        return None
    return m.group(1).replace(".base_layer.", ".")


# ======================================================================= inventory
@dataclass(frozen=True)
class VisionTensor:
    name: str                 # canonical name, e.g. visual.blocks.18.attn.qkv.weight
    shape: Tuple[int, ...]
    group: str                # patch_embed | pos_embed | block | merger
    block: Optional[int]
    module: str               # e.g. attn.qkv, mlp.linear_fc1, norm1, merger.linear_fc1
    param: str                # weight | bias
    gguf: str                 # tensor name in the llama.cpp mmproj GGUF

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def checkpoint_key(self) -> str:
        return "model." + self.name


_GGUF_BLOCK = {"norm1": "ln1", "attn.qkv": "attn_qkv", "attn.proj": "attn_out", "norm2": "ln2",
               "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}


def vision_inventory(vc: Optional[Mapping] = None, text_hidden: Optional[int] = None) -> List[VisionTensor]:
    """Every parameter of the Qwen3.5/Qwen3-VL vision tower, in checkpoint order, from the config alone.

    Shapes follow transformers' Qwen3_5VisionModel and vLLM's Qwen3_VisionTransformer; tests check
    them against a meta-device transformers model and the real checkpoint header totals.
    """
    vc = dict(ATLANTIC_VISION_CONFIG if vc is None else vc)
    if vc.get("deepstack_visual_indexes"):
        raise NotImplementedError("deepstack mergers are not in Atlantic's tower; extend the inventory first")
    h, inter, depth = int(vc["hidden_size"]), int(vc["intermediate_size"]), int(vc["depth"])
    p, tp, cin = int(vc["patch_size"]), int(vc["temporal_patch_size"]), int(vc["in_channels"])
    merge = int(vc["spatial_merge_size"])
    out_hidden = int(vc.get("out_hidden_size") or text_hidden or ATLANTIC_TEXT_HIDDEN)
    mh = h * merge * merge
    t: List[VisionTensor] = [
        VisionTensor("visual.patch_embed.proj.weight", (h, cin, tp, p, p), "patch_embed", None, "patch_embed.proj",
                     "weight", "v.patch_embd.weight"),
        VisionTensor("visual.patch_embed.proj.bias", (h,), "patch_embed", None, "patch_embed.proj", "bias",
                     "v.patch_embd.bias"),
        VisionTensor("visual.pos_embed.weight", (int(vc["num_position_embeddings"]), h), "pos_embed", None,
                     "pos_embed", "weight", "v.position_embd.weight"),
    ]
    shapes = {"norm1": [(h,), (h,)], "attn.qkv": [(3 * h, h), (3 * h,)], "attn.proj": [(h, h), (h,)],
              "norm2": [(h,), (h,)], "mlp.linear_fc1": [(inter, h), (inter,)],
              "mlp.linear_fc2": [(h, inter), (h,)]}
    for b in range(depth):
        for mod in ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.linear_fc1", "mlp.linear_fc2"):
            for param, shape in zip(("weight", "bias"), shapes[mod]):
                t.append(VisionTensor(f"visual.blocks.{b}.{mod}.{param}", shape, "block", b, mod, param,
                                      f"v.blk.{b}.{_GGUF_BLOCK[mod]}.{param}"))
    for mod, gg, shapes_m in (("norm", "v.post_ln", [(h,), (h,)]),
                              ("linear_fc1", "mm.0", [(mh, mh), (mh,)]),
                              ("linear_fc2", "mm.2", [(out_hidden, mh), (out_hidden,)])):
        for param, shape in zip(("weight", "bias"), shapes_m):
            t.append(VisionTensor(f"visual.merger.{mod}.{param}", shape, "merger", None, f"merger.{mod}", param,
                                  f"{gg}.{param}"))
    return t


def inventory_summary(inv: Optional[Sequence[VisionTensor]] = None) -> dict:
    inv = list(inv if inv is not None else vision_inventory())
    by_group: Dict[str, int] = {}
    for x in inv:
        by_group[x.group] = by_group.get(x.group, 0) + x.numel
    blocks = sorted({x.block for x in inv if x.block is not None})
    per_block = sum(x.numel for x in inv if x.block == blocks[0]) if blocks else 0
    per_block_mods = {m: sum(x.numel for x in inv if x.block == blocks[0] and x.module == m)
                      for m in ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.linear_fc1", "mlp.linear_fc2")}
    return {"tensors": len(inv), "params": sum(x.numel for x in inv), "by_group": by_group,
            "blocks": len(blocks), "params_per_block": per_block, "per_block_module_params": per_block_mods}


# ======================================================================= scope
def merged_space_cfg(space_cfg: Optional[Mapping]) -> dict:
    """DEFAULT_VISION["space"] deep-merged with space_cfg, except that `blocks` is replaced whole
    (merging {"last": 9} with {"fraction": 0.25} would otherwise keep both selectors)."""
    space_cfg = dict(space_cfg or {})
    s = deep_merge(DEFAULT_VISION["space"], space_cfg)
    if "blocks" in space_cfg:
        s["blocks"] = copy.deepcopy(space_cfg["blocks"])
    return s


def select_blocks(depth: int, spec: Mapping) -> List[int]:
    given = [k for k in ("indices", "last", "fraction") if spec.get(k) is not None]
    if len(given) != 1:
        raise ValueError(f"blocks needs exactly one of indices, last, fraction; got {given or 'none'}")
    if given[0] == "indices":
        idx = sorted({int(i) for i in spec["indices"]})
    elif given[0] == "last":
        n = int(spec["last"])
        idx = list(range(depth - n, depth))
    else:
        n = int(math.ceil(depth * float(spec["fraction"]) - 1e-9))
        idx = list(range(depth - n, depth))
    if any(i < 0 or i >= depth for i in idx):
        raise ValueError(f"block indices out of range 0..{depth - 1}: {idx}")
    return idx


def select_targets(inv: Sequence[VisionTensor], space_cfg: Mapping, depth: int) -> List[Tuple[VisionTensor, str, int]]:
    """(tensor, kind, rank) for every searched tensor, in inventory order."""
    s = merged_space_cfg(space_cfg)
    ranks = s["ranks"]
    blocks = set(select_blocks(depth, s["blocks"]))
    mods = set(s["block_modules"])
    unknown = mods - set(BLOCK_MODULES)
    if unknown:
        raise ValueError(f"unknown block modules {sorted(unknown)}; choose from {BLOCK_MODULES}")
    out = []
    for x in inv:
        if x.group == "block" and x.block in blocks and x.module in mods:
            if x.param == "weight":
                out.append((x, "matrix", int(ranks["block"])))
            elif s["block_biases"]:
                out.append((x, "vector", int(ranks["vector"])))
        elif x.group == "merger":
            if x.module.startswith("merger.linear_fc") and x.param == "weight" and s["merger_matrices"]:
                out.append((x, "matrix", int(ranks["merger"])))
            elif (x.param == "bias" or x.module == "merger.norm") and s["merger_vectors"]:
                out.append((x, "vector", int(ranks["vector"])))
        elif x.group == "patch_embed" and s["patch_embed"]:
            out.append((x, "matrix" if x.param == "weight" else "vector",
                        int(ranks["patch_embed"] if x.param == "weight" else ranks["vector"])))
        elif x.group == "pos_embed" and s["pos_embed"]:
            out.append((x, "matrix", int(ranks["pos_embed"])))
    if not out:
        raise ValueError("vision scope selects no tensors")
    return out


# ======================================================================= search space
@dataclass
class Target:
    name: str
    kind: str                 # matrix | vector
    shape: List[int]          # tensor shape as stored
    rows: int
    cols: int                 # 1 for vectors
    rank: int
    offset: int
    size: int                 # coordinates of z for this tensor
    scale: float
    base_sha256: str
    base_dtype: str
    gguf: str

    def key(self) -> str:
        return self.name


def orthonormal_basis(seed: int, rows: int, rank: int) -> np.ndarray:
    """Columns of Q from QR of a seeded Gaussian (rows x rank), signs fixed by diag(R) > 0. float64."""
    g = np.random.Generator(np.random.PCG64(int(seed))).standard_normal((int(rows), int(rank)))
    q, r = np.linalg.qr(g, mode="reduced")
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    return q * d[None, :]


def _matrix_view(shape: Sequence[int]) -> Tuple[int, int]:
    shape = tuple(int(s) for s in shape)
    if len(shape) == 1:
        return shape[0], 1
    return shape[0], int(np.prod(shape[1:]))


def _delta(kind: str, core: str, U: torch.Tensor, V: Optional[torch.Tensor], scale: float, c: torch.Tensor,
           shape: Sequence[int]) -> torch.Tensor:
    """s * U diag(c) V^T (or s * U C V^T, or s * U c) in the dtype and device of U, reshaped to shape."""
    if kind == "vector":
        return (U @ (c * scale)).reshape(tuple(shape))
    if core == "diag":
        d = (U * (c * scale).unsqueeze(0)) @ V.T
    else:
        r = U.shape[1]
        d = U @ (c.reshape(r, r) * scale) @ V.T
    return d.reshape(tuple(shape))


class VisionSpace:
    """Fixed bases, scales and base hashes for every searched vision tensor."""

    def __init__(self, targets: List[Target], U: Dict[str, torch.Tensor], V: Dict[str, torch.Tensor], meta: dict):
        self.targets = targets
        self.U = U
        self.V = V
        self.meta = meta
        self.core = meta["core"]
        self.dim = sum(t.size for t in targets)
        self.by_name = {t.name: t for t in targets}
        self._fp: Optional[str] = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, get_tensor: Callable[[str], torch.Tensor], space_cfg: Optional[Mapping] = None,
              vision_config: Optional[Mapping] = None, source: Optional[dict] = None) -> "VisionSpace":
        """get_tensor(canonical_name) returns the base tensor (any device; read once for scale and hash)."""
        s = merged_space_cfg(space_cfg)
        if s["core"] not in ("diag", "full"):
            raise ValueError("space.core must be diag or full")
        vc = dict(ATLANTIC_VISION_CONFIG if vision_config is None else vision_config)
        inv = vision_inventory(vc)
        chosen = select_targets(inv, s, int(vc["depth"]))
        targets: List[Target] = []
        U: Dict[str, torch.Tensor] = {}
        V: Dict[str, torch.Tensor] = {}
        offset = 0
        for x, kind, rank in chosen:
            w = get_tensor(x.name)
            if tuple(w.shape) != tuple(x.shape):
                raise ValueError(f"{x.name}: base tensor has shape {tuple(w.shape)}, inventory says {x.shape}")
            if w.dtype not in FLOAT_DTYPES:
                raise ValueError(f"{x.name}: unsupported base dtype {w.dtype}")
            rows, cols = _matrix_view(x.shape)
            w64 = w.detach().to("cpu", torch.float64)
            if kind == "matrix":
                rank = min(rank, rows, cols)
                scale = float(torch.linalg.vector_norm(w64)) / math.sqrt(min(rows, cols))
                size = rank if s["core"] == "diag" else rank * rank
            else:
                rank = min(rank, rows)
                scale = max(float(torch.linalg.vector_norm(w64)) / math.sqrt(rows), float(s["vector_scale_floor"]))
                size = rank
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{x.name}: degenerate scale {scale}")
            seed_u = derive_seed(s["basis_seed"], VISION_BASIS_TAG, x.name, "U")
            U[x.name] = torch.from_numpy(orthonormal_basis(seed_u, rows, rank).astype(np.float32))
            if kind == "matrix":
                seed_v = derive_seed(s["basis_seed"], VISION_BASIS_TAG, x.name, "V")
                V[x.name] = torch.from_numpy(orthonormal_basis(seed_v, cols, rank).astype(np.float32))
            name_dtype = [k for k, v in DTYPES.items() if v == w.dtype][0]
            targets.append(Target(name=x.name, kind=kind, shape=list(x.shape), rows=rows, cols=cols, rank=rank,
                                  offset=offset, size=size, scale=scale, base_sha256=tensor_sha256(w),
                                  base_dtype=name_dtype, gguf=x.gguf))
            offset += size
        meta = {"format": SPACE_FORMAT, "core": s["core"], "space_cfg": s, "vision_config": vc,
                "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "source": source or {}}
        return cls(targets, U, V, meta)

    @classmethod
    def from_snapshot(cls, snapshot_dir, space_cfg: Optional[Mapping] = None) -> "VisionSpace":
        reader = SnapshotReader(snapshot_dir)
        cfg = reader.config()
        vc = cfg.get("vision_config") or ATLANTIC_VISION_CONFIG
        return cls.build(reader.get, space_cfg, vc, source={"snapshot": str(snapshot_dir),
                                                            "config_sha256": reader.config_sha256()})

    @classmethod
    def from_module(cls, model: torch.nn.Module, space_cfg: Optional[Mapping] = None,
                    vision_config: Optional[Mapping] = None) -> "VisionSpace":
        params = resolve_vision_params(model)
        return cls.build(lambda n: params[n][1], space_cfg, vision_config, source={"module": type(model).__name__})

    # ------------------------------------------------------------------ io
    def fingerprint(self) -> str:
        if self._fp is None:
            h = hashlib.sha256()
            h.update(json.dumps({"format": SPACE_FORMAT, "core": self.core}, sort_keys=True).encode())
            for t in self.targets:
                h.update(json.dumps(asdict(t), sort_keys=True).encode())
                h.update(tensor_sha256(self.U[t.name]).encode())
                if t.name in self.V:
                    h.update(tensor_sha256(self.V[t.name]).encode())
            self._fp = h.hexdigest()
        return self._fp

    def save(self, out_dir) -> Path:
        from safetensors.torch import save_file
        out_dir = Path(out_dir)
        tmp = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        tensors = {f"U::{n}": u.contiguous() for n, u in self.U.items()}
        tensors.update({f"V::{n}": v.contiguous() for n, v in self.V.items()})
        save_file(tensors, str(tmp / "bases.safetensors"), metadata={"format": SPACE_FORMAT})
        doc = {"meta": self.meta, "dim": self.dim, "targets": [asdict(t) for t in self.targets],
               "fingerprint": self.fingerprint()}
        (tmp / "space.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
        return out_dir

    @classmethod
    def load(cls, space_dir) -> "VisionSpace":
        from safetensors.torch import load_file
        space_dir = Path(space_dir)
        doc = json.loads((space_dir / "space.json").read_text())
        if doc.get("meta", {}).get("format") != SPACE_FORMAT:
            raise ValueError(f"{space_dir} is not a {SPACE_FORMAT} directory")
        tensors = load_file(str(space_dir / "bases.safetensors"), device="cpu")
        targets = [Target(**t) for t in doc["targets"]]
        U = {t.name: tensors[f"U::{t.name}"] for t in targets}
        V = {t.name: tensors[f"V::{t.name}"] for t in targets if f"V::{t.name}" in tensors}
        obj = cls(targets, U, V, doc["meta"])
        if obj.fingerprint() != doc["fingerprint"]:
            raise ValueError(f"vision space {space_dir} failed its fingerprint check")
        return obj

    # ------------------------------------------------------------------ math
    def zeros(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)

    def check_vector(self, c) -> torch.Tensor:
        """Coefficients as a CPU FP32 tensor. A candidate's vision state is defined by its FP32 vector,
        so the vLLM worker (which receives FP32 bytes), the HF path and the export all agree."""
        if isinstance(c, torch.Tensor):
            t = c.detach().to("cpu")
        else:
            t = torch.from_numpy(np.array(c, copy=True))
        if tuple(t.shape) != (self.dim,):
            raise ValueError(f"vision coefficients have shape {tuple(t.shape)}, expected ({self.dim},)")
        t = t.to(torch.float32)
        if not bool(torch.isfinite(t).all()):
            raise ValueError("non-finite vision coefficients")
        return t

    def delta(self, name: str, c, dtype=torch.float64, device="cpu") -> torch.Tensor:
        """The additive change s * U diag(c_t) V^T for one tensor, in its stored shape."""
        t = self.by_name[name]
        cv = self.check_vector(c).to(device=device, dtype=dtype)[t.offset:t.offset + t.size]
        U = self.U[name].to(device=device, dtype=dtype)
        V = self.V[name].to(device=device, dtype=dtype) if name in self.V else None
        return _delta(t.kind, self.core, U, V, t.scale, cv, t.shape)

    def evolved(self, name: str, w0: torch.Tensor, c, compute_dtype=torch.float64) -> torch.Tensor:
        """W0 + delta computed in compute_dtype and cast back to W0's dtype (the exact served value)."""
        d = self.delta(name, c, compute_dtype, w0.device)
        return (w0.to(compute_dtype) + d).to(w0.dtype)

    def tag(self, c) -> str:
        """Short identity of a vision state: sha256 over the space fingerprint and the FP32 coefficients."""
        a = np.ascontiguousarray(np.asarray(c, dtype=np.float32))
        if not np.any(a):
            return "base"
        return hashlib.sha256(self.fingerprint().encode() + a.tobytes()).hexdigest()[:16]

    def summary(self) -> dict:
        kinds: Dict[str, int] = {}
        for t in self.targets:
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        touched = sum(int(np.prod(t.shape)) for t in self.targets)
        basis = sum(u.numel() for u in self.U.values()) + sum(v.numel() for v in self.V.values())
        blocks = sorted({int(t.name.split(".")[2]) for t in self.targets if t.name.startswith("visual.blocks.")})
        return {"dim": self.dim, "core": self.core, "targets": len(self.targets), "by_kind": kinds,
                "blocks": blocks, "touched_params": touched,
                "touched_fraction_of_tower": touched / float(ATLANTIC_VISION_PARAMS),
                "originals_bytes_bf16": 2 * touched, "basis_floats": basis, "basis_bytes_fp32": 4 * basis,
                "fingerprint": self.fingerprint()}


# ======================================================================= module binding
def resolve_vision_params(model: torch.nn.Module) -> Dict[str, Tuple[str, torch.nn.Parameter]]:
    """canonical name -> (full parameter name, parameter); refuses ambiguous matches."""
    out: Dict[str, Tuple[str, torch.nn.Parameter]] = {}
    for full, p in model.named_parameters(remove_duplicate=False):
        c = canonical_name(full)
        if c is None:
            continue
        if c in out and out[c][1] is not p:
            raise ValueError(f"ambiguous vision parameter {c}: {out[c][0]} and {full}")
        out[c] = (full, p)
    return out


class VisionAttachError(RuntimeError):
    """The served weights do not match the search space; nothing was changed."""


class VisionPatcher:
    """Applies and restores candidate vision deltas in place on a live module.

    originals: "device" keeps the pinned copies next to the weights (fast restore), "cpu" keeps them in
    page-locked host memory, "auto" chooses device when free memory exceeds three times the need.
    """

    def __init__(self, model: torch.nn.Module, space: VisionSpace, originals: str = "auto",
                 compute_dtype: str = "float64", verify_on_attach: bool = True, audit_restore: bool = True):
        self.space = space
        self.compute_dtype = DTYPES[compute_dtype]
        self.audit_restore = bool(audit_restore)
        found = resolve_vision_params(model)
        self.params: Dict[str, torch.nn.Parameter] = {}
        self.param_names: Dict[str, str] = {}
        problems = []
        for t in space.targets:
            if t.name not in found:
                problems.append(f"{t.name}: not found in the served model")
                continue
            full, p = found[t.name]
            if tuple(p.shape) != tuple(t.shape):
                problems.append(f"{t.name}: served shape {tuple(p.shape)} != {tuple(t.shape)} (sharded or fused?)")
            elif p.dtype not in FLOAT_DTYPES:
                problems.append(f"{t.name}: served dtype {p.dtype} is not a float type (quantized tower?)")
            elif p.device.type == "meta":
                problems.append(f"{t.name}: parameter is on the meta device")
            elif not p.is_contiguous():
                problems.append(f"{t.name}: parameter is not contiguous")
            self.params[t.name] = p
            self.param_names[t.name] = full
        if not problems and verify_on_attach:
            for t in space.targets:
                got = tensor_sha256(self.params[t.name])
                if got != t.base_sha256:
                    problems.append(f"{t.name}: served bytes differ from the snapshot (sha256 {got[:12]} != "
                                    f"{t.base_sha256[:12]}); layout, dtype or base revision mismatch")
        if problems:
            raise VisionAttachError("vision attach refused:\n  " + "\n  ".join(problems[:20]))
        device = next(iter(self.params.values())).device
        self.device = device
        need = sum(p.numel() * p.element_size() for p in self.params.values())
        if originals == "auto":
            originals = "device"
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(device)
                if free < 3 * need:
                    originals = "cpu"
        if originals not in ("device", "cpu"):
            raise ValueError("originals must be auto, device or cpu")
        self.originals_where = originals
        self.originals: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in self.params.items():
                if originals == "device" or p.device.type == "cpu":
                    self.originals[name] = p.detach().clone()
                else:
                    o = torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True)
                    o.copy_(p.detach())
                    self.originals[name] = o
        self.bases_dtype = torch.float64 if self.compute_dtype == torch.float64 else torch.float32
        self.U = {n: space.U[n].to(device=device, dtype=self.bases_dtype) for n in self.params}
        self.V = {n: space.V[n].to(device=device, dtype=self.bases_dtype) for n in space.V if n in self.params}
        self.state = "base"
        self.tag = "base"
        self.applies = 0
        self.restores = 0
        self.pinned_bytes = need

    @contextlib.contextmanager
    def candidate(self, c, tag: Optional[str] = None) -> Iterator["VisionPatcher"]:
        self.apply(c, tag)
        try:
            yield self
        finally:
            self.restore()

    def apply(self, c, tag: Optional[str] = None) -> str:
        """Write W0 + delta(c) into every touched tensor. Always starts from the pinned original."""
        cv = self.space.check_vector(c)
        tag = tag or self.space.tag(cv.numpy())
        cd = cv.to(device=self.device, dtype=self.bases_dtype)
        with torch.no_grad():
            for t in self.space.targets:
                p = self.params[t.name]
                w0 = self.originals[t.name].to(p.device, non_blocking=True)
                d = _delta(t.kind, self.space.core, self.U[t.name], self.V.get(t.name), t.scale,
                           cd[t.offset:t.offset + t.size], t.shape)
                p.copy_((w0.to(self.compute_dtype) + d.to(self.compute_dtype)).to(p.dtype))
        self._sync()
        self.state = "candidate"
        self.tag = tag
        self.applies += 1
        return tag

    def restore(self) -> None:
        """Copy the pinned originals back: bit-exact."""
        with torch.no_grad():
            for name, p in self.params.items():
                p.copy_(self.originals[name], non_blocking=True)
        self._sync()
        self.state = "base"
        self.tag = "base"
        self.restores += 1
        if self.audit_restore and self.originals_where == "device":
            bad = self.mismatches()
            if bad:
                raise RuntimeError(f"vision restore audit failed for {bad[:5]}")

    def mismatches(self) -> List[str]:
        """Touched tensors whose current bytes differ from the pinned originals."""
        bad = []
        with torch.no_grad():
            for name, p in self.params.items():
                o = self.originals[name]
                if o.device != p.device:
                    o = o.to(p.device)
                if not torch.equal(p.detach().view(_int_view(p.dtype)), o.view(_int_view(o.dtype))):
                    bad.append(name)
        return bad

    def sha256s(self) -> Dict[str, str]:
        return {name: tensor_sha256(p) for name, p in self.params.items()}

    def status(self) -> dict:
        return {"state": self.state, "tag": self.tag, "applies": self.applies, "restores": self.restores,
                "fingerprint": self.space.fingerprint(), "dim": self.space.dim, "targets": len(self.params),
                "device": str(self.device), "originals": self.originals_where, "pinned_bytes": self.pinned_bytes,
                "compute_dtype": str(self.compute_dtype)}

    def detach(self) -> None:
        if self.state != "base":
            self.restore()
        self.originals.clear()
        self.U.clear()
        self.V.clear()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _int_view(dtype: torch.dtype) -> torch.dtype:
    return {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32,
            torch.float64: torch.int64}[dtype]


# ======================================================================= ES on z
def vision_noise_seeds(run_seed: int, gen: int, n_pairs: int) -> List[int]:
    return [derive_seed(run_seed, VISION_NOISE_TAG, gen, j) for j in range(n_pairs)]


def vision_noise(seed: int, dim: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(int(seed))).standard_normal(int(dim), dtype=np.float32)


def vision_noise_fingerprint() -> str:
    return hashlib.sha256(vision_noise(NOISE_FINGERPRINT_SEED, 4096).tobytes()).hexdigest()


def candidate_names(n_pairs: int) -> List[str]:
    """Same names and order as es_loop.ESRun.candidates: parent, p000+, p000-, p001+, ..."""
    names = ["parent"]
    for j in range(n_pairs):
        names += [f"p{j:03d}+", f"p{j:03d}-"]
    return names


class VisionES:
    """FP32 master z, its optimizer and the per-generation update. Shaping is shared with the text block."""

    def __init__(self, dim: int, es_cfg: Optional[Mapping], run_seed: int, state: Optional[dict] = None):
        self.cfg = deep_merge(DEFAULT_VISION["es"], dict(es_cfg or {}))
        self.dim = int(dim)
        self.run_seed = int(run_seed)
        self.sigma = float(self.cfg["sigma"])
        if self.sigma <= 0:
            raise ValueError("vision sigma must be positive")
        o = self.cfg["optimizer"]
        if o["kind"] not in ("sgd", "adam"):
            raise ValueError(f"unknown vision optimizer {o['kind']!r}")
        state = state or {"gen": 0, "z": np.zeros(self.dim, dtype=np.float32), "t": 0,
                          "m": np.zeros(self.dim), "v": np.zeros(self.dim)}
        self.gen = int(state["gen"])
        self.z = np.asarray(state["z"], dtype=np.float32).copy()
        self.t = int(state["t"])
        self.m = np.asarray(state["m"], dtype=np.float64).copy()
        self.v = np.asarray(state["v"], dtype=np.float64).copy()
        if self.z.shape != (self.dim,):
            raise ValueError(f"vision state has dim {self.z.shape}, space has {self.dim}")

    # ------------------------------------------------------------- candidates
    def lr(self) -> float:
        o = self.cfg["optimizer"]
        if o.get("lr_sigma2_coef"):
            return float(o["lr_sigma2_coef"]) * self.sigma ** 2
        if o.get("lr") is None:
            raise ValueError("set vision es.optimizer.lr or lr_sigma2_coef")
        return float(o["lr"])

    def seeds(self, gen: int, n_pairs: int) -> List[int]:
        return vision_noise_seeds(self.run_seed, gen, n_pairs)

    def candidates(self, gen: int, n_pairs: int) -> List[Tuple[str, np.ndarray]]:
        """(name, c) with c in float64; c = z for the parent and z +/- sigma_v eps_j for pair j."""
        z = self.z.astype(np.float64)
        out = [("parent", z.copy())]
        for j, s in enumerate(self.seeds(gen, n_pairs)):
            e = vision_noise(s, self.dim).astype(np.float64)
            out.append((f"p{j:03d}+", z + self.sigma * e))
            out.append((f"p{j:03d}-", z - self.sigma * e))
        return out

    def coefficients(self, u_plus, u_minus) -> np.ndarray:
        up = np.asarray(u_plus, dtype=np.float64)
        um = np.asarray(u_minus, dtype=np.float64)
        if up.shape != um.shape or up.ndim != 1:
            raise ValueError("u_plus and u_minus must be equal-length vectors")
        return (up - um) / (2.0 * up.shape[0] * self.sigma)

    def gradient(self, coeffs, seeds) -> np.ndarray:
        g = np.zeros(self.dim, dtype=np.float64)
        for c, s in zip(coeffs, seeds):
            g += float(c) * vision_noise(int(s), self.dim).astype(np.float64)
        return g

    # ------------------------------------------------------------- update
    def update(self, gen: int, u_plus, u_minus) -> dict:
        """One step from the shared shaped utilities of generation gen. Mutates z; returns journal fields."""
        if gen != self.gen:
            raise ValueError(f"vision state is at generation {self.gen}, update asked for {gen}")
        n = len(u_plus)
        seeds = self.seeds(gen, n)
        coeffs = self.coefficients(u_plus, u_minus)
        g = self.gradient(coeffs, seeds)
        lr = self.lr()
        o = self.cfg["optimizer"]
        self.t += 1
        if o["kind"] == "adam":
            b1, b2, eps = float(o["beta1"]), float(o["beta2"]), float(o["eps"])
            self.m = b1 * self.m + (1.0 - b1) * g
            self.v = b2 * self.v + (1.0 - b2) * g * g
            upd = lr * (self.m / (1.0 - b1 ** self.t)) / (np.sqrt(self.v / (1.0 - b2 ** self.t)) + eps)
        else:
            mom = float(o.get("momentum") or 0.0)
            if mom:
                self.m = mom * self.m + g
                upd = lr * self.m
            else:
                upd = lr * g
        mx = self.cfg.get("max_update_norm")
        if mx:
            nrm = float(np.linalg.norm(upd))
            if nrm > float(mx):
                upd = upd * (float(mx) / nrm)
        before = self.z.copy()
        awd = float(self.cfg.get("awd") or 0.0)
        new = (1.0 - awd) * (before.astype(np.float64) + upd)
        if self.cfg.get("z_max_abs") is not None:
            lim = float(self.cfg["z_max_abs"])
            new = np.clip(new, -lim, lim)
        new32 = new.astype(np.float32)
        if not np.all(np.isfinite(new32)):
            raise FloatingPointError(f"generation {gen}: non-finite vision z; nothing committed")
        self.z = new32
        self.gen = gen + 1
        return {"seeds": seeds, "sigma": self.sigma, "lr": lr, "awd": awd, "coeffs": [float(c) for c in coeffs],
                "grad_norm": float(np.linalg.norm(g)), "step_norm": float(np.linalg.norm(upd)),
                "z_norm": float(np.linalg.norm(new32)), "z_max_abs": float(np.abs(new32).max()) if self.dim else 0.0,
                "z_sha256_before": array_sha256(before), "z_sha256_after": array_sha256(new32), "opt_t": self.t}

    # ------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"gen": self.gen, "z": self.z.copy(), "t": self.t, "m": self.m.copy(), "v": self.v.copy()}

    def save(self, path, extra: Optional[dict] = None) -> str:
        """Atomic .npz with a sidecar sha256 in the JSON metadata entry. Returns the z sha256."""
        import io
        path = Path(path)
        meta = {"format": STATE_FORMAT, "gen": self.gen, "t": self.t, "dim": self.dim,
                "z_sha256": array_sha256(self.z), "noise_fingerprint": vision_noise_fingerprint()}
        meta.update(extra or {})
        buf = io.BytesIO()
        np.savez(buf, z=self.z, m=self.m, v=self.v, meta=np.frombuffer(json.dumps(meta, sort_keys=True).encode(),
                                                                       dtype=np.uint8))
        _atomic_bytes(path, buf.getvalue())
        return meta["z_sha256"]

    @classmethod
    def load(cls, path, es_cfg: Optional[Mapping], run_seed: int, expect: Optional[dict] = None) -> "VisionES":
        with np.load(Path(path), allow_pickle=False) as f:
            meta = json.loads(bytes(f["meta"]).decode())
            z, m, v = f["z"].copy(), f["m"].copy(), f["v"].copy()
        if meta.get("format") != STATE_FORMAT:
            raise ValueError(f"{path} is not a {STATE_FORMAT} file")
        if array_sha256(z) != meta["z_sha256"]:
            raise ValueError(f"{path}: z failed its sha256 check")
        if meta.get("noise_fingerprint") != vision_noise_fingerprint():
            raise RuntimeError("vision noise generator differs from the checkpoint's (numpy version?)")
        for k, want in (expect or {}).items():
            if meta.get(k) != want:
                raise RuntimeError(f"vision checkpoint {k} is {meta.get(k)!r}, expected {want!r}")
        return cls(z.shape[0], es_cfg, run_seed, {"gen": meta["gen"], "z": z, "t": meta["t"], "m": m, "v": v})

    @staticmethod
    def replay(dim: int, es_cfg: Optional[Mapping], run_seed: int, entries: Sequence[dict]) -> np.ndarray:
        """Rebuild z from journal entries' vision.coeffs and vision.seeds; checks every recorded hash."""
        es = VisionES(dim, es_cfg, run_seed)
        for e in entries:
            v = e["vision"] if "vision" in e else e
            if array_sha256(es.z) != v["z_sha256_before"]:
                raise ValueError(f"replay: z hash before generation {e.get('gen')} does not match the journal")
            g = es.gradient(v["coeffs"], v["seeds"])
            es.t += 1
            o = es.cfg["optimizer"]
            if o["kind"] != "sgd" or float(o.get("momentum") or 0.0) or es.cfg.get("max_update_norm"):
                raise NotImplementedError("replay covers plain SGD (the protocol optimizer)")
            new = (1.0 - float(v["awd"])) * (es.z.astype(np.float64) + float(v["lr"]) * g)
            if es.cfg.get("z_max_abs") is not None:
                new = np.clip(new, -float(es.cfg["z_max_abs"]), float(es.cfg["z_max_abs"]))
            es.z = new.astype(np.float32)
            if array_sha256(es.z) != v["z_sha256_after"]:
                raise ValueError(f"replay: z hash after generation {e.get('gen')} does not match the journal")
        return es.z


# ======================================================================= waves
def plan_waves(cand_names: Sequence[str], n_text: int, n_image: int, mode: str = "per_candidate") -> List[dict]:
    """Group the (candidate, item) requests of one generation into engine calls.

    Vision weights are one global state per engine, so the image requests of candidate c must run while
    c's vision delta is applied. Text requests never run the tower and may go in any wave.

      per_candidate (default): one wave per candidate with all of its requests, text and image, under
          its own vision state. One LoRA per wave (no adapter rotation), 128 requests per call. This is
          es_loop's existing wave_size = 1 path with controller.candidate(c) around generate().
      serial: one wave with every text request of every candidate (vision at base, as in the text-only
          engine), then one small wave per candidate with only its image requests.

    Returns [{"vision": name or None, "requests": [(candidate, "text"|"image", item_index), ...]}];
    every (candidate, kind, item) appears exactly once.
    """
    cands = list(cand_names)
    if mode == "per_candidate":
        return [{"vision": c if n_image else None,
                 "requests": [(c, "text", i) for i in range(n_text)] + [(c, "image", i) for i in range(n_image)]}
                for c in cands]
    if mode != "serial":
        raise ValueError("mode must be per_candidate or serial")
    text = [(c, "text", i) for c in cands for i in range(n_text)]
    waves = [{"vision": None, "requests": text}] if text else []
    if n_image:
        waves += [{"vision": c, "requests": [(c, "image", i) for i in range(n_image)]} for c in cands]
    return waves


# ======================================================================= snapshots and export
class SnapshotReader:
    """Reads tensors from a Hugging Face safetensors snapshot (sharded or single file)."""

    def __init__(self, snapshot_dir):
        self.dir = Path(snapshot_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.index = json.loads(idx.read_text())
            self.weight_map: Dict[str, str] = dict(self.index["weight_map"])
        elif (self.dir / "model.safetensors").exists():
            from safetensors import safe_open
            self.index = None
            with safe_open(str(self.dir / "model.safetensors"), framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        else:
            raise FileNotFoundError(f"no safetensors snapshot in {self.dir}")

    def config(self) -> dict:
        p = self.dir / "config.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def config_sha256(self) -> Optional[str]:
        p = self.dir / "config.json"
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

    def key_for(self, name: str) -> str:
        for k in (name, "model." + name):
            if k in self.weight_map:
                return k
        raise KeyError(f"{name} not in snapshot {self.dir}")

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open
        key = self.key_for(name)
        with safe_open(str(self.dir / self.weight_map[key]), framework="pt") as f:
            return f.get_tensor(key)

    def vision_keys(self) -> List[str]:
        return sorted(k for k in self.weight_map if canonical_name(k) is not None)


MMPROJ_COMMAND = (
    "python convert_hf_to_gguf.py {src} --mmproj --outtype f16 --outfile {out}\n"
)


def evolved_vision_tensors(space: VisionSpace, z, reader: SnapshotReader,
                           compute_dtype: torch.dtype = torch.float64) -> Dict[str, torch.Tensor]:
    """Every model.visual.* tensor of the snapshot; searched ones replaced by W0 + delta(z), same dtype."""
    zz = space.check_vector(z)
    out: Dict[str, torch.Tensor] = {}
    for key in reader.vision_keys():
        name = canonical_name(key)
        w0 = reader.get(key)
        if name in space.by_name:
            t = space.by_name[name]
            if tensor_sha256(w0) != t.base_sha256:
                raise ValueError(f"{key}: snapshot differs from the space's base tensor")
            out[key] = space.evolved(name, w0, zz, compute_dtype).contiguous()
        else:
            out[key] = w0.contiguous()
    return out


def export_vision(space: VisionSpace, z, base_snapshot, out_dir, mode: str = "mmproj",
                  label: str = "", extra: Optional[dict] = None) -> Path:
    """Write the evolved vision weights as a Hugging Face snapshot that llama.cpp can convert.

    mode "mmproj": config.json, the preprocessor configs and one model.safetensors holding only the
        333 model.visual.* tensors (the mmproj converter reads nothing else). Small and fast.
    mode "full": the complete model with the shards that hold visual tensors rewritten and every
        other file symlinked to the base snapshot, so transformers or vLLM can load the evolved model.
    Both write VISION_ES_EXPORT.json (sha256 of every vision tensor, which ones changed, z, space) and
    MMPROJ_COMMAND.txt.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(base_snapshot)
    out_dir = Path(out_dir)
    reader = SnapshotReader(base)
    tensors = evolved_vision_tensors(space, z, reader)
    tmp = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    small = [p for p in sorted(base.iterdir()) if p.is_file() and not p.name.endswith(".safetensors")
             and p.name != "model.safetensors.index.json"]
    shards_written: Dict[str, str] = {}
    if mode == "mmproj":
        for p in small:
            if p.name.endswith(".json") or p.name.endswith(".jinja") or p.name in ("LICENSE", "README.md"):
                shutil.copy2(p, tmp / p.name)
        save_file(tensors, str(tmp / "model.safetensors"), metadata={"format": "pt"})
        shards_written["model.safetensors"] = _file_sha256(tmp / "model.safetensors")
    elif mode == "full":
        for p in small:
            shutil.copy2(p, tmp / p.name)
        if reader.index is not None:
            shutil.copy2(base / "model.safetensors.index.json", tmp / "model.safetensors.index.json")
        files = sorted(set(reader.weight_map.values()))
        touched_files = {reader.weight_map[k] for k in tensors}
        for fname in files:
            if fname in touched_files:
                with safe_open(str(base / fname), framework="pt") as f:
                    meta = f.metadata() or {"format": "pt"}
                    shard = {k: (tensors[k] if k in tensors else f.get_tensor(k)) for k in f.keys()}
                save_file(shard, str(tmp / fname), metadata=meta)
                shards_written[fname] = _file_sha256(tmp / fname)
            else:
                os.symlink((base / fname).resolve(), tmp / fname)
    else:
        raise ValueError("mode must be mmproj or full")
    zz = np.asarray(z, dtype=np.float32)
    manifest = {
        "format": EXPORT_FORMAT, "label": label, "mode": mode, "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION, "base_snapshot": str(base), "space_fingerprint": space.fingerprint(),
        "z_sha256": array_sha256(zz), "z_norm": float(np.linalg.norm(zz)), "dim": space.dim,
        "tensors": {k: {"sha256": tensor_sha256(v), "dtype": str(v.dtype).replace("torch.", ""),
                        "shape": list(v.shape), "evolved": canonical_name(k) in space.by_name}
                    for k, v in sorted(tensors.items())},
        "shards_written": shards_written,
        "mmproj_command": MMPROJ_COMMAND.format(src=str(out_dir), out=str(mmproj_path(out_dir))).strip(),
    }
    manifest.update(extra or {})
    np.save(tmp / "vision_z.npy", zz)
    (tmp / "VISION_ES_EXPORT.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (tmp / "MMPROJ_COMMAND.txt").write_text(
        "# llama.cpp b12407 or newer, from the llama.cpp checkout:\n" + manifest["mmproj_command"] + "\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return out_dir


def mmproj_path(export_dir) -> Path:
    export_dir = Path(export_dir)
    return export_dir.parent / f"mmproj-{export_dir.name}-f16.gguf"


def _file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ======================================================================= config
def load_vision_config(path, overrides: Sequence[str] = ()) -> dict:
    """Load a YAML that may name a base config with `extends` (resolved relative to the file).

    The returned dict is the base config deep-merged with this file, with `vision` filled from
    DEFAULT_VISION. `overrides` are "dotted.key=yaml" strings applied last.
    """
    import yaml
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    base = {}
    if raw.get("extends"):
        parent = (path.parent / raw["extends"]).resolve()
        base = yaml.safe_load(parent.read_text()) or {}
    cfg = deep_merge(base, {k: v for k, v in raw.items() if k != "extends"})
    cfg["vision"] = deep_merge(DEFAULT_VISION, cfg.get("vision") or {})
    for o in overrides:
        key, _, val = o.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def vision_plan(space_cfg: Optional[Mapping] = None, vision_config: Optional[Mapping] = None,
                text_dim: Optional[int] = None) -> dict:
    """Dimensions and memory of a scope without reading any weights."""
    s = merged_space_cfg(space_cfg)
    vc = dict(ATLANTIC_VISION_CONFIG if vision_config is None else vision_config)
    chosen = select_targets(vision_inventory(vc), s, int(vc["depth"]))
    dim = touched = basis = 0
    rows_out = []
    for x, kind, rank in chosen:
        rows, cols = _matrix_view(x.shape)
        if kind == "matrix":
            r = min(rank, rows, cols)
            size = r if s["core"] == "diag" else r * r
            basis += r * (rows + cols)
        else:
            r = min(rank, rows)
            size = r
            basis += r * rows
        dim += size
        touched += x.numel
        rows_out.append({"name": x.name, "kind": kind, "shape": list(x.shape), "rank": r, "coords": size})
    out = {"dim": dim, "targets": len(chosen), "blocks": select_blocks(int(vc["depth"]), s["blocks"]),
           "touched_params": touched, "touched_fraction_of_tower": touched / float(ATLANTIC_VISION_PARAMS),
           "originals_mb_bf16": round(2 * touched / 1e6, 1), "basis_mb_fp32": round(4 * basis / 1e6, 1),
           "basis_mb_fp64_on_device": round(8 * basis / 1e6, 1), "per_target": rows_out}
    if text_dim:
        out["combined_dim"] = dim + int(text_dim)
        out["vision_share_of_dims"] = dim / float(dim + int(text_dim))
    return out


# ======================================================================= CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m es.vision", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inventory", help="vision tensors, shapes and parameter counts")
    a.add_argument("--hf-config", help="config.json of the base model (default: the pinned Atlantic config)")
    a.add_argument("--table", action="store_true")
    a = sub.add_parser("plan", help="dimensions and memory of the configured scope")
    a.add_argument("--config", required=True)
    a.add_argument("--set", action="append", default=[])
    a.add_argument("--text-dim", type=int)
    a = sub.add_parser("build", help="build and save the search space from the base snapshot")
    a.add_argument("--config", required=True)
    a.add_argument("--set", action="append", default=[])
    a.add_argument("--base-snapshot")
    a.add_argument("--out", required=True)
    a = sub.add_parser("export", help="write the evolved vision weights for llama.cpp")
    a.add_argument("--space", required=True)
    a.add_argument("--state", help="VisionES .npz checkpoint (or --z)")
    a.add_argument("--z", help=".npy file with the FP32 z vector")
    a.add_argument("--base-snapshot", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--mode", default="mmproj", choices=("mmproj", "full"))
    args = ap.parse_args(argv)

    if args.cmd == "inventory":
        vc = None
        if args.hf_config:
            vc = json.loads(Path(args.hf_config).read_text())["vision_config"]
        inv = vision_inventory(vc)
        print(json.dumps(inventory_summary(inv), indent=1))
        if args.table:
            for x in inv:
                print(f"{x.name:48s} {str(list(x.shape)):24s} {x.numel:>12,d}  {x.gguf}")
        return 0
    cfg = load_vision_config(args.config, args.set) if args.cmd in ("plan", "build") else None
    if args.cmd == "plan":
        print(json.dumps(vision_plan(cfg["vision"]["space"], text_dim=args.text_dim), indent=1))
        return 0
    if args.cmd == "build":
        snap = args.base_snapshot or cfg["backend"]["base_model"]
        space = VisionSpace.from_snapshot(snap, cfg["vision"]["space"])
        space.save(args.out)
        print(json.dumps(space.summary(), indent=1))
        return 0
    if args.cmd == "export":
        space = VisionSpace.load(args.space)
        if args.state:
            with np.load(args.state, allow_pickle=False) as f:
                z = f["z"].copy()
        elif args.z:
            z = np.load(args.z)
        else:
            ap.error("give --state or --z")
        out = export_vision(space, z, args.base_snapshot, args.out, args.mode, label="cli export")
        print(out)
        print(MMPROJ_COMMAND.format(src=out, out=mmproj_path(out)).strip())
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
