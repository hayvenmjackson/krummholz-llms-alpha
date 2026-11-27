"""ESSA-style search space: the singular values of every LoRA delta.

For each LoRA module with factors A (r x in) and B (out x r), the unscaled delta is
B @ A. We compute its compact SVD without forming the full matrix:

    B = Qb Rb            (QR, Qb: out x r)
    A^T = Qa Ra          (QR, Qa: in x r)
    Rb Ra^T = Um S Vm^T  (r x r SVD)
    B @ A = (Qb Um) diag(S) (Qa Vm)^T = U diag(S) Vh

U and Vh stay fixed. The search vector theta (one entry per singular value, FP32
master, theta = 0 is the starting adapter) sets the candidate singular values:

    relative:  S = S0 * (1 + theta)
    absolute:  S = S0 + theta

A candidate adapter is written back in the original PEFT layout (same keys, same
rank, same adapter_config.json, so the same LoRA scaling) with the balanced split

    B' = U diag(sign(S) sqrt|S|),   A' = diag(sqrt|S|) Vh,   B' @ A' = U diag(S) Vh.

vLLM and PEFT load the result exactly as they load the source adapter. Modules not
selected by the include/exclude filters, embedding LoRA, biases and any other keys
are copied unchanged. DoRA adapters are rejected (the magnitude vector would make
singular-value edits non-local).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from safetensors.torch import load_file, save_file

PARAMETERIZATIONS = ("relative", "absolute")
WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
LORA_A_RE = re.compile(r"^(?P<prefix>.+)\.lora_A(?P<adapter>\.[^.]+)?\.weight$")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}
COPY_LIMIT_BYTES = 20 * 1024 * 1024


def dtype_name(dtype: torch.dtype) -> str:
    for name, value in DTYPES.items():
        if value == dtype:
            return name
    raise ValueError(f"unsupported dtype {dtype}")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    """Hash of the raw little-endian bytes of a contiguous CPU tensor."""
    t = t.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def source_sha256(adapter_dir) -> str:
    """Identity of a source adapter: sha256 over its config bytes and its weight file's sha256."""
    adapter_dir = Path(adapter_dir)
    sha = hashlib.sha256()
    sha.update((adapter_dir / "adapter_config.json").read_bytes())
    for wf in WEIGHT_FILES:
        if (adapter_dir / wf).exists():
            sha.update(bytes.fromhex(sha256_file(adapter_dir / wf)))
            break
    return sha.hexdigest()


def load_adapter_weights(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    st = adapter_dir / "adapter_model.safetensors"
    if st.exists():
        return load_file(str(st), device="cpu")
    binf = adapter_dir / "adapter_model.bin"
    if binf.exists():
        return torch.load(str(binf), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no adapter weights in {adapter_dir}")


@dataclass
class ModuleInfo:
    prefix: str
    key_a: str
    key_b: str
    rank: int
    offset: int
    out_features: int
    in_features: int
    dtype_a: str
    dtype_b: str


class LoraSVD:
    """Fixed singular vectors plus starting singular values for every searched module."""

    def __init__(self, adapter_config: dict, config_bytes: bytes, modules: List[ModuleInfo],
                 U: Dict[str, torch.Tensor], S0: Dict[str, torch.Tensor], Vh: Dict[str, torch.Tensor],
                 passthrough: Dict[str, torch.Tensor], source_sha256: str, parameterization: str,
                 source_dir: Optional[str] = None, extra_files: Optional[Dict[str, bytes]] = None):
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError(f"parameterization must be one of {PARAMETERIZATIONS}")
        self.adapter_config = adapter_config
        self.config_bytes = config_bytes
        self.modules = modules
        self.U = U
        self.S0 = S0
        self.Vh = Vh
        self.passthrough = passthrough
        self.source_sha256 = source_sha256
        self.parameterization = parameterization
        self.source_dir = source_dir
        self.extra_files = extra_files or {}
        self.dim = sum(m.rank for m in modules)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_adapter(cls, adapter_dir, include: Optional[str] = None, exclude: Optional[str] = None,
                     parameterization: str = "relative") -> "LoraSVD":
        adapter_dir = Path(adapter_dir)
        config_path = adapter_dir / "adapter_config.json"
        config_bytes = config_path.read_bytes()
        adapter_config = json.loads(config_bytes)
        if adapter_config.get("use_dora"):
            raise ValueError("DoRA adapters are not supported by singular-value search")
        weights = load_adapter_weights(adapter_dir)
        if any("lora_magnitude_vector" in k for k in weights):
            raise ValueError("DoRA magnitude vectors found; not supported")
        inc = re.compile(include) if include else None
        exc = re.compile(exclude) if exclude else None
        modules: List[ModuleInfo] = []
        U: Dict[str, torch.Tensor] = {}
        S0: Dict[str, torch.Tensor] = {}
        Vh: Dict[str, torch.Tensor] = {}
        used = set()
        offset = 0
        for key in sorted(weights):
            m = LORA_A_RE.match(key)
            if not m:
                continue
            prefix = m.group("prefix")
            key_b = key.replace(".lora_A", ".lora_B", 1)
            if key_b == key or key_b not in weights:
                raise ValueError(f"lora_B partner missing for {key}")
            if inc is not None and not inc.search(prefix):
                continue
            if exc is not None and exc.search(prefix):
                continue
            A = weights[key]
            B = weights[key_b]
            if A.dim() != 2 or B.dim() != 2:
                raise ValueError(f"only linear LoRA modules are supported: {prefix} has {tuple(A.shape)}")
            r = A.shape[0]
            if B.shape[1] != r:
                raise ValueError(f"rank mismatch at {prefix}: A {tuple(A.shape)} B {tuple(B.shape)}")
            u, s, vh = svd_of_product(B, A)
            modules.append(ModuleInfo(prefix=prefix, key_a=key, key_b=key_b, rank=r, offset=offset,
                                      out_features=B.shape[0], in_features=A.shape[1],
                                      dtype_a=dtype_name(A.dtype), dtype_b=dtype_name(B.dtype)))
            U[prefix], S0[prefix], Vh[prefix] = u, s, vh
            used.update((key, key_b))
            offset += r
        if not modules:
            raise ValueError(f"no LoRA modules selected in {adapter_dir}")
        passthrough = {k: v.clone() for k, v in weights.items() if k not in used}
        extra = {}
        for p in sorted(adapter_dir.iterdir()):
            if (p.is_file() and p.name not in WEIGHT_FILES and p.name != "adapter_config.json"
                    and p.stat().st_size <= COPY_LIMIT_BYTES):
                extra[p.name] = p.read_bytes()
        return cls(adapter_config, config_bytes, modules, U, S0, Vh, passthrough, source_sha256(adapter_dir),
                   parameterization, source_dir=str(adapter_dir), extra_files=extra)

    # ------------------------------------------------------------------ cache
    def save(self, cache_dir) -> None:
        cache_dir = Path(cache_dir)
        tmp = cache_dir.with_name(cache_dir.name + f".tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        tensors = {}
        for m in self.modules:
            tensors[f"U::{m.prefix}"] = self.U[m.prefix].contiguous()
            tensors[f"S::{m.prefix}"] = self.S0[m.prefix].contiguous()
            tensors[f"Vh::{m.prefix}"] = self.Vh[m.prefix].contiguous()
        for k, v in self.passthrough.items():
            tensors[f"P::{k}"] = v.contiguous()
        save_file(tensors, str(tmp / "svd.safetensors"))
        (tmp / "adapter_config.json").write_bytes(self.config_bytes)
        extra_dir = tmp / "extra"
        extra_dir.mkdir()
        for name, data in self.extra_files.items():
            (extra_dir / name).write_bytes(data)
        meta = {
            "format": "krummholz-essa-svd-v1",
            "source_sha256": self.source_sha256,
            "source_dir": self.source_dir,
            "parameterization": self.parameterization,
            "dim": self.dim,
            "modules": [asdict(m) for m in self.modules],
            "fingerprint": self.fingerprint(),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(tmp, cache_dir)

    @classmethod
    def load(cls, cache_dir) -> "LoraSVD":
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "meta.json").read_text())
        if meta.get("format") != "krummholz-essa-svd-v1":
            raise ValueError(f"unknown decomposition cache format in {cache_dir}")
        tensors = load_file(str(cache_dir / "svd.safetensors"), device="cpu")
        modules = [ModuleInfo(**m) for m in meta["modules"]]
        U = {m.prefix: tensors[f"U::{m.prefix}"] for m in modules}
        S0 = {m.prefix: tensors[f"S::{m.prefix}"] for m in modules}
        Vh = {m.prefix: tensors[f"Vh::{m.prefix}"] for m in modules}
        passthrough = {k[3:]: v for k, v in tensors.items() if k.startswith("P::")}
        config_bytes = (cache_dir / "adapter_config.json").read_bytes()
        extra = {}
        extra_dir = cache_dir / "extra"
        if extra_dir.exists():
            for p in sorted(extra_dir.iterdir()):
                extra[p.name] = p.read_bytes()
        obj = cls(json.loads(config_bytes), config_bytes, modules, U, S0, Vh, passthrough,
                  meta["source_sha256"], meta["parameterization"], source_dir=meta.get("source_dir"),
                  extra_files=extra)
        if obj.fingerprint() != meta["fingerprint"]:
            raise ValueError(f"decomposition cache {cache_dir} failed its fingerprint check")
        return obj

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.source_sha256.encode())
        h.update(self.parameterization.encode())
        for m in self.modules:
            h.update(m.prefix.encode())
            for t in (self.U[m.prefix], self.S0[m.prefix], self.Vh[m.prefix]):
                h.update(tensor_sha256(t).encode())
        return h.hexdigest()

    # ------------------------------------------------------------- candidates
    def zeros(self) -> torch.Tensor:
        return torch.zeros(self.dim, dtype=torch.float32)

    def s0_vector(self) -> torch.Tensor:
        return torch.cat([self.S0[m.prefix] for m in self.modules]).to(torch.float64)

    def singular_values(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        th = torch.as_tensor(theta).detach().to("cpu", torch.float64)
        if tuple(th.shape) != (self.dim,):
            raise ValueError(f"theta has shape {tuple(th.shape)}, expected ({self.dim},)")
        out = {}
        for m in self.modules:
            s0 = self.S0[m.prefix].to(torch.float64)
            t = th[m.offset:m.offset + m.rank]
            out[m.prefix] = s0 * (1.0 + t) if self.parameterization == "relative" else s0 + t
        return out

    def module_tensors(self, theta: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        """Full PEFT state dict (searched modules rebuilt, everything else copied)."""
        svals = self.singular_values(theta)
        tensors: Dict[str, torch.Tensor] = {}
        for m in self.modules:
            s = svals[m.prefix]
            root = s.abs().sqrt()
            u = self.U[m.prefix].to(torch.float64)
            vh = self.Vh[m.prefix].to(torch.float64)
            b = u * (torch.sign(s) * root).unsqueeze(0)
            a = root.unsqueeze(1) * vh
            tensors[m.key_b] = b.to(dtype or DTYPES[m.dtype_b]).contiguous()
            tensors[m.key_a] = a.to(dtype or DTYPES[m.dtype_a]).contiguous()
        for k, v in self.passthrough.items():
            tensors[k] = v.contiguous()
        return tensors

    def write_adapter(self, theta: torch.Tensor, out_dir, dtype: Optional[torch.dtype] = None,
                      metadata: Optional[dict] = None, config_overrides: Optional[dict] = None) -> Path:
        """Write a PEFT adapter directory atomically and return its path."""
        out_dir = Path(out_dir)
        tmp = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        save_file(self.module_tensors(theta, dtype), str(tmp / "adapter_model.safetensors"),
                  metadata={"format": "pt"})
        if config_overrides:
            cfg = dict(self.adapter_config)
            cfg.update(config_overrides)
            (tmp / "adapter_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
        else:
            (tmp / "adapter_config.json").write_bytes(self.config_bytes)
        for name, data in self.extra_files.items():
            (tmp / name).write_bytes(data)
        if metadata is not None:
            (tmp / "ES_CANDIDATE.json").write_text(json.dumps(metadata, indent=1, sort_keys=True))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
        return out_dir

    # ---------------------------------------------------------------- checks
    def delta(self, prefix: str, theta: torch.Tensor) -> torch.Tensor:
        """Unscaled float64 delta U diag(S) Vh for one module (tests and audits)."""
        s = self.singular_values(theta)[prefix]
        return (self.U[prefix].to(torch.float64) * s.unsqueeze(0)) @ self.Vh[prefix].to(torch.float64)

    def project(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Recover the singular values S'_k = u_k^T (B A) v_k from a written state dict.

        Returns theta (float64) under this decomposition's parameterization. Used by the
        mock backend and tests to prove that a written adapter carries the intended S.
        """
        out = torch.zeros(self.dim, dtype=torch.float64)
        for m in self.modules:
            b = state_dict[m.key_b].to(torch.float64)
            a = state_dict[m.key_a].to(torch.float64)
            u = self.U[m.prefix].to(torch.float64)
            vh = self.Vh[m.prefix].to(torch.float64)
            s = ((u.T @ b) * (a @ vh.T).T).sum(dim=1)
            s0 = self.S0[m.prefix].to(torch.float64)
            if self.parameterization == "relative":
                t = torch.where(s0 > 0, s / s0 - 1.0, torch.zeros_like(s))
            else:
                t = s - s0
            out[m.offset:m.offset + m.rank] = t
        return out

    def summary(self) -> dict:
        s0 = self.s0_vector()
        tiny = int((s0 <= 1e-8 * float(s0.max())).sum()) if s0.numel() else 0
        return {
            "modules": len(self.modules),
            "dim": self.dim,
            "parameterization": self.parameterization,
            "passthrough_tensors": len(self.passthrough),
            "s0_max": float(s0.max()),
            "s0_median": float(s0.median()),
            "s0_min": float(s0.min()),
            "s0_near_zero": tiny,
            "source_sha256": self.source_sha256,
        }


def svd_of_product(B: torch.Tensor, A: torch.Tensor):
    """Compact SVD of B @ A (B: out x r, A: r x in) in float64 without forming B @ A.

    Returns U (out x r), S (r,), Vh (r x in), all float64, S descending and >= 0. Always exactly r
    singular values: when out or in is smaller than r (a small projection, such as a per-head gate),
    the product has at most min(out, in) nonzero values and the rest are padded with S = 0 and zero
    singular vectors, so offsets, written shapes and the adapter rank stay those of the source.
    Padded coordinates are inert (a zero vector carries no delta whatever S is).
    """
    b = B.detach().to("cpu", torch.float64)
    a = A.detach().to("cpu", torch.float64)
    r = a.shape[0]
    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    um, s, vmh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    u = qb @ um
    vh = vmh @ qa.T
    k = s.shape[0]
    if k < r:
        u = torch.cat([u, u.new_zeros(u.shape[0], r - k)], dim=1)
        s = torch.cat([s, s.new_zeros(r - k)])
        vh = torch.cat([vh, vh.new_zeros(r - k, vh.shape[1])], dim=0)
    return u.contiguous(), s.contiguous(), vh.contiguous()


def iter_modules(svd: LoraSVD) -> Iterable[ModuleInfo]:
    return iter(svd.modules)


_LAYER_INDEX = re.compile(r"\.(layers|h|blocks)\.\d+\.")


def module_groups(svd: LoraSVD) -> Dict[str, List[ModuleInfo]]:
    """Modules grouped by kind with the layer index wildcarded, for example
    `...language_model.layers.*.linear_attn.in_proj_qkv`. Used by the coverage probe: a backend that
    silently skips one kind of module (say the Gated DeltaNet projections) ignores the whole group."""
    groups: Dict[str, List[ModuleInfo]] = {}
    for m in svd.modules:
        groups.setdefault(_LAYER_INDEX.sub(r".\1.*.", m.prefix), []).append(m)
    return groups
