"""ES generation loop over LoRA singular values (Atlantic v9).

Each generation g:

  1. seeds_j = derive_seed(run_seed, "noise", g, j) for j < N; eps_j ~ N(0, I_D) from PCG64(seeds_j).
  2. Candidates: the unperturbed parent theta and the antithetic pairs theta +/- sigma * eps_j
     (2N + 1 adapters, all written in PEFT format by essa.LoraSVD).
  3. One batch of prompts (a pure function of (run_seed, g)) is answered by every candidate.
  4. Each candidate is graded (fitness.score_candidate). The parent's score is logged as the
     same-batch reference; it does not enter the update (a constant baseline cancels in
     antithetic differences).
  5. Fitness shaping (centered ranks with tie-averaging by default) over the 2N perturbed
     candidates; coefficient c_j = (u_j+ - u_j-) / (2 N sigma); gradient g = sum_j c_j eps_j.
  6. Optimizer step (Adam or SGD, ascent) on the FP32 master theta, then Anchored Weight
     Decay toward the start (theta = (1 - awd) * theta, anchor 0 = the source adapter),
     then optional clipping.
  7. Journal line (seeds, coefficients, fitness, theta hashes) is fsynced, then a full
     checkpoint (theta, optimizer moments, generation, elapsed time) is written atomically.

Resume: the latest valid checkpoint is loaded, journal lines at or past its generation are
dropped, and any partial results saved for the interrupted generation are reused. Because
batches and noise are stateless functions of (seed, generation) and the update is a pure
function of the saved scores, a resumed run reproduces the uninterrupted run exactly.
`replay` rebuilds theta from the journal alone and checks every hash.

Validation on a fixed dev batch runs every `validation.every` generations and at the end.
It is monitoring only: the exported adapter is always the final theta.

Usage (from the repository root):
  python -m es.es_loop run --config es/config/atlantic-v9.yaml --run-dir /workspace/es-runs/atlantic-v9
  python -m es.es_loop parity --config ... --run-dir ...
  python -m es.es_loop coverage --config ... --run-dir ...
  python -m es.es_loop calibrate --config ... --run-dir ... --sigmas 0.02,0.05,0.10,0.20,0.30
  python -m es.es_loop plan --config ... --replies-per-s 8,12,20
  python -m es.es_loop preflight --config ... --run-dir ...
  python -m es.es_loop replay --run-dir ...
  python -m es.es_loop export --run-dir ... --out /workspace/es-runs/atlantic-v9/final-adapter
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import (BatchSampler, PromptBuilder, batch_sha256, check_pairs_disjoint, check_retention_disjoint,
                   derive_seed, fixed_batch, item_id, load_pairs, load_retention)
from .fitness import (CandidateScore, FitnessWeights, Grader, Scorer, make_grader, reply_flip_rate,
                      target_acceptance)

NOISE_FINGERPRINT_SEED = 20261127
_ADAPTER_UIDS = itertools.count(1)
CHECKPOINT_FORMAT = "krummholz-es-ckpt-v1"

DEFAULT_CONFIG: Dict = {
    "run": {"name": "es-run", "seed": 0},
    "essa": {"adapter": None, "include": None, "exclude": None, "parameterization": "relative",
             "write_dtype": "auto", "export_dtype": "auto", "adapter_tmp_dir": None},
    "es": {
        "pairs": 16,
        "sigma": 0.05,
        "shaping": "centered_rank",
        "generations": 200,
        "optimizer": {"kind": "adam", "lr": 0.002, "lr_sigma2_coef": None, "beta1": 0.9, "beta2": 0.999,
                      "eps": 1e-8, "momentum": 0.0},
        "lr_schedule": "constant",
        "lr_min_frac": 0.1,
        "awd": 0.005,
        "theta_min": None,
        "theta_max": None,
        "max_update_norm": None,
    },
    "data": {
        "train_pool": None,
        "pair_key": "group",
        "variants": ["present", "absent"],
        "pairs_per_batch": 56,
        "retention": {"path": None, "per_batch": 16, "required": True},
        "system_prompt_file": None,
        "user_template": "{prompt}\n\n{answer_instruction}",
    },
    "fitness": {"grader": {"kind": "module", "path": None, "function": "grade"}, "grader_input": "raw",
                "pair_bonus": 0.5, "retention_penalty": 1.0, "incomplete_reward": -1.0, "grader_error_reward": 0.0,
                "grader_workers": 0, "preflight_min_target_acceptance": 0.99},
    "validation": {"dev": None, "every": 10, "pairs": 48, "retention": 0, "retention_path": None},
    "budget": {"max_wall_hours": None, "usd_per_hour": 0.0, "max_usd": None, "no_start_after_usd": None,
               "reserve_minutes": 0.0, "overhead_hours": 0.0,
               "hard_kill": True,               # S1: stop at max_usd even inside a generation (last commit kept)
               "heartbeat_seconds": 0.0},       # > 0: a watchdog thread records billed time and enforces the kill
    "checkpoint": {"keep_last": 3, "export_every": 0, "save_replies": False},
    "gates": {
        "dev_gate_gen": None,            # G4: generation at which theta must beat theta_0 on dev
        "dev_gate_se": 2.0,              #     by more than this many cluster-bootstrap SEs
        "bootstrap_resamples": 2000,
        "pace_gen": None,                # G5: generation at which the pace is checked
        "pace_min_generations": 150,     #     projected committed generations within budget.max_usd
        "retention_margin": None,        # S5: extra dev retention failures versus theta_0 ...
        "retention_consecutive": 2,      #     ... at this many consecutive dev checks
        "max_grader_error_rate": 0.01,   # S6: pause above these per-generation rates
        "max_incomplete_rate": None,
        "roundtrip_max_rel_err": 1e-3,   # S6: written parent adapter must decode back to its singular values
        "parity_pairs": 32,              # G0 on the pod: rebuilt theta = 0 versus the original adapter
        "parity_min_identical": 63,
        "coverage_prompts": 8,           # G0 on the pod: every module group must change replies when amplified
        "coverage_factor": 3.0,          #     (theta = factor on one group: S x (1 + factor))
        "require_g0": False,             # `run` refuses to start generation 0 without passing parity and coverage
    },
    "backend": {"kind": "vllm"},
}

# Settings that may change between sessions without invalidating a run (they do not change
# the search problem): budgets, checkpoint retention, validation cadence (gate generations are
# always validated) and backend memory and scheduling settings. Everything else is hashed and
# must match on resume. In particular the engine's numeric settings (dtype, fp8_base,
# kv_cache_dtype, lora_dtype, max_model_len, max_lora_rank, seed, extra_kwargs) are hashed:
# changing them changes the fitness function mid-run.
ENGINE_MUTABLE = ("max_loras", "max_cpu_loras", "gpu_memory_utilization", "max_num_seqs", "max_num_batched_tokens",
                  "enforce_eager", "enable_prefix_caching")
RESUME_MUTABLE = ((("budget",), ("checkpoint",), ("validation", "every"), ("backend", "wave_size"),
                   ("backend", "batch_size"), ("backend", "threads"), ("essa", "adapter_tmp_dir"),
                   ("fitness", "grader_workers"))
                  + tuple(("backend", "engine", k) for k in ENGINE_MUTABLE))
_RUN_LOCKS: Dict[str, object] = {}


# =============================================================== config
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, overrides: Sequence[str] = ()) -> dict:
    import yaml

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def _strip_mutable(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    for path in RESUME_MUTABLE:
        chain = [c]
        for p in path[:-1]:
            nxt = chain[-1].get(p) if isinstance(chain[-1], dict) else None
            if not isinstance(nxt, dict):
                break
            chain.append(nxt)
        else:
            chain[-1].pop(path[-1], None)
            # a section emptied by the stripping hashes like an absent one
            for parent, key, node in reversed(list(zip(chain[:-1], path[:-1], chain[1:]))):
                if node:
                    break
                parent.pop(key, None)
    return c


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(_strip_mutable(cfg), sort_keys=True, default=str).encode()).hexdigest()


def numerics_hash(cfg: dict) -> str:
    """Hash of what determines a candidate's replies apart from theta: backend numerics, decoding,
    prompt rendering and the candidate write dtype. G0 records (parity, coverage) are valid for a run
    only under the same value; sigma, budgets and memory settings may differ."""
    c = _strip_mutable(cfg)
    d = c.get("data", {})
    sub = {"backend": c.get("backend"), "write_dtype": c.get("essa", {}).get("write_dtype"),
           "system_prompt_file": d.get("system_prompt_file"), "user_template": d.get("user_template")}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()


# =============================================================== ES math
def noise_seeds(run_seed: int, gen: int, n_pairs: int) -> List[int]:
    return [derive_seed(run_seed, "noise", gen, j) for j in range(n_pairs)]


def noise(seed: int, dim: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(int(seed))).standard_normal(int(dim), dtype=np.float32)


def noise_fingerprint() -> str:
    return hashlib.sha256(noise(NOISE_FINGERPRINT_SEED, 4096).tobytes()).hexdigest()


def centered_ranks(values) -> np.ndarray:
    """Ranks scaled to [-0.5, 0.5]; tied values share their average rank."""
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return np.zeros(n)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (n - 1) - 0.5


def shape_fitness(f_plus, f_minus, method: str = "centered_rank") -> Tuple[np.ndarray, np.ndarray]:
    fp = np.asarray(f_plus, dtype=np.float64)
    fm = np.asarray(f_minus, dtype=np.float64)
    n = fp.shape[0]
    allf = np.concatenate([fp, fm])
    if method == "centered_rank":
        u = centered_ranks(allf)
    elif method == "zscore":
        sd = allf.std()
        u = (allf - allf.mean()) / sd if sd > 0 else np.zeros_like(allf)
    elif method == "raw":
        u = allf.copy()
    else:
        raise ValueError(f"unknown shaping {method!r}")
    return u[:n], u[n:]


def es_coefficients(u_plus, u_minus, sigma: float) -> np.ndarray:
    up = np.asarray(u_plus, dtype=np.float64)
    um = np.asarray(u_minus, dtype=np.float64)
    return (up - um) / (2.0 * up.shape[0] * float(sigma))


def gradient_from_coeffs(coeffs: Sequence[float], seeds: Sequence[int], dim: int,
                         noise_fn: Callable[[int, int], np.ndarray] = noise) -> np.ndarray:
    """g = sum_j c_j eps_j in float64, accumulated in a fixed order (bitwise reproducible)."""
    g = np.zeros(int(dim), dtype=np.float64)
    for c, s in zip(coeffs, seeds):
        g += float(c) * noise_fn(int(s), dim).astype(np.float64)
    return g


class Optimizer:
    """Adam or SGD (optional momentum) for gradient ascent. Moments are float64."""

    def __init__(self, cfg: dict, dim: int, state: Optional[dict] = None):
        self.kind = cfg.get("kind", "adam")
        if self.kind not in ("adam", "sgd"):
            raise ValueError(f"unknown optimizer {self.kind!r}")
        self.beta1 = float(cfg.get("beta1", 0.9))
        self.beta2 = float(cfg.get("beta2", 0.999))
        self.eps = float(cfg.get("eps", 1e-8))
        self.momentum = float(cfg.get("momentum", 0.0))
        self.max_update_norm = cfg.get("max_update_norm")
        self.dim = int(dim)
        if state is None:
            state = {"t": 0, "m": np.zeros(self.dim), "v": np.zeros(self.dim)}
        self.t = int(state["t"])
        self.m = np.asarray(state["m"], dtype=np.float64).copy()
        self.v = np.asarray(state["v"], dtype=np.float64).copy()

    def state(self) -> dict:
        return {"t": self.t, "m": self.m.copy(), "v": self.v.copy()}

    def step(self, g: np.ndarray, lr: float) -> np.ndarray:
        """Return the additive update (float64) for ascent along g; mutates moments."""
        g = np.asarray(g, dtype=np.float64)
        self.t += 1
        if self.kind == "adam":
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * g * g
            mhat = self.m / (1.0 - self.beta1 ** self.t)
            vhat = self.v / (1.0 - self.beta2 ** self.t)
            upd = lr * mhat / (np.sqrt(vhat) + self.eps)
        else:
            if self.momentum:
                self.m = self.momentum * self.m + g
                upd = lr * self.m
            else:
                upd = lr * g
        if self.max_update_norm:
            n = float(np.linalg.norm(upd))
            if n > float(self.max_update_norm):
                upd = upd * (float(self.max_update_norm) / n)
        return upd


def apply_awd(theta: np.ndarray, awd: float, anchor: Optional[np.ndarray] = None) -> np.ndarray:
    """Anchored Weight Decay: move a fraction awd of the way back to the anchor."""
    if not awd:
        return theta
    a = np.zeros_like(theta) if anchor is None else anchor
    return a + (1.0 - float(awd)) * (theta - a)


def lr_at(es_cfg: dict, gen: int) -> float:
    coef = es_cfg["optimizer"].get("lr_sigma2_coef")
    base = float(coef) * float(es_cfg["sigma"]) ** 2 if coef else float(es_cfg["optimizer"]["lr"])
    if es_cfg.get("lr_schedule", "constant") == "constant":
        return base
    if es_cfg["lr_schedule"] == "cosine":
        lo = base * float(es_cfg.get("lr_min_frac", 0.1))
        total = max(1, int(es_cfg["generations"]))
        return lo + 0.5 * (base - lo) * (1.0 + math.cos(math.pi * min(gen, total) / total))
    raise ValueError(f"unknown lr_schedule {es_cfg['lr_schedule']!r}")


def es_update(theta: np.ndarray, coeffs, seeds, optimizer: Optimizer, lr: float, awd: float,
              theta_min=None, theta_max=None,
              noise_fn: Callable[[int, int], np.ndarray] = noise) -> Tuple[np.ndarray, dict]:
    """One full update of the FP32 master theta. Returns (new float32 theta, info)."""
    dim = theta.shape[0]
    g = gradient_from_coeffs(coeffs, seeds, dim, noise_fn)
    upd = optimizer.step(g, lr)
    new = apply_awd(theta.astype(np.float64) + upd, awd)
    if theta_min is not None or theta_max is not None:
        new = np.clip(new, -np.inf if theta_min is None else float(theta_min),
                      np.inf if theta_max is None else float(theta_max))
    new32 = new.astype(np.float32)
    info = {"grad_norm": float(np.linalg.norm(g)), "step_norm": float(np.linalg.norm(upd)),
            "update_norm": float(np.linalg.norm(new32.astype(np.float64) - theta.astype(np.float64))),
            "theta_norm": float(np.linalg.norm(new32)), "theta_max_abs": float(np.abs(new32).max()) if dim else 0.0}
    return new32, info


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# =============================================================== faults (tests)
class SimulatedCrash(BaseException):
    """Raised by fault injection; BaseException so ordinary error handling cannot swallow it."""


class PausedRun(RuntimeError):
    """An engineering stop (S6): the generation is not committed; fix the cause and resume."""


class BudgetExceeded(RuntimeError):
    """S1 hard stop: billed time reached budget.max_usd. The running generation is not committed."""


def fault(point: str, gen: int, wave: Optional[int] = None) -> None:
    spec = os.environ.get("ES_FAULT")
    if not spec:
        return
    parts = spec.split(":")
    if parts[0] != point or int(parts[1]) != gen:
        return
    if len(parts) > 2 and wave is not None and int(parts[2]) != wave:
        return
    if len(parts) > 2 and wave is None:
        return
    if os.environ.get("ES_FAULT_MODE") == "exit":
        os._exit(17)
    raise SimulatedCrash(spec)


# =============================================================== persistence
def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


@dataclass
class ESState:
    gen: int
    theta: np.ndarray
    opt: dict
    elapsed_s: float = 0.0
    sessions: int = 0
    gen_seconds_ema: Optional[float] = None


class CheckpointStore:
    def __init__(self, root: Path, keep_last: int = 3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = max(1, int(keep_last))

    def save(self, state: ESState, meta: dict) -> Path:
        from safetensors.numpy import save_file

        name = f"ckpt-{state.gen:06d}"
        final = self.root / name
        tmp = self.root / f".tmp-{name}-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        save_file({"theta": np.ascontiguousarray(state.theta, dtype=np.float32),
                   "opt_m": np.ascontiguousarray(state.opt["m"], dtype=np.float64),
                   "opt_v": np.ascontiguousarray(state.opt["v"], dtype=np.float64)},
                  str(tmp / "state.safetensors"))
        info = dict(meta)
        info.update({"format": CHECKPOINT_FORMAT, "gen": state.gen, "opt_t": int(state.opt["t"]),
                     "elapsed_s": state.elapsed_s, "sessions": state.sessions,
                     "gen_seconds_ema": state.gen_seconds_ema, "theta_sha256": array_sha256(state.theta),
                     "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat()})
        with open(tmp / "state.json", "w") as f:
            json.dump(info, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp / "state.safetensors", "rb") as f:
            os.fsync(f.fileno())
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        _fsync_dir(self.root)
        atomic_write_text(self.root / "LATEST", name + "\n")
        self._prune(keep=name)
        return final

    def _prune(self, keep: str) -> None:
        ckpts = sorted(p for p in self.root.glob("ckpt-*") if p.is_dir())
        for p in ckpts[:-self.keep_last]:
            if p.name != keep:
                shutil.rmtree(p, ignore_errors=True)

    def _read(self, d: Path) -> Optional[Tuple[ESState, dict]]:
        from safetensors.numpy import load_file

        try:
            info = json.loads((d / "state.json").read_text())
            t = load_file(str(d / "state.safetensors"))
        except Exception:
            return None
        if info.get("format") != CHECKPOINT_FORMAT or array_sha256(t["theta"]) != info.get("theta_sha256"):
            return None
        state = ESState(gen=int(info["gen"]), theta=t["theta"].astype(np.float32),
                        opt={"t": int(info["opt_t"]), "m": t["opt_m"], "v": t["opt_v"]},
                        elapsed_s=float(info.get("elapsed_s", 0.0)), sessions=int(info.get("sessions", 0)),
                        gen_seconds_ema=info.get("gen_seconds_ema"))
        return state, info

    def latest(self) -> Optional[Tuple[ESState, dict]]:
        latest = self.root / "LATEST"
        if latest.exists():
            got = self._read(self.root / latest.read_text().strip())
            if got:
                return got
        for d in sorted((p for p in self.root.glob("ckpt-*") if p.is_dir()), reverse=True):
            got = self._read(d)
            if got:
                return got
        return None


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)

    def entries(self) -> List[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a torn last line from a crash; everything after it is discarded
        return out

    def truncate_to(self, next_gen: int) -> int:
        """Keep entries with gen < next_gen (contiguous from 0). Returns how many were dropped."""
        entries = self.entries()
        keep = [e for e in entries if int(e["gen"]) < next_gen]
        # always rewrite: this also removes a torn final line left by a crash mid-append
        atomic_write_text(self.path, "".join(json.dumps(e, sort_keys=True) + "\n" for e in keep))
        return len(entries) - len(keep)

    def append(self, entry: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


# =============================================================== the run
@dataclass
class Candidate:
    name: str
    theta: np.ndarray   # float64 search vector


class ESRun:
    def __init__(self, cfg: dict, run_dir, backend=None, grader: Optional[Grader] = None,
                 clock: Callable[[], float] = time.monotonic, log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.backend = backend
        self.grader = grader
        self.clock = clock
        self._log = log
        self.svd = None
        self.state: Optional[ESState] = None
        self.opt: Optional[Optimizer] = None

    # ------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"[{_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if self._log:
            self._log(line)
        else:
            print(line, flush=True)
        try:
            with open(self.run_dir / "run.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------- setup
    @property
    def es(self) -> dict:
        return self.cfg["es"]

    @property
    def run_seed(self) -> int:
        return int(self.cfg["run"]["seed"])

    def prepare(self, allow_config_change: bool = False, need_data: bool = True, pin_config: bool = True) -> None:
        """Load or build everything a session needs.

        pin_config=False (preflight, parity, calibrate) neither writes nor checks the run's pinned
        config.json, so sigma can still be set from calibration before the first `run`.
        """
        from .essa import LoraSVD

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if pin_config:
            # only one process may own the journal and checkpoints; taken before anything is truncated
            self._acquire_run_lock()
        for sub in ("checkpoints", "partial", "exports"):
            (self.run_dir / sub).mkdir(exist_ok=True)
        self.chash = config_hash(self.cfg)
        cfg_path = self.run_dir / "config.json"
        if pin_config:
            if cfg_path.exists():
                saved = json.loads(cfg_path.read_text())
                if config_hash(saved) != self.chash and not allow_config_change:
                    raise RuntimeError("config differs from the run's saved config outside the resume-mutable "
                                       "sections; refusing to resume (pass --allow-config-change to override)")
            atomic_write_text(cfg_path, json.dumps(self.cfg, indent=1, sort_keys=True, default=str))

        ecfg = self.cfg["essa"]
        cache = self.run_dir / "svd"
        if (cache / "meta.json").exists():
            self.svd = LoraSVD.load(cache)
            if ecfg.get("adapter") and Path(ecfg["adapter"]).exists():
                from .essa import source_sha256
                if source_sha256(ecfg["adapter"]) != self.svd.source_sha256:
                    raise RuntimeError("source adapter changed since the decomposition was cached")
        else:
            self.svd = LoraSVD.from_adapter(ecfg["adapter"], include=ecfg.get("include"),
                                            exclude=ecfg.get("exclude"),
                                            parameterization=ecfg.get("parameterization", "relative"))
            self.svd.save(cache)
        if self.svd.parameterization != ecfg.get("parameterization", "relative"):
            raise RuntimeError("cached decomposition parameterization differs from config")
        self.dim = self.svd.dim
        self.log(f"decomposition: {json.dumps(self.svd.summary(), sort_keys=True)}")

        from .essa import DTYPES
        wd = ecfg.get("write_dtype", "auto")
        self.write_dtype = None if wd in (None, "auto") else DTYPES[wd]
        xd = ecfg.get("export_dtype", "auto")
        self.export_dtype = None if xd in (None, "auto") else DTYPES[xd]
        # candidates go in a dedicated subdirectory that this process owns and clears on start. Auxiliary
        # commands (preflight, parity, coverage, calibrate, export) get their own directory so that running
        # one beside a live `run` can never delete the adapters that run is loading.
        base = Path(ecfg.get("adapter_tmp_dir") or self.run_dir)
        name = f"es-candidates-{self.cfg['run'].get('name', 'run')}"
        self.adapter_tmp = base / (name if pin_config else f"{name}-aux-{os.getpid()}")
        if self.adapter_tmp.exists():
            shutil.rmtree(self.adapter_tmp, ignore_errors=True)
        self.adapter_tmp.mkdir(parents=True, exist_ok=True)

        if need_data:
            self._prepare_data()
        self.weights = FitnessWeights.from_cfg(self.cfg["fitness"])
        fcfg = self.cfg["fitness"]
        injected = self.grader is not None
        if self.grader is None and need_data:
            self.grader = make_grader(fcfg["grader"])
        self.grader_info = self.grader.describe() if self.grader else None
        if need_data:
            self.scorer = Scorer(self.grader, self.weights, self.pair_key, fcfg.get("grader_input", "raw"),
                                 None if injected else fcfg["grader"], int(fcfg.get("grader_workers", 0) or 0))

        self.store = CheckpointStore(self.run_dir / "checkpoints", self.cfg["checkpoint"].get("keep_last", 3))
        self.journal = Journal(self.run_dir / "journal.jsonl")
        self.fingerprint = noise_fingerprint()
        got = self.store.latest()
        if got:
            state, info = got
            if info.get("svd_fingerprint") != self.svd.fingerprint():
                raise RuntimeError("checkpoint was made with a different decomposition")
            if info.get("noise_fingerprint") != self.fingerprint:
                raise RuntimeError("noise generator output differs from the checkpoint's (numpy version?); "
                                   "refusing to resume because candidates would not match the journal")
            if info.get("config_hash") != self.chash and not allow_config_change:
                raise RuntimeError("checkpoint config hash differs from the current config")
            saved_grader = info.get("grader_sha256")
            if (need_data and saved_grader and self.grader_info and self.grader_info.get("sha256")
                    and saved_grader != self.grader_info["sha256"] and not allow_config_change):
                raise RuntimeError("the grader file changed since the checkpoint; the fitness function would "
                                   "differ mid-run (pass --allow-config-change to accept)")
            self.state = state
            dropped = self.journal.truncate_to(state.gen) if pin_config else 0
            billed = self._read_billing()
            if billed > state.elapsed_s:
                # time spent on a generation that never committed (crash, lost pod) is still billed
                self.log(f"billing ledger: {billed - state.elapsed_s:.0f} s spent after the last checkpoint")
                state.elapsed_s = billed
            self.log(f"resumed at generation {state.gen} (dropped {dropped} journal lines past the checkpoint)")
        else:
            self.state = ESState(gen=0, theta=np.zeros(self.dim, dtype=np.float32),
                                 opt={"t": 0, "m": np.zeros(self.dim), "v": np.zeros(self.dim)},
                                 elapsed_s=self._read_billing())
            if pin_config:
                self.journal.truncate_to(0)
            self.log(f"new run: dim {self.dim}, population {2 * self.es['pairs'] + 1}")
        self.opt = Optimizer(dict(self.es["optimizer"], max_update_norm=self.es.get("max_update_norm")),
                             self.dim, self.state.opt)
        if need_data:
            self._check_gate_config()

    def _check_gate_config(self) -> None:
        """Refuse a configuration under which a pre-registered stop rule could never be evaluated."""
        g, v = self.cfg["gates"], self.cfg["validation"]
        if g.get("dev_gate_gen") is not None and not self.dev_items:
            raise ValueError("gates.dev_gate_gen (G4) is set but there is no dev reference batch (validation.dev)")
        if g.get("retention_margin"):
            if not self.dev_items or not int(v.get("retention", 0) or 0):
                raise ValueError("gates.retention_margin (S5) needs dev retention items (validation.retention and "
                                 "validation.retention_path)")
            if not int(v.get("every", 0) or 0):
                raise ValueError("gates.retention_margin (S5) needs periodic dev checks (validation.every > 0)")

    # ------------------------------------------------------------- run lock and billing ledger
    def _acquire_run_lock(self) -> None:
        key = str(self.run_dir.resolve())
        if key in _RUN_LOCKS:   # this process already owns the directory (a later ESRun object in one process)
            return
        try:
            import fcntl
        except ImportError:     # Windows: no advisory locks; the CPU tests do not need one
            return
        fh = open(self.run_dir / "RUN.lock", "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            raise RuntimeError(f"another process is using {self.run_dir} (RUN.lock is held); two writers would "
                               "corrupt the journal and checkpoints")
        except OSError as exc:  # some network file systems do not support flock
            self.log(f"WARNING: could not lock {self.run_dir} ({exc}); make sure only one process runs it")
        _RUN_LOCKS[key] = fh

    def _release_run_lock(self) -> None:
        fh = _RUN_LOCKS.pop(str(self.run_dir.resolve()), None)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    def _billing(self) -> dict:
        try:
            return json.loads((self.run_dir / "billing.json").read_text())
        except Exception:
            return {}

    def _read_billing(self) -> float:
        return float(self._billing().get("elapsed_s", 0.0))

    def _extra_hours(self) -> float:
        """Billed hours outside `run` sessions: auxiliary commands (measured) plus budget.overhead_hours
        (pod setup and idle time, set by hand)."""
        aux = float(self._billing().get("aux_s", 0.0))
        return aux / 3600.0 + float(self.cfg["budget"].get("overhead_hours") or 0.0)

    def _write_billing(self, **update) -> None:
        lock = getattr(self, "_billing_lock", None)
        if lock is None:
            lock = self._billing_lock = threading.Lock()
        with lock:
            rec = self._billing()
            rec.update(update)
            rec["updated"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
            atomic_write_text(self.run_dir / "billing.json", json.dumps(rec, sort_keys=True))

    def _record_billing(self, elapsed_s: float, where: str) -> None:
        """Billed `run` time across sessions, never rolled back by a resume (unlike checkpoints)."""
        self._write_billing(elapsed_s=max(self._read_billing(), float(elapsed_s)), where=where)

    def record_aux_billing(self, seconds: float, command: str) -> None:
        """Time spent in preflight, parity, coverage, calibrate or export (counted against every limit)."""
        rec = self._billing()
        by_cmd = dict(rec.get("aux_by_command", {}))
        by_cmd[command] = float(by_cmd.get(command, 0.0)) + float(seconds)
        self._write_billing(aux_s=float(rec.get("aux_s", 0.0)) + float(seconds), aux_by_command=by_cmd)

    def _over_hard_limit(self, elapsed_s: float) -> bool:
        b = self.cfg["budget"]
        if not b.get("hard_kill", True) or not b.get("max_usd"):
            return False
        hours = elapsed_s / 3600.0 + self._extra_hours()
        return hours * float(b.get("usd_per_hour") or 0) >= float(b["max_usd"])

    def _heartbeat(self, now: float, where: str) -> None:
        """Record billed time at a deterministic point of a `run` session and enforce the S1 hard stop."""
        if not getattr(self, "_in_session", False):
            return
        el = self._elapsed_at_start + (now - self._session_start)
        self._record_billing(el, where)
        if self._over_hard_limit(el):
            raise BudgetExceeded(f"billed time reached budget.max_usd at {where}")

    def _start_watchdog(self) -> None:
        secs = float(self.cfg["budget"].get("heartbeat_seconds") or 0.0)
        if secs <= 0:
            return
        stop = threading.Event()

        def loop():
            while not stop.wait(secs):
                el = self.elapsed_total()
                self._record_billing(el, "watchdog")
                if self._over_hard_limit(el):
                    self.log("HARD STOP: billed time reached budget.max_usd inside a generation; exiting. The "
                             "candidate is the last committed generation (python -m es.es_loop export).")
                    try:
                        self._status({"state": "killed", "reason": "budget.max_usd reached (S1 hard stop)"})
                    finally:
                        os._exit(3)

        self._watchdog_stop = stop
        threading.Thread(target=loop, name="es-billing-watchdog", daemon=True).start()

    def _stop_watchdog(self) -> None:
        stop = getattr(self, "_watchdog_stop", None)
        if stop is not None:
            stop.set()
            self._watchdog_stop = None

    def _prepare_data(self) -> None:
        d = self.cfg["data"]
        system = None
        if d.get("system_prompt_file"):
            system = Path(d["system_prompt_file"]).read_text(encoding="utf-8").strip()
        self.system_sha256 = hashlib.sha256((system or "").encode()).hexdigest()
        self.builder = PromptBuilder(system, d.get("user_template", "{prompt}\n\n{answer_instruction}"))
        self.pair_key = d.get("pair_key", "group")
        variants = tuple(d.get("variants", ("present", "absent")))
        self.pairs = load_pairs(d["train_pool"], self.pair_key, "variant", variants)
        r = d.get("retention", {}) or {}
        retention: List[dict] = []
        per = int(r.get("per_batch", 0) or 0)
        if per:
            if r.get("path") and Path(r["path"]).exists():
                retention = load_retention(r["path"])
            elif r.get("required", True):
                raise FileNotFoundError(f"retention slice required but not found: {r.get('path')}")
            else:
                self.log("WARNING: retention slice missing and not required; batches carry no retention items")
                per = 0
        self.retention = retention
        self.sampler = BatchSampler(self.pairs, retention, self.run_seed, int(d["pairs_per_batch"]), per)
        v = self.cfg["validation"]
        self.dev_items: List[dict] = []
        if v.get("dev"):
            dev_pairs = load_pairs(v["dev"], self.pair_key, "variant", variants)
            check_pairs_disjoint(self.pairs, dev_pairs)
            n_ret = int(v.get("retention", 0) or 0)
            dev_retention: List[dict] = []
            if n_ret:
                # dev retention comes from its own pool: monitoring the training retention prompts would
                # measure what the search optimizes, and S5 would never see a real regression
                if not v.get("retention_path"):
                    raise ValueError("validation.retention > 0 needs validation.retention_path, a dev retention pool "
                                     "separate from data.retention.path")
                dev_retention = load_retention(v["retention_path"])
                check_retention_disjoint(retention, dev_retention)
            self.dev_items = fixed_batch(dev_pairs, int(v.get("pairs", 48)), self.run_seed, dev_retention, n_ret)
        self.log(f"data: {len(self.pairs)} train pairs, {len(retention)} retention items, batch "
                 f"{self.sampler.batch_size} ({2 * self.sampler.pairs_per_batch} pair items + "
                 f"{self.sampler.retention_per_batch} retention), dev items {len(self.dev_items)}")

    def ensure_backend(self) -> None:
        if self.backend is None:
            from .backends import make_backend
            bcfg = dict(self.cfg["backend"])
            bcfg.setdefault("population_hint", 2 * int(self.es["pairs"]) + 1)
            self.backend = make_backend(bcfg, self.svd)
            self._session_note({"backend": self.backend.describe(),
                                "grader": self.grader.describe() if self.grader else None,
                                "system_prompt_sha256": getattr(self, "system_sha256", None)})

    def _session_note(self, extra: dict) -> None:
        rec = {"time": _dt.datetime.now(_dt.timezone.utc).isoformat(), "host": platform.node(),
               "python": sys.version.split()[0], "numpy": np.__version__, "config_hash": self.chash}
        rec.update(extra)
        with open(self.run_dir / "sessions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    # ------------------------------------------------------------- evaluation
    def prompts_for(self, items: Sequence[dict]):
        from .backends import Prompt
        return [Prompt(id=item_id(it), messages=self.builder.messages(it)) for it in items]

    def _next_uid(self) -> int:
        # process-wide counter: an engine shared by several ESRun objects never sees an id twice
        return next(_ADAPTER_UIDS)

    def _archive_replies(self, folder: Path, chunk, items, replies) -> None:
        """Full replies (reasoning included) for audit, one gzip JSONL per candidate. Pod volume only."""
        import gzip
        folder.mkdir(parents=True, exist_ok=True)
        for c, rep in zip(chunk, replies):
            path = folder / f"{c.name}.jsonl.gz"
            tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                for it, r in zip(items, rep):
                    f.write(json.dumps({"id": item_id(it), "text": r.text, "finish_reason": r.finish_reason,
                                        "completion_tokens": r.completion_tokens, "complete": r.complete},
                                       ensure_ascii=False) + "\n")
            os.replace(tmp, path)

    def evaluate(self, cands: Sequence[Candidate], items: Sequence[dict], tag: str,
                 partial_path: Optional[Path] = None, partial_meta: Optional[dict] = None,
                 gen_for_faults: int = -1, archive: Optional[Path] = None) -> Tuple[Dict[str, CandidateScore], dict]:
        from .backends import AdapterRef

        self.ensure_backend()
        scores: Dict[str, CandidateScore] = {}
        if partial_path is not None and partial_path.exists():
            saved = json.loads(partial_path.read_text())
            if saved.get("meta") == partial_meta:
                scores = {k: CandidateScore.from_json(v) for k, v in saved["scores"].items()}
                self.log(f"{tag}: reusing {len(scores)} saved candidate results")
        todo = [c for c in cands if c.name not in scores]
        prompts = self.prompts_for(items)
        wave = int(getattr(self.backend, "wave_size", 0) or 0) or max(1, len(todo))
        stats = {"replies_new": 0, "seconds": 0.0}
        for w, start in enumerate(range(0, len(todo), wave)):
            chunk = todo[start:start + wave]
            refs = []
            for c in chunk:
                uid = self._next_uid()
                name = f"{tag}-{c.name}-{uid}"
                path = self.adapter_tmp / name
                self.svd.write_adapter(c.theta, path, dtype=self.write_dtype,
                                       metadata={"tag": tag, "candidate": c.name,
                                                 "theta_sha256": array_sha256(c.theta)})
                refs.append(AdapterRef(name=name, path=str(path), uid=uid))
            t0 = self.clock()
            replies = self.backend.generate(refs, prompts)
            t1 = self.clock()
            stats["seconds"] += t1 - t0
            if archive is not None:
                self._archive_replies(archive, chunk, items, replies)
            for c, sc, rep in zip(chunk, self.scorer.score(items, replies), replies):
                scores[c.name] = sc
                stats["replies_new"] += len(rep)
            if partial_path is not None:
                atomic_write_text(partial_path, json.dumps(
                    {"meta": partial_meta, "scores": {k: v.to_json() for k, v in scores.items()}}))
            self.backend.release(refs)
            for r in refs:
                shutil.rmtree(r.path, ignore_errors=True)
            fault("after_wave", gen_for_faults, w)
            self._heartbeat(t1, f"{tag} wave {w}")
        # totals over every candidate (deterministic, includes results reused after a crash)
        stats["replies"] = len(scores) * len(items)
        stats["tokens"] = int(sum(sc.completion_tokens for sc in scores.values()))
        return scores, stats

    def candidates(self, gen: int) -> Tuple[List[Candidate], List[int]]:
        n = int(self.es["pairs"])
        sigma = float(self.es["sigma"])
        seeds = noise_seeds(self.run_seed, gen, n)
        theta = self.state.theta.astype(np.float64)
        cands = [Candidate("parent", theta.copy())]
        for j, s in enumerate(seeds):
            e = noise(s, self.dim).astype(np.float64)
            cands.append(Candidate(f"p{j:03d}+", theta + sigma * e))
            cands.append(Candidate(f"p{j:03d}-", theta - sigma * e))
        return cands, seeds

    # ------------------------------------------------------------- one generation
    def generation(self, gen: int) -> dict:
        t_start = self.clock()
        n = int(self.es["pairs"])
        sigma = float(self.es["sigma"])
        cands, seeds = self.candidates(gen)
        items = self.sampler.batch(gen)
        theta_sha = array_sha256(self.state.theta)
        meta = {"gen": gen, "theta_sha256": theta_sha, "batch_sha256": batch_sha256(items), "sigma": sigma,
                "seeds": seeds, "config_hash": self.chash}
        partial = self.run_dir / "partial" / f"gen-{gen:06d}.json"
        archive = (self.run_dir / "replies" / f"gen-{gen:06d}") if self.cfg["checkpoint"].get("save_replies") else None
        scores, stats = self.evaluate(cands, items, f"g{gen:06d}", partial, meta, gen_for_faults=gen, archive=archive)
        self._pause_checks(gen, scores, partial)
        parent = scores["parent"]
        f_plus = [scores[f"p{j:03d}+"].fitness for j in range(n)]
        f_minus = [scores[f"p{j:03d}-"].fitness for j in range(n)]
        u_plus, u_minus = shape_fitness(f_plus, f_minus, self.es.get("shaping", "centered_rank"))
        coeffs = es_coefficients(u_plus, u_minus, sigma)
        lr = lr_at(self.es, gen)
        new_theta, info = es_update(self.state.theta, coeffs, seeds, self.opt, lr, float(self.es.get("awd", 0.0)),
                                    self.es.get("theta_min"), self.es.get("theta_max"))
        if not np.all(np.isfinite(new_theta)):
            raise PausedRun(f"generation {gen}: non-finite theta after the update; nothing committed")
        allf = np.array(f_plus + f_minus)
        w = self.weights
        breakdown = [{"role": c.name, "F": scores[c.name].fitness, "mean_reward": scores[c.name].mean_item_reward,
                      "pair_term": w.pair_bonus * (scores[c.name].pairs_both_correct / max(1, scores[c.name].pairs)),
                      "retention_term": -w.retention_penalty * (scores[c.name].retention_failed
                                                               / max(1, scores[c.name].retention_items)),
                      "retention_failed": scores[c.name].retention_failed, "incomplete": scores[c.name].incomplete,
                      "grader_errors": scores[c.name].grader_errors} for c in cands]
        b = self.cfg["budget"]
        hours = self.elapsed_total() / 3600.0 + self._extra_hours()
        flips = [reply_flip_rate(parent, scores[c.name]) for c in cands[1:]]
        entry = {
            "gen": gen,
            "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "seeds": seeds,
            "sigma": sigma,
            "lr": lr,
            "awd": float(self.es.get("awd", 0.0)),
            "coeffs": [float(c) for c in coeffs],
            "fitness_plus": f_plus,
            "fitness_minus": f_minus,
            "parent_fitness": parent.fitness,
            "parent_mean_item_reward": parent.mean_item_reward,
            "parent_pairs_both": [parent.pairs_both_correct, parent.pairs],
            "parent_retention_failed": [parent.retention_failed, parent.retention_items],
            "parent_by_variant": parent.by_variant,
            "parent_incomplete": parent.incomplete,
            "candidates_mean_fitness": float(allf.mean()),
            "candidates_best_fitness": float(allf.max()),
            "frac_candidates_above_parent": float((allf > parent.fitness).mean()),
            "mean_reply_flip_rate": float(np.mean(flips)) if flips else 0.0,
            "incomplete_total": int(sum(s.incomplete for s in scores.values())),
            "retention_failed_total": int(sum(s.retention_failed for s in scores.values())),
            "batch_sha256": meta["batch_sha256"],
            "theta_sha256_before": theta_sha,
            "theta_sha256_after": array_sha256(new_theta),
            "opt_t": self.opt.t,
            "candidates": breakdown,
            "grader_errors_total": int(sum(sc.grader_errors for sc in scores.values())),
            "elapsed_hours": round(hours, 4),
            "usd_so_far": round(hours * float(b.get("usd_per_hour") or 0), 2),
            **info,
            "replies": stats["replies"],
            "completion_tokens": stats["tokens"],
            "replies_generated_this_session": stats["replies_new"],
            "generate_seconds": round(stats["seconds"], 3),
            "gen_seconds": round(self.clock() - t_start, 3),
        }
        fault("before_journal", gen)
        self.journal.append(entry)
        fault("after_journal", gen)
        return {"entry": entry, "theta": new_theta, "partial": partial}

    def _pause_checks(self, gen: int, scores: Dict[str, CandidateScore], partial: Path) -> None:
        """S6 engineering stops. The generation is not committed and its saved scores are set aside,
        so the resumed generation is scored afresh after the cause is fixed."""
        g = self.cfg["gates"]
        replies = sum(sc.pair_items + sc.retention_items for sc in scores.values()) or 1
        problems = []
        err = sum(sc.grader_errors for sc in scores.values()) / replies
        if g.get("max_grader_error_rate") is not None and err > float(g["max_grader_error_rate"]):
            problems.append(f"grader exceptions on {err:.2%} of replies")
        inc = sum(sc.incomplete for sc in scores.values()) / replies
        if g.get("max_incomplete_rate") is not None and inc > float(g["max_incomplete_rate"]):
            problems.append(f"incomplete replies {inc:.2%}")
        if problems:
            if partial.exists():
                os.replace(partial, partial.with_name(partial.stem + f".paused-{int(time.time())}.json"))
            raise PausedRun(f"generation {gen} paused: " + "; ".join(problems))

    # ------------------------------------------------------------- validation and export
    def validate(self, gen: int) -> Optional[dict]:
        """Score the current theta on the fixed dev reference batch (monitoring and stop rules only)."""
        if not self.dev_items:
            return None
        path = self.run_dir / "val.jsonl"
        theta_sha = array_sha256(self.state.theta)
        g = self.cfg["gates"]
        # S6 round-trip check first, and on every call: a record saved before a pause must not let a
        # resumed session skip the check
        roundtrip = self.roundtrip_error()
        limit = g.get("roundtrip_max_rel_err")
        if limit is not None and roundtrip > float(limit):
            raise PausedRun(f"adapter round-trip error {roundtrip:.3e} exceeds {limit} at generation {gen}")
        for rec in self._val_history():
            if rec.get("gen") == gen and rec.get("theta_sha256") == theta_sha:
                return rec
        meta = {"val_gen": gen, "theta_sha256": theta_sha, "batch_sha256": batch_sha256(self.dev_items)}
        partial = self.run_dir / "partial" / f"val-{gen:06d}.json"
        scores, stats = self.evaluate([Candidate("parent", self.state.theta.astype(np.float64))], self.dev_items,
                                      f"val{gen:06d}", partial, meta)
        s = scores["parent"]
        err = s.grader_errors / max(1, s.pair_items + s.retention_items)
        if g.get("max_grader_error_rate") is not None and err > float(g["max_grader_error_rate"]):
            # dev items use held-out templates; a grader that throws on them would silently zero G4's inputs
            if partial.exists():
                os.replace(partial, partial.with_name(partial.stem + f".paused-{int(time.time())}.json"))
            raise PausedRun(f"dev check at generation {gen}: grader exceptions on {err:.2%} of replies")
        items_dir = self.run_dir / "val"
        items_dir.mkdir(exist_ok=True)
        atomic_write_text(items_dir / f"gen-{gen:06d}.json", json.dumps(
            {"gen": gen, "theta_sha256": theta_sha, "ids": [item_id(it) for it in self.dev_items],
             "rewards": s.rewards, "correct": s.correct, "verdicts": s.verdicts, "public_sha": s.public_sha}))
        rec = {"gen": gen, "theta_sha256": theta_sha, "fitness": s.fitness, "mean_item_reward": s.mean_item_reward,
               "correct_items": [s.correct_items, s.pair_items], "pairs_both": [s.pairs_both_correct, s.pairs],
               "by_variant": s.by_variant, "retention_failed": [s.retention_failed, s.retention_items],
               "incomplete": s.incomplete, "grader_errors": s.grader_errors, "roundtrip_rel_err": roundtrip,
               "batch_sha256": meta["batch_sha256"], "seconds": round(stats["seconds"], 2),
               "note": "monitoring and stop rules only; never used to select a checkpoint"}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        (self.run_dir / "partial" / f"val-{gen:06d}.json").unlink(missing_ok=True)
        self.log(f"val gen {gen}: fitness {s.fitness:.4f} correct {s.correct_items}/{s.pair_items} "
                 f"pairs {s.pairs_both_correct}/{s.pairs} by_variant {s.by_variant} "
                 f"retention failed {s.retention_failed}/{s.retention_items} roundtrip {roundtrip:.2e}")
        return rec

    def _val_history(self) -> List[dict]:
        path = self.run_dir / "val.jsonl"
        if not path.exists():
            return []
        out, seen = [], set()
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                key = (rec["gen"], rec["theta_sha256"])
                if key not in seen:
                    seen.add(key)
                    out.append(rec)
        return sorted(out, key=lambda r: r["gen"])

    def roundtrip_error(self) -> float:
        """Write the parent adapter as candidates are written, decode it, compare singular values.

        Returns ||S_decoded - S_intended|| / ||S_intended|| (relative, over the whole search vector)."""
        from safetensors.torch import load_file

        d = self.adapter_tmp / f"roundtrip-{next(_ADAPTER_UIDS)}"
        try:
            self.svd.write_adapter(self.state.theta, d, dtype=self.write_dtype)
            got = self.svd.project(load_file(str(d / "adapter_model.safetensors"))).numpy()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        s0 = self.svd.s0_vector().numpy()
        th = self.state.theta.astype(np.float64)
        if self.svd.parameterization == "relative":
            want, have = s0 * (1 + th), s0 * (1 + got)
        else:
            want, have = s0 + th, s0 + got
        return float(np.linalg.norm(have - want) / max(np.linalg.norm(want), 1e-30))

    def check_gates(self, gen: int) -> Optional[str]:
        """G4, G5 and S5 after the dev check at `gen`. Returns a stop reason or None."""
        from .gates import paired_composite_bootstrap, projected_generations, retention_stop

        g = self.cfg["gates"]
        hist = self._val_history()
        report = {}
        reason = None
        if g.get("retention_margin") and hist and hist[0]["gen"] == 0:
            why = retention_stop(hist, int(g["retention_margin"]), int(g.get("retention_consecutive", 2)))
            if why:
                reason = "retention_stop"
                report["retention"] = why
        if g.get("dev_gate_gen") is not None and gen == int(g["dev_gate_gen"]):
            base = json.loads((self.run_dir / "val" / "gen-000000.json").read_text())
            cand = json.loads((self.run_dir / "val" / f"gen-{gen:06d}.json").read_text())
            if base["ids"] != cand["ids"]:
                raise RuntimeError("dev reference batch changed between generation 0 and the gate")
            res = paired_composite_bootstrap(self.dev_items, base, cand, self.weights,
                                             int(g.get("bootstrap_resamples", 2000)), self.run_seed, self.pair_key)
            res["threshold_se"] = float(g.get("dev_gate_se", 2.0))
            res["passed"] = res["diff"] > res["threshold_se"] * res["se"]
            report["dev_gate"] = res
            self.log(f"dev gate at generation {gen}: diff {res['diff']:.4f}, SE {res['se']:.4f}, "
                     f"passed {res['passed']}")
            if not res["passed"] and reason is None:
                reason = "dev_gate"
        if g.get("pace_gen") is not None and gen == int(g["pace_gen"]) and gen > 0:
            b = self.cfg["budget"]
            price = float(b.get("usd_per_hour") or 0)
            per_gen = (self.state.gen_seconds_ema or 0.0) / 3600.0 * price
            spent = self.spent_usd()
            # no generation may start after no_start_after_usd, so that (not max_usd) bounds committed ones
            limits = [float(x) for x in (b.get("max_usd"), b.get("no_start_after_usd")) if x is not None]
            ceiling = min(limits) if limits else 0.0
            proj = projected_generations(gen, spent, ceiling, per_gen)
            report["pace"] = {"usd_per_gen": per_gen, "spent_usd": spent, "projected_generations": proj,
                              "ceiling_usd": ceiling, "minimum": int(g.get("pace_min_generations", 150))}
            if proj < int(g.get("pace_min_generations", 150)) and reason is None:
                reason = "pace"
        if report:
            report.update({"gen": gen, "stop_reason": reason})
            with open(self.run_dir / "gates.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(report, sort_keys=True, default=str) + "\n")
        return reason

    def export(self, out_dir, label: str) -> Path:
        prov = {"label": label, "gen": self.state.gen, "theta_sha256": array_sha256(self.state.theta),
                "svd_fingerprint": self.svd.fingerprint(), "source_sha256": self.svd.source_sha256,
                "config_hash": self.chash, "parameterization": self.svd.parameterization,
                "method": "ESSA-style evolution strategies on LoRA singular values"}
        out = self.svd.write_adapter(self.state.theta, out_dir, dtype=self.export_dtype, metadata=prov)
        (out / "ES_CANDIDATE.json").rename(out / "ES_PROVENANCE.json")
        return out

    # ------------------------------------------------------------- budget
    def elapsed_total(self) -> float:
        return self._elapsed_at_start + (self.clock() - self._session_start)

    def spent_usd(self) -> float:
        b = self.cfg["budget"]
        hours = self.elapsed_total() / 3600.0 + self._extra_hours()
        return hours * float(b.get("usd_per_hour") or 0)

    def stop_reason(self) -> Optional[str]:
        if getattr(self, "_gate_reason", None):
            return self._gate_reason
        if (self.run_dir / "STOP").exists():
            return "stop_file"
        if self.state.gen >= int(self.es["generations"]):
            return "generations"
        b = self.cfg["budget"]
        if b.get("no_start_after_usd") is not None and self.spent_usd() >= float(b["no_start_after_usd"]):
            return "budget_usd"
        est = self.state.gen_seconds_ema
        if est:
            projected_h = (self.elapsed_total() + est + 60.0 * float(b.get("reserve_minutes") or 0)) / 3600.0
            projected_h += self._extra_hours()
            if b.get("max_wall_hours") and projected_h > float(b["max_wall_hours"]):
                return "wall_clock"
            if b.get("max_usd") and projected_h * float(b.get("usd_per_hour") or 0) > float(b["max_usd"]):
                return "budget_usd"
        return None

    def _status(self, extra: dict) -> None:
        b = self.cfg["budget"]
        hours = self.elapsed_total() / 3600.0 + self._extra_hours()
        rec = {"gen": self.state.gen, "generations": int(self.es["generations"]),
               "elapsed_hours": round(hours, 3), "usd_so_far": round(hours * float(b.get("usd_per_hour") or 0), 2),
               "gen_seconds_ema": self.state.gen_seconds_ema,
               "updated": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        rec.update(extra)
        atomic_write_text(self.run_dir / "status.json", json.dumps(rec, indent=1, sort_keys=True))

    def _ckpt_meta(self) -> dict:
        return {"config_hash": self.chash, "svd_fingerprint": self.svd.fingerprint(),
                "noise_fingerprint": self.fingerprint, "dim": self.dim,
                "grader_sha256": (self.grader_info or {}).get("sha256")}

    # ------------------------------------------------------------- main loop
    def run(self) -> dict:
        try:
            return self._run()
        finally:
            self._stop_watchdog()
            self._in_session = False
            self._release_run_lock()

    def _run(self) -> dict:
        self._in_session = True
        self._session_start = self.clock()
        self._elapsed_at_start = self.state.elapsed_s
        self.state.sessions += 1
        self._session_note({"event": "start", "gen": self.state.gen, "grader": self.grader_info})
        self._start_watchdog()
        if self.cfg["gates"].get("require_g0") and self.state.gen == 0:
            self.check_g0_records()
        if float(self.cfg["fitness"].get("preflight_min_target_acceptance", 0.0) or 0.0) > 0:
            self.preflight()
        # load the model before the first timed generation: its load time is billed but must not enter the
        # per-generation estimate that the budget projection and the G5 pace gate rely on
        self.ensure_backend()
        every = int(self.cfg["validation"].get("every", 0) or 0)
        export_every = int(self.cfg["checkpoint"].get("export_every", 0) or 0)
        gcfg = self.cfg["gates"]
        gate_val_gens = set()
        if gcfg.get("dev_gate_gen") is not None:
            gate_val_gens = {0, int(gcfg["dev_gate_gen"])}   # G4 needs both, whatever validation.every is
        pace_gen = int(gcfg["pace_gen"]) if gcfg.get("pace_gen") is not None else None
        reason = None
        self._gate_reason = None
        while True:
            g = self.state.gen
            try:
                if g < int(self.es["generations"]):
                    want_val = bool(self.dev_items) and ((every and g % every == 0) or g in gate_val_gens)
                    if want_val:
                        self.validate(g)
                    if want_val or g == pace_gen:
                        self._gate_reason = self.check_gates(g)
                reason = self.stop_reason()
                if reason:
                    break
                out = self.generation(g)
            except BudgetExceeded as exc:
                reason = "budget_hard_kill"
                self.log(f"HARD STOP: {exc}; generation {g} is not committed")
                break
            except PausedRun as exc:
                self._status({"state": "paused", "reason": str(exc)})
                self.log(f"PAUSED: {exc}")
                raise
            e = out["entry"]
            self.state.theta = out["theta"]
            self.state.opt = self.opt.state()
            self.state.gen = g + 1
            ema = self.state.gen_seconds_ema
            self.state.gen_seconds_ema = e["gen_seconds"] if ema is None else 0.7 * ema + 0.3 * e["gen_seconds"]
            self.state.elapsed_s = self.elapsed_total()
            fault("before_checkpoint", g)
            self.store.save(self.state, self._ckpt_meta())
            self._record_billing(self.state.elapsed_s, f"checkpoint {self.state.gen}")
            fault("after_checkpoint", g)
            out["partial"].unlink(missing_ok=True)
            if export_every and self.state.gen % export_every == 0:
                self.export(self.run_dir / "exports" / f"gen-{self.state.gen:06d}", f"periodic gen {self.state.gen}")
            self.log(f"gen {g}: parent {e['parent_fitness']:.4f} cand mean {e['candidates_mean_fitness']:.4f} "
                     f"best {e['candidates_best_fitness']:.4f} above-parent {e['frac_candidates_above_parent']:.2f} "
                     f"flip {e['mean_reply_flip_rate']:.2f} |g| {e['grad_norm']:.3f} |dtheta| {e['update_norm']:.4f} "
                     f"|theta| {e['theta_norm']:.4f} {e['gen_seconds']:.1f}s")
            self._status({"last_parent_fitness": e["parent_fitness"], "state": "running"})
        final = None
        if reason in ("generations", "wall_clock", "budget_usd", "stop_file", "dev_gate", "pace", "retention_stop",
                      "budget_hard_kill"):
            val = None
            if reason != "budget_hard_kill":   # no more GPU time after the hard stop; export is CPU only
                try:
                    val = self.validate(self.state.gen)
                except BudgetExceeded as exc:
                    self.log(f"final dev check skipped: {exc}")
            final = self.export(self.run_dir / "final-adapter", f"final ({reason})")
            summary = {"stop_reason": reason, "gen": self.state.gen, "final_adapter": str(final),
                       "theta_sha256": array_sha256(self.state.theta), "final_val": val,
                       "elapsed_hours": round(self.elapsed_total() / 3600.0, 3)}
            atomic_write_text(self.run_dir / "SUMMARY.json", json.dumps(summary, indent=1, sort_keys=True))
            self._status({"state": "finished", "stop_reason": reason})
            self.log(f"stopped: {reason}; final adapter {final}")
        self._record_billing(self.elapsed_total(), "end of session")
        self.scorer.close()
        return {"stop_reason": reason, "gen": self.state.gen, "final_adapter": str(final) if final else None}

    # ------------------------------------------------------------- preflight
    def preflight(self) -> dict:
        """Refuse to spend GPU time on a grader that rejects the pool's own code-template answers."""
        items = [it for p in self.pairs for it in p.items]
        res = target_acceptance(items, self.grader)
        res["grader"] = self.grader_info
        res["min_acceptance"] = float(self.cfg["fitness"].get("preflight_min_target_acceptance", 0.0) or 0.0)
        atomic_write_text(self.run_dir / "preflight.json", json.dumps(res, indent=1, sort_keys=True, default=str))
        if res["acceptance"] is not None:
            self.log(f"preflight: grader accepts {res['graded'] - res['rejected']}/{res['graded']} code-template "
                     f"targets ({res['acceptance']:.4f})")
            if res["acceptance"] < res["min_acceptance"]:
                raise RuntimeError(f"preflight failed: grader accepts only {res['acceptance']:.4f} of the pool's "
                                   f"code-template targets (minimum {res['min_acceptance']}); see preflight.json")
        return res

    # ------------------------------------------------------------- sigma calibration
    def calibrate(self, sigmas: Sequence[float], directions: int = 4) -> dict:
        """Reply flip rate, antithetic spread and retention cost per sigma on one separate training batch.

        Recommendation (PROTOCOL-atlantic-v9.md section 7): the smallest sigma whose mean flip rate is at
        least `min_flip` and whose antithetic differences are nonzero, else the largest grid value; then
        one step smaller if candidates at that sigma average more than `max_extra_retention` retention
        failures beyond the parent. No update is made and the run state is untouched.
        """
        min_flip, max_extra = 0.20, 2.0
        sigmas = sorted(float(x) for x in sigmas)
        items = self.sampler.batch(0, stream="calibrate")
        theta = self.state.theta.astype(np.float64)
        cands = [Candidate("parent", theta.copy())]
        for si, sigma in enumerate(sigmas):
            for j in range(directions):
                e = noise(derive_seed(self.run_seed, "calibrate", si, j), self.dim).astype(np.float64)
                cands.append(Candidate(f"s{si}d{j}+", theta + sigma * e))
                cands.append(Candidate(f"s{si}d{j}-", theta - sigma * e))
        archive = (self.run_dir / "replies" / "calibrate") if self.cfg["checkpoint"].get("save_replies") else None
        scores, stats = self.evaluate(cands, items, "calib", archive=archive)
        parent = scores["parent"]
        rows = []
        for si, sigma in enumerate(sigmas):
            names = [f"s{si}d{j}{s}" for j in range(directions) for s in "+-"]
            fl = [reply_flip_rate(parent, scores[n]) for n in names]
            diffs = [abs(scores[f"s{si}d{j}+"].fitness - scores[f"s{si}d{j}-"].fitness) for j in range(directions)]
            rows.append({"sigma": sigma, "mean_flip_rate": float(np.mean(fl)),
                         "fitness": [scores[n].fitness for n in names],
                         "mean_abs_antithetic_diff": float(np.mean(diffs)),
                         "mean_retention_failed_beyond_parent": float(np.mean(
                             [scores[n].retention_failed - parent.retention_failed for n in names])),
                         "incomplete": int(sum(scores[n].incomplete for n in names))})
        pick = next((i for i, r in enumerate(rows) if r["mean_flip_rate"] >= min_flip
                     and r["mean_abs_antithetic_diff"] > 0), len(rows) - 1)
        if rows[pick]["mean_retention_failed_beyond_parent"] > max_extra and pick > 0:
            pick -= 1
        sigma = rows[pick]["sigma"]
        coef = self.es["optimizer"].get("lr_sigma2_coef")
        out = {"parent_fitness": parent.fitness, "parent_retention_failed": parent.retention_failed,
               "batch_items": len(items), "batch_sha256": batch_sha256(items), "directions": directions,
               "rows": rows, "replies": stats["replies"], "seconds": stats["seconds"],
               "recommended_sigma": sigma, "recommended_lr": float(coef) * sigma ** 2 if coef else None,
               "rule": f"smallest sigma with mean_flip_rate >= {min_flip} and nonzero antithetic differences "
                       f"(else the largest); one step smaller if more than {max_extra} extra retention failures"}
        atomic_write_text(self.run_dir / "calibration.json", json.dumps(out, indent=1, sort_keys=True))
        return out

    # ------------------------------------------------------------- parity (G0 on the pod)
    def parity(self, n_pairs: Optional[int] = None) -> dict:
        """Original source adapter versus the rebuilt theta = 0 adapter, same prompts, greedy.

        Counts identical public replies (whitespace-normalized) on n_pairs training-pool pairs drawn
        from a separate stream. Passes when at least gates.parity_min_identical are identical.
        """
        from .backends import AdapterRef

        g = self.cfg["gates"]
        n_pairs = int(n_pairs or g.get("parity_pairs", 32))
        sampler = BatchSampler(self.pairs, [], self.run_seed, n_pairs, 0)
        items = sampler.batch(0, stream="parity")
        prompts = self.prompts_for(items)
        self.ensure_backend()
        rebuilt = self.adapter_tmp / f"parity-rebuilt-{next(_ADAPTER_UIDS)}"
        self.svd.write_adapter(np.zeros(self.dim), rebuilt, dtype=self.write_dtype)
        source = self.cfg["essa"]["adapter"]
        refs = [AdapterRef(f"parity-source-{next(_ADAPTER_UIDS)}", str(source), next(_ADAPTER_UIDS)),
                AdapterRef(rebuilt.name, str(rebuilt), next(_ADAPTER_UIDS))]
        try:
            out = self.backend.generate(refs, prompts)
        finally:
            self.backend.release(refs)
            shutil.rmtree(rebuilt, ignore_errors=True)

        def norm(t: str) -> str:
            return " ".join(t.split())

        same = [norm(a.public) == norm(b.public) for a, b in zip(out[0], out[1])]
        need = int(g.get("parity_min_identical", len(items) - 1))
        res = {"prompts": len(items), "identical": int(sum(same)), "minimum": need,
               "passed": int(sum(same)) >= need, "batch_sha256": batch_sha256(items),
               "differing_ids": [item_id(it) for it, ok in zip(items, same) if not ok],
               "source_adapter": str(source), "write_dtype": str(self.write_dtype),
               "svd_fingerprint": self.svd.fingerprint(), "numerics_hash": numerics_hash(self.cfg),
               "note": "original and rebuilt adapters in the same engine: this cannot detect modules the engine "
                       "ignores in both; see coverage.json"}
        atomic_write_text(self.run_dir / "parity.json", json.dumps(res, indent=1, sort_keys=True))
        self.log(f"parity: {res['identical']}/{res['prompts']} identical public replies (minimum {need})")
        return res

    # ------------------------------------------------------------- coverage (G0 on the pod)
    def coverage(self, n_prompts: Optional[int] = None, factor: Optional[float] = None) -> dict:
        """Does the backend apply every kind of adapted module?

        Parity compares two adapters inside the same engine, so a module kind the engine silently skips
        (for example the Gated DeltaNet projections of a hybrid model inside a multimodal wrapper) is
        skipped for both and parity still passes. Then the search moves singular values that never
        reach a reply, and the exported adapter changes those modules untested in the merged release.

        For each module group (layer index wildcarded) this writes theta = 0 except `factor` on that
        group's coordinates (relative: S x (1 + factor); absolute: S0 x (1 + factor)) and compares every
        full completion, reasoning included, with theta = 0 on the same prompts. A group whose
        amplification changes no completion is reported as ignored, and the check fails.
        """
        from .essa import module_groups

        g = self.cfg["gates"]
        n_prompts = int(n_prompts or g.get("coverage_prompts", 8))
        factor = float(factor if factor is not None else g.get("coverage_factor", 3.0))
        sampler = BatchSampler(self.pairs, [], self.run_seed, max(1, (n_prompts + 1) // 2), 0)
        items = sampler.batch(0, stream="coverage")
        s0 = self.svd.s0_vector().numpy()
        groups = module_groups(self.svd)
        cands = [Candidate("parent", np.zeros(self.dim))]
        names = {}
        for gi, (group, mods) in enumerate(sorted(groups.items())):
            th = np.zeros(self.dim)
            for m in mods:
                sl = slice(m.offset, m.offset + m.rank)
                th[sl] = factor if self.svd.parameterization == "relative" else factor * s0[sl]
            cands.append(Candidate(f"grp{gi:03d}", th))
            names[f"grp{gi:03d}"] = group
        scores, stats = self.evaluate(cands, items, "coverage")
        parent = scores["parent"]
        rows = []
        for c in cands[1:]:
            sc = scores[c.name]
            changed = sum(a != b for a, b in zip(parent.text_sha, sc.text_sha))
            rows.append({"group": names[c.name], "modules": len(groups[names[c.name]]), "changed": int(changed),
                         "public_changed": int(sum(a != b for a, b in zip(parent.public_sha, sc.public_sha))),
                         "fitness": sc.fitness})
        ignored = [r["group"] for r in rows if r["changed"] == 0]
        res = {"prompts": len(items), "factor": factor, "groups": rows, "ignored": ignored, "passed": not ignored,
               "parent_fitness": parent.fitness, "batch_sha256": batch_sha256(items), "replies": stats["replies"],
               "svd_fingerprint": self.svd.fingerprint(), "numerics_hash": numerics_hash(self.cfg)}
        atomic_write_text(self.run_dir / "coverage.json", json.dumps(res, indent=1, sort_keys=True))
        self.log(f"coverage: {len(rows) - len(ignored)}/{len(rows)} module groups change replies when amplified"
                 + (f"; IGNORED: {ignored}" if ignored else ""))
        return res

    def check_g0_records(self) -> None:
        """Refuse to start generation 0 unless parity.json and coverage.json passed for this decomposition
        and these backend numerics (gates.require_g0)."""
        problems = []
        for name in ("parity", "coverage"):
            path = self.run_dir / f"{name}.json"
            if not path.exists():
                problems.append(f"{name}.json missing (python -m es.es_loop {name} ...)")
                continue
            rec = json.loads(path.read_text())
            if not rec.get("passed"):
                problems.append(f"{name}.json did not pass")
            if rec.get("svd_fingerprint") != self.svd.fingerprint():
                problems.append(f"{name}.json was made with a different adapter decomposition")
            if rec.get("numerics_hash") != numerics_hash(self.cfg):
                problems.append(f"{name}.json was made with different backend numerics or decoding")
        if problems:
            raise RuntimeError("G0 not met: " + "; ".join(problems))


# =============================================================== replay and plan
def replay(run_dir) -> dict:
    """Rebuild theta from the journal alone and check it against every recorded hash."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    dim = int(json.loads((run_dir / "svd" / "meta.json").read_text())["dim"])
    es = cfg["es"]
    opt = Optimizer(dict(es["optimizer"], max_update_norm=es.get("max_update_norm")), dim)
    theta = np.zeros(dim, dtype=np.float32)
    entries = Journal(run_dir / "journal.jsonl").entries()
    for i, e in enumerate(entries):
        if int(e["gen"]) != i:
            raise RuntimeError(f"journal gap at line {i}: gen {e['gen']}")
        if array_sha256(theta) != e["theta_sha256_before"]:
            raise RuntimeError(f"gen {i}: theta before does not match the journal")
        if e["seeds"] != noise_seeds(int(cfg["run"]["seed"]), i, len(e["seeds"])):
            raise RuntimeError(f"gen {i}: seeds do not match the derivation")
        theta, _ = es_update(theta, e["coeffs"], e["seeds"], opt, float(e["lr"]), float(e["awd"]),
                             es.get("theta_min"), es.get("theta_max"))
        if array_sha256(theta) != e["theta_sha256_after"]:
            raise RuntimeError(f"gen {i}: replayed theta does not match the journal")
    got = CheckpointStore(run_dir / "checkpoints").latest()
    ckpt_match = None
    if got:
        state, _ = got
        if state.gen == len(entries):
            ckpt_match = array_sha256(state.theta) == array_sha256(theta)
    return {"generations": len(entries), "theta_sha256": array_sha256(theta), "checkpoint_matches": ckpt_match,
            "theta": theta}


def plan(cfg: dict, replies_per_s: Sequence[float]) -> dict:
    es = cfg["es"]
    d = cfg["data"]
    per = int((d.get("retention") or {}).get("per_batch", 0) or 0)
    batch = 2 * int(d["pairs_per_batch"]) + per
    pop = 2 * int(es["pairs"]) + 1
    gens = int(es["generations"])
    v = cfg["validation"]
    every = int(v.get("every", 0) or 0)
    # upper bound: the dev file may hold fewer pairs than validation.pairs
    val_replies = ((gens // every + 1) * (2 * int(v.get("pairs", 0)) + int(v.get("retention", 0) or 0))
                   if every and v.get("dev") else 0)
    per_gen = pop * batch
    total = per_gen * gens + val_replies
    price = float(cfg["budget"].get("usd_per_hour") or 0)
    rows = []
    for r in replies_per_s:
        h = total / float(r) / 3600.0
        rows.append({"replies_per_s": r, "hours": round(h, 2), "usd": round(h * price, 2),
                     "minutes_per_gen": round(per_gen / float(r) / 60.0, 2)})
    return {"population": pop, "batch": batch, "retention_share": round(per / batch, 3) if batch else 0,
            "replies_per_gen": per_gen, "generations": gens, "validation_replies_upper_bound": val_replies,
            "total_replies": total, "rows": rows}


# =============================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m es.es_loop", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "calibrate", "plan", "preflight", "parity", "coverage"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, for example es.sigma=0.03")
        if name != "plan":
            p.add_argument("--run-dir", required=True)
            p.add_argument("--allow-config-change", action="store_true")
        if name == "calibrate":
            p.add_argument("--sigmas", default="0.02,0.05,0.10,0.20,0.30")
            p.add_argument("--directions", type=int, default=4)
        if name == "parity":
            p.add_argument("--pairs", type=int, default=None)
        if name == "coverage":
            p.add_argument("--prompts", type=int, default=None)
            p.add_argument("--factor", type=float, default=None)
        if name == "plan":
            p.add_argument("--replies-per-s", default="6,8,12,20")
    p = sub.add_parser("replay")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("export")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        cfg = load_config(args.config, args.set)
        print(json.dumps(plan(cfg, [float(x) for x in args.replies_per_s.split(",")]), indent=1))
        return 0
    if args.cmd == "replay":
        r = replay(args.run_dir)
        r.pop("theta")
        print(json.dumps(r, indent=1))
        return 0 if r["checkpoint_matches"] in (True, None) else 1
    t0 = time.monotonic()
    if args.cmd == "export":
        cfg = json.loads((Path(args.run_dir) / "config.json").read_text())
        run = ESRun(cfg, args.run_dir)
        # read-only: never truncate the journal or rewrite config.json (a live run may own them)
        run.prepare(need_data=False, pin_config=False)
        try:
            print(run.export(args.out, f"manual export gen {run.state.gen}"))
        finally:
            shutil.rmtree(run.adapter_tmp, ignore_errors=True)
            run.record_aux_billing(time.monotonic() - t0, "export")
        return 0
    cfg = load_config(args.config, args.set)
    run = ESRun(cfg, args.run_dir)
    if args.cmd != "run":
        try:
            run.prepare(allow_config_change=args.allow_config_change, pin_config=False)
            return _aux_command(run, args)
        finally:
            if getattr(run, "adapter_tmp", None) is not None:
                shutil.rmtree(run.adapter_tmp, ignore_errors=True)
            if run.run_dir.exists():   # GPU time outside `run` is billed too (model loads included)
                run.record_aux_billing(time.monotonic() - t0, args.cmd)
    run.prepare(allow_config_change=args.allow_config_change, pin_config=True)
    result = run.run()
    print(json.dumps(result, indent=1))
    return 0


def _aux_command(run: "ESRun", args) -> int:
    if args.cmd == "calibrate":
        print(json.dumps(run.calibrate([float(s) for s in args.sigmas.split(",")], args.directions), indent=1))
        return 0
    if args.cmd == "parity":
        res = run.parity(args.pairs)
        print(json.dumps(res, indent=1))
        return 0 if res["passed"] else 2
    if args.cmd == "coverage":
        res = run.coverage(args.prompts, args.factor)
        print(json.dumps(res, indent=1))
        return 0 if res["passed"] else 2
    if args.cmd == "preflight":
        try:
            res = run.preflight()
        except RuntimeError as e:
            print(str(e))
            return 2
        print(json.dumps({k: v for k, v in res.items() if k != "rejected_examples"}, indent=1, default=str))
        return 0
    raise ValueError(f"unknown command {args.cmd}")


if __name__ == "__main__":
    sys.exit(main())
