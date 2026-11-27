"""Deterministic mock backend for fast loop tests (no model).

Each candidate adapter is read back from disk and decoded into theta with
LoraSVD.project, so the mock also checks that the written files carry the intended
singular values. The reply reports a score that peaks at a hidden target theta:

    score_p = 1 - ||theta - (target + jitter_p)||^2 / ||target||^2

and the mock grader (fitness.MockGrader) turns "SCORE x" into a reward.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .base import AdapterRef, Backend, Prompt, Reply, make_reply


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little") >> 1


class MockBackend(Backend):
    def __init__(self, cfg: dict, svd):
        if svd is None:
            raise ValueError("mock backend needs the LoraSVD")
        self.svd = svd
        dim = svd.dim
        rng = np.random.Generator(np.random.PCG64(int(cfg.get("target_seed", 7))))
        direction = rng.standard_normal(dim)
        self.target = direction / np.linalg.norm(direction) * float(cfg.get("target_norm", 0.3))
        self.jitter = float(cfg.get("jitter", 0.02))
        self.wave_size = int(cfg.get("wave_size", 0))
        self.sleep_s = float(cfg.get("sleep_s", 0.0))   # simulate a slow generate() call (watchdog tests)
        self.calls = 0
        self.released: List[int] = []

    def score(self, theta: np.ndarray, prompt_id: str) -> float:
        rng = np.random.Generator(np.random.PCG64(_seed("mock-prompt", prompt_id)))
        t = self.target + self.jitter * rng.standard_normal(self.target.shape[0])
        return float(1.0 - np.sum((theta - t) ** 2) / np.sum(self.target ** 2))

    def generate(self, adapters: Sequence[AdapterRef], prompts: Sequence[Prompt]) -> List[List[Reply]]:
        from safetensors.torch import load_file

        self.calls += 1
        if self.sleep_s:
            import time
            time.sleep(self.sleep_s)
        out = []
        for ad in adapters:
            sd = load_file(str(Path(ad.path) / "adapter_model.safetensors"))
            theta = self.svd.project(sd).numpy()
            row = []
            for p in prompts:
                value = self.score(theta, p.id)
                row.append(make_reply(f"<think>mock</think>SCORE {value:.9f}", "stop", 4, False))
            out.append(row)
        return out

    def release(self, adapters: Sequence[AdapterRef]) -> None:
        self.released.extend(a.uid for a in adapters)

    def describe(self) -> dict:
        return {"kind": "mock", "dim": self.svd.dim, "target_norm": float(np.linalg.norm(self.target))}
