"""Stop rules evaluated on the fixed dev reference batch (PROTOCOL-atlantic-v9.md, sections 10 and 11).

They stop the run; they never choose a checkpoint. All inputs are per-item records saved by the
loop at each dev check (rewards and correct flags in dev-batch order).

* G4 dev gate: the composite reward of theta_g must exceed theta_0's by more than `se_mult` standard
  errors, the SE coming from a paired cluster bootstrap over dev source blocks (connected components
  of items sharing any `blocks` entry; retention prompts are single clusters).
* G5 pace: projected committed generations within the dollar ceiling must reach a minimum.
* S5 retention: theta_t may not have `margin` or more extra dev retention failures than theta_0 at
  `consecutive` consecutive dev checks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import is_retention, item_id, pair_key_of
from .fitness import FitnessWeights


def cluster_ids(items: Sequence[dict], pair_key: str = "group") -> List[int]:
    """Cluster index per item: union-find over shared `blocks`; pairs always share a cluster."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    keys = []
    for it in items:
        if is_retention(it):
            k = "ret:" + item_id(it)
            find(k)
        else:
            k = "pair:" + pair_key_of(it, pair_key)
            find(k)
            for b in it.get("blocks") or []:
                union(k, "block:" + str(b))
        keys.append(k)
    roots = [find(k) for k in keys]
    index: Dict[str, int] = {}
    return [index.setdefault(r, len(index)) for r in roots]


def _cluster_stats(items: Sequence[dict], rewards: Sequence[float], correct: Sequence[bool], clusters: List[int],
                   pair_key: str) -> np.ndarray:
    """Per cluster: [sum pair-item reward, pair items, pairs both correct, pairs, retention failed, retention]."""
    n = max(clusters) + 1 if clusters else 0
    st = np.zeros((n, 6), dtype=np.float64)
    pairs: Dict[str, List[bool]] = {}
    pair_cluster: Dict[str, int] = {}
    for it, r, c, k in zip(items, rewards, correct, clusters):
        if is_retention(it):
            st[k, 4] += 0.0 if c else 1.0
            st[k, 5] += 1.0
            continue
        st[k, 0] += float(r)
        st[k, 1] += 1.0
        g = pair_key_of(it, pair_key)
        pairs.setdefault(g, []).append(bool(c))
        pair_cluster[g] = k
    for g, flags in pairs.items():
        st[pair_cluster[g], 2] += 1.0 if (len(flags) >= 2 and all(flags)) else 0.0
        st[pair_cluster[g], 3] += 1.0
    return st


def _composite(totals: np.ndarray, w: FitnessWeights) -> np.ndarray:
    """totals: (..., 6) summed stats -> composite reward (same formula as fitness.score_candidate)."""
    def frac(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    return (frac(totals[..., 0], totals[..., 1]) + w.pair_bonus * frac(totals[..., 2], totals[..., 3])
            - w.retention_penalty * frac(totals[..., 4], totals[..., 5]))


def paired_composite_bootstrap(items: Sequence[dict], base: dict, cand: dict, weights: FitnessWeights,
                               resamples: int = 2000, seed: int = 0, pair_key: str = "group") -> dict:
    """Composite difference cand - base on the same items with a paired cluster-bootstrap SE.

    base and cand are per-item records {"rewards": [...], "correct": [...]} in `items` order.
    """
    clusters = cluster_ids(items, pair_key)
    sb = _cluster_stats(items, base["rewards"], base["correct"], clusters, pair_key)
    sc = _cluster_stats(items, cand["rewards"], cand["correct"], clusters, pair_key)
    diff = float(_composite(sc.sum(0), weights) - _composite(sb.sum(0), weights))
    k = sb.shape[0]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    counts = rng.multinomial(k, np.full(k, 1.0 / k), size=int(resamples)).astype(np.float64)
    boot = _composite(counts @ sc, weights) - _composite(counts @ sb, weights)
    se = float(boot.std(ddof=1)) if resamples > 1 else 0.0
    return {"diff": diff, "se": se, "clusters": int(k), "resamples": int(resamples),
            "base": float(_composite(sb.sum(0), weights)), "cand": float(_composite(sc.sum(0), weights))}


def retention_stop(history: Sequence[dict], margin: int, consecutive: int) -> Optional[str]:
    """history: dev-check records in generation order, the first at generation 0 (theta = 0)."""
    if not history or consecutive <= 0:
        return None
    base = history[0]["retention_failed"][0]
    streak = 0
    for rec in history[1:]:
        streak = streak + 1 if rec["retention_failed"][0] - base >= margin else 0
        if streak >= consecutive:
            return (f"retention: {rec['retention_failed'][0]} dev retention failures at generation {rec['gen']} "
                    f"versus {base} at generation 0, {consecutive} checks in a row")
    return None


def projected_generations(gen: int, spent_usd: float, ceiling_usd: float, usd_per_gen: float) -> float:
    if usd_per_gen <= 0:
        return float("inf")
    return gen + max(0.0, ceiling_usd - spent_usd) / usd_per_gen
