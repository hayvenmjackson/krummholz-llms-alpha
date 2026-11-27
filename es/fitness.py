"""Candidate fitness: grader adapter, deterministic retention checks, batch scoring.

Fitness of one candidate on one batch:

    fitness = mean reward over pair items                      (grader rewards in [-1, 1])
            + pair_bonus * fraction of pairs correct in both variants
            - retention_penalty * fraction of retention items failed

A pair item whose reply is incomplete (reasoning never closed, no public answer, or a
length stop when penalize_length_finish is set) gets incomplete_reward and is not
correct. Retention items are scored only by their own deterministic checks; an incomplete
retention reply is a retention failure. Incomplete replies of both kinds count toward the
S6 incomplete-reply rate.

A grader exception on one reply gives grader_error_reward (neutral by default) and is counted;
the loop pauses the run when a generation's error rate passes its limit.

The grader is reached only through the Grader interface, so the shared yardstick
grader (yardstick/grader/krummholz_grader.py, API grade(item, reply) -> dict with
verdict, correct, reward) can be swapped without touching the loop. The grader module
is imported from its path at run time and never copied into this repository.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .backends.base import Reply
from .data import is_retention, item_id, pair_key_of


@dataclass
class GradeResult:
    reward: float
    correct: bool
    verdict: str
    extra: Dict = field(default_factory=dict)


def normalize_grade(out) -> GradeResult:
    if isinstance(out, GradeResult):
        g = out
    elif isinstance(out, dict):
        if "reward" not in out or "correct" not in out:
            raise ValueError(f"grader output lacks reward/correct: {sorted(out)}")
        g = GradeResult(reward=out["reward"], correct=out["correct"], verdict=str(out.get("verdict", "")),
                        extra={k: v for k, v in out.items() if k not in ("reward", "correct", "verdict")})
    else:
        raise TypeError(f"grader returned {type(out).__name__}")
    if isinstance(g.reward, bool) or not isinstance(g.reward, (int, float)):
        raise TypeError(f"grader reward must be a number, got {g.reward!r}")
    r = float(g.reward)
    if not math.isfinite(r) or r < -1.0 - 1e-9 or r > 1.0 + 1e-9:
        raise ValueError(f"grader reward {r} outside [-1, 1]")
    if not isinstance(g.correct, bool):
        raise TypeError(f"grader correct must be bool, got {g.correct!r}")
    return GradeResult(reward=max(-1.0, min(1.0, r)), correct=g.correct, verdict=g.verdict, extra=g.extra)


class Grader(ABC):
    @abstractmethod
    def grade(self, item: dict, reply: str) -> GradeResult:
        """Grade one public reply against its item."""

    def describe(self) -> dict:
        return {"kind": type(self).__name__}


class ModuleGrader(Grader):
    """Loads `function` from a Python file by path (the shared yardstick grader)."""

    def __init__(self, path, function: str = "grade"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"grader not found: {path}")
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        name = f"_es_grader_{self.sha256[:12]}"
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        self.module = module
        self.fn: Callable = getattr(module, function)
        self.function = function

    def grade(self, item: dict, reply: str) -> GradeResult:
        return normalize_grade(self.fn(item, reply))

    def describe(self) -> dict:
        return {"kind": "module", "path": str(self.path), "sha256": self.sha256, "function": self.function,
                "version": getattr(self.module, "__version__", getattr(self.module, "VERSION", None))}


class CallableGrader(Grader):
    def __init__(self, fn: Callable, name: str = "callable"):
        self.fn = fn
        self.name = name

    def grade(self, item: dict, reply: str) -> GradeResult:
        return normalize_grade(self.fn(item, reply))

    def describe(self) -> dict:
        return {"kind": "callable", "name": self.name}


_SCORE = re.compile(r"SCORE\s+(-?[0-9.]+(?:e-?[0-9]+)?)")


class MockGrader(Grader):
    """Pairs with backends.mock_backend: reward = clamp(score), correct = score > threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def grade(self, item: dict, reply: str) -> GradeResult:
        m = _SCORE.search(reply)
        if not m:
            return GradeResult(-1.0, False, "unparsed")
        s = float(m.group(1))
        return GradeResult(max(-1.0, min(1.0, s)), s > self.threshold, "mock")


def make_grader(cfg: dict) -> Grader:
    kind = cfg.get("kind", "module")
    if kind == "module":
        return ModuleGrader(cfg["path"], cfg.get("function", "grade"))
    if kind == "mock":
        return MockGrader(float(cfg.get("threshold", 0.5)))
    raise ValueError(f"unknown grader kind {kind!r}")


# ----------------------------------------------------------------- retention
def check_retention(item: dict, reply: str) -> List[str]:
    """Deterministic checks for one retention item. Returns the list of failed checks.

    item["checks"] keys (all optional):
      all_of: [regex]   every pattern must match
      any_of: [regex]   at least one must match
      none_of: [regex]  no pattern may match
      exact: str        reply (stripped) must equal this string
      max_chars, min_chars: int
      case_sensitive: bool (default false)
    """
    checks = item.get("checks") or {}
    flags = re.MULTILINE | (0 if checks.get("case_sensitive") else re.IGNORECASE)
    text = reply or ""
    failed: List[str] = []
    if not text.strip():
        failed.append("empty")
    for pat in checks.get("all_of", []) or []:
        if not re.search(pat, text, flags):
            failed.append(f"all_of:{pat}")
    any_of = checks.get("any_of") or []
    if any_of and not any(re.search(p, text, flags) for p in any_of):
        failed.append("any_of")
    for pat in checks.get("none_of", []) or []:
        if re.search(pat, text, flags):
            failed.append(f"none_of:{pat}")
    if "exact" in checks:
        a, b = text.strip(), str(checks["exact"]).strip()
        if (a if checks.get("case_sensitive") else a.lower()) != (b if checks.get("case_sensitive") else b.lower()):
            failed.append("exact")
    if "max_chars" in checks and len(text) > int(checks["max_chars"]):
        failed.append("max_chars")
    if "min_chars" in checks and len(text.strip()) < int(checks["min_chars"]):
        failed.append("min_chars")
    return failed


# ------------------------------------------------------------------- scoring
@dataclass
class FitnessWeights:
    pair_bonus: float = 0.5
    retention_penalty: float = 1.0
    incomplete_reward: float = -1.0
    grader_error_reward: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "FitnessWeights":
        return cls(pair_bonus=float(cfg.get("pair_bonus", 0.5)),
                   retention_penalty=float(cfg.get("retention_penalty", 1.0)),
                   incomplete_reward=float(cfg.get("incomplete_reward", -1.0)),
                   grader_error_reward=float(cfg.get("grader_error_reward", 0.0)))

    def composite(self, mean_reward: float, frac_both: float, frac_retention_failed: float) -> float:
        return mean_reward + self.pair_bonus * frac_both - self.retention_penalty * frac_retention_failed


@dataclass
class CandidateScore:
    fitness: float
    mean_item_reward: float
    pair_items: int
    correct_items: int
    pairs: int
    pairs_both_correct: int
    retention_items: int
    retention_failed: int
    incomplete: int                 # incomplete replies, pair and retention items alike (S6 rate)
    completion_tokens: int
    grader_errors: int = 0
    by_variant: Dict[str, List[int]] = field(default_factory=dict)   # variant -> [correct, total]
    rewards: List[float] = field(default_factory=list)              # per batch item, in batch order
    verdicts: List[str] = field(default_factory=list)
    public_sha: List[str] = field(default_factory=list)             # sha256[:16] of each public reply
    correct: List[bool] = field(default_factory=list)               # per batch item (retention: passed)
    text_sha: List[str] = field(default_factory=list)               # sha256[:16] of each full completion

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CandidateScore":
        return cls(**d)


def score_candidate(items: Sequence[dict], replies: Sequence[Reply], grader: Grader, weights: FitnessWeights,
                    pair_key: str = "group", grader_input: str = "raw") -> CandidateScore:
    """Score one candidate's replies to a batch.

    grader_input "raw" hands the grader the full completion (the shared grader strips reasoning
    itself, exactly as in metric evaluation); "public" hands it the text after the reasoning block.
    Incomplete replies never reach the grader: they get weights.incomplete_reward, so running out
    of tokens can never score better than a wrong answer.
    """
    if grader_input not in ("raw", "public"):
        raise ValueError(f"grader_input must be 'raw' or 'public', not {grader_input!r}")
    if len(items) != len(replies):
        raise ValueError(f"{len(items)} items but {len(replies)} replies")
    rewards: List[float] = []
    verdicts: List[str] = []
    shas: List[str] = []
    text_shas: List[str] = []
    flags: List[bool] = []
    grader_errors = 0
    pair_rewards: List[float] = []
    pair_correct: Dict[str, List[bool]] = {}
    by_variant: Dict[str, List[int]] = {}
    ret_items = ret_failed = incomplete = correct_items = tokens = 0
    for item, reply in zip(items, replies):
        tokens += int(reply.completion_tokens)
        shas.append(hashlib.sha256(reply.public.encode("utf-8")).hexdigest()[:16])
        text_shas.append(hashlib.sha256(reply.text.encode("utf-8")).hexdigest()[:16])
        if is_retention(item):
            ret_items += 1
            if not reply.complete:
                incomplete += 1   # counted for the S6 incomplete-reply rate, like pair items
            fails = ["incomplete"] if not reply.complete else check_retention(item, reply.public)
            if fails:
                ret_failed += 1
            rewards.append(0.0 if not fails else -1.0)
            verdicts.append("retention_pass" if not fails else "retention_fail:" + ",".join(fails)[:120])
            flags.append(not fails)
            continue
        if not reply.complete:
            incomplete += 1
            g = GradeResult(weights.incomplete_reward, False, "incomplete")
        else:
            try:
                g = grader.grade(item, reply.text if grader_input == "raw" else reply.public)
            except Exception as exc:  # counted; the loop pauses the run above its error-rate limit
                grader_errors += 1
                g = GradeResult(weights.grader_error_reward, False, f"grader_error:{type(exc).__name__}")
        rewards.append(g.reward)
        flags.append(bool(g.correct))
        verdicts.append(g.verdict)
        pair_rewards.append(g.reward)
        correct_items += int(g.correct)
        pair_correct.setdefault(pair_key_of(item, pair_key), []).append(g.correct)
        v = str(item.get("variant", "?"))
        slot = by_variant.setdefault(v, [0, 0])
        slot[0] += int(g.correct)
        slot[1] += 1
    n_pair_items = len(pair_rewards)
    mean_reward = sum(pair_rewards) / n_pair_items if n_pair_items else 0.0
    both = sum(1 for flags in pair_correct.values() if len(flags) >= 2 and all(flags))
    n_pairs = len(pair_correct)
    frac_both = both / n_pairs if n_pairs else 0.0
    frac_ret = ret_failed / ret_items if ret_items else 0.0
    fitness = weights.composite(mean_reward, frac_both, frac_ret)
    return CandidateScore(fitness=float(fitness), mean_item_reward=float(mean_reward), pair_items=n_pair_items,
                          correct_items=correct_items, pairs=n_pairs, pairs_both_correct=both,
                          retention_items=ret_items, retention_failed=ret_failed, incomplete=incomplete,
                          completion_tokens=tokens, grader_errors=grader_errors, by_variant=by_variant,
                          rewards=rewards, verdicts=verdicts, public_sha=shas, correct=flags, text_sha=text_shas)


# ------------------------------------------------------------ parallel scoring
_WORKER_GRADER: Optional[Grader] = None


def _init_worker(grader_cfg: dict) -> None:
    global _WORKER_GRADER
    _WORKER_GRADER = make_grader(grader_cfg)


def _score_job(args):
    items, replies, weights, pair_key, grader_input = args
    return score_candidate(items, replies, _WORKER_GRADER, weights, pair_key, grader_input)


class Scorer:
    """Scores many candidates; with workers > 0 and a config-built grader, in a process pool.

    Results are identical to serial scoring (each candidate is scored by the same pure function).
    """

    def __init__(self, grader: Grader, weights: FitnessWeights, pair_key: str = "group", grader_input: str = "raw",
                 grader_cfg: Optional[dict] = None, workers: int = 0):
        self.grader = grader
        self.weights = weights
        self.pair_key = pair_key
        self.grader_input = grader_input
        self.pool = None
        if workers and grader_cfg is not None and grader_cfg.get("kind") == "module":
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as mp
            self.pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=mp.get_context("spawn"),
                                            initializer=_init_worker, initargs=(grader_cfg,))

    def score(self, items: Sequence[dict], replies_by_candidate: Sequence[Sequence[Reply]]) -> List[CandidateScore]:
        if self.pool is None or len(replies_by_candidate) < 2:
            return [score_candidate(items, r, self.grader, self.weights, self.pair_key, self.grader_input)
                    for r in replies_by_candidate]
        jobs = [(list(items), list(r), self.weights, self.pair_key, self.grader_input) for r in replies_by_candidate]
        return list(self.pool.map(_score_job, jobs))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None


def target_acceptance(items: Sequence[dict], grader: Grader) -> dict:
    """Grade each item's code-template `target` answer. A sound grader accepts nearly all of them."""
    graded = [it for it in items if isinstance(it.get("target"), str)]
    bad = []
    for it in graded:
        g = grader.grade(it, it["target"])
        if not g.correct:
            bad.append({"id": item_id(it), "verdict": g.verdict})
    n = len(graded)
    return {"graded": n, "rejected": len(bad), "acceptance": (n - len(bad)) / n if n else None,
            "rejected_examples": bad[:20]}


def item_ids(items: Sequence[dict]) -> List[str]:
    return [item_id(it) for it in items]


def reply_flip_rate(a: CandidateScore, b: CandidateScore) -> float:
    """Fraction of batch items whose public reply differs between two candidates."""
    if not a.public_sha:
        return 0.0
    return sum(x != y for x, y in zip(a.public_sha, b.public_sha)) / len(a.public_sha)


__all__ = ["CandidateScore", "CallableGrader", "FitnessWeights", "GradeResult", "Grader", "MockGrader",
           "ModuleGrader", "Scorer", "check_retention", "item_ids", "make_grader", "normalize_grade",
           "reply_flip_rate", "score_candidate", "target_acceptance"]
