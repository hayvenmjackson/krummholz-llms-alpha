"""Training pool, dev and retention loading; deterministic, stateless batch sampling.

Item files live outside this public repository (see config/atlantic-v9.yaml) and are
referenced only by path. The yardstick v1 schema (yardstick/heldout/FORMAT.md):
two lines per pair sharing `group`, `variant` present/absent, user text is
`prompt + "\\n\\n" + answer_instruction`.

The terminal held-out set and the conflict set are refused outright: any file named
terminal.jsonl or conflict.jsonl and any item whose `split` is "terminal" or "conflict" raises
TerminalSetError. Training and dev pools (and the two retention pools) are checked for overlap
at load time (check_pairs_disjoint, check_retention_disjoint).

Batches are a pure function of (seed, generation): pairs are drawn from per-epoch
permutations of the pool, so a resumed run sees exactly the batches an uninterrupted
run would, and every pair is used once before any pair repeats.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

TERMINAL_FILENAMES = {"terminal.jsonl", "conflict.jsonl"}
FORBIDDEN_SPLITS = {"terminal", "conflict"}
DISJOINT_KEYS = ("record_ids", "entities", "blocks")


class TerminalSetError(RuntimeError):
    pass


class LeakageError(RuntimeError):
    """Two pools that must be disjoint (train and dev, training and dev retention) share material."""


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from any parts (sha256 of their ':'-joined string)."""
    data = ":".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") & ((1 << 63) - 1)


def load_jsonl(path) -> List[dict]:
    path = Path(path)
    if path.name in TERMINAL_FILENAMES:
        raise TerminalSetError(f"refusing to read the terminal held-out set: {path}")
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("split", "")).lower() in FORBIDDEN_SPLITS:
                raise TerminalSetError(f"{path}:{n} is a terminal held-out item; never used for search")
            items.append(item)
    return items


def item_id(item: dict) -> str:
    for key in ("id", "item_id", "item_key"):
        if item.get(key):
            return str(item[key])
    raise KeyError("item has no id")


@dataclass
class Pair:
    group: str
    items: List[dict]   # ordered as `variants` (present first)


def pair_key_of(item: dict, pair_key: str = "group") -> str:
    if item.get(pair_key):
        return str(item[pair_key])
    iid = item_id(item)
    if "/" in iid:
        return iid.rsplit("/", 1)[0]
    raise KeyError(f"item {iid} has no {pair_key!r} and no '/<variant>' id suffix")


def load_pairs(path, pair_key: str = "group", variant_key: str = "variant",
               variants: Sequence[str] = ("present", "absent")) -> List[Pair]:
    groups: Dict[str, Dict[str, dict]] = {}
    for item in load_jsonl(path):
        g = pair_key_of(item, pair_key)
        v = str(item.get(variant_key))
        if v not in variants:
            raise ValueError(f"{path}: item {item_id(item)} has variant {v!r}, expected one of {list(variants)}")
        slot = groups.setdefault(g, {})
        if v in slot:
            raise ValueError(f"{path}: group {g} has two {v!r} items")
        slot[v] = item
    pairs = []
    for g in sorted(groups):
        slot = groups[g]
        missing = [v for v in variants if v not in slot]
        if missing:
            raise ValueError(f"{path}: group {g} lacks {missing}")
        pairs.append(Pair(group=g, items=[slot[v] for v in variants]))
    if not pairs:
        raise ValueError(f"{path}: no pairs")
    return pairs


def load_retention(path) -> List[dict]:
    items = load_jsonl(path)
    for it in items:
        item_id(it)
        if "checks" not in it:
            raise ValueError(f"retention item {item_id(it)} has no 'checks'")
        if "messages" not in it and "prompt" not in it:
            raise ValueError(f"retention item {item_id(it)} has neither 'messages' nor 'prompt'")
        it.setdefault("split", "retention")
        if it["split"] != "retention":
            raise ValueError(f"retention item {item_id(it)} has split {it['split']!r}")
    ids = [item_id(it) for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate retention ids")
    return items


def is_retention(item: dict) -> bool:
    return item.get("split") == "retention"


def _norm_text(text) -> str:
    return " ".join(str(text or "").lower().split())


def retention_text(item: dict) -> str:
    """Normalized user-visible text of a retention item (for disjointness checks)."""
    if "messages" in item:
        return _norm_text(" ".join(str(m.get("content", "")) for m in item["messages"] if m.get("role") != "system"))
    return _norm_text(item.get("prompt", ""))


def check_pairs_disjoint(train: List["Pair"], dev: List["Pair"], keys: Sequence[str] = DISJOINT_KEYS) -> dict:
    """Refuse train and dev pools that share a group, an item id, or any record id, entity or block.

    FORMAT.md guarantees zero shared record ids, entities and blocks between splits; this re-checks it
    on the copies the run actually reads, so a mis-copied or mis-pointed file cannot leak dev into fitness.
    """
    def collect(pairs):
        out = {"group": set(), "id": set()}
        for k in keys:
            out[k] = set()
        for p in pairs:
            out["group"].add(p.group)
            for it in p.items:
                out["id"].add(item_id(it))
                for k in keys:
                    out[k].update(str(x) for x in (it.get(k) or []))
        return out

    a, b = collect(train), collect(dev)
    shared = {k: sorted(a[k] & b[k])[:5] for k in a if a[k] & b[k]}
    if shared:
        raise LeakageError(f"train and dev pools overlap: {shared}")
    return {k: 0 for k in a}


def check_retention_disjoint(train: List[dict], dev: List[dict]) -> None:
    """Refuse a dev retention pool that shares an id or a normalized prompt with the training retention pool."""
    ids = {item_id(it) for it in train} & {item_id(it) for it in dev}
    texts = {retention_text(it) for it in train} & {retention_text(it) for it in dev}
    texts.discard("")
    if ids or texts:
        raise LeakageError(f"training and dev retention pools overlap: ids {sorted(ids)[:5]}, "
                           f"{len(texts)} shared prompts")


def _epoch_take(n: int, k: int, gen: int, seed: int, stream: str) -> List[int]:
    """Indices for batch `gen`: positions gen*k .. gen*k+k-1 of the concatenated epoch permutations.

    Duplicates inside one batch (only possible across an epoch boundary) are skipped and
    replaced by the next position, so a batch never contains the same element twice.
    """
    if k == 0:
        return []
    if k > n:
        raise ValueError(f"batch needs {k} distinct elements but the pool has {n}")
    out: List[int] = []
    seen = set()
    pos = gen * k
    perms: Dict[int, np.ndarray] = {}
    while len(out) < k:
        epoch, off = divmod(pos, n)
        if epoch not in perms:
            perms[epoch] = np.random.Generator(np.random.PCG64(derive_seed(seed, stream, epoch))).permutation(n)
        idx = int(perms[epoch][off])
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
        pos += 1
    return out


class BatchSampler:
    def __init__(self, pairs: List[Pair], retention: List[dict], seed: int, pairs_per_batch: int,
                 retention_per_batch: int):
        self.pairs = pairs
        self.retention = retention
        self.seed = int(seed)
        self.pairs_per_batch = int(pairs_per_batch)
        self.retention_per_batch = int(retention_per_batch)
        if self.retention_per_batch and not retention:
            raise ValueError("retention_per_batch > 0 but no retention items were loaded")

    @property
    def batch_size(self) -> int:
        return 2 * self.pairs_per_batch + self.retention_per_batch

    def batch(self, gen: int, stream: str = "train") -> List[dict]:
        items: List[dict] = []
        for i in _epoch_take(len(self.pairs), self.pairs_per_batch, gen, self.seed, f"{stream}-pairs"):
            items.extend(self.pairs[i].items)
        for i in _epoch_take(len(self.retention), self.retention_per_batch, gen, self.seed, f"{stream}-retention"):
            items.append(self.retention[i])
        return items


def fixed_batch(pairs: List[Pair], n_pairs: int, seed: int, retention: Optional[List[dict]] = None,
                n_retention: int = 0) -> List[dict]:
    """A fixed monitoring batch: the first n pairs by sha256(seed|group) (all pairs if fewer)."""
    ranked = sorted(pairs, key=lambda p: hashlib.sha256(f"{seed}|{p.group}".encode()).hexdigest())
    items: List[dict] = []
    for p in ranked[:n_pairs]:   # all pairs when the file has fewer than n_pairs
        items.extend(p.items)
    if n_retention:
        r = sorted(retention or [], key=lambda it: hashlib.sha256(f"{seed}|{item_id(it)}".encode()).hexdigest())
        if len(r) < n_retention:
            raise ValueError(f"dev batch wants {n_retention} retention items, pool has {len(r)}")
        items.extend(r[:n_retention])
    return items


def batch_sha256(items: Sequence[dict]) -> str:
    return hashlib.sha256("\n".join(item_id(it) for it in items).encode()).hexdigest()


_FIELD = re.compile(r"\{(prompt|answer_instruction)\}")


class PromptBuilder:
    """Chat messages for an item: optional system prompt, then the user text.

    Items with `messages` are used as given (a system message is prepended when the
    item has none and a system prompt is configured). Otherwise the user text is the
    template with {prompt} and {answer_instruction} filled from the item.
    """

    def __init__(self, system_prompt: Optional[str], user_template: str = "{prompt}\n\n{answer_instruction}"):
        self.system_prompt = system_prompt
        self.user_template = user_template

    def user_text(self, item: dict) -> str:
        values = {"prompt": str(item.get("prompt", "")), "answer_instruction": str(item.get("answer_instruction") or "")}
        text = _FIELD.sub(lambda m: values[m.group(1)], self.user_template)
        return text.strip()

    def messages(self, item: dict) -> List[Dict[str, str]]:
        if "messages" in item:
            msgs = [dict(m) for m in item["messages"]]
            if self.system_prompt and not any(m.get("role") == "system" for m in msgs):
                msgs.insert(0, {"role": "system", "content": self.system_prompt})
            return msgs
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": self.user_text(item)})
        return msgs
