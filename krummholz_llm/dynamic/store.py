"""Per-user local state for dynamic weights (ADR-004 section 4).

  memory.sqlite        facts with provenance; retrieved and cited, never trained
  examples.jsonl       append-only, hash-chained record of everything the adapter learned
  review_queue.jsonl   candidates awaiting the user's review
  ledger.jsonl         activations, rollbacks, forgets and review cycles

Encryption at rest is the platform's job (Android Keystore-backed storage, desktop
OS keychain); this module keeps files private to the profile directory and never
logs example content into the ledger.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

from .manifest import canonical

FACT_BASIS = frozenset({"user_stated", "user_confirmed"})
EXAMPLE_KINDS = frozenset({"sft", "dpo"})
EXAMPLE_ORIGINS = frozenset({"correction", "confirmation", "edit", "remember"})
GENESIS = "0" * 64


class StoreError(ValueError):
    pass


def now_iso(clock=None):
    return (clock() if clock else dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc).isoformat()


def _private(path):
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class LocalStore:
    def __init__(self, root, clock=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.examples_path = self.root / "examples.jsonl"
        self.queue_path = self.root / "review_queue.jsonl"
        self.ledger_path = self.root / "ledger.jsonl"
        self.db = sqlite3.connect(self.root / "memory.sqlite")
        self.db.execute("""CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, basis TEXT NOT NULL, source_ref TEXT,
            learned_at TEXT NOT NULL, approved_at TEXT NOT NULL, expires_at TEXT,
            status TEXT NOT NULL DEFAULT 'active', embedding_model TEXT, embedding BLOB)""")
        self.db.commit()
        for p in (self.root / "memory.sqlite", self.examples_path, self.queue_path, self.ledger_path):
            if p.exists():
                _private(p)

    # ---- ledger (no content, only ids, hashes and reasons)
    def log(self, event, **details):
        row = {"at": now_iso(self.clock), "event": event, **details}
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _private(self.ledger_path)
        return row

    def ledger(self):
        return _read_jsonl(self.ledger_path)

    # ---- memory
    def add_fact(self, text, basis, source_ref=None, learned_at=None, expires_at=None):
        if basis not in FACT_BASIS:
            raise StoreError("facts_come_only_from_the_user")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise StoreError("invalid_fact_text")
        fid = "fact-" + uuid.uuid4().hex[:16]
        stamp = now_iso(self.clock)
        self.db.execute("INSERT INTO facts (id, text, basis, source_ref, learned_at, approved_at, expires_at) VALUES (?,?,?,?,?,?,?)",
                        (fid, text.strip(), basis, source_ref, learned_at or stamp, stamp, expires_at))
        self.db.commit()
        self.log("fact_added", id=fid, basis=basis)
        return fid

    def facts(self, include_expired=False):
        rows = self.db.execute("SELECT id, text, basis, source_ref, learned_at, approved_at, expires_at FROM facts WHERE status='active'").fetchall()
        stamp = now_iso(self.clock)
        keys = ("id", "text", "basis", "source_ref", "learned_at", "approved_at", "expires_at")
        return [dict(zip(keys, r)) for r in rows if include_expired or r[6] is None or r[6] > stamp]

    def forget_fact(self, fid):
        if self.db.execute("DELETE FROM facts WHERE id=?", (fid,)).rowcount != 1:
            raise StoreError("unknown_fact")
        self.db.commit()
        self.db.execute("VACUUM")  # do not leave the deleted text in free pages
        self.log("fact_forgotten", id=fid)

    # ---- example log
    def examples(self):
        return _read_jsonl(self.examples_path)

    def append_example(self, kind, messages, origin, target=None, chosen=None, rejected=None, approved_at=None):
        if kind not in EXAMPLE_KINDS or origin not in EXAMPLE_ORIGINS:
            raise StoreError("invalid_example_kind_or_origin")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            raise StoreError("messages_must_end_with_the_user_turn")
        if kind == "sft" and not (isinstance(target, str) and target.strip() and chosen is None and rejected is None):
            raise StoreError("sft_needs_target_only")
        if kind == "dpo" and not (isinstance(chosen, str) and isinstance(rejected, str) and chosen != rejected and target is None):
            raise StoreError("dpo_needs_distinct_chosen_and_rejected")
        rows = self.examples()
        record = {"id": "ex-" + uuid.uuid4().hex[:16], "kind": kind, "origin": origin, "messages": messages,
                  "created_at": now_iso(self.clock), "approved_at": approved_at or now_iso(self.clock),
                  "prev_sha256": rows[-1]["sha256"] if rows else GENESIS}
        if kind == "sft":
            record["target"] = target
        else:
            record.update(chosen=chosen, rejected=rejected)
        record["sha256"] = hashlib.sha256(canonical(record)).hexdigest()
        with open(self.examples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _private(self.examples_path)
        self.log("example_added", id=record["id"], kind=kind, origin=origin, sha256=record["sha256"])
        return record["id"]

    def verify_chain(self):
        prev = GENESIS
        for row in self.examples():
            body = {k: v for k, v in row.items() if k != "sha256"}
            if row.get("prev_sha256") != prev or hashlib.sha256(canonical(body)).hexdigest() != row.get("sha256"):
                return False
            prev = row["sha256"]
        return True

    def examples_sha256(self):
        """Pinned in the L4 manifest; any change to the log changes this hash."""
        return hashlib.sha256(self.examples_path.read_bytes() if self.examples_path.exists() else b"").hexdigest()

    def forget_example(self, eid):
        """Delete an example and re-chain the log. The adapter must then be rebuilt from the
        remaining log so the deleted item's influence is gone, not just hidden."""
        rows = self.examples()
        keep = [r for r in rows if r["id"] != eid]
        if len(keep) == len(rows):
            raise StoreError("unknown_example")
        prev, out = GENESIS, []
        for r in keep:
            body = {k: v for k, v in r.items() if k != "sha256"}
            body["prev_sha256"] = prev
            body["sha256"] = hashlib.sha256(canonical(body)).hexdigest()
            prev = body["sha256"]
            out.append(body)
        tmp = self.examples_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        os.replace(tmp, self.examples_path)
        _private(self.examples_path)
        self.log("example_forgotten", id=eid, retrain_required=True)

    # ---- review queue
    def queue(self):
        return _read_jsonl(self.queue_path)

    def enqueue(self, candidate):
        with open(self.queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        _private(self.queue_path)

    def dequeue(self, cid):
        rows = self.queue()
        keep = [r for r in rows if r["id"] != cid]
        if len(keep) == len(rows):
            raise StoreError("unknown_candidate")
        tmp = self.queue_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
        os.replace(tmp, self.queue_path)
        return next(r for r in rows if r["id"] == cid)

    def close(self):
        self.db.close()


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
