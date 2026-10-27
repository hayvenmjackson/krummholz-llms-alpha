"""Capture and review: only explicit user actions become candidates, only accepted ones learn.

ADR-004 decisions: review is offered daily with at most three review-and-update cycles
per user per day; each cycle has an item cap. Retrieved pages, documents and tool
results can inform an answer but can never be captured here.
"""
from __future__ import annotations

import datetime as dt
import uuid

from .store import LocalStore, StoreError, now_iso

MAX_CYCLES_PER_DAY = 3
DEFAULT_ITEMS_PER_CYCLE = 50
CANDIDATE_KINDS = {
    "fact": "remember",            # "remember that ..."      -> memory store
    "behavior": "confirmation",    # "always / shorter / format like this" -> SFT example
    "correction": "correction",    # user fixed an answer      -> DPO pair (and fact if sourced)
    "edit": "edit",                # user edited an answer     -> DPO pair
}
# Untrusted origins are refused at capture time, whatever a caller claims.
FORBIDDEN_ORIGINS = frozenset({"retrieved", "tool_result", "web", "document", "email", "message", "model"})


class ReviewError(ValueError):
    pass


def capture(store: LocalStore, kind, *, actor, messages=None, fact=None, target=None, original=None, revised=None,
            source_ref=None):
    """Record a candidate from an explicit user action. `actor` must be "user"."""
    if actor != "user" or actor in FORBIDDEN_ORIGINS:
        raise ReviewError("only_the_user_can_create_learning_candidates")
    if kind not in CANDIDATE_KINDS:
        raise ReviewError("unknown_candidate_kind")
    if kind == "fact" and not (isinstance(fact, str) and fact.strip()):
        raise ReviewError("fact_text_required")
    if kind == "behavior" and not (messages and isinstance(target, str) and target.strip()):
        raise ReviewError("behavior_needs_messages_and_target")
    if kind in {"correction", "edit"} and not (messages and isinstance(original, str) and isinstance(revised, str) and original != revised):
        raise ReviewError("correction_needs_original_and_revised")
    candidate = {"id": "cand-" + uuid.uuid4().hex[:16], "kind": kind, "created_at": now_iso(store.clock),
                 "messages": messages, "fact": fact, "target": target, "original": original, "revised": revised,
                 "source_ref": source_ref, "actor": "user"}
    store.enqueue(candidate)
    store.log("candidate_captured", id=candidate["id"], kind=kind)
    return candidate["id"]


class ReviewCycle:
    """One "What I learned today" session. Opening a fourth cycle on the same UTC day fails."""

    def __init__(self, store: LocalStore, items_cap=DEFAULT_ITEMS_PER_CYCLE):
        today = now_iso(store.clock)[:10]
        opened = [e for e in store.ledger() if e["event"] == "review_cycle_opened" and e["at"][:10] == today]
        if len(opened) >= MAX_CYCLES_PER_DAY:
            raise ReviewError("daily_review_cycle_limit_reached")
        if not 1 <= items_cap <= 500:
            raise ReviewError("invalid_items_cap")
        self.store, self.cap, self.accepted, self.closed = store, items_cap, 0, False
        self.id = "cycle-" + uuid.uuid4().hex[:12]
        store.log("review_cycle_opened", id=self.id, cycle_of_day=len(opened) + 1)

    def pending(self):
        return self.store.queue()

    def decide(self, cid, decision, edited=None):
        if self.closed:
            raise ReviewError("cycle_closed")
        if decision not in {"accept", "edit", "reject"}:
            raise ReviewError("unknown_decision")
        if decision != "reject" and self.accepted >= self.cap:
            raise ReviewError("cycle_item_cap_reached")
        c = self.store.dequeue(cid)
        if decision == "reject":
            self.store.log("candidate_rejected", id=cid, kind=c["kind"])
            return None
        if decision == "edit":
            if not isinstance(edited, str) or not edited.strip():
                raise ReviewError("edit_requires_new_text")
            field = {"fact": "fact", "behavior": "target", "correction": "revised", "edit": "revised"}[c["kind"]]
            c[field] = edited
        stamp = now_iso(self.store.clock)
        if c["kind"] == "fact":
            ref = self.store.add_fact(c["fact"], "user_stated", source_ref=c.get("source_ref"))
        elif c["kind"] == "behavior":
            ref = self.store.append_example("sft", c["messages"], "confirmation", target=c["target"], approved_at=stamp)
        else:
            ref = self.store.append_example("dpo", c["messages"], CANDIDATE_KINDS[c["kind"]], chosen=c["revised"],
                                            rejected=c["original"], approved_at=stamp)
        self.accepted += 1
        self.store.log("candidate_accepted", id=cid, kind=c["kind"], stored_as=ref, cycle=self.id)
        return ref

    def close(self):
        """Ends the cycle. Returns True when there is something new to train on."""
        self.closed = True
        self.store.log("review_cycle_closed", id=self.id, accepted=self.accepted, train_requested=self.accepted > 0)
        return self.accepted > 0
