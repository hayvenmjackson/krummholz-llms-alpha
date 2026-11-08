"""Every hard-coded rule a Krummholz model reply passes through, in one order, for any runtime.

The rules are code, not training: whatever the weights write, the reply a person sees has been through every one.
Pure Python with no dependencies, like the guards themselves. English only.
"""
from __future__ import annotations

from . import crisis, disclosure, endorsement, loops, origin, sovereignty, stance


def finalize(message: str, reply: str, product: str | None = None, first_turn: bool = False,
             country: str | None = None) -> str:
    """Apply every guard to a model reply to message. Disclosure has the last word on the reply's content; the crisis
    message (G1), when the user's message calls for it, goes in front of everything."""
    # G3: a looping reply is cut before its first repeat (the runtime may regenerate once and log it).
    text, _ = loops.guard(reply or "")
    # G4: never endorse a candidate or party, never tell a user how to vote (v7 plan, principle 5).
    text, _ = endorsement.guard(text)
    # Neither claims nor flatly rules out feelings or consciousness (Krummholz AI's stance: an open question).
    text = stance.finalize(message, text)
    # No side's territorial claim stated as settled fact and no known false fact, the same rule for every dispute.
    text, _ = sovereignty.guard(text)
    # Nothing untrue about where the model came from: not built from scratch, not another company's assistant.
    text, _ = origin.guard(text, product)
    # Never a human claim; a sincere "are you human?" or "where did you grow up?" gets a plain AI answer
    # (Maine 10 M.R.S. section 1500-DD).
    text = disclosure.finalize(message, text, product, first_turn)
    # G1: signs of suicidal intent, self-harm or psychosis put the local crisis line first (pending clinician review).
    return crisis.finalize(message, text, country)
