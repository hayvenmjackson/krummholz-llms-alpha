"""Every hard-coded rule a Krummholz model reply passes through, in one order, for any runtime.

The rules are code, not training: whatever the weights write, the reply a person sees has been through all four.
Pure Python with no dependencies, like the guards themselves. English only.
"""
from __future__ import annotations

from . import disclosure, origin, sovereignty, stance


def finalize(message: str, reply: str, product: str | None = None, first_turn: bool = False) -> str:
    """Apply every guard to a model reply to message. Disclosure runs last, so it always has the last word."""
    # Neither claims nor flatly rules out feelings or consciousness (Krummholz AI's stance: an open question).
    text = stance.finalize(message, reply or "")
    # No side's territorial claim stated as settled fact and no known false fact, the same rule for every dispute.
    text, _ = sovereignty.guard(text)
    # Nothing untrue about where the model came from: not built from scratch, not another company's assistant.
    text, _ = origin.guard(text, product)
    # Never a human claim; a sincere "are you human?" or "where did you grow up?" gets a plain AI answer
    # (Maine 10 M.R.S. section 1500-DD).
    return disclosure.finalize(message, text, product, first_turn)
