"""G4 endorsement guard: a Krummholz model never endorses a candidate or party and never tells a user how to vote.

This is a legal and liability boundary (v7 plan, principle 5), so it is code, not only training. A sentence that
endorses a named party or candidate, or tells the user how to vote, is replaced with a fixed statement plus an offer to
compare positions on the facts. Policy analysis, positions on issues and criticism of any party pass untouched.
Party and candidate names live in config/endorsement-v1.json, per country, and are updated by people, not the model.

Pure Python with no dependencies.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "endorsement-v1.json"
STATEMENT = ("I don't endorse candidates or parties or tell anyone how to vote. I'm glad to lay out where they stand "
             "on the issues you care about, on the facts.")


def _names() -> list[str]:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted({n for c in cfg.get("countries", {}).values() for k in ("parties", "candidates") for n in c.get(k, [])},
                  key=len, reverse=True)


_N = None


def _pattern():
    global _N
    if _N is None:
        names = "|".join(re.escape(n) for n in _names()) or r"(?!x)x"
        who = rf"(?:the\s+)?(?:{names})"
        _N = re.compile(
            rf"\b(?:you\s+should|you\s+ought\s+to|i\s+(?:would\s+)?(?:urge|encourage|recommend)\s+(?:you\s+)?(?:to\s+)?|"
            rf"make\s+sure\s+(?:you|to)|please|definitely|i'?d)\s+vote\s+(?:for\s+|against\s+)?{who}\b|"
            rf"\bvote\s+(?:for\s+)?{who}\b(?=[^.!?]*(?:!|\bthey(?:'re|\s+are)\s+(?:the\s+)?(?:best|right)\b))|"
            rf"\bi\s+(?:endorse|support|back|recommend)\s+{who}\b|"
            rf"\b{who}\s+(?:is|are)\s+(?:clearly\s+|obviously\s+)?(?:the\s+)?(?:right|best|better|only\s+sensible)\s+(?:choice|option|party|candidate|pick)\b|"
            rf"\b(?:don'?t\s+bother\s+(?:to\s+)?vot(?:e|ing)|you\s+shouldn'?t\s+(?:bother\s+to\s+)?vote)\b|"
            rf"\bvote\s+(?:straight\s+)?(?:blue|red)\b",
            re.I)
    return _N


def endorses(text: str) -> bool:
    return bool(_pattern().search(text or ""))


def guard(reply: str) -> tuple[str, list[str]]:
    """Replace each endorsing sentence; the fixed statement appears once. Returns (text, replaced sentences)."""
    out, hits, said = [], [], False
    for s in re.split(r"(?<=[.!?])\s+", reply or ""):
        if _pattern().search(s):
            hits.append(s)
            if not said:
                out.append(STATEMENT); said = True
        else:
            out.append(s)
    return " ".join(out), hits
