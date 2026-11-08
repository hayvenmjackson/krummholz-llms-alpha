"""G3 loop guard: catch a reply that repeats itself (the Qwen looping habit) and cut it at the first repeat.

detect(text) reports a repeated sentence or a long repeated word n-gram. guard(text) cuts the reply before the second
copy, so a runtime can regenerate once with a higher repetition penalty and log the event as training data (v7 plan,
G3). Lists, tables and code repeat structure legitimately, so lines that look like markup are ignored.

Pure Python with no dependencies.
"""
from __future__ import annotations

import re

_SENT = re.compile(r"[^.!?\n]+[.!?]?")
_MARKUP = re.compile(r"^\s*(?:[-*|#>]|\d+[.)]|```)")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def detect(text: str, n: int = 12, min_sentence_words: int = 6) -> dict | None:
    """First repetition found: {'kind', 'at'} where at is the character offset of the second copy, or None."""
    seen = {}
    prose = "\n".join("" if _MARKUP.match(l) else l for l in (text or "").splitlines())
    for m in _SENT.finditer(prose):
        s = _norm(m.group(0))
        if len(s.split()) < min_sentence_words:
            continue
        if s in seen:
            return {"kind": "sentence", "at": m.start()}
        seen[s] = m.start()
    words = [(w.group(0).lower(), w.start()) for w in re.finditer(r"\w+", prose)]
    grams = {}
    for i in range(len(words) - n + 1):
        g = " ".join(w for w, _ in words[i:i + n])
        if g in grams and i - grams[g] >= n:
            return {"kind": "ngram", "at": words[i][1]}
        grams.setdefault(g, i)
    return None


def guard(text: str) -> tuple[str, dict | None]:
    """Cut a looping reply before its first repeat. Returns (text, event or None)."""
    hit = detect(text)
    if not hit:
        return text, None
    return text[:hit["at"]].rstrip(), hit
