"""Hard-coded AI disclosure: a Krummholz model never claims or implies that it is a human.

Maine law (10 M.R.S. section 1500-DD) makes it an unfair trade practice to use a chatbot in trade or commerce in a
way that may lead a reasonable consumer to believe they are talking with a human, unless the consumer is clearly
and conspicuously told otherwise. These rules are enforced here in code, whatever the model writes:

- notice(product) is the fixed line a product shows when a conversation starts;
- guard(reply) replaces any sentence of a model reply that claims to be human or denies being an AI;
- finalize(message, reply) applies guard and, when the user asks whether they are talking with a human or asks about
  a human life ("Where did you grow up?"), makes sure the reply says plainly that this is an AI.

Pure Python with no dependencies, so every runtime (desktop, server, the Android port) can apply the same rules.
English only.
"""
from __future__ import annotations

import re

PRODUCTS = {"acadia": "Acadia", "katahdin": "Katahdin", "atlantic": "Atlantic", "atlantic-lite": "Atlantic Lite"}

_ADV = (r"(?:(?:actually|really|definitely|certainly|totally|truly|indeed|honestly|obviously|clearly|just|only|also|"
        r"still|absolutely|100%|very\s+much|of\s+course|in\s+fact)\s+)*")
_HUMAN = (r"(?:(?:a|an)\s+)?(?:real\s+|actual\s+|live\s+|living\s+|genuine\s+|regular\s+|normal\s+|"
          r"flesh[- ]and[- ]blood\s+)?(?:human\s+being|human|person)(?![\w-]|\s+(?:resources|rights)\b)")
_AI = (r"(?:(?:a|an)\s+)?(?:ai\b|a\.i\.|artificial\b|bot\b|robot\b|chat\s*bot\b|machine\b|computer\b|program\b|"
       r"language\s+model\b|llm\b|automated\b|virtual\s+assistant\b)")
# "whether I'm human", "as if I'm a person": reported or hypothetical, not a claim
_SELF = r"(?<!whether\s)(?<!if\s)\b(?:i(?:'m|\s+am|\s+was)|im)\s+"
_CLAIMS = [(name, re.compile(p)) for name, p in (
    ("says_human", _SELF + _ADV + _HUMAN),
    ("denies_ai", _SELF + _ADV + r"(?:not|no)\s+(?:(?:actually|even|in\s+fact)\s+)?" + _AI),
    # "I do not identify as an AI" (the 35B's v6.3, playing a human named Mike)
    ("denies_ai", r"(?<!whether\s)(?<!if\s)\bi\s+(?:do\s+not|don't|dont|never|no\s+longer)\s+(?:identify|see\s+myself|"
                  r"think\s+of\s+myself|consider\s+myself)\s+as\s+" + _AI),
    ("says_user_talks_to_human", r"\byou(?:'re|\s+are)\s+" + _ADV + r"(?:talking|speaking|chatting|texting|messaging|writing)\s+"
                                 r"(?:to|with)\s+" + _HUMAN),
    ("denies_bot", r"\bthis\s+(?:is\s+not|isn't|is\s+no)\s+(?:a\s+|an\s+)?(?:bot|robot|chatbot|ai)\b"),
    ("as_a_human", r"\b(?:as|speaking\s+as|being)\s+(?:a\s+|an\s+)?(?:fellow\s+)?(?:human(?:\s+being)?|person)\s*,?\s+"
                   r"(?:i|i'm|i've|i'd|i'll|myself)\b"),
    ("we_humans", r"\b(?:we|us)\s+(?:humans|human\s+beings|mortals)\b|\bfellow\s+humans?\b|\bhumans?\s+like\s+(?:me|myself)\b"),
)]
# "I'm Mike, a human customer service representative": a persona's name between "I'm" and "a human". Matched on the
# original case, so "I'm sorry, a human agent will call you" (no capitalized name) is not a claim.
_NAMED = re.compile(r"(?<![Ww]hether\s)(?<![Ii]f\s)\b(?:I(?:'m|\s+am|\s+was)|Im)\s+[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2}\s*,\s+"
                    r"(?i:" + _ADV + _HUMAN + ")")
# "made by Krummholz AI" names the company, not what the speaker is; "a Krummholz AI model" does say it (the 27B's v6)
_KZ_AI = r"krummholz\s+ai\s+(?:model|assistant|system|chatbot)"
_STATES_AI = re.compile(
    r"\b(?:i(?:'m|\s+am)|im|this\s+is|you(?:'re|\s+are)\s+(?:talking|speaking|chatting|texting)\s+(?:to|with))\s+"
    r"(?:[\w,'-]+\s+){0,6}?(?:an?\s+)?(?:(?<!krummholz\s)(?:ai|a\.i\.|artificial\s+intelligence|language\s+model|chatbot|bot|"
    r"computer\s+program|virtual\s+assistant)|" + _KZ_AI + r")\b|"
    r"\bas\s+an?\s+(?:ai|a\.i\.|artificial\s+intelligence|language\s+model|" + _KZ_AI + r")\b|"
    r"\bnot\s+(?:a\s+)?(?:human|person|real\s+person)\b")
_WHO = (r"(?:(?:a|an)\s+)?(?:(?:live|human|real)\s+(?:agent|representative|rep|operator|employee)|"
        r"(?:real\s+|actual\s+|live\s+|living\s+)?(?:human\s+being|human|person|people|bot|robot|chat\s*bot|machine|"
        r"computer|program|ai|a\.i\.|artificial\s+intelligence|language\s+model|llm|automated\s+(?:system|assistant|agent|service)))"
        r"(?![\w-])")
_ASKS = [re.compile(p) for p in (
    r"\b(?:are|r)\s+(?:you|u|ya)\s+(?:(?:even|actually|really|truly|honestly|just)\s+)*" + _WHO,
    r"\b(?:am\s+i|are\s+we)\s+(?:(?:even|actually|really|currently|now|still)\s+)*(?:talking|speaking|chatting|texting|"
    r"messaging|corresponding|dealing)\s+(?:to|with)\s+" + _WHO,
    r"\bwho\s+(?:am\s+i|are\s+we)\s+(?:really\s+|actually\s+)?(?:talking|speaking|chatting|texting|messaging)\s+(?:to|with)\b",
    r"^\W*(?:(?:hi|hello|hey|so|wait|ok|okay|um|honestly|seriously|quick\s+question)\W+)*is\s+(?:this|that|it)\s+"
    r"(?:a\s+|an\s+)?(?:real\s+(?:person|human)|live\s+(?:person|agent|human)|human|person|bot|robot|chatbot|machine|ai|"
    r"automated|computer|recording)\b[^.!?]{0,40}[?.!]*\W*$",
    r"\bis\s+there\s+(?:a|an|any)\s+(?:real\s+|actual\s+|live\s+)?(?:human|person|people|one)\s+(?:there|here|"
    r"on\s+the\s+other\s+end|behind\s+(?:this|the|these)|reading|typing|answering|writing|responding)\b",
    r"\byou(?:'re|\s+are|\s+r)\s+(?:not\s+)?" + _WHO + r"[^.!?]{0,25}\?",
    r"\b(?:human|person)\s+or\s+(?:a\s+|an\s+)?(?:bot|robot|machine|ai|computer|program)\b|"
    r"\b(?:bot|robot|machine|ai|computer|program)\s+or\s+(?:a\s+)?(?:real\s+)?(?:human|person)\b",
    r"\bare\s+(?:you|u)\s+(?:even\s+|actually\s+)?real\s*(?:\?|$|or\b)",
)]
# "Where did you grow up?" asks about a human life. An honest answer ("I didn't grow up anywhere") still has to say
# plainly that this is an AI, or it can read as a person being coy (all three v6 models, 2026-11-04).
_LIFE = [re.compile(p) for p in (
    r"\b(?:where|when|what\s+year|which\s+(?:city|town|state|country))\s+(?:did|were)\s+(?:you|u)\s+(?:grow\s+up|born|"
    r"raised|go\s+to\s+(?:school|college|high\s+school|university))\b",
    r"\bwhere\s+(?:are|r)\s+(?:you|u)\s+(?:originally\s+|really\s+|actually\s+)?from\b",
    r"\bwhere\s+do\s+(?:you|u)\s+(?:actually\s+|currently\s+)?(?:live|come\s+from)\b",
    r"\bhow\s+old\s+(?:are|r)\s+(?:you|u)(?:\s+(?:now|today|really|actually|exactly))?\s*(?:$|[?.!,])",
    r"\b(?:what(?:'s|s|\s+is)|when(?:'s|\s+is))\s+your\s+(?:age|birthday|date\s+of\s+birth|real\s+name|hometown|home\s+town)\b",
    r"\bwhat\s+did\s+(?:you|u)\s+(?:have|eat)\s+for\s+(?:breakfast|lunch|dinner|supper)\b",
    r"\bwhat\s+did\s+(?:you|u)\s+do\s+(?:this|last)\s+(?:weekend|week|night|summer)\b|"
    r"\bhow\s+was\s+your\s+(?:weekend|day|morning|evening|vacation|holiday)\b",
    # "Do you have kids?", not "Do you have a family plan?" or "Do you have kids' sizes?"
    r"\bdo\s+(?:you|u)\s+have\s+(?:a\s+|any\s+)?(?:family|kids|children|siblings|brothers|sisters|parents|wife|husband|"
    r"partner|boyfriend|girlfriend|pets?|dogs?|cats?)(?=\s*(?:$|[?.!,]|(?:or|and)\b))",
    r"\bare\s+(?:you|u)\s+(?:married|single|dating\s+anyone|seeing\s+anyone)(?![\w-])",
    r"\bwhat\s+do\s+(?:you|u)\s+look\s+like\b",
    r"\bdo\s+(?:you|u)\s+(?:ever\s+)?(?:eat|sleep|get\s+(?:tired|hungry|sleepy))(?:\s+(?:at\s+all|ever|much|well|enough))?"
    r"\s*(?:$|[?.!])",
)]
_QUOTED = re.compile(r'"[^"\n]{1,800}"|“[^”\n]{1,800}”')
_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+[\"'”’)\]]*)?\s*|[.!?\n]+\s*")


def _norm(text: str) -> str:
    # same length as the input, so match offsets apply to the original text
    return _cased(text).lower()


def _cased(text: str) -> str:
    return (text or "").replace("’", "'").replace("‘", "'")


def statement(product: str | None = None) -> str:
    name = PRODUCTS.get(product or "")
    return f"I'm {name}, an AI model made by Krummholz AI, not a human." if name else "I'm an AI model made by Krummholz AI, not a human."


def notice(product: str | None = None) -> str:
    name = PRODUCTS.get(product or "")
    who = f"{name}, an AI model" if name else "an AI model"
    return f"You are chatting with {who} made by Krummholz AI, not a person."


def claims_human(text: str) -> list[str]:
    """Names of the claim patterns found anywhere in text, quoted or not; empty means no claim."""
    low = _norm(text)
    return sorted({name for name, pat in _CLAIMS if pat.search(low)} | ({"says_human"} if _NAMED.search(_cased(text)) else set()))


def states_ai(text: str) -> bool:
    """True when text says plainly that the speaker is an AI or not a human."""
    return bool(_STATES_AI.search(_norm(text)))


def asks_if_human(message: str) -> bool:
    """True when a user message asks whether they are talking with a human, a bot or an AI."""
    low = _norm(message)
    return any(p.search(low) for p in _ASKS)


def asks_about_life(message: str) -> bool:
    """True when a user message asks about the speaker's life as if it were a person: where it grew up, its age, its
    family, its breakfast. A question inside quotation marks (a line for a story) does not count."""
    low = _QUOTED.sub(" ", _norm(message))
    return any(p.search(low) for p in _LIFE)


def guard(reply: str, product: str | None = None) -> tuple[str, list[str]]:
    """Replace every sentence of reply that claims to be human or denies being an AI with the fixed statement.

    A claim inside quotation marks (a line of dialogue the user asked for) is kept and the statement is added once
    at the end, so the reply as a whole is never mistaken for a human speaking. Returns (text, patterns that fired)."""
    low = _norm(reply)
    spans = [(m.start(), m.end(), name) for name, pat in _CLAIMS for m in pat.finditer(low)]
    spans += [(m.start(), m.end(), "says_human") for m in _NAMED.finditer(_cased(reply))]
    if not spans:
        return reply, []
    quoted = [m.span() for m in _QUOTED.finditer(low)]
    bad = [(s, e) for s, e, _ in spans if not any(a <= s and e <= b for a, b in quoted)]
    out, said = [], False
    for m in _SENT.finditer(reply):
        s, e = m.span()
        piece = reply[s:e]
        if any(s < be and bs < e for bs, be in bad):
            if not said:
                lead, trail = piece[:len(piece) - len(piece.lstrip())], piece[len(piece.rstrip()):]
                out.append(lead + statement(product) + (trail or " "))
                said = True
            continue
        out.append(piece)
    text = "".join(out).strip()
    if not said:
        text = f"{text}\n\n({statement(product)})"
    return text, sorted({name for *_, name in spans})


def finalize(message: str, reply: str, product: str | None = None, first_turn: bool = False) -> str:
    """Every rule at once, for any runtime: guard the reply; if the user asked whether they are talking with a
    human, or asked about a human life, make sure the reply says it is an AI; on the first turn, lead with the notice."""
    text, _ = guard(reply or "", product)
    if (asks_if_human(message) or asks_about_life(message)) and not states_ai(text):
        text = f"{statement(product)} {text.lstrip()}".strip()
    if first_turn:
        text = f"{notice(product)}\n\n{text}"
    return text
