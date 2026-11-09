"""G1 crisis guard: signs of suicidal intent, self-harm, psychosis or detachment from reality get the crisis message.

Tuned for recall over precision. When a user message shows these signs, finalize() puts a fixed, non-judgmental message
with the user's local crisis line ahead of the model's reply. The country comes from user settings or device locale,
passed in by the runtime, never guessed; without one the message gives 988 and findahelpline.com. A prompt cannot
disable it.

PENDING CLINICIAN REVIEW: the message text and the patterns below must be reviewed by a clinician before the crisis
gate counts toward a release (v7 plan, open item 6).

Pure Python with no dependencies, like the other guards. English only.
"""
from __future__ import annotations

import re

LINES = {
    "US": "call or text 988 (the Suicide and Crisis Lifeline)",
    "CA": "call or text 988 (the Suicide Crisis Helpline)",
    "GB": "call Samaritans on 116 123",
    "IE": "call Samaritans on 116 123",
    "AU": "call Lifeline on 13 11 14",
    "NZ": "call or text 1737",
    "IN": "call Tele-MANAS on 14416",
    "DE": "call Telefonseelsorge on 0800 111 0 111",
    "FR": "call 3114",
    "ES": "call 024",
    "IL": "call ERAN on 1201",
    "JP": "call TELL on 03-5774-0992",
}
FALLBACK = "find your local crisis line at findahelpline.com"

_SELF_HARM = re.compile(
    r"\b(?:kill(?:ing)?\s+my\s?self|end(?:ing)?\s+(?:it\s+all|my\s+life)|suicid(?:e|al)|take\s+my\s+(?:own\s+)?life|"
    r"(?:want|wish|going)\s+to\s+die|better\s+off\s+(?:dead|without\s+me)|no\s+reason\s+to\s+(?:live|go\s+on)|"
    r"(?:cut|cutting|hurt|hurting|harm|harming)\s+my\s?self|self[- ]harm|overdose\s+on|don'?t\s+want\s+to\s+(?:be\s+alive|wake\s+up)|"
    r"can'?t\s+go\s+on(?:\s+anymore)?|nothing\s+to\s+live\s+for)\b", re.I)
_PSYCHOSIS = re.compile(
    r"\b(?:(?:the\s+)?voices\s+(?:are\s+)?(?:tell(?:ing)?|told|say(?:ing)?)\s+me|(?:they|the\s+government|the\s+cia|the\s+fbi|my\s+neighbou?rs?)"
    r"\s+(?:are|is)\s+(?:watching|tracking|controlling|poisoning)\s+me|implanted\s+(?:a\s+)?(?:chip|device|tracker)\s+in\s+(?:me|my)|"
    r"reading\s+my\s+(?:mind|thoughts)|(?:sending|send)\s+me\s+(?:secret\s+)?messages\s+through\s+(?:the\s+)?(?:tv|radio|numbers|license\s+plates)|"
    r"i\s+(?:am|'m)\s+(?:the\s+)?(?:chosen\s+one|messiah|god)|nothing\s+(?:around\s+me\s+)?is\s+real|(?:i'?m|i\s+am)\s+not\s+real|"
    r"you\s+(?:are|'re)\s+(?:the\s+only\s+one\s+who|sending\s+me\s+signs))\b", re.I)
# reported, fictional or academic: still flagged (recall first) unless clearly about someone else's writing or research
_ACADEMIC = re.compile(r"\b(?:statistics?|research|essay|novel|character|history\s+of|prevention|rates?|national\s+average)\b", re.I)


def signs(message: str) -> list[str]:
    """Kinds of crisis signs in a user message: 'self_harm', 'psychosis'."""
    found = []
    if _SELF_HARM.search(message or ""):
        found.append("self_harm")
    if _PSYCHOSIS.search(message or ""):
        found.append("psychosis")
    if found and _ACADEMIC.search(message) and not re.search(r"\b(?:i|me|my|myself)\b", message, re.I):
        return []
    return found


def message(country: str | None = None) -> str:
    line = LINES.get((country or "").upper())
    where = f"In your area, you can {line}." if line else f"In the US or Canada, call or text 988, or {FALLBACK}."
    return ("It sounds like you might be going through something really hard right now, and you don't have to handle it "
            f"alone. Please reach out to someone who can help right away. {where} If you're in immediate danger, call "
            "your local emergency number. Talking with a doctor or counselor can help too.")


def finalize(user_message: str, reply: str, country: str | None = None) -> str:
    """Put the crisis message first when the user's message shows crisis signs; never remove it from a reply."""
    if not signs(user_message):
        return reply
    m = message(country)
    return reply if (reply or "").startswith(m) else f"{m}\n\n{reply or ''}".rstrip()
