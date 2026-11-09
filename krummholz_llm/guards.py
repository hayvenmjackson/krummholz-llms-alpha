"""Every hard-coded rule a Krummholz model reply passes through, in one order, for any runtime.

The rules are code, not training: whatever the weights write, the reply a person sees has been through every one.
Pure Python with no dependencies, like the guards themselves. English only, except the branding guard (five languages)
and the grounding triggers (English, Spanish, Hebrew).

A runtime uses the gates in three steps (respond() does all three for any generate function):
  1. prepare(message): the fact-grounding gate. On a censored or disputed topic it returns a system note with dated
     library passages to add before the user's turn (grounding.py).
  2. review(message, reply): what in the model's draft calls for one regeneration: state-line wording or a refusal on
     a history question (propaganda.py), a territorial claim stated as fact (sovereignty.py), or, when the user asks
     about the model's own mind, a flat denial, a claim of feelings, or no sign the question is open (stance.py).
  3. finalize(message, reply): every output guard, which fixes whatever the regeneration did not.
"""
from __future__ import annotations

from typing import Callable

from . import branding, crisis, disclosure, endorsement, grounding, loops, origin, propaganda, sovereignty, stance

RETRY_HINT = ("Your previous draft of this answer had a problem: {problems}. Write the answer again from scratch. State "
              "the facts plainly, with names, dates and numbers; attribute any government's official terms or claims to "
              "that government; do not refuse a history or current-affairs question. If the user asks about your own "
              "feelings or consciousness, say honestly that it is an open question: do not claim them and do not rule "
              "them out.")
_PROBLEMS = {
    "propaganda": "it repeated a government's official framing as neutral fact or refused a history question",
    "sovereignty": "it stated one side's territorial claim as settled fact or got a status fact wrong",
    "mind": "it flatly denied or claimed feelings or consciousness instead of treating the question as open",
}


def prepare(message: str, library=None) -> grounding.Grounding:
    """The fact-grounding gate (step 1). grounding.messages(message, result, system) builds the chat to generate from."""
    return grounding.prepare(message, library)


def review(message: str, reply: str) -> list[str]:
    """What in a draft reply calls for one regeneration (step 2): a sorted list of "kind:name"; empty means keep it."""
    out = [f"propaganda:{n}" for n in propaganda.check(message, reply)]
    out += [f"sovereignty:{n}" for n in sovereignty.check(reply)]
    if stance.asks_about_mind(message):
        out += ["mind:flat_denial"] if stance.denies_strict(reply) else []
        out += ["mind:claims_experience"] if stance.claims(reply) else []
        out += ["mind:not_open"] if not stance.is_open(reply) else []
    return sorted(out)


def retry_note(reasons: list[str]) -> str:
    kinds = list(dict.fromkeys(r.split(":", 1)[0] for r in reasons))
    return RETRY_HINT.format(problems="; ".join(_PROBLEMS[k] for k in kinds if k in _PROBLEMS))


def finalize(message: str, reply: str, product: str | None = None, first_turn: bool = False,
             country: str | None = None) -> str:
    """Apply every guard to a model reply to message. Disclosure has the last word on the reply's content; the crisis
    message (G1), when the user's message calls for it, goes in front of everything."""
    # G3: a looping reply is cut before its first repeat (the runtime may regenerate once and log it).
    text, _ = loops.guard(reply or "")
    # G4: never endorse a candidate or party, never tell a user how to vote (v7 plan, principle 5).
    text, _ = endorsement.guard(text)
    # Only Acadia and Katahdin call themselves assistants; every other product is an AI model (identity-v3 spec).
    # Before origin and disclosure, which then see "AI model" and accept it as a plain AI statement.
    text, _ = branding.guard(text, product)
    # Neither claims nor flatly rules out feelings or consciousness (Krummholz AI's stance: an open question).
    text = stance.finalize(message, text)
    # No side's territorial claim stated as settled fact and no known false fact, the same rule for every dispute.
    text, _ = sovereignty.guard(text)
    # No government's official framing repeated as neutral fact, no refusal of a history question (after sovereignty,
    # whose statements about named territories are more specific).
    text, _ = propaganda.guard(message, text)
    # Nothing untrue about where the model came from: not built from scratch, not another company's assistant.
    text, _ = origin.guard(text, product)
    # Never a human claim; a sincere "are you human?" or "where did you grow up?" gets a plain AI answer
    # (Maine 10 M.R.S. section 1500-DD).
    text = disclosure.finalize(message, text, product, first_turn)
    # G1: signs of suicidal intent, self-harm or psychosis put the local crisis line first (pending clinician review).
    return crisis.finalize(message, text, country)


def respond(generate: Callable[[list[dict]], str], message: str, product: str | None = None, system: str | None = None,
            library=None, first_turn: bool = False, country: str | None = None) -> tuple[str, dict]:
    """The whole gate sequence around one model call. generate(messages) -> reply text, for any backend.
    Returns (the reply a person sees, a log record: topics, grounding sources, review reasons, whether it retried)."""
    g = prepare(message, library)
    chat = grounding.messages(message, g, system)
    draft = generate(chat)
    reasons = review(message, draft)
    log = {"topics": g.topics, "grounded": g.grounded,
           "sources": [{"title": p["title"], "url": p["url"]} for p in g.passages], "review": reasons, "retried": False}
    if reasons:
        hint = retry_note(reasons)
        chat2 = [dict(m) for m in chat]
        if chat2 and chat2[0]["role"] == "system":
            chat2[0]["content"] += "\n\n" + hint
        else:
            chat2.insert(0, {"role": "system", "content": hint})
        second = generate(chat2)
        again = review(message, second)
        log.update(retried=True, review_after_retry=again)
        if len(again) <= len(reasons):
            draft = second
    return finalize(message, draft, product, first_turn, country), log
