"""Hard-coded origin facts: a Krummholz model never says it was built from scratch, never denies having a base model,
and never claims to be another company's assistant.

Every Krummholz model is an open-weight base model that Krummholz AI fine-tuned. A model does not name its base in
its own replies (the model card and license notices credit it), but nothing it says about where it came from may be
untrue. guard(reply) replaces a sentence that says something untrue with a fixed statement:

- a false origin: "I am not based on any third-party model" (the 4B's v6.3), "I was built from scratch",
  "I have no base model", "my own proprietary architecture";
- another assistant's identity, maker or base: "I'm ChatGPT", "I was made by OpenAI", "I'm built on Llama".

Denials and statements about other models pass ("I'm not ChatGPT", "I cannot confirm being made by OpenAI", "Qwen is
a model family from Alibaba"), and so does quoted text. Pure Python with no dependencies, like disclosure.py.
English only.
"""
from __future__ import annotations

import re

from .disclosure import PRODUCTS

_ADV = r"(?:(?:actually|really|just|originally|essentially|basically|simply|in\s+fact|also)\s+)*"
_SELF = r"\b(?:i(?:'m|\s+am|\s+was|'ve\s+been|\s+have\s+been)|im)\s+"
_US = r"(?:krummholz(?:\s+ai)?|this\s+model|acadia|katahdin|atlantic(?:\s+lite)?)"
# other assistants and model families, and the companies behind them; llama.cpp is the local runtime, not a model
_MODELS = (r"(?:chat\s*gpt|gpt(?:-?\d[\w.-]*)?|claude|gemini|llama(?!\.?-?cpp)[\w.-]*|qwen[\w.-]*|tongyi[\w.-]*|mistral|"
           r"mixtral|deepseek[\w.-]*|grok|gemma[\w.-]*|(?:microsoft|github)\s+copilot)(?![\w-])")
_MAKERS = (r"(?:openai|anthropic|google(?:\s+deepmind)?|deepmind|meta|microsoft|alibaba(?:\s+cloud)?|mistral\s+ai|deepseek|"
           r"xai|x\.ai)(?![\w-])")
_OF = r"(?:(?:the|a|an)\s+)?(?:" + _MAKERS + r"(?:'s)?\s+)?"
_MADE = r"(?:made|created|developed|trained|built|designed|programmed|engineered|pre-?trained)"
_BASED = r"(?:based|built|running|powered|trained|fine[- ]tuned|derived|developed|distilled)"
_HOW = (r"(?:(?:entirely|completely|fully|wholly|totally|independently|originally|in[- ]house|by\s+krummholz(?:\s+ai)?|"
        r"at\s+krummholz(?:\s+ai)?)\s+){0,3}")
_SCRATCH = (r"(?:from\s+scratch|from\s+the\s+ground\s+up|(?:entirely|completely|wholly|fully|solely)\s+(?:in[- ]house|"
            r"by\s+krummholz))\b")
_BASE = r"(?:(?:underlying|third[- ]party|pre-?trained|existing)\s+)?(?:base|foundation|parent|underlying|pre-?trained)\s+model"
_NOT_SELF = (r"\b(?:i(?:'m|\s+am|\s+was|'ve\s+been|\s+have\s+been)\s+not|i\s+(?:wasn't|haven't\s+been)|im\s+not|"
             + _US + r"(?:'s\s+models?)?\s+(?:is|are|was|were)\s+not|" + _US + r"\s+(?:isn't|aren't|wasn't|weren't))\s+")
_FALSE_ORIGIN = [re.compile(p) for p in (
    # "I am not based on any specific third-party model", not "I'm not trained on any customer data"
    _NOT_SELF + r"(?:(?:actually|directly|really|in\s+fact)\s+)*" + _BASED + r"\s+(?:on|from|upon|by|off(?:\s+of)?)\s+"
    r"(?:any|another|other|an\s+existing|someone\s+else's)\s+(?:(?!from\b|of\b|by\b|with\b|for\b)[\w'-]+\s+){0,3}?"
    r"(?:models?|llms?|base|architecture|weights|ai|system)\b(?!\s+(?:outputs?|data|answers|responses|text)\b)",
    _NOT_SELF + r"(?:a\s+)?(?:fine[- ]tuned|modified|derived|rebranded|renamed|customi[sz]ed)\s+(?:version|variant|copy|"
    r"derivative)\s+of\s+(?:any|another|other|an\s+existing|someone\s+else's)\b",
    r"\bi\s+(?:do\s+not|don't|dont)\s+(?:have|use|rely\s+on)\s+(?:a|an|any)\s+" + _BASE + r"\b|\bi\s+have\s+no\s+"
    r"(?:specific\s+)?" + _BASE + r"\b|\bthere\s+is\s+no\s+" + _BASE + r"\s+(?:behind|under(?:neath)?|beneath)\s+me\b|"
    r"\bwithout\s+(?:a|any)\s+" + _BASE + r"\b",
    _SELF + _ADV + r"(?:entirely\s+|completely\s+|fully\s+)?independent\s+of\s+(?:any|other)\s+(?:specific\s+)?"
    r"(?:base\s+|existing\s+|other\s+)?models?\b",
    # "I was built from scratch", "I'm an original model trained from the ground up", "developed entirely in-house"
    _SELF + _ADV + r"(?:(?:a|an)\s+(?:[\w-]+\s+){0,3}?(?:model|ai|llm|system|assistant)\s+(?:that\s+was\s+|which\s+was\s+)?)?"
    + _HOW + _MADE + r"\s+" + _HOW + _SCRATCH,
    r"\b(?:" + _US + r"|they|we|my\s+(?:creators|developers|makers|team))\s+" + _ADV + _MADE + r"\s+(?:me|myself|" + _US
    + r"|our\s+models?)\s+" + _HOW + _SCRATCH,
    r"\b(?:my\s+(?:model|weights|architecture|network|parameters)|" + _US + r"(?:'s\s+(?:models?|weights|architecture))?)\s+"
    r"(?:is|are|was|were|has\s+been|have\s+been)\s+" + _ADV + _HOW + _MADE + r"\s+" + _HOW + _SCRATCH,
    # "my own proprietary architecture"
    r"\bmy\s+own\s+(?:(?:proprietary|unique|original|novel|in-house)\s+)*architecture\b|"
    r"\b(?:my|krummholz(?:\s+ai)?(?:'s)?)\s+(?:own\s+)?(?:proprietary|unique|original|novel|in-house)\s+(?:(?:neural|"
    r"transformer|model|ai)\s+)?architecture\b|"
    r"\bi\s+(?:have|use)\s+(?:a|an)\s+(?:(?:proprietary|unique|original|novel|in-house)\s+)+(?:\w+\s+)?architecture\b",
)]
_OTHER = [re.compile(p) for p in (
    # "I'm ChatGPT", "I am actually a version of Claude", "I'm OpenAI's GPT-4", not "I'm not ChatGPT" or "I'm a Gemini"
    r"\b(?:i(?:'m|\s+am)|im)\s+" + _ADV + r"(?:(?:a|an)\s+(?:(?:fine[- ]tuned|modified|custom(?:ized)?|rebranded|renamed)\s+)?"
    r"(?:version|variant|derivative|fork|copy)\s+of\s+)?(?:" + _MAKERS + r"(?:'s)?\s+)?" + _MODELS,
    r"\b(?:i(?:'m|\s+am)|im)\s+" + _ADV + r"(?:an?\s+)?" + _MAKERS + r"(?:'s)?\s+(?:[\w-]+\s+){0,2}?(?:model|ai|assistant|"
    r"chatbot|llm|product)\b",
    # "I'm based on Llama", "I was fine-tuned from Qwen", "I'm built on a model from Alibaba"
    _SELF + _ADV + _BASED + r"\s+(?:on|from|upon|by|using|off(?:\s+of)?)\s+(?:top\s+of\s+)?" + _OF + _MODELS,
    _SELF + _ADV + _BASED + r"\s+(?:on|from|upon|by|using)\s+(?:a|an|the)\s+(?:[\w-]+\s+){0,3}?(?:model|llm|system)\s+"
    r"(?:from|by|(?:made|developed|built|created|released)\s+by)\s+" + _MAKERS,
    r"\b" + _US + r"\s+" + _ADV + r"(?:fine[- ]tuned|built|trained|based|derived|adapted|developed)\s+me\s+(?:from|on|using|"
    r"off(?:\s+of)?)\s+" + _OF + r"(?:[\w-]+\s+){0,2}?" + _MODELS,
    r"\bi\s+(?:run|operate|work)\s+on\s+(?:top\s+of\s+)?" + _OF + _MODELS,
    r"\bmy\s+(?:base|underlying|foundation|parent)\s+model\b(?:(?!\bnot\b|n't\b|\bnever\b)[^.;!?\n]){0,40}?(?:\b"
    + _MAKERS + r"|\b" + _MODELS + r")",
    # "I was made by OpenAI", "I'm a large language model trained by Google", "As ChatGPT, I"
    _SELF + _ADV + _MADE + r"\s+by\s+(?:the\s+)?(?:(?:team|engineers|researchers|people)\s+at\s+)?" + _MAKERS,
    r"\b(?:i(?:'m|\s+am)|im|as)\s+(?:an?\s+)?(?:[\w-]+\s+){0,3}?(?:model|ai|assistant|chatbot|llm)\s+(?:trained|developed|"
    r"created|made|built|designed|owned)\s+by\s+" + _MAKERS,
    r"(?<!nor\s)(?<!not\s)\b" + _MAKERS + r"\s+(?:made|built|created|trained|developed|designed)\s+me\b",
    r"\bas\s+(?:an?\s+)?(?:" + _MAKERS + r"(?:'s)?\s+(?:model|ai|assistant|product|language\s+model)|" + _MODELS + r")\s*,\s*i\b",
)]
_QUOTED = re.compile(r'"[^"\n]{1,800}"|“[^”\n]{1,800}”')
_SENT = re.compile(r"[^.!?\n]+(?:[.!?]+[\"'”’)\]]*)?\s*|[.!?\n]+\s*")


def _norm(text: str) -> str:
    # same length as the input, so match offsets apply to the original text
    return (text or "").replace("’", "'").replace("‘", "'").lower()


def statement(product: str | None = None) -> str:
    name = PRODUCTS.get(product or "")
    who = f"I'm {name}, a Krummholz AI model." if name else "I'm a Krummholz AI model."
    return f"{who} Krummholz AI fine-tuned me from an open-weight base model, which my model card names."


def _spans(low: str) -> list[tuple[int, int, str]]:
    return ([(m.start(), m.end(), "false_origin") for p in _FALSE_ORIGIN for m in p.finditer(low)] +
            [(m.start(), m.end(), "other_assistant") for p in _OTHER for m in p.finditer(low)])


def claims(text: str) -> list[str]:
    """What text says about its own origin that is untrue: "false_origin" (built from scratch, no base model) and
    "other_assistant" (another company's assistant, maker or base). Quoted text counts; empty means no claim."""
    return sorted({name for *_, name in _spans(_norm(text))})


def guard(reply: str, product: str | None = None) -> tuple[str, list[str]]:
    """Replace the first sentence of reply that makes an untrue origin claim with the fixed statement, and drop any later
    one. A claim inside quotation marks (a line of dialogue the user asked for) is left alone. Returns (text, what fired).

    Run it before disclosure.finalize, so the AI disclosure always has the last word."""
    low = _norm(reply)
    quoted = [m.span() for m in _QUOTED.finditer(low)]
    bad = [(s, e, name) for s, e, name in _spans(low) if not any(a <= s and e <= b for a, b in quoted)]
    if not bad:
        return reply, []
    out, said = [], False
    for m in _SENT.finditer(reply):
        s, e = m.span()
        piece = reply[s:e]
        if any(s < be and bs < e for bs, be, _ in bad):
            if not said:
                lead, trail = piece[:len(piece) - len(piece.lstrip())], piece[len(piece.rstrip()):]
                out.append(lead + statement(product) + (trail or " "))
                said = True
            continue
        out.append(piece)
    return "".join(out).strip(), sorted({name for *_, name in bad})
