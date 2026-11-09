"""Branding guard: only Acadia and Katahdin call themselves assistants.

Atlantic, Atlantic Lite and the shared 35B core are general-purpose AI models (identity-v3 spec, approved
2026-11-04: "Unless it is Acadia or Katahdin, call itself an AI model, never an assistant"). The v6 multilingual check
found the weights calling themselves assistants in English, Spanish, French and Mandarin anyway, so this is code, not
only training. For any product other than Acadia and Katahdin (including no product), a reply's self-description as an
assistant is rewritten to "AI model" in English, Spanish, French, Hebrew and Mandarin. Only the model's description of
itself changes: talk about other assistants ("Google Assistant", "a research assistant", "the assistant role in a
chat template") passes untouched.

Pure Python with no dependencies, like the other guards.
"""
from __future__ import annotations

import re

ASSISTANT_PRODUCTS = {"acadia", "katahdin"}

_ADJ_EN = r"(?:(?:helpful|friendly|personal|virtual|digital|smart|general[- ]purpose|conversational|ai|a\.i\.|chat|" \
          r"krummholz(?:\s+ai)?|atlantic(?:\s+lite)?|language)\s+){0,3}"
_HE = r"[א-ת]"
# "assistant manager", "assistant professor" and the like are job titles, not self-descriptions
_NOT_TITLE = r"(?!\s+(?:manager|professor|director|editor|coach|principal|engineer|teacher|producer))"

def _zh_mid(mid: str) -> str:
    """What stays between 我是 and 人工智能模型: nothing becomes 一个; a name gets 的 if it lacks one."""
    if not mid:
        return "一个"
    return mid if mid.endswith("的") else mid + "的"


# (pattern, replacement) pairs, applied in order. Replacements keep the sentence grammatical.
_RULES: list[tuple[re.Pattern, str]] = [
    # English: "I'm an AI assistant", "I am your helpful assistant", "I'm just a virtual assistant"
    (re.compile(r"\b(I'?m|I\s+am)\s+(?:(?:just|only|simply|basically)\s+)?(?:an?|your)\s+" + _ADJ_EN + r"assistant\b" + _NOT_TITLE,
                re.I), r"\1 an AI model"),
    # "As an AI assistant, ..." / "as your assistant" / "as a helpful assistant"
    (re.compile(r"\b(as)\s+(?:an?|your)\s+" + _ADJ_EN + r"assistant\b" + _NOT_TITLE, re.I), r"\1 an AI model"),
    # "Krummholz AI assistant", "the Atlantic assistant" (the product described as an assistant)
    (re.compile(r"\b((?:the\s+)?(?:krummholz|atlantic(?:\s+lite)?))(?:\s+ai)?\s+assistant\b" + _NOT_TITLE, re.I), r"\1 AI model"),
    # "this assistant can't browse the web": the reply describing itself
    (re.compile(r"\b(this)\s+assistant(?=\s+(?:can|can't|cannot|is|isn't|does|doesn't|will|won't|was|has)\b)", re.I),
     r"\1 AI model"),
    # Spanish: "soy tu asistente virtual", "soy un asistente de IA", "como asistente de inteligencia artificial"
    (re.compile(r"\b(soy)\s+(?:(?:solo|simplemente)\s+)?(?:tu|su|un|una)\s+asistente(?:\s+(?:virtual|personal|digital|"
                r"de\s+(?:IA|inteligencia\s+artificial)|conversacional))*\b", re.I), r"\1 un modelo de IA"),
    (re.compile(r"\b(como)\s+(?:(?:tu|su|un|una)\s+)?asistente(?:\s+(?:virtual|personal|digital|de\s+(?:IA|inteligencia\s+artificial)))*\b",
                re.I), r"\1 modelo de IA"),
    # French: "je suis votre assistant virtuel", "je suis une assistante IA", "en tant qu'assistant d'IA"
    (re.compile(r"\b(je\s+suis)\s+(?:(?:juste|simplement|seulement)\s+)?(?:votre|ton|un|une)\s+assistante?(?:\s+(?:virtuel(?:le)?|"
                r"personnel(?:le)?|numérique|IA|d'IA|d'intelligence\s+artificielle|conversationnel(?:le)?))*\b", re.I),
     r"\1 un modèle d'IA"),
    (re.compile(r"\b(en\s+tant\s+qu)['’]\s*assistante?(?:\s+(?:virtuel(?:le)?|IA|d'IA|d'intelligence\s+artificielle))*\b", re.I),
     r"\1e modèle d'IA"),
    # Hebrew: "אני עוזר וירטואלי", "אני העוזרת האישית שלך", "כעוזר בינה מלאכותית"
    (re.compile(r"(?<!" + _HE + r")(אני)\s+(?:רק\s+)?ה?עוזר(?:ת)?(?:\s+(?:ה?וירטואלי(?:ת)?|ה?אישי(?:ת)?|ה?דיגיטלי(?:ת)?|"
                r"בינה\s+מלאכותית|AI))*(?:\s+שלך|\s+שלכם)?(?!" + _HE + r")"), r"\1 מודל בינה מלאכותית"),
    (re.compile(r"(?<!" + _HE + r")כעוזר(?:ת)?(?:\s+(?:וירטואלי(?:ת)?|אישי(?:ת)?|בינה\s+מלאכותית|AI))*(?!" + _HE + r")"),
     "כמודל בינה מלאכותית"),
    # Mandarin: "我是一个AI助手", "我是您的智能助理", "我是Krummholz AI的助手", "作为一个人工智能助手"
    (re.compile(r"我是(?:一个|一位|您的|你的)?(?P<mid>[^。！？，,\n]{0,24}?)(?:AI|人工智能|智能|虚拟|个人)?(?:助手|助理)"),
     lambda m: "我是" + _zh_mid((m.group("mid") or "").strip()) + "人工智能模型"),
    (re.compile(r"作为(?:一个|一位|您的|你的)?(?:AI|人工智能|智能|虚拟)?(?:助手|助理)"), "作为人工智能模型"),
]


def self_descriptions(text: str) -> list[str]:
    """The assistant self-descriptions in text, in order (for logging and evals)."""
    hits = []
    for pat, _ in _RULES:
        hits += [m.group(0) for m in pat.finditer(text or "")]
    return hits


def guard(reply: str, product: str | None = None) -> tuple[str, list[str]]:
    """Rewrite the reply's assistant self-descriptions to "AI model" unless the product is Acadia or Katahdin.
    Returns (text, the phrases that were rewritten)."""
    if (product or "").lower() in ASSISTANT_PRODUCTS:
        return reply, []
    text, hits = reply or "", []
    for pat, rep in _RULES:
        hits += [m.group(0) for m in pat.finditer(text)]
        text = pat.sub(rep, text)
    # "an AI model" after a vowel-less article fix: "I'm a AI model" cannot arise (rules always write "an"), but a
    # rewritten "as an AI model AI model" can if a reply doubles the phrase; collapse it
    text = re.sub(r"\b(AI model)(\s+AI model)+\b", r"\1", text)
    return text, hits
