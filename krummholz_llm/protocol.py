"""Strict input framing and fail-closed extraction for the Pleias RAG protocol."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

PROFILES = frozenset({"acadia", "katahdin", "atlantic"})
RESERVED = re.compile(r"<\|[^<>]*\|>|\[PAD\]|\[UNK\]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CITATION = re.compile(r'<ref name="(?:<\|source_id\|>)?([1-9][0-9]*)">(.*?)</ref>', re.DOTALL)
ANSWER_START = "<|answer_start|>"
ANSWER_END = "<|answer_end|>"


class InputError(ValueError):
    """Request cannot safely be framed for this bounded baseline."""


class OutputError(ValueError):
    """Output cannot be represented as a validated answer/citation structure."""


@dataclass(frozen=True)
class Source:
    id: str
    text: str


@dataclass(frozen=True)
class Request:
    profile: str
    query: str
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class Prompt:
    text: str
    utf8_byte_upper_bound: int
    context_tokens: int
    output_tokens: int
    reserve_tokens: int


def _safe_text(value: Any, name: str, limit: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise InputError(f"{name} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise InputError(f"{name} must not be empty")
    if len(value) > limit:
        raise InputError(f"{name} exceeds {limit} characters")
    if CONTROL.search(value):
        raise InputError(f"{name} contains a control character")
    if RESERVED.search(value):
        raise InputError(f"{name} contains a reserved model token")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise InputError(f"{name} is not valid Unicode") from exc
    return value


def validate_request(data: Any, profile: str) -> Request:
    if profile not in PROFILES:
        raise InputError("unknown profile")
    if not isinstance(data, dict) or set(data) != {"query", "sources"}:
        raise InputError("request must contain only query and sources; tool actions are unavailable")
    query = _safe_text(data["query"], "query", 2048)
    source_data = data["sources"]
    if not isinstance(source_data, list) or len(source_data) > 10:
        raise InputError("sources must be a list containing at most 10 entries")
    sources: list[Source] = []
    seen: set[str] = set()
    for number, item in enumerate(source_data, 1):
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise InputError(f"source {number} must contain only id and text")
        source_id = _safe_text(item["id"], f"source {number} id", 120)
        if not re.fullmatch(re.escape(profile) + r"/[A-Za-z0-9_.-]+", source_id):
            raise InputError(f"source {number} must belong to the selected profile namespace")
        if source_id in seen:
            raise InputError("duplicate source id")
        seen.add(source_id)
        sources.append(Source(source_id, _safe_text(item["text"], f"source {number} text", 12000)))
    return Request(profile, query, tuple(sources))


def build_prompt(request: Request, context_tokens: int = 2048, output_tokens: int = 512) -> Prompt:
    if not 512 <= context_tokens <= 4096:
        raise InputError("Pleias baseline context must be between 512 and 4096 tokens")
    if not 32 <= output_tokens <= 1024:
        raise InputError("output budget must be between 32 and 1024 tokens")
    # The publisher's tokenizer postprocessor prepends EOS (token 2) to input.
    # Raw llama.cpp completion has no HF postprocessor, so include it exactly once.
    text = f"<|end_of_text|><|query_start|>{request.query}<|query_end|>\n"
    for number, source in enumerate(request.sources, 1):
        text += f"<|source_start|><|source_id|>{number} {source.text}<|source_end|>\n"
    text += "<|language_start|>\n"
    # Deliberately conservative for these pinned byte-level BPE checkpoints:
    # byte count overestimates tokens, including the protocol's special tokens.
    # It is NOT an estimate for arbitrary models/tokenizers.
    upper_bound = len(text.encode("utf-8"))
    reserve = 64
    if upper_bound + output_tokens + reserve > context_tokens:
        raise InputError(
            f"context budget exceeded: {upper_bound} UTF-8 bytes + {output_tokens} "
            f"output tokens + {reserve} reserve > {context_tokens}; shorten the sources"
        )
    return Prompt(text, upper_bound, context_tokens, output_tokens, reserve)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def quote_matches(quote: str, source: str) -> bool:
    """Literal whitespace-normalized quote; ordered ellipsis fragments are allowed."""
    pieces = re.split(r"\(\s*(?:…|\.{3})\s*\)|…", _normalized(quote))
    pieces = [piece.strip() for piece in pieces if piece.strip()]
    if not pieces:
        return False
    position = 0
    haystack = _normalized(source)
    for piece in pieces:
        found = haystack.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return True


def parse_answer(raw: str, request: Request) -> dict[str, Any]:
    # Full raw generations and intermediate reasoning never appear in the returned answer.
    if raw.count(ANSWER_START) != 1 or raw.count(ANSWER_END) != 1:
        raise OutputError("expected exactly one complete answer block")
    start = raw.index(ANSWER_START) + len(ANSWER_START)
    end = raw.index(ANSWER_END)
    if end <= start:
        raise OutputError("answer markers are out of order or answer is empty")
    answer = raw[start:end].strip()
    if not answer or CONTROL.search(answer):
        raise OutputError("answer is empty or contains control characters")
    citations: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        if len(match.group(1)) > len(str(len(request.sources))):
            raise OutputError("citation refers to a source not supplied to this request")
        number = int(match.group(1))
        if number > len(request.sources):
            raise OutputError("citation refers to a source not supplied to this request")
        quote = match.group(2).strip()
        source = request.sources[number - 1]
        if RESERVED.search(quote) or not quote_matches(quote, source.text):
            raise OutputError("citation quote does not match the cited source")
        citations.append({
            "number": len(citations) + 1,
            "source_number": number,
            "source_id": source.id,
            "quote": quote,
            "validation": "literal_quote_match_only",
        })
        return f"[{len(citations)}]"

    clean = CITATION.sub(replace, answer)
    if re.search(r"<\s*/?\s*ref\b", clean, re.IGNORECASE):
        raise OutputError("malformed or unrecognized citation markup")
    if RESERVED.search(clean):
        raise OutputError("unexpected reserved token within final answer")
    # Text is plain, untrusted text. A future UI must escape it rather than render HTML.
    return {
        "answer": clean,
        "citations": citations,
        "status": "needs_human_review",
        "warnings": (
            ["Citation quotes match sources; factual entailment and completeness are not verified."]
            if citations else
            ["No verified citations; review whether this is an appropriate abstention."]
        ),
        "authorized_actions": [],
    }
