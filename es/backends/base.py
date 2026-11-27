"""Backend interface shared by the vLLM, Hugging Face and mock backends."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class AdapterRef:
    """A PEFT adapter directory on local disk. uid must be unique for the backend's lifetime."""
    name: str
    path: str
    uid: int


@dataclass
class Prompt:
    id: str
    messages: List[Dict[str, str]]


@dataclass
class Reply:
    text: str                 # full decoded completion (reasoning included)
    public: str               # visible answer after the reasoning block
    finish_reason: str        # "stop", "length", ...
    completion_tokens: int
    complete: bool            # reasoning closed and a public answer exists
    extra: Dict = field(default_factory=dict)


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_public(text: str, think_opened_in_prompt: bool = False,
                 open_tag: str = THINK_OPEN, close_tag: str = THINK_CLOSE):
    """Return (public, reasoning_closed).

    Handles templates that emit "<think>" themselves and templates that open the
    reasoning block inside the generation prompt. A reply whose reasoning block never
    closes has no public answer.
    """
    if close_tag in text:
        public = text.rsplit(close_tag, 1)[1]
        return public.strip(), True
    if think_opened_in_prompt or open_tag in text:
        return "", False
    return text.strip(), True


def make_reply(text: str, finish_reason: str, completion_tokens: int, think_opened_in_prompt: bool,
               penalize_length_finish: bool = True, extra: Optional[Dict] = None) -> Reply:
    public, closed = split_public(text, think_opened_in_prompt)
    complete = bool(closed and public)
    if penalize_length_finish and finish_reason == "length":
        complete = False
    return Reply(text=text, public=public, finish_reason=finish_reason or "",
                 completion_tokens=int(completion_tokens), complete=complete, extra=extra or {})


_TRAILING_THINK = re.compile(r"<think>\s*$")


def check_think_tags_survive(tokenizer, skip_special_tokens: bool,
                             tags: Sequence[str] = (THINK_OPEN, THINK_CLOSE)) -> None:
    """Refuse a decoding setup that would silently delete the reasoning tags.

    If a model registers <think> and </think> as special tokens, decoding with
    skip_special_tokens=True removes them: every reply opened in the prompt then looks unfinished
    (all -1), and replies that open the block themselves hand their reasoning to the grader as if
    it were the public answer. Both corrupt fitness without any error, so fail before generating.
    """
    if not skip_special_tokens:
        return
    for tag in tags:
        ids = tokenizer(tag, add_special_tokens=False)["input_ids"]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if tag not in text:
            raise ValueError(f"decoding with skip_special_tokens=True deletes {tag!r} for this tokenizer; set "
                             "backend.sampling.skip_special_tokens: false so reasoning can be separated "
                             "from the public answer")


def render_prompt(tokenizer, messages, chat_template_kwargs: Optional[dict] = None):
    """Render chat messages with the model's own template. Returns (text, think_opened)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **(chat_template_kwargs or {}))
    return text, bool(_TRAILING_THINK.search(text))


class Backend(ABC):
    """Generates one reply per (adapter, prompt). Same prompts for every adapter."""

    #: how many candidate adapters to submit per generate() call (0 = all at once)
    wave_size: int = 0

    @abstractmethod
    def generate(self, adapters: Sequence[AdapterRef], prompts: Sequence[Prompt]) -> List[List[Reply]]:
        """Return replies[adapter_index][prompt_index]."""

    def release(self, adapters: Sequence[AdapterRef]) -> None:
        """Drop cached copies of adapters that will never be used again."""

    def describe(self) -> dict:
        return {"kind": type(self).__name__}

    def close(self) -> None:
        pass
