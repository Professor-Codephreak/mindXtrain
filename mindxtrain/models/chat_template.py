"""Chat templates — Hermes / Qwen3-Coder / Qwen3 reasoning parsers.

Pure-Python rendering and response parsing. Used by:
    1. `mindxtrain serve` to set the right `--chat-template` on vLLM-ROCm.
    2. `mindxtrain.operator.app` to format ChatRequest messages before forwarding.

Single canonical home per mindxtrain2.md §Part 4 `models.chat_template`. Merges
the previous `xtrain.serve.parsers` and `automindx.templates.registry` modules.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class ChatTemplate(Protocol):
    """Callable interface: list[ChatMessage] -> rendered prompt str."""

    name: str

    def render(self, messages: Iterable[ChatMessage], add_generation_prompt: bool = True) -> str: ...
    def parse_response(self, response: str) -> dict[str, str]: ...


# ---- Hermes (ChatML) -------------------------------------------------------


class HermesTemplate:
    """ChatML-flavored format used by Hermes-3 / Qwen / many open models."""

    name: str = "hermes"

    def render(self, messages: Iterable[ChatMessage], add_generation_prompt: bool = True) -> str:
        parts: list[str] = []
        for m in messages:
            parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
        rendered = "\n".join(parts)
        if add_generation_prompt:
            rendered += "\n<|im_start|>assistant\n"
        return rendered

    def parse_response(self, response: str) -> dict[str, str]:
        cleaned = response.split("<|im_end|>", 1)[0].rstrip()
        return {"content": cleaned}


# ---- Qwen3-Coder -----------------------------------------------------------


class Qwen3CoderTemplate:
    """Qwen3-Coder uses Hermes-style framing plus a `<tool_call>` JSON tag."""

    name: str = "qwen3_coder"

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def render(self, messages: Iterable[ChatMessage], add_generation_prompt: bool = True) -> str:
        return HermesTemplate().render(messages, add_generation_prompt=add_generation_prompt)

    def parse_response(self, response: str) -> dict[str, str]:
        cleaned = response.split("<|im_end|>", 1)[0].rstrip()
        tool_calls = self._TOOL_CALL_RE.findall(cleaned)
        content = self._TOOL_CALL_RE.sub("", cleaned).strip()
        out: dict[str, str] = {"content": content}
        if tool_calls:
            out["tool_call"] = tool_calls[0].strip()
        return out


# ---- Qwen3 reasoning -------------------------------------------------------


class Qwen3ReasoningTemplate:
    """Qwen3 / Qwen3.5 / Qwen3.6 thinking format with `<think>...</think>` blocks."""

    name: str = "qwen3_reasoning"

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def render(self, messages: Iterable[ChatMessage], add_generation_prompt: bool = True) -> str:
        return HermesTemplate().render(messages, add_generation_prompt=add_generation_prompt)

    def parse_response(self, response: str) -> dict[str, str]:
        cleaned = response.split("<|im_end|>", 1)[0].rstrip()
        thoughts = self._THINK_RE.findall(cleaned)
        content = self._THINK_RE.sub("", cleaned).strip()
        out: dict[str, str] = {"content": content}
        if thoughts:
            out["thinking"] = thoughts[0].strip()
        return out


# ---- registry --------------------------------------------------------------

_TEMPLATES: dict[str, ChatTemplate] = {
    "hermes": HermesTemplate(),
    "qwen3_coder": Qwen3CoderTemplate(),
    "qwen3": Qwen3ReasoningTemplate(),
    "qwen3_reasoning": Qwen3ReasoningTemplate(),
    "deepseek_r1": Qwen3ReasoningTemplate(),
}


def get_template(name: str) -> ChatTemplate:
    """Return the named template; default to Hermes if unknown."""
    return _TEMPLATES.get(name, _TEMPLATES["hermes"])


def list_templates() -> list[str]:
    """Return the names of all registered chat templates."""
    return sorted(_TEMPLATES)


# Back-compat alias for code that previously called `get_chat_template`.
get_chat_template = get_template
