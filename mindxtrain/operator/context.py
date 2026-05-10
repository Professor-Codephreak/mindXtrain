"""ContextManager — token-budgeted compaction window (ml-intern pattern).

Tracks per-message token estimates, summarizes oldest turns when the window
exceeds `target_tokens`, preserves the system prompt and the most recent
user/tool turns verbatim. Pure stdlib; optional summarizer callback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContextManagerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens: int = Field(default=170_000, ge=1024)
    keep_recent: int = Field(
        default=8,
        ge=1,
        description="messages from the end to keep verbatim",
    )
    chars_per_token: float = Field(default=4.0, gt=0.0)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Cheap whitespace/char-rate token estimate (no tokenizer dep)."""
    if not text:
        return 0
    # 4 chars/token is the OpenAI / Anthropic rule of thumb for English text.
    return max(1, int(len(text) / chars_per_token))


def message_tokens(message: dict[str, Any], chars_per_token: float = 4.0) -> int:
    """Estimate tokens for one chat message."""
    text_parts: list[str] = [str(message.get("role", "")), str(message.get("content", "") or "")]
    for c in message.get("tool_calls") or []:
        fn = c.get("function", {})
        text_parts.append(str(fn.get("name", "")))
        text_parts.append(str(fn.get("arguments", "")))
    return estimate_tokens(" ".join(text_parts), chars_per_token)


SummarizeFn = Callable[[list[dict[str, Any]]], str]


class ContextManager:
    """Trim message lists to fit `target_tokens`."""

    def __init__(
        self,
        cfg: ContextManagerConfig | None = None,
        summarize: SummarizeFn | None = None,
    ) -> None:
        self.cfg = cfg or ContextManagerConfig()
        self.summarize = summarize

    def estimate(self, messages: list[dict[str, Any]]) -> int:
        return sum(message_tokens(m, self.cfg.chars_per_token) for m in messages)

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a trimmed message list whose token estimate is <= target_tokens.

        Strategy:
            1. Always keep the leading system message (if present) verbatim.
            2. Always keep the trailing `keep_recent` messages verbatim.
            3. The middle band is replaced with a single summary turn (via
               `summarize` callback) or simply elided with a meta note.
        """
        if not messages:
            return []

        if self.estimate(messages) <= self.cfg.target_tokens:
            return list(messages)

        head: list[dict[str, Any]] = []
        if messages and messages[0].get("role") == "system":
            head = [messages[0]]
            body = messages[1:]
        else:
            body = list(messages)

        if len(body) <= self.cfg.keep_recent:
            return head + body

        recent = body[-self.cfg.keep_recent :]
        elided = body[: -self.cfg.keep_recent]

        if self.summarize is not None:
            summary_text = self.summarize(elided)
        else:
            summary_text = (
                f"[context compacted: {len(elided)} earlier messages elided"
                f" to fit {self.cfg.target_tokens}-token window]"
            )

        summary_msg = {"role": "system", "content": summary_text}
        compacted = [*head, summary_msg, *recent]

        # If still over, recurse with a tighter `keep_recent` until it fits or is minimal.
        if self.estimate(compacted) > self.cfg.target_tokens and self.cfg.keep_recent > 1:
            tighter_cfg = self.cfg.model_copy(update={"keep_recent": max(1, self.cfg.keep_recent // 2)})
            return ContextManager(tighter_cfg, self.summarize).compact(messages)

        return compacted


__all__ = ["ContextManager", "ContextManagerConfig", "estimate_tokens", "message_tokens"]
