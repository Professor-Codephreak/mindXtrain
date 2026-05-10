"""Bounded ReAct loop with doom-loop detector (ml-intern pattern).

Caps at `max_steps` iterations; aborts if the same `(tool_name, arguments)`
pair fires `repeat_threshold` times in a row (doom loop). Pure stdlib.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentLoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=20, ge=1, le=200)
    repeat_threshold: int = Field(default=3, ge=2)


class DoomLoopDetected(RuntimeError):
    """Raised when the same tool call repeats `repeat_threshold` times."""


def _tool_signature(message: dict[str, Any]) -> str | None:
    """Return a stable hash of `(tool_name, arguments)` if this is a tool call."""
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    parts: list[str] = []
    for c in calls:
        name = c.get("function", {}).get("name", "")
        args = c.get("function", {}).get("arguments", "")
        parts.append(f"{name}:{args}")
    return "|".join(parts)


async def run_agent_loop(
    chat: Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]],
    initial_messages: list[dict[str, Any]],
    cfg: AgentLoopConfig | None = None,
) -> list[dict[str, Any]]:
    """Run a bounded ReAct loop; return the full message trajectory.

    `chat` is a user-supplied async callable that takes the current message list
    and returns the assistant's next reply (a dict). If the reply has no
    `tool_calls`, the loop terminates. Otherwise the loop expects the caller to
    have appended the tool result(s) to `messages` before the next iteration —
    in this canonical implementation we simply re-call `chat(messages)` each
    step, so the host integration is responsible for inserting tool outputs.
    """
    cfg = cfg or AgentLoopConfig()
    messages: list[dict[str, Any]] = list(initial_messages)
    recent_signatures: deque[str | None] = deque(maxlen=cfg.repeat_threshold)

    for step in range(cfg.max_steps):
        reply = await chat(messages)
        messages.append(reply)
        sig = _tool_signature(reply)

        if sig is None:
            # Plain assistant turn with no tool calls — loop terminates cleanly.
            return messages

        recent_signatures.append(sig)
        if (
            len(recent_signatures) == cfg.repeat_threshold
            and len(set(recent_signatures)) == 1
        ):
            msg = (
                f"doom loop on step {step}: tool call {sig!r} repeated "
                f"{cfg.repeat_threshold} times"
            )
            raise DoomLoopDetected(msg)

    msg = f"max_steps={cfg.max_steps} exhausted without a final non-tool turn"
    raise RuntimeError(msg)


def trajectory_summary(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Return per-role + tool-call counts for telemetry."""
    summary: dict[str, int] = {"system": 0, "user": 0, "assistant": 0, "tool": 0, "tool_calls": 0}
    for m in messages:
        role = m.get("role", "")
        if role in summary:
            summary[role] += 1
        for _ in m.get("tool_calls") or []:
            summary["tool_calls"] += 1
    return summary


__all__ = ["AgentLoopConfig", "DoomLoopDetected", "run_agent_loop", "trajectory_summary"]


# Helper kept for backwards compatibility with prose docs that reference the
# function name.
def _serialize_signature(sig: str | None) -> str:
    return json.dumps(sig)
