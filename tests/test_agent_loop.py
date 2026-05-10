"""Bounded ReAct loop + doom-loop detector."""

from __future__ import annotations

import pytest

from mindxtrain.operator.agent_loop import (
    AgentLoopConfig,
    DoomLoopDetected,
    run_agent_loop,
    trajectory_summary,
)


@pytest.mark.asyncio
async def test_loop_terminates_on_no_tool_calls():
    async def chat(messages):
        return {"role": "assistant", "content": "done"}

    out = await run_agent_loop(chat, [{"role": "user", "content": "hi"}])
    assert out[-1]["content"] == "done"


@pytest.mark.asyncio
async def test_loop_doom_detected_on_repeat():
    call_count = {"n": 0}

    async def chat(messages):
        call_count["n"] += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
        }

    with pytest.raises(DoomLoopDetected):
        await run_agent_loop(
            chat,
            [{"role": "user", "content": "x"}],
            AgentLoopConfig(max_steps=10, repeat_threshold=3),
        )
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_loop_max_steps_exhausted():
    async def chat(messages):
        # Different tool every step → no doom detection, but eventually max_steps.
        n = len([m for m in messages if m.get("role") == "assistant"])
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": f"t{n}", "arguments": "{}"}}],
        }

    with pytest.raises(RuntimeError, match="max_steps"):
        await run_agent_loop(chat, [{"role": "user", "content": "x"}], AgentLoopConfig(max_steps=4))


def test_trajectory_summary_counts_roles():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "content": "r"},
        {"role": "assistant", "content": "ok"},
    ]
    s = trajectory_summary(msgs)
    assert s["assistant"] == 2
    assert s["tool"] == 1
    assert s["tool_calls"] == 1
