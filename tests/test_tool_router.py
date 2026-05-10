"""ToolRouter dispatch."""

from __future__ import annotations

import pytest

from mindxtrain.operator.tool_router import ToolRouter, ToolSpec


@pytest.mark.asyncio
async def test_dispatch_invokes_handler():
    async def add(a: int, b: int) -> int:
        return a + b

    router = ToolRouter([ToolSpec(name="add", description="sum", handler=add)])
    result = await router.dispatch("add", {"a": 2, "b": 3})
    assert result == 5


@pytest.mark.asyncio
async def test_unknown_tool_raises():
    router = ToolRouter()
    with pytest.raises(KeyError):
        await router.dispatch("missing", {})


def test_register_duplicate_raises():
    async def noop():
        return None

    spec = ToolSpec(name="t", description="", handler=noop)
    router = ToolRouter([spec])
    with pytest.raises(ValueError):
        router.register(spec)
