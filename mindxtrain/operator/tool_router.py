"""ToolRouter + ToolSpec — bounded tool dispatch (ml-intern pattern).

Canonical mindxtrain2.md §Part 4 `operator.tool_router`. Adapted from the
ml-intern Tool/ToolRouter pattern: Pydantic-typed tool specs, name-based
dispatch, run-id-scoped invocation logging.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    handler: Callable[..., Awaitable[Any]]


class ToolRouter:
    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {t.name: t for t in (tools or [])}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            msg = f"tool {tool.name!r} already registered"
            raise ValueError(msg)
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            available = ", ".join(self.names()) or "(none)"
            msg = f"unknown tool {name!r}. available: {available}"
            raise KeyError(msg)
        return await self._tools[name].handler(**arguments)
