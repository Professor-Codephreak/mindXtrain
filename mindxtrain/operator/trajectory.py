"""TrajectoryWriter — JSONL append-only run log.

Canonical mindxtrain2.md §Part 4 `operator.trajectory`. Each line is one
ReAct step (request, response, tool calls, latency, tokens) keyed by run_id.
Useful for replay, regression analysis, and post-hoc training data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrajectoryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    step: int
    ts: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class TrajectoryWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TrajectoryEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False))
            f.write("\n")
