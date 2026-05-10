"""Model-version registry — content-addressed deploy slots.

JSON-file-backed registry with atomic writes (`os.replace`). Three slots:
`live`, `staged`, `canary`. Pure stdlib.

The registry tracks (run_id, blake3, activated_at) per slot plus a
`previous_live` slot so `mindxtrain.deploy.hot_swap.rollback_live` is a
single-step operation.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SlotName = Literal["live", "staged", "canary", "previous_live"]


class Slot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: SlotName
    run_id: str
    blake3: str
    activated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class RegistryState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    slots: dict[str, Slot] = Field(default_factory=dict)


class DeployRegistry:
    """Atomic JSON-file-backed deploy registry."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> RegistryState:
        if not self.path.exists():
            return RegistryState()
        return RegistryState.model_validate_json(self.path.read_text())

    def _write(self, state: RegistryState) -> None:
        # Atomic write: tmp file in same directory, then os.replace.
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.path.parent,
            delete=False,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        ) as tmp:
            tmp.write(state.model_dump_json(indent=2))
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.path)

    def get(self, slot: SlotName) -> Slot | None:
        return self._read().slots.get(slot)

    def set(self, slot: Slot) -> None:
        state = self._read()
        state.slots[slot.name] = slot
        self._write(state)

    def clear(self, slot: SlotName) -> None:
        state = self._read()
        state.slots.pop(slot, None)
        self._write(state)

    def all(self) -> dict[str, Slot]:
        return dict(self._read().slots)


__all__ = ["DeployRegistry", "RegistryState", "Slot", "SlotName"]
