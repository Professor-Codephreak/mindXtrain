"""StorageProvider — uniform interface across local fs / HF Hub / Lighthouse / IPFS.

Canonical mindxtrain2.md §Part 4 `storage.provider`. Concrete implementations
live alongside (`local_fs`, `hf_hub`, `lighthouse`, `ipfs`). Selected at
runtime via `cfg.publish.storage_provider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class StorageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    uri: str


class StorageProvider(ABC):
    """Pluggable storage backend."""

    name: str

    @abstractmethod
    def put_dir(self, src: Path, key: str) -> StorageRef: ...

    @abstractmethod
    def get_dir(self, ref: StorageRef, dest: Path) -> Path: ...
