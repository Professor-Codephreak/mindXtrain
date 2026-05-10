"""Always-available local filesystem `StorageProvider` implementation.

Canonical mindxtrain2.md §Part 4 `storage.local_fs`. Acts as the dev/CI fallback
when no remote provider is configured.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from mindxtrain.storage.provider import StorageProvider, StorageRef


class LocalFsProvider(StorageProvider):
    name = "local_fs"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_dir(self, src: Path, key: str) -> StorageRef:
        dest = self.root / key
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return StorageRef(provider=self.name, uri=str(dest))

    def get_dir(self, ref: StorageRef, dest: Path) -> Path:
        if ref.provider != self.name:
            msg = f"ref provider {ref.provider!r} != {self.name!r}"
            raise ValueError(msg)
        src = Path(ref.uri)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest
