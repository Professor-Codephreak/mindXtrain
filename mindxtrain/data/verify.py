"""Dataset verification — recompute BLAKE3 over the local files referenced by
a shard manifest, surface mismatches.

Pure stdlib + `mindxtrain.provenance.hashing`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mindxtrain.provenance.hashing import blake3_file


class VerifyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: Path
    matched: int = 0
    mismatched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


def verify_dataset(manifest_path: Path, root: Path) -> VerifyResult:
    """Walk the manifest, recompute hashes, return mismatches.

    Manifest format expected:

        {"shards": [{"path": "shard-00000.tar", "blake3": "...."}, ...]}

    `path` is resolved relative to `root`.
    """
    manifest_path = Path(manifest_path)
    root = Path(root)
    raw = json.loads(manifest_path.read_text())
    shards = raw.get("shards") or []
    result = VerifyResult(manifest_path=manifest_path)
    for shard in shards:
        rel = shard.get("path", "")
        expected = shard.get("blake3", "")
        if not rel or not expected:
            continue
        local = root / rel
        if not local.exists():
            result.missing.append(rel)
            continue
        actual = blake3_file(local)
        if actual != expected:
            result.mismatched.append(rel)
        else:
            result.matched += 1
    return result


__all__ = ["VerifyResult", "verify_dataset"]
