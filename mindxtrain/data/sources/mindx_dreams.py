"""mindX dream-cycle JSONL training corpus adapter.

mindX's `agents/machine_dreaming.py` writes per-agent training data to
`<mindx_home>/data/memory/ltm/<agent>/<timestamp>_training.jsonl` on every
dream cycle (Phase 5b — `_write_training_data`). Each line is an OpenAI chat
example:

    {"messages": [{"role": "system|user|assistant", "content": "..."}]}

This adapter walks the glob, deduplicates by content hash, and yields the
rows. No heavyweight deps — pure stdlib so the CLI/Coach can validate a
recipe pointing at this source even on a CPU-only laptop.

Config example::

    data:
      source: mindx_dreams
      path: /home/hacker/mindX/data/memory
      max_samples: null   # null = all examples
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_DEFAULT_GLOB = "ltm/**/*_training.jsonl"


def _row_fingerprint(row: dict[str, Any]) -> str:
    """Stable hash over the messages payload for cross-cycle deduplication."""
    payload = json.dumps(row.get("messages"), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def iter_mindx_dreams(
    root: Path,
    *,
    glob: str = _DEFAULT_GLOB,
    max_samples: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield deduplicated chat-format rows from a mindX dream JSONL tree.

    `root` is the mindX `data/memory` directory. `glob` is relative to it.
    Cross-file dedup is by BLAKE2b of the sorted messages payload —
    identical patterns emitted by different agents/days collapse to one row,
    keeping the corpus from being dominated by very-frequent insights.

    Bad JSON lines are skipped silently (the dream writer is lossy and the
    LTM tree may contain partial flushes).
    """
    root = Path(root).expanduser()
    if not root.exists():
        msg = f"mindX dream-corpus root not found: {root}"
        raise FileNotFoundError(msg)

    seen: set[str] = set()
    emitted = 0
    for jsonl_path in sorted(root.glob(glob)):
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or "messages" not in row:
                        continue
                    if not isinstance(row["messages"], list) or not row["messages"]:
                        continue
                    fp = _row_fingerprint(row)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    yield row
                    emitted += 1
                    if max_samples is not None and emitted >= max_samples:
                        return
        except OSError:
            continue


def count_mindx_dreams(root: Path, *, glob: str = _DEFAULT_GLOB) -> dict[str, int]:
    """Cheap statistics about a corpus root — files, raw lines, unique rows.

    Used by the Coach UI / CLI to surface "corpus has N examples" before a
    training run is dispatched. Walks the tree once; safe to call from a
    request handler.
    """
    root = Path(root).expanduser()
    files = 0
    raw_lines = 0
    unique = 0
    seen: set[str] = set()
    for jsonl_path in root.glob(glob):
        files += 1
        try:
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    raw_lines += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or "messages" not in row:
                        continue
                    fp = _row_fingerprint(row)
                    if fp not in seen:
                        seen.add(fp)
                        unique += 1
        except OSError:
            continue
    return {"files": files, "raw_lines": raw_lines, "unique_rows": unique}


__all__ = ["count_mindx_dreams", "iter_mindx_dreams"]
