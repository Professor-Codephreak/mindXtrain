"""BLAKE3 helpers for content-addressed provenance.

Files: streamed, returns a hex digest.
Directories: hash sorted (relpath, file-hash) pairs to make the dir hash
deterministic regardless of filesystem walk order or mtime.
"""

from __future__ import annotations

from pathlib import Path

from blake3 import blake3

_CHUNK = 1 << 20  # 1 MiB


def blake3_file(path: Path) -> str:
    """Return BLAKE3 hex digest of a single file."""
    h = blake3()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def blake3_dir(root: Path) -> str:
    """Return BLAKE3 hex digest of a directory (sorted-relpath + file-hash composition)."""
    if not root.is_dir():
        msg = f"not a directory: {root}"
        raise NotADirectoryError(msg)

    entries: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            entries.append((rel, blake3_file(p)))

    h = blake3()
    for rel, file_hash in entries:
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(file_hash.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()
