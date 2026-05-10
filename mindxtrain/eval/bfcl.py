"""BFCL — Berkeley Function-Calling Leaderboard adapter.

Subprocess `bfcl evaluate`. Lazy availability check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _bfcl_available() -> bool:
    return shutil.which("bfcl") is not None


def run_bfcl(
    checkpoint: Path,
    *,
    version: str = "v4",
    out_dir: Path | None = None,
    categories: list[str] | None = None,
) -> dict[str, float]:
    """Run the BFCL evaluation suite; return per-category scores."""
    if not _bfcl_available():
        msg = "bfcl not installed; run `uv sync --extra eval`."
        raise RuntimeError(msg)
    out_dir = Path(out_dir or checkpoint / "bfcl")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bfcl",
        "evaluate",
        "--model",
        str(checkpoint),
        "--test-category",
        ",".join(categories or ["simple", "parallel", "multiple", "multi_turn"]),
        "--version",
        version,
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    out: dict[str, float] = {}
    for p in sorted(out_dir.glob("*.json")):
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for k, v in raw.items():
            if isinstance(v, int | float):
                out[f"{p.stem}/{k}"] = float(v)
    return out


__all__ = ["run_bfcl"]
