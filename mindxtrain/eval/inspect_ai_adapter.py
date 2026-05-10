"""Inspect-AI adapter — agentic evaluations.

Subprocess `inspect eval`; emits per-task JSON under `out_dir`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _inspect_available() -> bool:
    return shutil.which("inspect") is not None


def run_inspect(
    checkpoint: Path,
    tasks: list[str],
    *,
    out_dir: Path | None = None,
) -> dict[str, float]:
    """Run an inspect-ai task suite; return summary metrics."""
    if not _inspect_available():
        msg = "inspect-ai not installed; run `uv sync --extra eval`."
        raise RuntimeError(msg)
    out_dir = Path(out_dir or checkpoint / "inspect")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "inspect",
        "eval",
        *tasks,
        "--model",
        f"hf/{checkpoint}",
        "--log-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    out: dict[str, float] = {}
    for p in sorted(out_dir.glob("**/*.json")):
        try:
            raw = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scores = raw.get("results", {}).get("scores", {})
        for k, v in scores.items():
            if isinstance(v, dict) and "value" in v and isinstance(v["value"], int | float):
                out[f"{p.stem}/{k}"] = float(v["value"])
    return out


__all__ = ["run_inspect"]
