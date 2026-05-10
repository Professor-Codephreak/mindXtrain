"""Lighteval adapter — standard LM benchmarks via the HF Lighteval harness.

Subprocess `lighteval` CLI; emits a `results.json` under `out_dir`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _lighteval_available() -> bool:
    return shutil.which("lighteval") is not None


def run_lighteval(
    checkpoint: Path,
    tasks: list[str],
    *,
    out_dir: Path | None = None,
) -> dict[str, float]:
    """Run lighteval; return summary metrics."""
    if not _lighteval_available():
        msg = "lighteval not installed; run `uv sync --extra eval`."
        raise RuntimeError(msg)
    out_dir = Path(out_dir or checkpoint / "lighteval")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lighteval",
        "accelerate",
        "--model_args",
        f"pretrained={checkpoint}",
        "--tasks",
        ",".join(tasks),
        "--output_dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    candidates = sorted(out_dir.glob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}
    raw = json.loads(candidates[-1].read_text())
    out: dict[str, float] = {}
    for task, metrics in (raw.get("results") or {}).items():
        for k, v in metrics.items():
            if isinstance(v, int | float):
                out[f"{task}/{k}"] = float(v)
    return out


__all__ = ["run_lighteval"]
