"""τ-Bench adapter — multi-turn agentic evaluation (Sierra)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _tau_bench_available() -> bool:
    return shutil.which("tau-bench") is not None or shutil.which("tau_bench") is not None


def run_tau_bench(
    checkpoint: Path,
    *,
    scenario: str = "airline",
    out_dir: Path | None = None,
) -> dict[str, float]:
    """Run τ-Bench against `checkpoint`; return scenario scores."""
    if not _tau_bench_available():
        msg = "tau_bench not installed; install per https://github.com/sierra-research/tau-bench"
        raise RuntimeError(msg)
    out_dir = Path(out_dir or checkpoint / "tau_bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_name = shutil.which("tau-bench") or shutil.which("tau_bench") or "tau-bench"
    cmd = [
        bin_name,
        "--task",
        scenario,
        "--model",
        str(checkpoint),
        "--output",
        str(out_dir / "results.json"),
    ]
    subprocess.run(cmd, check=True)
    rp = out_dir / "results.json"
    if not rp.exists():
        return {}
    raw = json.loads(rp.read_text())
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, int | float):
            out[k] = float(v)
    return out


__all__ = ["run_tau_bench"]
