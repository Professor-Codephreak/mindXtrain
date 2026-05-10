"""lm-evaluation-harness wrapper.

Subprocess `lm_eval --model hf --tasks <comma-sep> --model_args pretrained=<dir>`.
Output JSON written under `<out_dir>/lm_eval.json`. Lazy availability check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _lm_eval_available() -> bool:
    return shutil.which("lm_eval") is not None or shutil.which("lm-eval") is not None


def run_lm_eval(
    model_dir: Path,
    tasks: list[str],
    *,
    out_dir: Path | None = None,
    batch_size: str = "auto",
) -> Path:
    """Run lm-eval-harness against `model_dir`; return path to results JSON."""
    if not _lm_eval_available():
        msg = "lm-eval not installed; run `uv sync --extra eval`."
        raise RuntimeError(msg)

    out_dir = Path(out_dir or model_dir / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        shutil.which("lm_eval") or "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_dir}",
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        batch_size,
        "--output_path",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    # lm-eval writes a JSON named `results-<timestamp>.json` per the harness;
    # find the newest one and rename to `lm_eval.json` for stable downstream use.
    candidates = sorted(out_dir.glob("results*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        msg = f"lm-eval ran but no results*.json found under {out_dir}"
        raise RuntimeError(msg)
    target = out_dir / "lm_eval.json"
    target.write_text(candidates[-1].read_text())
    return target


def parse_summary(results_json: Path) -> dict[str, float]:
    """Flatten `results.json` into a `{task: metric}` dict."""
    raw = json.loads(Path(results_json).read_text())
    out: dict[str, float] = {}
    for task, metrics in (raw.get("results") or {}).items():
        for k, v in metrics.items():
            if isinstance(v, int | float):
                out[f"{task}/{k}"] = float(v)
    return out


__all__ = ["parse_summary", "run_lm_eval"]
