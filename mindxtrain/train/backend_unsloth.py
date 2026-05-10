"""Unsloth backend — fastest single-MI300X LoRA path.

Subprocess wrapper around `unsloth.cli.train`. Lazy availability check;
opt-in via `uv add unsloth` (the OneClickAMD partnership wheel).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.schema import XTrainConfig
from mindxtrain.train.axolotl_compile import compile_axolotl_yaml


def _unsloth_available() -> bool:
    return importlib.util.find_spec("unsloth") is not None


def run_unsloth(cfg: XTrainConfig, plan: AutotunePlan, out_dir: Path) -> Path:
    """Run an Unsloth LoRA training job; return the checkpoint directory."""
    if not _unsloth_available():
        msg = (
            "unsloth not installed; install the OneClickAMD wheel per "
            "https://github.com/unslothai/unsloth (ROCm support is opt-in)."
        )
        raise RuntimeError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the Axolotl YAML as a portable trainer config — Unsloth consumes a
    # near-identical schema.
    yaml_payload = compile_axolotl_yaml(cfg, plan)
    yaml_path = out_dir / f"{cfg.meta.run_name}.unsloth.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

    log_path = out_dir / "train.log"
    env = dict(os.environ)
    env["PYTORCH_ROCM_ARCH"] = "gfx942"

    # Unsloth ships its own `unsloth-cli` entry; fall back to python -m if missing.
    if shutil.which("unsloth-cli"):
        cmd = ["unsloth-cli", "train", str(yaml_path)]
    else:
        cmd = ["python", "-m", "unsloth.cli.train", str(yaml_path)]

    with log_path.open("w") as log:
        log.write(f"# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)

    if proc.returncode != 0:
        sys.stderr.write(f"unsloth returned {proc.returncode}; see {log_path}\n")
        raise SystemExit(proc.returncode)

    return out_dir / yaml_payload.get("output_dir", "checkpoint")


__all__ = ["run_unsloth"]
