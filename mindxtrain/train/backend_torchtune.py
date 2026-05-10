"""torchtune backend — modular reference recipes.

Subprocess wrapper around `tune run`. Lazy availability check; opt-in via
`uv pip install torchtune`.
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


def _torchtune_available() -> bool:
    return importlib.util.find_spec("torchtune") is not None or shutil.which("tune") is not None


def run_torchtune(cfg: XTrainConfig, plan: AutotunePlan, out_dir: Path) -> Path:
    """Run a torchtune training job; return the checkpoint directory."""
    if not _torchtune_available():
        msg = (
            "torchtune not installed; `uv pip install torchtune`. Note: Qwen3 "
            "recipes are not yet upstream — ship a small PR as a side deliverable."
        )
        raise RuntimeError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_payload = compile_axolotl_yaml(cfg, plan)
    yaml_path = out_dir / f"{cfg.meta.run_name}.torchtune.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

    log_path = out_dir / "train.log"
    env = dict(os.environ)
    env["PYTORCH_ROCM_ARCH"] = "gfx942"

    cmd = ["tune", "run", "--config", str(yaml_path)]
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"torchtune returned {proc.returncode}; see {log_path}\n")
        raise SystemExit(proc.returncode)

    return out_dir / yaml_payload.get("output_dir", "checkpoint")


__all__ = ["run_torchtune"]
