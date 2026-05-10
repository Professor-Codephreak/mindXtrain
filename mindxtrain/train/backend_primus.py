"""Primus / Primus-Turbo backend — pretraining-scale (>70 B params).

Subprocess wrapper that expects the rocm/primus:v26.2 container to be active
(Primus, AITER, Composable Kernel, hipBLASLt, FP8 transformer-engine support
all live there). Lazy availability check.
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


def _primus_available() -> bool:
    return (
        importlib.util.find_spec("primus_turbo") is not None
        or shutil.which("primus") is not None
        or os.environ.get("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32") is not None
    )


def run_primus(cfg: XTrainConfig, plan: AutotunePlan, out_dir: Path) -> Path:
    """Run a Primus pretraining job; return the checkpoint directory."""
    if not _primus_available():
        msg = (
            "primus_turbo not installed and not running inside rocm/primus:v26.2; "
            "see ops/containerfiles/containerfile_train and pull the container."
        )
        raise RuntimeError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_payload = compile_axolotl_yaml(cfg, plan)
    yaml_path = out_dir / f"{cfg.meta.run_name}.primus.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

    log_path = out_dir / "train.log"
    env = dict(os.environ)
    env["PYTORCH_ROCM_ARCH"] = "gfx942"
    env["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "1"
    if plan.rccl_config == "8gpu_xgmi":
        env["NCCL_MIN_NCHANNELS"] = "112"

    cmd = ["python", "-m", "primus_turbo.train", "--config", str(yaml_path)]
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if proc.returncode != 0:
        sys.stderr.write(f"primus returned {proc.returncode}; see {log_path}\n")
        raise SystemExit(proc.returncode)

    return out_dir / yaml_payload.get("output_dir", "checkpoint")


__all__ = ["run_primus"]
