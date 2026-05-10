"""Axolotl SFT/LoRA backend.

Real subprocess wrapper around `accelerate launch -m axolotl.cli.train`. Steps:
    1. Compile `XTrainConfig` + `AutotunePlan` to an Axolotl YAML via
       `mindxtrain.train.axolotl_compile.compile_axolotl_yaml`.
    2. Inject MI300X env vars from the autotune plan.
    3. Spawn the subprocess; tee stdout/stderr to `runs/<run_id>/train.log`.
    4. Return the checkpoint directory.

Two entry points share the same prep helpers (`prepare_run`):

- `run_axolotl(cfg, plan, out_dir)` — synchronous; blocks until completion;
  used by the `mindxtrain train` CLI verb.
- `prepare_run(cfg, plan, out_dir)` — returns the cmd + env + paths so the
  Coach UI's streaming launch path can hand them to
  `mindxtrain.operator.runs.spawn_subprocess_streaming`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.schema import XTrainConfig
from mindxtrain.train.axolotl_compile import compile_axolotl_yaml

_BASE_ENV = {
    "PYTORCH_ROCM_ARCH": "gfx942",
    "HSA_NO_SCRATCH_RECLAIM": "1",
    "HIP_FORCE_DEV_KERNARG": "1",
    "GPU_MAX_HW_QUEUES": "1",
}


def _accelerate_available() -> bool:
    return shutil.which("accelerate") is not None


def _plan_env(plan: AutotunePlan) -> dict[str, str]:
    env: dict[str, str] = dict(_BASE_ENV)
    if plan.rccl_config == "8gpu_xgmi":
        env["NCCL_MIN_NCHANNELS"] = "112"
    if plan.attention_backend == "ck":
        env["NVTE_CK_USES_BWD_V3"] = "1"
        env["NVTE_CK_IS_V3_ATOMIC_FP32"] = "1"
        env["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "1"
    return env


@dataclass(frozen=True)
class PreparedRun:
    """Resolved cmd + env + paths for an Axolotl run.

    The streaming launch path consumes this directly; the synchronous
    `run_axolotl` does too.
    """

    cmd: list[str]
    env: dict[str, str]
    yaml_path: Path
    log_path: Path
    checkpoint_dir: Path


def prepare_run(cfg: XTrainConfig, plan: AutotunePlan, out_dir: Path) -> PreparedRun:
    """Compile YAML, materialize on disk, and return the cmd/env/paths.

    Does NOT spawn the subprocess. Raises `RuntimeError` if `accelerate` is
    not on PATH (the same condition the synchronous wrapper checks).
    """
    if not _accelerate_available():
        msg = (
            "accelerate not found on PATH; install with `uv sync --extra ml` "
            "and ensure the `axolotl` package is reachable in the same venv."
        )
        raise RuntimeError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_payload = compile_axolotl_yaml(cfg, plan)
    yaml_path = out_dir / f"{cfg.meta.run_name}.axolotl.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_payload, sort_keys=False))

    log_path = out_dir / "train.log"

    env = dict(os.environ)
    env.update(_plan_env(plan))

    cmd = [
        "accelerate",
        "launch",
        "-m",
        "axolotl.cli.train",
        str(yaml_path),
    ]

    return PreparedRun(
        cmd=cmd,
        env=env,
        yaml_path=yaml_path,
        log_path=log_path,
        checkpoint_dir=out_dir / yaml_payload.get("output_dir", "checkpoint"),
    )


def _run_streaming(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    on_line: Callable[[str], None],
) -> int:
    """Run `cmd`, tee each stdout line to `log_path` and call `on_line(line)`.

    Returns the subprocess return code. Used by the synchronous CLI path;
    the Coach UI streaming launch path uses
    `mindxtrain.operator.runs.spawn_subprocess_streaming` instead, which
    runs the reader in a thread.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        log.write(f"# cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            log.flush()
            on_line(raw.rstrip("\n"))
        return proc.wait()


def run_axolotl(
    cfg: XTrainConfig,
    plan: AutotunePlan,
    out_dir: Path,
    *,
    on_line: Callable[[str], None] | None = None,
) -> Path:
    """Run an Axolotl training job and return the checkpoint directory.

    Blocks until completion. If `on_line` is provided, it is called once per
    stdout line (the same lines that get written to `train.log`); the CLI
    path passes `None`.
    """
    prepared = prepare_run(cfg, plan, out_dir)
    sink = on_line if on_line is not None else (lambda _line: None)
    rc = _run_streaming(cmd=prepared.cmd, env=prepared.env, log_path=prepared.log_path, on_line=sink)
    if rc != 0:
        sys.stderr.write(f"axolotl returned {rc}; see {prepared.log_path}\n")
        raise SystemExit(rc)
    return prepared.checkpoint_dir


__all__ = ["PreparedRun", "prepare_run", "run_axolotl"]
