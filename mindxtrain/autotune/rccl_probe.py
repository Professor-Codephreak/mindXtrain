"""RCCL collective config selection.

MI300X xGMI bandwidth is asymmetric on 2- and 4-GPU groupings; FSDP shard
topology must be 1-GPU or 8-GPU. We hard-fail anything else here so the
training dispatch refuses to launch a misconfigured run.

GPU count is auto-detected at probe time (torch first, then `rocminfo`), so
`mindxtrain bench` self-selects '1gpu_noop' vs '8gpu_xgmi' on the box it runs
on. Pass an explicit `gpu_count` to override (used by tests).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess

from mindxtrain.autotune.plan import RcclConfig


def detect_gpu_count() -> int:
    """Best-effort GPU count: torch.cuda first, then `rocminfo`, else 0.

    On ROCm, `torch.cuda.device_count()` reports HIP devices. When torch isn't
    installed (typical CPU dev box), fall back to counting GPU agents in
    `rocminfo`. Returns 0 when nothing is detectable.
    """
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except (RuntimeError, ImportError, OSError):
            pass

    rocminfo = shutil.which("rocminfo")
    if rocminfo is not None:
        try:
            out = subprocess.run(
                [rocminfo],
                capture_output=True,
                text=True,
                timeout=15.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return 0
        if out.returncode == 0:
            # Each GPU agent block reports `Device Type: GPU`.
            return sum(
                1
                for line in out.stdout.splitlines()
                if "Device Type:" in line and "GPU" in line
            )
    return 0


def probe_rccl(gpu_index: int = 0, gpu_count: int | None = None) -> RcclConfig:
    """Pick the RCCL config; refuse 2/4-GPU sharding.

    `gpu_count=None` auto-detects. A detected count of 0 (no GPU, CPU dev box)
    maps to the single-device no-op config so the plan stays consumable.
    """
    _ = gpu_index
    if gpu_count is None:
        gpu_count = detect_gpu_count()
    if gpu_count in (0, 1):
        return "1gpu_noop"
    if gpu_count == 8:
        return "8gpu_xgmi"
    msg = (
        f"FSDP on {gpu_count} GPUs is unsafe on MI300X due to xGMI bandwidth asymmetry. "
        "Use 1 or 8 GPUs (mindXtrain2.md §13)."
    )
    raise RuntimeError(msg)
