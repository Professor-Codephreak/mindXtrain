"""ResourceBudget — psutil + rocm-smi-derived training-job sizing helper.

Reads host RAM, CPU, GPU memory; emits a recommended `micro_batch_size`
that fits under the budget for a given `seq_len`. psutil + rocm-smi are
both lazy / optional — falls back to conservative defaults if neither is
available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import cast

from pydantic import BaseModel, ConfigDict, Field


class ResourceBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host_ram_gb: float = Field(ge=0.0)
    cpu_cores: int = Field(ge=1)
    gpu_count: int = Field(ge=0)
    gpu_mem_gb_each: float = Field(ge=0.0)


def _host_ram_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024**3)
    except ImportError:
        return 0.0


def _cpu_cores() -> int:
    try:
        import psutil

        return int(psutil.cpu_count(logical=True) or 1)
    except ImportError:
        import os

        return os.cpu_count() or 1


def _gpu_mem_gb() -> tuple[int, float]:
    """Return (gpu_count, gpu_mem_gb_each)."""
    if shutil.which("rocm-smi") is None:
        return (0, 0.0)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return (0, 0.0)
    if out.returncode != 0:
        return (0, 0.0)
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return (0, 0.0)
    cards = [k for k in data if k.startswith("card")]
    if not cards:
        return (0, 0.0)
    total_bytes = 0.0
    for c in cards:
        for k, v in (data[c] or {}).items():
            if "total" in k.lower():
                try:
                    total_bytes = max(total_bytes, float(v))
                    break
                except ValueError:
                    continue
    if total_bytes == 0.0:
        return (len(cards), 0.0)
    return (len(cards), total_bytes / (1024**3))


def detect() -> ResourceBudget:
    """Return a `ResourceBudget` snapshot of the current host."""
    gpu_count, gpu_mem = _gpu_mem_gb()
    return ResourceBudget(
        host_ram_gb=_host_ram_gb(),
        cpu_cores=_cpu_cores(),
        gpu_count=gpu_count,
        gpu_mem_gb_each=gpu_mem,
    )


def recommend_micro_batch(budget: ResourceBudget, seq_len: int, *, dtype_bytes: int = 2) -> int:
    """Return a conservative `micro_batch_size` that fits under the budget.

    Heuristic: each token consumes ~`dtype_bytes` bytes for activations + a
    factor for KV cache + gradients. We leave 30% headroom.
    """
    if budget.gpu_mem_gb_each <= 0:
        return 1
    bytes_per_token = dtype_bytes * 16  # rough activations + grads + KV factor
    available = budget.gpu_mem_gb_each * 0.7 * (1024**3)
    bs = int(available / (seq_len * bytes_per_token))
    return cast(int, max(1, bs))


__all__ = ["ResourceBudget", "detect", "recommend_micro_batch"]
