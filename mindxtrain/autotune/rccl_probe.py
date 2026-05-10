"""RCCL collective config selection.

MI300X xGMI bandwidth is asymmetric on 2- and 4-GPU groupings; FSDP shard
topology must be 1-GPU or 8-GPU. We hard-fail anything else here so the
training dispatch refuses to launch a misconfigured run.

Day 2 implementation: detect GPU count via `rocminfo` and either return
'1gpu_noop' or '8gpu_xgmi' with NCCL_MIN_NCHANNELS=112 set in the plan notes.
"""

from __future__ import annotations

from mindxtrain.autotune.plan import RcclConfig


def probe_rccl(gpu_index: int = 0, gpu_count: int = 1) -> RcclConfig:
    """Pick the RCCL config; refuse 2/4-GPU sharding."""
    _ = gpu_index
    if gpu_count == 1:
        return "1gpu_noop"
    if gpu_count == 8:
        return "8gpu_xgmi"
    msg = (
        f"FSDP on {gpu_count} GPUs is unsafe on MI300X due to xGMI bandwidth asymmetry. "
        "Use 1 or 8 GPUs (mindXtrain2.md §13)."
    )
    raise RuntimeError(msg)
