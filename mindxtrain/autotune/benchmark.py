"""Autotune orchestrator — runs three probes and emits an AutotunePlan.

Day 1: returns a hardcoded reference plan (`dry_run=True` always).
Day 2 (on MI300X): wires up the real CK-vs-Triton attention probe and replaces
the hardcoded GEMM/RCCL decisions with documented heuristics.
"""

from __future__ import annotations

from mindxtrain.autotune.attention_probe import probe_attention
from mindxtrain.autotune.gemm_probe import probe_gemm
from mindxtrain.autotune.plan import AutotunePlan, ProbeTiming
from mindxtrain.autotune.rccl_probe import probe_rccl


def run_autotune(gpu_index: int = 0, dry_run: bool = False) -> AutotunePlan:
    """Run the 60-second probe sequence and return an AutotunePlan.

    On Day 1 (dry_run=True or no GPU) returns a static reference plan that
    the training dispatch can consume to exercise its code path.
    """
    if dry_run:
        return _reference_plan()

    attention_backend, attention_timings = probe_attention(gpu_index=gpu_index)
    gemm_heuristic = probe_gemm(gpu_index=gpu_index)
    rccl_config = probe_rccl(gpu_index=gpu_index)

    return AutotunePlan(
        attention_backend=attention_backend,
        gemm_heuristic=gemm_heuristic,
        rccl_config=rccl_config,
        probe_timings=attention_timings,
        notes=[
            "Day 2 real probe: attention measured, GEMM/RCCL from documented AMD heuristics.",
        ],
    )


def _reference_plan() -> AutotunePlan:
    """Hardcoded Day-1 reference plan; values are sane defaults for MI300X gfx942."""
    return AutotunePlan(
        attention_backend="ck",
        gemm_heuristic="hipblaslt_default",
        rccl_config="1gpu_noop",
        fsdp_shard_width=1,
        suggested_lora_rank=16,
        suggested_micro_batch_size=4,
        probe_timings=[
            ProbeTiming(label="dry-run-reference", backend="ck", median_ms=0.0, iterations=1),
        ],
        notes=["dry-run reference plan; replace with real probe output on MI300X via Day-2 bench"],
    )
