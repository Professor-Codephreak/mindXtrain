"""AutotunePlan — output of the 60s MI300X probe, consumed by the training layer.

The plan is the contract between the autotune module (Day 2 work) and the
training dispatch (Day 3 work). It must be pure data so a plan generated on
one MI300X can be replayed on another, and so a plan can be hand-written for
testing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AttentionBackend = Literal["ck", "triton"]
GemmHeuristic = Literal["hipblaslt_default", "hipblaslt_tuned", "rocblas_fallback"]
RcclConfig = Literal["1gpu_noop", "8gpu_xgmi", "unsupported_2_4_gpu"]


class ProbeTiming(BaseModel):
    """Single probe measurement (one shape, one backend)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    backend: str
    median_ms: float = Field(ge=0.0)
    iterations: int = Field(ge=1)


class AutotunePlan(BaseModel):
    """Static AOT plan written by `mindxtrain bench` and read by `mindxtrain train`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    gpu_arch: str = Field(default="gfx942")
    rocm_version: str = Field(default="7.2.1")

    attention_backend: AttentionBackend = "ck"
    gemm_heuristic: GemmHeuristic = "hipblaslt_default"
    rccl_config: RcclConfig = "1gpu_noop"

    fsdp_shard_width: Literal[1, 8] = 1
    suggested_lora_rank: int = Field(default=16, ge=1, le=512)
    suggested_micro_batch_size: int = Field(default=4, ge=1)

    probe_timings: list[ProbeTiming] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
