"""AutotunePlan JSON round-trip + dry-run reference plan."""

from __future__ import annotations

import json

import pytest

from mindxtrain.autotune.benchmark import run_autotune
from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.autotune.rccl_probe import probe_rccl


def test_dry_run_returns_reference_plan():
    plan = run_autotune(dry_run=True)
    assert isinstance(plan, AutotunePlan)
    assert plan.attention_backend == "ck"
    assert plan.fsdp_shard_width in (1, 8)
    assert plan.gpu_arch == "gfx942"


def test_plan_json_round_trip():
    original = run_autotune(dry_run=True)
    blob = original.model_dump_json()
    restored = AutotunePlan.model_validate(json.loads(blob))
    assert restored == original


def test_rccl_probe_rejects_2gpu():
    """xGMI gotcha: 2-GPU and 4-GPU FSDP are unsupported."""
    with pytest.raises(RuntimeError, match="xGMI"):
        probe_rccl(gpu_count=2)
    with pytest.raises(RuntimeError):
        probe_rccl(gpu_count=4)


def test_rccl_probe_accepts_1_and_8():
    assert probe_rccl(gpu_count=1) == "1gpu_noop"
    assert probe_rccl(gpu_count=8) == "8gpu_xgmi"


def test_rccl_probe_zero_gpus_maps_to_noop():
    """A CPU box (0 detected GPUs) maps to the single-device no-op config."""
    assert probe_rccl(gpu_count=0) == "1gpu_noop"


def test_rccl_probe_autodetects_when_count_omitted(monkeypatch):
    from mindxtrain.autotune import rccl_probe

    monkeypatch.setattr(rccl_probe, "detect_gpu_count", lambda: 8)
    assert probe_rccl() == "8gpu_xgmi"
    monkeypatch.setattr(rccl_probe, "detect_gpu_count", lambda: 0)
    assert probe_rccl() == "1gpu_noop"


def test_detect_gpu_count_returns_int():
    """On a CPU dev box this is 0; the call must never raise."""
    from mindxtrain.autotune.rccl_probe import detect_gpu_count

    n = detect_gpu_count()
    assert isinstance(n, int)
    assert n >= 0


def test_gemm_probe_cpu_returns_default_no_timings():
    """No GPU → documented default heuristic, empty timings (deterministic)."""
    from mindxtrain.autotune.gemm_probe import probe_gemm

    heuristic, timings = probe_gemm()
    assert heuristic in ("hipblaslt_default", "hipblaslt_tuned", "rocblas_fallback")
    # On the CI/CPU box there is no GPU, so no timing is recorded.
    if not timings:
        assert heuristic == "hipblaslt_default"
