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
