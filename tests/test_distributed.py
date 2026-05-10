"""FSDP / DeepSpeed config builders."""

from __future__ import annotations

import pytest

from mindxtrain.train.distributed import build_deepspeed_config, build_fsdp_config


def test_fsdp_8gpu_config_has_correct_strategy():
    cfg = build_fsdp_config(8)
    assert cfg["num_processes"] == 8
    assert cfg["distributed_type"] == "FSDP"
    assert cfg["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"


def test_fsdp_1gpu_config_is_no_distributed():
    cfg = build_fsdp_config(1)
    assert cfg["num_processes"] == 1
    assert cfg["distributed_type"] == "NO"


def test_fsdp_2gpu_rejected_xgmi():
    with pytest.raises(ValueError, match="1 or 8"):
        build_fsdp_config(2)  # type: ignore[arg-type]


def test_fsdp_4gpu_rejected_xgmi():
    with pytest.raises(ValueError, match="1 or 8"):
        build_fsdp_config(4)  # type: ignore[arg-type]


def test_deepspeed_zero3_default():
    cfg = build_deepspeed_config()
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["bf16"]["enabled"] is True


def test_deepspeed_invalid_stage_rejected():
    with pytest.raises(ValueError):
        build_deepspeed_config(zero_stage=4)  # type: ignore[arg-type]
