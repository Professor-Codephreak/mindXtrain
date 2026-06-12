"""Device resolution for the trl_local lane (CPU-testable, no GPU required)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mindxtrain.config.loader import load_config, render_recipe
from mindxtrain.train import backend_trl_cpu as b


def _local_cfg():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.yaml"
        p.write_text(render_recipe("mindx_fallback_qwen3_1_5b_local"))
        return load_config(p)


class _FakeCuda:
    def __init__(self, available: bool, bf16: bool = True):
        self._available = available
        self._bf16 = bf16

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"

    def __init__(self, available: bool, bf16: bool = True):
        self.cuda = _FakeCuda(available, bf16)


def test_recipe_uses_trl_local_backend():
    cfg = _local_cfg()
    assert cfg.train.backend == "trl_local"
    assert cfg.hardware.gpus == 1
    assert cfg.train.precision == "bfloat16"


def test_resolve_device_cpu_when_no_accelerator():
    cfg = _local_cfg()
    dp = b._resolve_device(cfg, _FakeTorch(available=False), force_cpu=False)
    assert dp.device_map == {"": "cpu"}
    assert dp.torch_dtype == "fp32"
    assert dp.attn_impl == "eager"
    assert dp.apply_cpu_throttle is True
    assert dp.batch_cap == 2
    assert dp.grad_checkpointing is False
    assert dp.bf16 is False and dp.fp16 is False


def test_resolve_device_gpu_bf16():
    cfg = _local_cfg()
    dp = b._resolve_device(cfg, _FakeTorch(available=True, bf16=True), force_cpu=False)
    assert dp.device_map == {"": "cuda"}
    assert dp.torch_dtype == "bf16"
    assert dp.attn_impl == "sdpa"
    assert dp.bf16 is True and dp.fp16 is False
    assert dp.apply_cpu_throttle is False
    assert dp.batch_cap is None
    # cfg.train.gradient_checkpointing is True in the local recipe.
    assert dp.grad_checkpointing is True


def test_resolve_device_gpu_fp16_when_no_bf16():
    cfg = _local_cfg()
    dp = b._resolve_device(cfg, _FakeTorch(available=True, bf16=False), force_cpu=False)
    assert dp.torch_dtype == "fp16"
    assert dp.bf16 is False and dp.fp16 is True


def test_force_cpu_overrides_available_gpu():
    cfg = _local_cfg()
    dp = b._resolve_device(cfg, _FakeTorch(available=True, bf16=True), force_cpu=True)
    assert dp.device_map == {"": "cpu"}
    assert dp.apply_cpu_throttle is True


def test_env_force_cpu_flag(monkeypatch):
    monkeypatch.setenv("MINDXTRAIN_FORCE_CPU", "1")
    assert b._env_force_cpu() is True
    monkeypatch.setenv("MINDXTRAIN_FORCE_CPU", "off")
    assert b._env_force_cpu() is False
    monkeypatch.delenv("MINDXTRAIN_FORCE_CPU", raising=False)
    assert b._env_force_cpu() is False


@pytest.mark.parametrize(
    ("per_device", "cap", "expected"),
    [(8, None, 8), (8, 2, 2), (1, 2, 1), (4, None, 4), (0, 2, 1)],
)
def test_capped_batch(per_device, cap, expected):
    assert b._capped_batch(per_device, cap) == expected


def test_dispatch_routes_trl_local(monkeypatch):
    """dispatch_training routes trl_local to run_trl_local."""
    from mindxtrain.train import dispatch

    called = {}

    def _fake_run_local(cfg, plan, out_dir):
        called["hit"] = True
        return out_dir / "checkpoint"

    monkeypatch.setattr(b, "run_trl_local", _fake_run_local)
    cfg = _local_cfg()
    from mindxtrain.autotune.benchmark import run_autotune

    dispatch.dispatch_training(cfg, run_autotune(dry_run=True), Path("/tmp/x"))
    assert called.get("hit") is True
