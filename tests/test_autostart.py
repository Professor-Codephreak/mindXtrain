"""Hands-free CPU autostart — operator launches a run at boot, no button.

`autostart_cpu_training` is gated on MINDXTRAIN_AUTOSTART so test clients
and CI never spawn a real trainer. These tests patch `coach.api._SPAWN`
with a fake so the launch path runs without a subprocess.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mindxtrain.operator import runs as _runs
from mindxtrain.operator.coach import api as coach_api
from mindxtrain.operator.coach import run_metrics as rm


@pytest.fixture(autouse=True)
def _restore(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the real spawn fn + clear autostart env after each test."""
    original = coach_api._SPAWN
    monkeypatch.delenv("MINDXTRAIN_AUTOSTART", raising=False)
    monkeypatch.delenv("MINDXTRAIN_AUTOSTART_RECIPE", raising=False)
    yield
    coach_api._SPAWN = original


def _fake_spawn(run: _runs.Run, _cfg: Any, _plan: Any) -> None:
    """Stand-in spawn — marks the run running without a subprocess."""
    coach_api._REGISTRY.publish(
        run.id,
        _runs.StatusEvent(run_id=run.id, status="running", message="fake"),
    )


async def _cleanup_samplers() -> None:
    """Stop any per-run metrics samplers the launch path started."""
    for tid in list(rm._TASKS):
        await rm.stop_metrics_sampler(tid)


# ---- env-driven gates ----------------------------------------------------


def test_autostart_disabled_by_default() -> None:
    assert coach_api.autostart_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_autostart_enabled_recognises_truthy(
    monkeypatch: pytest.MonkeyPatch, val: str,
) -> None:
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", val)
    assert coach_api.autostart_enabled() is True


def test_autostart_recipe_defaults_to_smoke() -> None:
    assert coach_api.autostart_recipe() == "mindx_fallback_qwen3_1_5b_cpu_smoke"


def test_autostart_recipe_honours_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART_RECIPE", "qwen3_8b_sft_lora")
    assert coach_api.autostart_recipe() == "qwen3_8b_sft_lora"


# ---- autostart_cpu_training behaviour ------------------------------------


@pytest.mark.asyncio
async def test_autostart_noop_when_disabled() -> None:
    """No env flag → returns None, launches nothing."""
    assert await coach_api.autostart_cpu_training() is None


@pytest.mark.asyncio
async def test_autostart_launches_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Flag set → the smoke recipe is launched and a Run comes back."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    coach_api._SPAWN = _fake_spawn
    try:
        run = await coach_api.autostart_cpu_training()
        assert run is not None
        assert run.recipe == "mindx_fallback_qwen3_1_5b_cpu_smoke"
        assert run.status == "running"
    finally:
        await _cleanup_samplers()


@pytest.mark.asyncio
async def test_autostart_idempotent_when_run_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A run already pending/running → autostart skips (no stacked trainers)."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    coach_api._SPAWN = _fake_spawn
    # Seed an already-active run directly in the registry.
    existing = coach_api._REGISTRY.create("qwen3_8b_sft_lora", tmp_path / "r")
    assert existing.status == "pending"  # counts as active
    run = await coach_api.autostart_cpu_training()
    assert run is None
