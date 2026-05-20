"""Hands-free CPU autostart, gated on the mindX SEA agent's decision.

`autostart_cpu_training` has two gates: `MINDXTRAIN_AUTOSTART` arms
autonomous mode, and the StrategicEvolutionAgent's decision file is the
actual decider. Tests patch `coach.api._SPAWN` with a fake so the launch
path runs without a subprocess, and point `MINDXTRAIN_SEA_DECISION` at a
temp file so no real mindX install is touched.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator import runs as _runs
from mindxtrain.operator.app import app
from mindxtrain.operator.coach import api as coach_api
from mindxtrain.operator.coach import run_metrics as rm

client = TestClient(app)


@pytest.fixture(autouse=True)
def _restore(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the real spawn fn + clear all autostart/SEA env each test."""
    original = coach_api._SPAWN
    for var in (
        "MINDXTRAIN_AUTOSTART",
        "MINDXTRAIN_AUTOSTART_RECIPE",
        "MINDXTRAIN_SEA_DECISION",
        "MINDXTRAIN_MINDX_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    coach_api._SPAWN = original
    # Drain any runs the fake spawn left "running" — otherwise the
    # idempotency check in autostart_cpu_training trips on stale runs
    # carried into the next test by the module-global registry.
    for run in coach_api._REGISTRY.list_runs():
        if run.status in ("pending", "running"):
            coach_api._REGISTRY.publish(
                run.id,
                _runs.StatusEvent(
                    run_id=run.id, status="cancelled", message="test teardown",
                ),
            )


def _fake_spawn(run: _runs.Run, _cfg: Any, _plan: Any) -> None:
    """Stand-in spawn — marks the run running without a subprocess."""
    coach_api._REGISTRY.publish(
        run.id,
        _runs.StatusEvent(run_id=run.id, status="running", message="fake"),
    )


def _write_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **fields: Any,
) -> Path:
    """Write a SEA decision file and point the operator at it."""
    record: dict[str, Any] = {
        "recommend": True,
        "recipe": None,
        "reason": "test recommendation",
        "decided_by": "strategic_evolution_agent",
        "ts": time.time(),
        "ttl_s": 3600,
    }
    record.update(fields)
    path = tmp_path / "training_recommendation.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("MINDXTRAIN_SEA_DECISION", str(path))
    return path


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


# ---- SEA decision gate ---------------------------------------------------


def test_sea_gate_closed_when_no_decision_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINDXTRAIN_SEA_DECISION", str(tmp_path / "missing.json"))
    gate = coach_api.sea_training_gate()
    assert gate["open"] is False
    assert gate["available"] is False


def test_sea_gate_open_on_fresh_recommendation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_decision(tmp_path, monkeypatch, recommend=True, reason="corpus grew")
    gate = coach_api.sea_training_gate()
    assert gate["open"] is True
    assert "corpus grew" in gate["reason"]


def test_sea_gate_closed_when_sea_says_no(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_decision(tmp_path, monkeypatch, recommend=False, reason="loss still high")
    gate = coach_api.sea_training_gate()
    assert gate["open"] is False
    assert gate["available"] is True
    assert "loss still high" in gate["reason"]


def test_sea_gate_closed_when_recommendation_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_decision(
        tmp_path, monkeypatch,
        recommend=True, ts=time.time() - 10_000, ttl_s=60,
    )
    gate = coach_api.sea_training_gate()
    assert gate["open"] is False
    assert "stale" in gate["reason"]


def test_sea_gate_handles_corrupt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    bad = tmp_path / "training_recommendation.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("MINDXTRAIN_SEA_DECISION", str(bad))
    gate = coach_api.sea_training_gate()
    assert gate["open"] is False  # unreadable → treated as no decision


# ---- autostart_cpu_training behaviour ------------------------------------


@pytest.mark.asyncio
async def test_autostart_noop_when_disabled() -> None:
    """No env flag → returns None, launches nothing."""
    assert await coach_api.autostart_cpu_training() is None


@pytest.mark.asyncio
async def test_autostart_noop_when_sea_gate_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Autostart armed but SEA has not recommended → no launch."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    monkeypatch.setenv("MINDXTRAIN_SEA_DECISION", str(tmp_path / "missing.json"))
    coach_api._SPAWN = _fake_spawn
    assert await coach_api.autostart_cpu_training() is None


@pytest.mark.asyncio
async def test_autostart_noop_when_sea_declines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """SEA decision present but recommend=false → no launch."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    _write_decision(tmp_path, monkeypatch, recommend=False)
    coach_api._SPAWN = _fake_spawn
    assert await coach_api.autostart_cpu_training() is None


@pytest.mark.asyncio
async def test_autostart_launches_when_sea_recommends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Both gates open → the smoke recipe is launched and a Run comes back."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    _write_decision(tmp_path, monkeypatch, recommend=True)
    coach_api._SPAWN = _fake_spawn
    try:
        run = await coach_api.autostart_cpu_training()
        assert run is not None
        assert run.recipe == "mindx_fallback_qwen3_1_5b_cpu_smoke"
        assert run.status == "running"
    finally:
        await _cleanup_samplers()


@pytest.mark.asyncio
async def test_autostart_uses_sea_chosen_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """SEA's `recipe` field overrides the MINDXTRAIN_AUTOSTART_RECIPE default."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    _write_decision(tmp_path, monkeypatch, recommend=True, recipe="qwen3_8b_sft_lora")
    coach_api._SPAWN = _fake_spawn
    try:
        run = await coach_api.autostart_cpu_training()
        assert run is not None
        assert run.recipe == "qwen3_8b_sft_lora"
    finally:
        await _cleanup_samplers()


@pytest.mark.asyncio
async def test_autostart_idempotent_when_run_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A run already pending/running → autostart skips (no stacked trainers)."""
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    _write_decision(tmp_path, monkeypatch, recommend=True)
    coach_api._SPAWN = _fake_spawn
    existing = coach_api._REGISTRY.create("qwen3_8b_sft_lora", tmp_path / "r")
    assert existing.status == "pending"  # counts as active
    assert await coach_api.autostart_cpu_training() is None


# ---- /coach/api/sea-decision endpoint ------------------------------------


def test_sea_decision_endpoint_reports_closed_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINDXTRAIN_SEA_DECISION", str(tmp_path / "missing.json"))
    r = client.get("/coach/api/sea-decision")
    assert r.status_code == 200
    body = r.json()
    assert body["open"] is False
    assert body["autostart_enabled"] is False
    assert "decision_path" in body


def test_sea_decision_endpoint_reports_open_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINDXTRAIN_AUTOSTART", "1")
    _write_decision(tmp_path, monkeypatch, recommend=True, reason="dream corpus +340")
    r = client.get("/coach/api/sea-decision")
    body = r.json()
    assert body["open"] is True
    assert body["autostart_enabled"] is True
    assert "dream corpus +340" in body["reason"]
    assert body["decision"]["decided_by"] == "strategic_evolution_agent"
