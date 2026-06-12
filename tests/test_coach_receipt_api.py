"""Coach receipt endpoint: /coach/api/receipt/{run_id}."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator import runs as _runs
from mindxtrain.operator.app import app
from mindxtrain.operator.coach import api as coach_api
from mindxtrain.provenance import manifest as _m

client = TestClient(app)


@pytest.fixture(autouse=True)
def _offline_attestation(monkeypatch):
    monkeypatch.setattr(_m, "_fetch_time_attestation", lambda: _m.TimeAttestation())


def _canned_spawn(tmp_dir: Path):
    """A spawn that writes a checkpoint + verifiable receipt, then succeeds."""

    def _spawn(run: _runs.Run, cfg, plan) -> None:
        ckpt = Path(run.out_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
        from mindxtrain.operator.receipt_emit import emit_run_receipt

        emit_run_receipt(coach_api._REGISTRY, run, cfg, plan)
        coach_api._REGISTRY.publish_threadsafe(
            run.id,
            _runs.StatusEvent(run_id=run.id, status="succeeded", message="done"),
        )
        coach_api._REGISTRY.close_subscribers(run.id)

    return _spawn


def test_receipt_verified_after_run(tmp_path, monkeypatch):
    out_dir = tmp_path / "run"
    monkeypatch.setattr(coach_api, "_SPAWN", _canned_spawn(out_dir))

    launched = client.post(
        "/coach/api/runs/launch",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke", "out_dir": str(out_dir)},
    )
    assert launched.status_code == 200, launched.text
    run_id = launched.json()["id"]

    r = client.get(f"/coach/api/receipt/{run_id}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["verified"] is True
    assert data["checks"]["checkpoint"] is True
    assert data["checks"]["autotune_plan"] is True
    # CPU run has no dataset/eval artifacts → empty hashes, pass-by-default.
    assert data["checks"]["dataset"] is True
    assert data["hashes"]["autotune_plan"]
    assert data["hashes"]["checkpoint"]
    assert data["hashes"]["dataset"] == ""


def test_receipt_404_for_unknown_run():
    r = client.get("/coach/api/receipt/does-not-exist")
    assert r.status_code == 404


def test_receipt_409_when_no_manifest(tmp_path, monkeypatch):
    out_dir = tmp_path / "norun"

    def _spawn_no_receipt(run, cfg, plan) -> None:
        # Run "exists" in the registry but never produces a manifest.
        coach_api._REGISTRY.publish_threadsafe(
            run.id,
            _runs.StatusEvent(run_id=run.id, status="running", message="mid-flight"),
        )

    monkeypatch.setattr(coach_api, "_SPAWN", _spawn_no_receipt)
    launched = client.post(
        "/coach/api/runs/launch",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke", "out_dir": str(out_dir)},
    )
    assert launched.status_code == 200, launched.text
    run_id = launched.json()["id"]

    r = client.get(f"/coach/api/receipt/{run_id}")
    assert r.status_code == 409
