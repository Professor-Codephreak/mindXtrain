"""Public /v1/training/jobs API.

Tests the versioned facade — including auth, recipe-vs-config-yaml-vs-config
input modes, list/get, and that the SSE events endpoint stays well-formed.

The spawner is monkey-patched so no real training runs; we verify the
request → registry → response shape.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator import runs as _runs
from mindxtrain.operator import training_api
from mindxtrain.operator.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test starts with a fresh registry and no API key requirement."""
    _runs.reset_default_registry()
    # Re-bind the training_api module's singleton to the new registry.
    monkeypatch.setattr(training_api, "_REGISTRY", _runs.default_registry())
    monkeypatch.delenv("MINDXTRAIN_API_KEY", raising=False)
    yield


def _no_op_spawn(run, cfg, plan):
    """Replace the real launcher; the registry still records the run."""
    return


def test_create_job_from_recipe_returns_job_id(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "job_id" in data
    assert data["recipe"] == "mindx_fallback_qwen3_1_5b_cpu_smoke"
    assert data["base_model"] == "HuggingFaceTB/SmolLM2-135M"
    assert data["backend"] == "trl_cpu"
    assert data["status"] in {"pending", "running"}


def test_unknown_recipe_404(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    r = client.post("/v1/training/jobs", json={"recipe": "does_not_exist"})
    assert r.status_code == 404


def test_exactly_one_source_required(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke", "config_yaml": "x: y"},
    )
    assert r.status_code == 422


def test_get_job_after_create(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    r = client.post("/v1/training/jobs", json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"})
    job_id = r.json()["job_id"]
    r2 = client.get(f"/v1/training/jobs/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id


def test_unknown_job_404():
    r = client.get("/v1/training/jobs/bogus-id")
    assert r.status_code == 404


def test_list_jobs(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    client.post("/v1/training/jobs", json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"})
    client.post("/v1/training/jobs", json={"recipe": "instella_3b_lora"})
    r = client.get("/v1/training/jobs")
    assert r.status_code == 200
    items = r.json()
    recipes_in_response = {item["recipe"] for item in items}
    assert "mindx_fallback_qwen3_1_5b_cpu_smoke" in recipes_in_response
    assert "instella_3b_lora" in recipes_in_response


def test_bearer_auth_required_when_key_set(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    monkeypatch.setenv("MINDXTRAIN_API_KEY", "secret-key")
    r = client.post("/v1/training/jobs", json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"})
    assert r.status_code == 401


def test_bearer_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    monkeypatch.setenv("MINDXTRAIN_API_KEY", "secret-key")
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert r.status_code == 200


def test_bearer_auth_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    monkeypatch.setenv("MINDXTRAIN_API_KEY", "secret-key")
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401


def test_bearer_auth_open_when_key_unset(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    assert os.environ.get("MINDXTRAIN_API_KEY", "") == ""
    r = client.post("/v1/training/jobs", json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"})
    assert r.status_code == 200


def test_config_yaml_input(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    from mindxtrain.config.loader import render_recipe
    cfg_yaml = render_recipe("mindx_fallback_qwen3_1_5b_cpu_smoke")
    r = client.post("/v1/training/jobs", json={"config_yaml": cfg_yaml})
    assert r.status_code == 200, r.text
    assert r.json()["recipe"].startswith("adhoc:")


def test_x402_gate_returns_402_when_unpaid(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    monkeypatch.setenv("MINDXTRAIN_X402_REQUIRED", "1")
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"},
    )
    assert r.status_code == 402, r.text
    detail = r.json()["detail"]
    assert detail["error"] == "payment required"
    assert detail["invoice"]["amount_usdc"] > 0
    assert detail["invoice"]["asset_id"] == 203977300


def test_x402_gate_off_by_default(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    assert os.environ.get("MINDXTRAIN_X402_REQUIRED", "") == ""
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"},
    )
    assert r.status_code == 200


def test_x402_gate_validates_settlement_when_tx_present(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    monkeypatch.setenv("MINDXTRAIN_X402_REQUIRED", "1")

    # Stub the on-chain validation (real path needs --extra chain + a live tx).
    from mindxtrain.provenance import x402

    def _fake_validate(tx_id, **_kw):
        return x402.Settlement(tx_id=tx_id, amount_usdc=1.0, confirmed=True)

    monkeypatch.setattr(x402, "validate_settlement", _fake_validate)
    r = client.post(
        "/v1/training/jobs",
        json={
            "recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke",
            "settlement_tx": "FAKE_TX_ID",
        },
    )
    assert r.status_code == 200, r.text


def test_trl_local_routes_to_inprocess(monkeypatch):
    """A trl_local recipe takes the in-process spawn branch (like trl_cpu)."""
    seen = {}

    def _fake_inprocess(run, cfg, plan):
        seen["backend"] = cfg.train.backend

    monkeypatch.setattr(training_api, "_spawn_inprocess_cpu", _fake_inprocess)
    r = client.post(
        "/v1/training/jobs",
        json={"recipe": "mindx_fallback_qwen3_1_5b_local"},
    )
    assert r.status_code == 200, r.text
    assert seen.get("backend") == "trl_local"


def test_cancel_job(monkeypatch):
    monkeypatch.setattr(training_api, "_spawn_for_backend", _no_op_spawn)
    r = client.post("/v1/training/jobs", json={"recipe": "mindx_fallback_qwen3_1_5b_cpu_smoke"})
    job_id = r.json()["job_id"]
    r2 = client.post(f"/v1/training/jobs/{job_id}/cancel")
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id
