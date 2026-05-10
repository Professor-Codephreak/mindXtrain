"""Coach UI deploy endpoints — env validation, spawn shim, 409 concurrency, SSE.

Mirrors the existing _SPAWN injection pattern in `tests/test_runs_sse.py`:
the three deploy spawn shims are monkeypatched per-test so we never invoke
real ssh/rsync/git/gh/httpx.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator import runs as _runs
from mindxtrain.operator.app import app
from mindxtrain.operator.coach import api as coach_api

client = TestClient(app)

_GITHUB_ENV = {
    "GITHUB_TOKEN": "ghp_TEST",
    "GITHUB_REPO": "professor-codephreak/mindXtrain",
    "GITHUB_DEFAULT_BRANCH": "main",
}

_DROPLET_ENV = {
    "DROPLET_HOST": "mi300x.test",
    "DROPLET_USER": "root",
    "DROPLET_SSH_KEY": "/dev/null",
    "DROPLET_REMOTE_PATH": "/workspace/mindxtrain",
    "DROPLET_CONTAINER": "rocm/primus:v26.2",
}

_AMD_DC_ENV = {
    "AMD_DEV_CLOUD_TOKEN": "dop_v1_TEST",
    "AMD_DEV_CLOUD_SSH_KEY_ID": "56216059",
    "AMD_DEV_CLOUD_REGION": "atl1",
    "AMD_DEV_CLOUD_SIZE": "gpu-mi300x8-1536gb-devcloud",
    "AMD_DEV_CLOUD_IMAGE": "vllm-0-17-1",
}


@pytest.fixture(autouse=True)
def _restore_deploy_spawns() -> Iterator[None]:
    g = coach_api._GITHUB_SPAWN
    s = coach_api._DROPLET_SYNC_SPAWN
    p = coach_api._DROPLET_PROVISION_SPAWN
    yield
    coach_api._GITHUB_SPAWN = g
    coach_api._DROPLET_SYNC_SPAWN = s
    coach_api._DROPLET_PROVISION_SPAWN = p


@pytest.fixture(autouse=True)
def _reset_registry() -> Iterator[None]:
    """Ensure no in-flight runs leak between tests (concurrency tests need this)."""
    yield
    # Mark every busy run as terminal so the next test sees a clean slate.
    for run in coach_api._REGISTRY.list_runs():
        if run.status in ("pending", "running"):
            coach_api._REGISTRY.publish(
                run.id,
                _runs.StatusEvent(run_id=run.id, status="cancelled", message="test teardown"),
            )
            coach_api._REGISTRY.close_subscribers(run.id)


def _parse_sse(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        kind = ""
        data = ""
        for ln in frame.splitlines():
            if ln.startswith("event: "):
                kind = ln[7:]
            elif ln.startswith("data: "):
                data = ln[6:]
        if kind and data:
            out.append({"event": kind, "data": json.loads(data)})
    return out


# ---- /api/github/status --------------------------------------------------


def test_github_status_unconfigured_lists_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    r = client.get("/coach/api/github/status")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is False
    assert "GITHUB_TOKEN" in body["missing"]
    assert "GITHUB_REPO" in body["missing"]


def test_github_status_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _GITHUB_ENV.items():
        monkeypatch.setenv(k, v)
    # Force "binaries present" by monkeypatching shutil.which for predictable CI.
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")
    r = client.get("/coach/api/github/status")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is True
    assert body["missing"] == []
    assert body["target"] == "professor-codephreak/mindXtrain"


# ---- /api/github/push ----------------------------------------------------


def test_github_push_503_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    r = client.post("/coach/api/github/push", json={})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "missing" in detail
    assert "GITHUB_TOKEN" in detail["missing"]


def test_github_push_200_with_fake_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _GITHUB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")

    captured: dict[str, str] = {}

    def _fake(run: _runs.Run, req: coach_api.GithubPushRequest) -> None:
        captured["run_id"] = run.id
        captured["msg"] = req.commit_message
        coach_api._REGISTRY.publish(
            run.id,
            _runs.LogEvent(run_id=run.id, line="=== step 1/11: probe-git ==="),
        )
        coach_api._REGISTRY.publish(
            run.id,
            _runs.StatusEvent(run_id=run.id, status="succeeded", message="rc=0"),
        )

    coach_api._GITHUB_SPAWN = _fake
    r = client.post("/coach/api/github/push", json={"commit_message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["recipe"] == "_github_push"
    assert body["id"] == captured["run_id"]
    assert captured["msg"] == "hello"


# ---- /api/droplet/status ------------------------------------------------


def test_droplet_status_returns_both_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _DROPLET_ENV.items():
        monkeypatch.setenv(k, v)
    for k, v in _AMD_DC_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")
    r = client.get("/coach/api/droplet/status")
    assert r.status_code == 200
    body = r.json()
    assert "sync" in body and "provision" in body
    assert body["sync"]["configured"] is True
    assert body["provision"]["configured"] is True
    assert "mi300x.test" in body["sync"]["target"]
    assert body["provision"]["target"] == "amd-dev-cloud:atl1:gpu-mi300x8-1536gb-devcloud"


def test_droplet_status_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (*_DROPLET_ENV, *_AMD_DC_ENV):
        monkeypatch.delenv(k, raising=False)
    r = client.get("/coach/api/droplet/status")
    assert r.status_code == 200
    body = r.json()
    assert body["sync"]["configured"] is False
    assert body["provision"]["configured"] is False
    assert "DROPLET_HOST" in body["sync"]["missing"]
    assert "AMD_DEV_CLOUD_TOKEN" in body["provision"]["missing"]


# ---- /api/droplet/sync --------------------------------------------------


def test_droplet_sync_503_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _DROPLET_ENV:
        monkeypatch.delenv(k, raising=False)
    r = client.post("/coach/api/droplet/sync", json={})
    assert r.status_code == 503
    assert "DROPLET_HOST" in r.json()["detail"]["missing"]


def test_droplet_sync_409_when_provision_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {**_DROPLET_ENV, **_AMD_DC_ENV}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")

    # Spawn a provision that stays in-flight.
    def _hang_provision(run: _runs.Run, _req: coach_api.DropletProvisionRequest) -> None:
        coach_api._REGISTRY.publish(
            run.id, _runs.StatusEvent(run_id=run.id, status="running", message="pretending")
        )

    coach_api._DROPLET_PROVISION_SPAWN = _hang_provision
    r1 = client.post("/coach/api/droplet/provision", json={})
    assert r1.status_code == 200, r1.text

    # Now a sync should 409.
    r2 = client.post("/coach/api/droplet/sync", json={})
    assert r2.status_code == 409
    detail = r2.json()["detail"]
    assert detail["active_recipe"] == "_droplet_provision"


def test_droplet_sync_200_with_fake_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _DROPLET_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")

    def _fake(run: _runs.Run, req: coach_api.DropletSyncRequest) -> None:
        assert req.run_bench is True
        coach_api._REGISTRY.publish(
            run.id, _runs.LogEvent(run_id=run.id, line="=== step 1/4: rsync ===")
        )
        coach_api._REGISTRY.publish(
            run.id, _runs.StatusEvent(run_id=run.id, status="succeeded", message="rc=0")
        )

    coach_api._DROPLET_SYNC_SPAWN = _fake
    r = client.post("/coach/api/droplet/sync", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["recipe"] == "_droplet_sync"


# ---- /api/droplet/provision ---------------------------------------------


def test_droplet_provision_503_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _AMD_DC_ENV:
        monkeypatch.delenv(k, raising=False)
    r = client.post("/coach/api/droplet/provision", json={})
    assert r.status_code == 503
    assert "AMD_DEV_CLOUD_TOKEN" in r.json()["detail"]["missing"]


def test_droplet_provision_passes_request_through_to_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _AMD_DC_ENV.items():
        monkeypatch.setenv(k, v)

    captured: dict[str, str] = {}

    def _fake(run: _runs.Run, req: coach_api.DropletProvisionRequest) -> None:
        captured["repo"] = req.repo or "default"
        captured["wait"] = str(req.wait_for_bootstrap)
        coach_api._REGISTRY.publish(
            run.id, _runs.StatusEvent(run_id=run.id, status="succeeded", message="rc=0")
        )

    coach_api._DROPLET_PROVISION_SPAWN = _fake
    r = client.post("/coach/api/droplet/provision", json={"repo": "owner/repo", "wait_for_bootstrap": False})
    assert r.status_code == 200
    assert captured["repo"] == "owner/repo"
    assert captured["wait"] == "False"


# ---- SSE replay over the deploy run -------------------------------------


def test_deploy_run_sse_replays_log_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _GITHUB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")

    def _fake(run: _runs.Run, _req: coach_api.GithubPushRequest) -> None:
        coach_api._REGISTRY.publish(
            run.id, _runs.LogEvent(run_id=run.id, line="=== step 1/11 ===")
        )
        coach_api._REGISTRY.publish(
            run.id, _runs.LogEvent(run_id=run.id, line="initialized empty git repo")
        )
        coach_api._REGISTRY.publish(
            run.id, _runs.StatusEvent(run_id=run.id, status="succeeded", message="rc=0")
        )

    coach_api._GITHUB_SPAWN = _fake
    r = client.post("/coach/api/github/push", json={})
    run_id = r.json()["id"]

    # Give the registry a moment for the synchronous fake to publish.
    time.sleep(0.05)

    es = client.get(f"/coach/api/runs/{run_id}/events", headers={"accept": "text/event-stream"})
    assert es.status_code == 200
    events = _parse_sse(es.text)
    kinds = [e["event"] for e in events]
    assert "log" in kinds
    assert "status" in kinds
    final = [e for e in events if e["event"] == "status"][-1]
    assert final["data"]["status"] == "succeeded"


# ---- /api/runs/{id}/cancel works on synthetic runs ----------------------


def test_cancel_endpoint_works_on_deploy_run(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _GITHUB_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")

    def _fake(run: _runs.Run, _req: coach_api.GithubPushRequest) -> None:
        coach_api._REGISTRY.publish(
            run.id, _runs.StatusEvent(run_id=run.id, status="running", message="busy")
        )

    coach_api._GITHUB_SPAWN = _fake
    r = client.post("/coach/api/github/push", json={})
    run_id = r.json()["id"]

    c = client.post(f"/coach/api/runs/{run_id}/cancel")
    # No real subprocess means cancel returns False, but the endpoint still 200s.
    assert c.status_code == 200
    assert "cancelled" in c.json()
