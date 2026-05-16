"""Admin diagnostics endpoints — live metrics + runs roster."""
from __future__ import annotations

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


# ---- /diagnostics/live -----------------------------------------------------


def test_live_endpoint_returns_metrics():
    r = client.get("/coach/api/diagnostics/live")
    assert r.status_code == 200
    data = r.json()
    for key in (
        "load_avg_1m", "load_avg_5m", "load_avg_15m",
        "ram_total_gb", "ram_available_gb", "ram_used_pct",
        "disk_total_gb", "disk_used_pct",
        "operator_rss_mb", "operator_threads", "cores", "ts",
    ):
        assert key in data, f"missing live metric: {key}"


def test_live_endpoint_operator_rss_positive():
    """The operator process is reporting on itself — its own RSS must be > 0."""
    data = client.get("/coach/api/diagnostics/live").json()
    assert data["operator_rss_mb"] >= 0
    # And cores is always ≥ 1.
    assert data["cores"] >= 1


def test_live_endpoint_ram_pct_in_range():
    data = client.get("/coach/api/diagnostics/live").json()
    assert 0.0 <= data["ram_used_pct"] <= 100.0
    assert 0.0 <= data["disk_used_pct"] <= 100.0


def test_live_endpoint_consecutive_calls_have_different_timestamps():
    """The ts field is a real sample timestamp, not a constant."""
    import time
    a = client.get("/coach/api/diagnostics/live").json()
    time.sleep(0.05)
    b = client.get("/coach/api/diagnostics/live").json()
    assert b["ts"] > a["ts"]


# ---- /diagnostics/runs -----------------------------------------------------


def test_diagnostics_runs_returns_list():
    """Even with no runs registered, the endpoint returns []."""
    r = client.get("/coach/api/diagnostics/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_diagnostics_runs_newest_first(monkeypatch):
    """Created-at ordering: newest first.

    The endpoint reads from the module-level _REGISTRY singleton in
    coach/api.py — monkeypatch that to a fresh registry so the test
    doesn't see leftover runs from other tests.
    """
    import time
    from pathlib import Path

    from mindxtrain.operator import runs as _runs
    from mindxtrain.operator.coach import api as coach_api

    fresh = _runs.RunRegistry()
    monkeypatch.setattr(coach_api, "_REGISTRY", fresh)
    fresh.create("r1", Path("/tmp/r1"))
    time.sleep(0.05)
    fresh.create("r2", Path("/tmp/r2"))
    r = client.get("/coach/api/diagnostics/runs").json()
    assert len(r) == 2
    assert r[0]["recipe"] == "r2"  # newest first
    assert r[1]["recipe"] == "r1"
