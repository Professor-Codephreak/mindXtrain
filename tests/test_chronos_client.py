"""Coach chronos_client + the three /coach/api/diagnostics/* endpoints
it powers. Tests stay independent of a running mindX backend by
patching httpx.AsyncClient via MockTransport — same approach used by
test_ollama_backend.py.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator import app as operator_app
from mindxtrain.operator.coach import chronos_client

client = TestClient(operator_app.app)


# ---- chronos_client direct ----------------------------------------------


@pytest.mark.asyncio
async def test_now_returns_promised_time_shape(monkeypatch):
    body = {
        "unix_18dp": "1779169999.123456789",
        "utc": "2026-05-19T05:53:19.123456789+00:00",
        "consensus": "correlated",
        "confidence_ms": 42.0,
        "sources": {"cpu": {"unix": 1779169999.0}},
        "anchor_count_24h": 12,
        "promised_by": "chronos.agent",
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    pt = await chronos_client.now()
    assert pt["consensus"] == "correlated"
    assert pt["confidence_ms"] == 42.0
    assert pt["anchor_count_24h"] == 12
    assert pt["promised_by"] == "chronos.agent"


@pytest.mark.asyncio
async def test_now_degrades_on_connection_error(monkeypatch):
    def _raise(*_a, **_kw):
        raise httpx.ConnectError("mindX down")
    transport = httpx.MockTransport(_raise)
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    pt = await chronos_client.now()
    # The contract: same keys present, consensus flips to "unavailable".
    assert pt["consensus"] == "unavailable"
    assert pt["confidence_ms"] == 0.0
    assert pt["anchor_count_24h"] == 0
    assert pt["promised_by"] == "chronos.agent"
    assert "error" in pt["sources"]


@pytest.mark.asyncio
async def test_now_fills_missing_keys_from_partial_response(monkeypatch):
    """Defensive against upstream shape drift — caller never KeyErrors."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"unix_18dp": "1.0"}),
    )
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    pt = await chronos_client.now()
    assert pt["unix_18dp"] == "1.0"
    assert pt["consensus"] == "unavailable"  # default fill
    assert pt["anchor_count_24h"] == 0
    assert pt["sources"] == {}


@pytest.mark.asyncio
async def test_anchors_returns_list_and_n(monkeypatch):
    body = {"anchors": [{"id": 1, "chain": "algorand"}], "n": 1}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    out = await chronos_client.anchors(limit=10)
    assert out["n"] == 1
    assert out["anchors"][0]["chain"] == "algorand"


@pytest.mark.asyncio
async def test_anchors_returns_empty_on_failure(monkeypatch):
    def _raise(*_a, **_kw):
        raise httpx.ConnectError("nope")
    transport = httpx.MockTransport(_raise)
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    out = await chronos_client.anchors(limit=10)
    assert out["n"] == 0
    assert out["anchors"] == []
    assert "error" in out


@pytest.mark.asyncio
async def test_drift_returns_history_shape(monkeypatch):
    body = {
        "hours": 24, "bucket_count": 2,
        "buckets": [{"ts_unix": 1000, "n": 3, "drift_mean_ms": 5.0, "drift_std_ms": 1.0}],
        "drift_std_ms": 1.5, "drift_max_abs_ms": 8.0, "anchor_count": 5,
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    out = await chronos_client.drift(hours=24)
    assert out["bucket_count"] == 2
    assert out["anchor_count"] == 5


@pytest.mark.asyncio
async def test_base_url_env_override(monkeypatch):
    """MINDX_BASE_URL must shape the request URL."""
    monkeypatch.setenv("MINDX_BASE_URL", "http://example.test:9999")
    seen: dict[str, str] = {}

    def _handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"hours": 24, "buckets": []})

    transport = httpx.MockTransport(_handler)
    orig = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    await chronos_client.drift()
    assert "example.test:9999" in seen["url"]


# ---- Coach endpoint integration -----------------------------------------


def test_coach_chronos_endpoint_merges_three_calls(monkeypatch):
    """/coach/api/diagnostics/chronos aggregates time + anchors + drift."""

    async def _fake_now(**_kw):
        return {"consensus": "correlated", "confidence_ms": 1.0, "promised_by": "x"}

    async def _fake_anchors(**_kw):
        return {"anchors": [{"id": 1}], "n": 1}

    async def _fake_drift(**_kw):
        return {"hours": 24, "anchor_count": 1, "buckets": []}

    monkeypatch.setattr(chronos_client, "now", _fake_now)
    monkeypatch.setattr(chronos_client, "anchors", _fake_anchors)
    monkeypatch.setattr(chronos_client, "drift", _fake_drift)

    r = client.get("/coach/api/diagnostics/chronos")
    assert r.status_code == 200
    body = r.json()
    assert body["promised_time"]["consensus"] == "correlated"
    assert body["anchor_count"] == 1
    assert body["drift_history"]["anchor_count"] == 1


def test_measurement_confidence_endpoint_returns_band():
    """psutil vs ps cross-check endpoint — should classify on a live host."""
    r = client.get("/coach/api/diagnostics/measurement-confidence")
    assert r.status_code == 200
    body = r.json()
    # Band is one of the four known categories.
    assert body["confidence_band"] in {"tight", "loose", "divergent", "unknown"}
    if body["ok"]:
        # Deltas are non-negative absolute values.
        assert body["cpu_delta_pp"] >= 0
        assert body["rss_delta_mb"] >= 0


def test_cli_samplers_endpoint_returns_sampler_records():
    """When mindX cli_time_samplers.py is reachable, all 6 records come through."""
    r = client.get("/coach/api/diagnostics/cli-samplers")
    assert r.status_code == 200
    body = r.json()
    # Either ok=True (mindX checkout present) or ok=False with a reason.
    assert "ok" in body and "samplers" in body
    if body["ok"]:
        # Six samplers expected.
        assert set(body["samplers"]) == {
            "nano_wall", "ntp_status", "uptime_load",
            "proc_snapshot", "cpu_jiffies", "monotonic_boot",
        }
