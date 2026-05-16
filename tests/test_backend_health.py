"""Backend reachability probes + /health + /readyz.

Production surface: `/health` always returns 200 with structured
`backend_ready`, while `/readyz` is the strict gate that fails closed
when inference isn't warm. Both consult the same generalized probes
(_ollama_reachable, _vllm_reachable) so they stay consistent.
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from mindxtrain.operator import app as operator_app

client = TestClient(operator_app.app)


# ---- _vllm_reachable / _vllm_first_model ---------------------------------


def test_vllm_reachable_true_on_200(monkeypatch):
    """vLLM exposes /v1/models — a 200 means inference is warm."""
    monkeypatch.delenv("MINDXTRAIN_VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("AUTOMINDX_VLLM_BASE_URL", raising=False)
    captured: dict[str, str] = {}

    def _handle(req):
        captured["url"] = str(req.url)
        return httpx.Response(
            200,
            json={"data": [{"id": "Qwen/Qwen3-8B", "object": "model"}]},
        )

    transport = httpx.MockTransport(_handle)
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app._vllm_reachable() is True
    # Default base URL is localhost:8000/v1, probe hits /v1/models.
    assert captured["url"].endswith("/v1/models")


def test_vllm_reachable_false_on_connection_error(monkeypatch):
    def _raise(*_a, **_kw):
        raise httpx.ConnectError("refused")
    transport = httpx.MockTransport(_raise)
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app._vllm_reachable() is False


def test_vllm_first_model_returns_id(monkeypatch):
    body = {"data": [{"id": "Qwen/Qwen3-8B"}, {"id": "fallback"}]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app._vllm_first_model() == "Qwen/Qwen3-8B"


def test_vllm_first_model_none_on_empty(monkeypatch):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"data": []}),
    )
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app._vllm_first_model() is None


# ---- backend_reachable / backend_first_model dispatch ---------------------


def test_backend_reachable_dispatches_to_ollama(monkeypatch):
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(operator_app, "_vllm_reachable", lambda: False)
    assert operator_app.backend_reachable("ollama") is True
    assert operator_app.backend_reachable("vllm") is False


def test_backend_reachable_unknown_returns_false(monkeypatch):
    """openai_compat and unknown names have no generic probe → False."""
    assert operator_app.backend_reachable("openai_compat") is False
    assert operator_app.backend_reachable("totally-made-up") is False


def test_backend_first_model_dispatches(monkeypatch):
    monkeypatch.setattr(operator_app, "ollama_first_model", lambda: "qwen3:0.6b")
    monkeypatch.setattr(operator_app, "_vllm_first_model", lambda: "Qwen/Qwen3-8B")
    assert operator_app.backend_first_model("ollama") == "qwen3:0.6b"
    assert operator_app.backend_first_model("vllm") == "Qwen/Qwen3-8B"
    assert operator_app.backend_first_model("openai_compat") is None


# ---- /health (always 200, structured readiness) --------------------------


def test_health_reports_backend_ready_when_reachable(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(operator_app, "ollama_first_model", lambda: "qwen3:0.6b")

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "ollama"
    assert body["backend_ready"] is True
    assert body["backend_model"] == "qwen3:0.6b"
    assert body["coach_url"] == "/coach/"


def test_health_stays_ok_when_backend_cold(monkeypatch):
    """A cold backend is *not* a process-liveness failure."""
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: False)
    monkeypatch.setattr(operator_app, "_vllm_reachable", lambda: False)

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"   # still alive
    assert body["backend_ready"] is False
    assert body["backend_model"] == ""


# ---- /readyz (strict gate) -----------------------------------------------


def test_readyz_returns_200_when_backend_warm(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)

    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["reachable"] is True


def test_readyz_returns_503_when_backend_cold(monkeypatch):
    """The whole point of /readyz: fail closed during inference outage."""
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: False)
    monkeypatch.setattr(operator_app, "_vllm_reachable", lambda: False)

    r = client.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["detail"]["reachable"] is False


# ---- /coach/api/health vllm path -----------------------------------------


def test_coach_health_reports_vllm_when_reachable(monkeypatch):
    """Production parity: vLLM-on-VPS flips chat card to live, same as ollama on laptop."""
    monkeypatch.setenv("MINDXTRAIN_BACKEND", "vllm")
    monkeypatch.setattr(operator_app, "_vllm_reachable", lambda: True)
    monkeypatch.setattr(operator_app, "_vllm_first_model", lambda: "Qwen/Qwen3-8B")

    r = client.get("/coach/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["chat_backend_name"] == "vllm"
    assert body["chat_backend_ready"] is True
    assert body["chat_backend_model"] == "Qwen/Qwen3-8B"
