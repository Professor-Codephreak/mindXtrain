"""Ollama backend + auto-detect resolution.

The ollama backend is a thin alias of OpenAICompatBackend. We exercise
both the backend's own defaults and the operator's auto-detect path
(which probes localhost:11434 and falls back to vllm).
"""
from __future__ import annotations

import httpx

from mindxtrain.operator import app as operator_app
from mindxtrain.operator.backends.ollama import OllamaBackend

# ---- OllamaBackend instantiation ----------------------------------------


def test_ollama_default_base_url(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_OLLAMA_BASE_URL", raising=False)
    b = OllamaBackend()
    assert b.base_url == "http://localhost:11434/v1"
    assert b.name == "ollama"


def test_ollama_env_override(monkeypatch):
    monkeypatch.setenv("MINDXTRAIN_OLLAMA_BASE_URL", "http://10.0.0.5:9999/v1")
    b = OllamaBackend()
    assert b.base_url == "http://10.0.0.5:9999/v1"


def test_ollama_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("MINDXTRAIN_OLLAMA_BASE_URL", "http://wrong:1/v1")
    b = OllamaBackend(base_url="http://right:2/v1")
    assert b.base_url == "http://right:2/v1"


def test_ollama_is_registered_in_factory():
    """build_backend('ollama', ...) must construct an OllamaBackend."""
    from mindxtrain.models.registry import build_backend
    b = build_backend("ollama")
    assert isinstance(b, OllamaBackend)


# ---- resolve_backend_name auto-detect logic ------------------------------


def test_resolve_explicit_env_wins(monkeypatch):
    """MINDXTRAIN_BACKEND overrides the probe entirely."""
    monkeypatch.setenv("MINDXTRAIN_BACKEND", "openai_compat")
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    # Even if ollama is reachable, the explicit env wins.
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)
    assert operator_app.resolve_backend_name() == "openai_compat"


def test_resolve_falls_back_to_legacy_automindx(monkeypatch):
    """AUTOMINDX_BACKEND back-compat — older deployments still work."""
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.setenv("AUTOMINDX_BACKEND", "vllm")
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)
    assert operator_app.resolve_backend_name() == "vllm"


def test_resolve_autodetect_picks_ollama_when_reachable(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)
    assert operator_app.resolve_backend_name() == "ollama"


def test_resolve_autodetect_falls_through_to_vllm(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: False)
    assert operator_app.resolve_backend_name() == "vllm"


# ---- _ollama_reachable probe --------------------------------------------


def test_ollama_reachable_true_on_200(monkeypatch):
    """A 200 response on /api/tags counts as reachable."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"models": []}),
    )
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app._ollama_reachable() is True


def test_ollama_reachable_false_on_connection_error(monkeypatch):
    """When ollama isn't running the connection should fail cleanly."""
    def _raise(*a, **kw):
        raise httpx.ConnectError("connection refused")
    transport = httpx.MockTransport(_raise)
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app._ollama_reachable() is False


def test_ollama_first_model_returns_local_model(monkeypatch):
    """Prefer non-cloud models in the listing."""
    body = {
        "models": [
            {"name": "glm-5.1:cloud"},
            {"name": "qwen3:0.6b"},
            {"name": "deepseek-v4-pro:cloud"},
        ],
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app.ollama_first_model() == "qwen3:0.6b"


def test_ollama_first_model_returns_none_on_failure(monkeypatch):
    """Probe failure → None (UI degrades gracefully without a model name)."""
    def _raise(*a, **kw):
        raise httpx.ConnectError("nope")
    transport = httpx.MockTransport(_raise)
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    assert operator_app.ollama_first_model() is None


# ---- Coach /api/health surfaces detected backend + model -----------------


def test_health_endpoint_reports_ollama_when_detected(monkeypatch):
    """The Chat card flips from '(no backend configured)' to live when
    ollama is reachable on the loopback."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(operator_app, "ollama_first_model", lambda: "qwen3:0.6b")

    client = TestClient(operator_app.app)
    body = client.get("/coach/api/health").json()
    assert body["chat_backend_name"] == "ollama"
    assert body["chat_backend_ready"] is True
    assert body["chat_backend_model"] == "qwen3:0.6b"


def test_health_endpoint_reports_vllm_fallthrough(monkeypatch):
    """When ollama isn't reachable and no explicit backend is set, the
    Coach health endpoint reports vllm (the production default)."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("MINDXTRAIN_BACKEND", raising=False)
    monkeypatch.delenv("AUTOMINDX_BACKEND", raising=False)
    monkeypatch.setattr(operator_app, "_ollama_reachable", lambda: False)

    client = TestClient(operator_app.app)
    body = client.get("/coach/api/health").json()
    assert body["chat_backend_name"] == "vllm"
    # No active vllm probe yet — readiness stays False, which is honest:
    # the dev laptop doesn't have vllm running anyway.
    assert body["chat_backend_ready"] is False
