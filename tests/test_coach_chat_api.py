"""Coach streaming chat + ollama controls."""

from __future__ import annotations

import shutil
import subprocess

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app
from mindxtrain.operator.coach import api as coach_api

client = TestClient(app)


class _FakeBackend:
    async def stream_chat(self, req):
        async def _gen():
            for tok in ["i ", "am ", "codephreak."]:
                yield tok
        return _gen()


def test_chat_stream_relays_tokens(monkeypatch):
    monkeypatch.setattr(coach_api, "_resolve_chat_backend", lambda: _FakeBackend())
    with client.stream("POST", "/coach/api/chat/stream", json={
        "model": "qwen3:0.6b",
        "messages": [{"role": "user", "content": "who are you?"}],
    }) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert 'data: "i "' in body
    assert 'data: "codephreak."' in body
    assert "data: [DONE]" in body


def test_chat_stream_surfaces_backend_error(monkeypatch):
    class _BoomBackend:
        async def stream_chat(self, req):
            raise RuntimeError("backend down")

    monkeypatch.setattr(coach_api, "_resolve_chat_backend", lambda: _BoomBackend())
    with client.stream("POST", "/coach/api/chat/stream", json={
        "model": "m", "messages": [{"role": "user", "content": "x"}],
    }) as r:
        body = "".join(r.iter_text())
    assert "event: error" in body
    assert "data: [DONE]" in body


def test_models_endpoint_sorts_local_first(monkeypatch):
    # The models list must rank local models ahead of :cloud ones.
    import httpx

    class _Resp:
        status_code = 200
        def raise_for_status(self): return None
        def json(self):
            return {"data": [{"id": "glm-5.1:cloud"}, {"id": "qwen3:0.6b"}]}

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    models = client.get("/coach/api/models").json()["models"]
    assert models[0] == "qwen3:0.6b"  # local before cloud


def test_ollama_status_shape():
    r = client.get("/coach/api/ollama/status")
    assert r.status_code == 200
    d = r.json()
    assert set(d) >= {"reachable", "has_ollama_bin", "serve_pids", "base_url"}
    assert isinstance(d["serve_pids"], list)


def test_ollama_start_when_not_running(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/ollama")
    monkeypatch.setattr(coach_api, "_ollama_serve_pids", lambda: [])
    started = {}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.setdefault("ran", True))
    r = client.post("/coach/api/ollama/start")
    assert r.status_code == 200
    assert r.json()["started"] is True
    assert started.get("ran")


def test_ollama_start_already_running(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _x: "/usr/bin/ollama")
    monkeypatch.setattr(coach_api, "_ollama_serve_pids", lambda: [123])
    r = client.post("/coach/api/ollama/start")
    assert r.json()["started"] is False


def test_ollama_start_missing_binary(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _x: None)
    r = client.post("/coach/api/ollama/start")
    assert r.status_code == 422


def test_ollama_stop(monkeypatch):
    monkeypatch.setattr(coach_api, "_ollama_serve_pids", lambda: [123])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    r = client.post("/coach/api/ollama/stop")
    assert r.json()["stopped"] is True


def test_ollama_stop_when_not_running(monkeypatch):
    monkeypatch.setattr(coach_api, "_ollama_serve_pids", lambda: [])
    r = client.post("/coach/api/ollama/stop")
    assert r.json()["stopped"] is False


def test_chat_card_has_streaming_controls():
    html = client.get("/coach/").text
    assert 'id="chat-model"' in html
    assert 'id="chat-transcript"' in html
    assert 'id="ollama-start"' in html and 'id="ollama-stop"' in html
    js = client.get("/coach/static/coach.js").text
    assert "/coach/api/chat/stream" in js
    assert "loadChatModels" in js
    assert "refreshOllamaStatus" in js
