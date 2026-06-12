"""Coach Modelfile-builder endpoints + the separate builder page."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


def test_modelfile_page_served():
    r = client.get("/coach/modelfile")
    assert r.status_code == 200
    assert "Modelfile builder" in r.text
    assert "/coach/static/modelfile.js" in r.text


def test_modelfile_params_endpoint():
    r = client.get("/coach/api/modelfile/params")
    assert r.status_code == 200
    params = r.json()["parameters"]
    names = {p["name"] for p in params}
    assert {"num_ctx", "temperature", "top_k", "repeat_penalty"} <= names
    assert all("type" in p and "description" in p for p in params)


def test_modelfile_build_renders_text():
    r = client.post("/coach/api/modelfile/build", json={
        "from_model": "qwen3:0.6b",
        "system": "You are Codephreak.",
        "parameters": {"temperature": 0.7, "num_ctx": 4096},
        "stop": ["<|im_end|>"],
        "messages": [{"role": "user", "content": "who are you?"}],
    })
    assert r.status_code == 200, r.text
    mf = r.json()["modelfile"]
    assert mf.startswith("FROM qwen3:0.6b")
    assert "PARAMETER temperature 0.7" in mf
    assert "PARAMETER num_ctx 4096" in mf
    assert 'PARAMETER stop "<|im_end|>"' in mf
    assert 'SYSTEM """You are Codephreak."""' in mf
    assert "MESSAGE user who are you?" in mf


def test_modelfile_build_requires_from():
    r = client.post("/coach/api/modelfile/build", json={"system": "x"})
    assert r.status_code == 422


def test_modelfile_create_requires_tag():
    r = client.post("/coach/api/modelfile/create", json={"spec": {"from_model": "m"}})
    assert r.status_code == 422


def test_coach_index_links_modelfile_builder():
    r = client.get("/coach/")
    assert 'id="open-modelfile-btn"' in r.text
    js = client.get("/coach/static/coach.js").text
    assert "openModelfileBuilder" in js
    assert "/coach/modelfile" in js
