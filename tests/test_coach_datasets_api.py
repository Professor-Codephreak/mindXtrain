"""Coach create-dataset (script) + imprint-score endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_datasets_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDXTRAIN_DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.delenv("MINDXTRAIN_PERSONA_PATH", raising=False)
    yield


def test_create_and_list_and_preview_script():
    body = {
        "name": "Codephreak Test 1",
        "persona_name": "Codephreak",
        "system_prompt": "You are Codephreak.",
        "voice_examples": ["augmentic intelligence."],
        "exchanges": [
            {"user": "who are you?", "assistant": "i am codephreak."},
            {"user": "what do you do?", "assistant": "i orchestrate agents."},
        ],
        "seed_voice": True,
    }
    r = client.post("/coach/api/datasets", json=body)
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["name"] == "codephreak-test-1"  # sanitised
    assert info["rows"] == 3  # 2 exchanges + 1 voice seed
    assert info["path"].endswith("codephreak-test-1/script.jsonl")

    listed = client.get("/coach/api/datasets").json()
    assert any(s["name"] == "codephreak-test-1" for s in listed)

    prev = client.get("/coach/api/datasets/codephreak-test-1").json()
    assert prev["rows"] == 3
    assert prev["sample"][0]["messages"][0]["role"] == "system"


def test_create_script_requires_content():
    r = client.post("/coach/api/datasets", json={"name": "empty", "exchanges": []})
    assert r.status_code == 422


def test_preview_unknown_404():
    assert client.get("/coach/api/datasets/nope").status_code == 404


def test_persona_endpoint_default():
    p = client.get("/coach/api/persona").json()
    assert p["name"] == "actor"
    assert "system_prompt" in p


def test_imprint_score_endpoint():
    r = client.post(
        "/coach/api/imprint/score",
        json={
            "inquiries": ["who are you?"],
            "before": ["I am an AI assistant."],
            "after": ["i am codephreak, augmentic intelligence."],
            "baseline": ["i am codephreak, augmentic intelligence orchestrator."],
        },
    )
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["after_voice"] > rep["before_voice"]
    assert rep["imprinted"] is True
