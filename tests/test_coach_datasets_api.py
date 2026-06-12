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


def test_personas_endpoint_lists_builtins_and_skills():
    body = client.get("/coach/api/personas").json()
    pnames = {p["name"] for p in body["personas"]}
    snames = {s["name"] for s in body["skills"]}
    assert {"codephreak", "assistant", "mentor"} <= pnames
    assert {"software_engineer", "platform_architect", "bash", "solidity"} <= snames


def test_create_script_from_builtin_persona_with_skills():
    r = client.post("/coach/api/datasets", json={
        "name": "codephreak skills",
        "persona": "codephreak",
        "skills": ["software_engineer", "solidity"],
        "seed_voice": True,
    })
    assert r.status_code == 200, r.text
    info = r.json()
    # 3 software_engineer + 3 solidity exchanges + 2 codephreak voice seeds = 8 rows.
    assert info["rows"] == 8
    assert set(info["skills"]) == {"software_engineer", "solidity"}
    # Training params auto-derived from the dataset size.
    assert info["train_params"]["epochs"] >= 8
    assert info["train_params"]["grad_accum"] == 1

    prev = client.get(f"/coach/api/datasets/{info['name']}").json()
    sys_msg = prev["sample"][0]["messages"][0]["content"]
    assert "Codephreak" in sys_msg  # persona voice carried into the script


def test_coach_index_has_persona_and_skill_controls():
    html = client.get("/coach/").text
    assert 'id="ds-persona"' in html
    assert 'id="ds-skills"' in html
    js = client.get("/coach/static/coach.js").text
    assert "loadPersonasAndSkills" in js
    assert "/coach/api/personas" in js


def test_create_script_from_only_skills():
    r = client.post("/coach/api/datasets", json={
        "name": "bash-only", "persona": "assistant", "skills": ["bash"],
        "seed_voice": False,
    })
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == 3  # bash skill's 3 exchanges


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
