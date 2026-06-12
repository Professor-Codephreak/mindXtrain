"""Coach classroom-evaluate + autotune-feedback endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_feedback(tmp_path, monkeypatch):
    from mindxtrain.autotune import feedback as fb
    monkeypatch.setattr(fb, "DEFAULT_FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    yield


def test_classroom_evaluate_endpoint():
    baseline = ["i am codephreak.", "i orchestrate agents."]
    r = client.post("/coach/api/classroom/evaluate", json={
        "inquiries": ["who?", "what?"],
        "before": ["I am an AI.", "I help."],
        "after": baseline,
        "baseline": baseline,
    })
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["passed"] is True
    assert rep["recall"] > rep["before_recall"]
    assert rep["persona_maintained"] is True


def test_autotune_feedback_endpoint_suggests_harder_on_failure():
    r = client.post("/coach/api/autotune/feedback", json={
        "run_id": "r1",
        "params": {"epochs": 12, "grad_accum": 4, "per_device": 1},
        "classroom_score": 0.0,
        "passed": False,
        "boardroom_outcome": "rejected",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recorded"] is True
    nxt = body["suggested_next_params"]
    assert nxt["epochs"] > 12 and nxt["grad_accum"] == 1


def test_autotune_feedback_keeps_on_pass():
    r = client.post("/coach/api/autotune/feedback", json={
        "run_id": "r2",
        "params": {"epochs": 12, "grad_accum": 1, "per_device": 1},
        "classroom_score": 0.4,
        "passed": True,
        "boardroom_outcome": "approved",
    })
    assert r.json()["suggested_next_params"] == {"epochs": 12, "grad_accum": 1, "per_device": 1}
