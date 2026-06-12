"""Coach governance endpoints: boardroom convene + dojo settle."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mindxtrain.governance import panel as P
from mindxtrain.operator.app import app

client = TestClient(app)


def test_boardroom_presets():
    r = client.get("/coach/api/boardroom/presets")
    assert r.status_code == 200
    presets = r.json()
    assert "classic_triad" in presets
    assert presets["classic_triad"] == ["advocate", "critic", "analyst"]


def test_convene_with_explicit_votes():
    r = client.post("/coach/api/boardroom/convene", json={
        "motion": "promote actor x",
        "members": [{"id": "a", "role": "advocate"}, {"id": "b", "role": "critic"},
                    {"id": "c", "role": "analyst"}],
        "votes": {"a": "approve", "b": "approve", "c": "reject"},
    })
    assert r.status_code == 200, r.text
    d = r.json()["decision"]
    assert d["outcome"] == "approved"
    assert d["approvals"] == 2 and d["rejections"] == 1


def test_convene_tie_is_disputed():
    r = client.post("/coach/api/boardroom/convene", json={
        "motion": "x",
        "members": [{"id": "a"}, {"id": "b"}],
        "votes": {"a": "approve", "b": "reject"},
    })
    assert r.status_code == 200
    assert r.json()["decision"]["disputed"] is True


def test_convene_with_models(monkeypatch):
    def _chat(model, messages, **k):
        return "VERDICT: APPROVE" if "advocate" in messages[0]["content"].lower() else "VERDICT: REJECT"
    monkeypatch.setattr(P, "chat_once", _chat)
    r = client.post("/coach/api/boardroom/convene", json={
        "motion": "promote actor x",
        "members": [{"id": "a", "role": "advocate", "model": "m"},
                    {"id": "b", "role": "critic", "model": "m"}],
        "use_models": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["deliberations"]) == 2
    assert body["decision"]["approvals"] == 1 and body["decision"]["rejections"] == 1


def test_convene_requires_votes_or_models():
    r = client.post("/coach/api/boardroom/convene", json={
        "motion": "x", "members": [{"id": "a"}],
    })
    assert r.status_code == 422


def test_convene_rejects_bad_role():
    r = client.post("/coach/api/boardroom/convene", json={
        "motion": "x", "members": [{"id": "a", "role": "wizard"}],
        "votes": {"a": "approve"},
    })
    assert r.status_code == 422


def test_dojo_settle_with_votes():
    r = client.post("/coach/api/dojo/settle", json={
        "motion": "promote actor x", "size": 3,
        "votes": {"judge-0": "approve", "judge-1": "approve", "judge-2": "reject"},
    })
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["settled"] is True
    assert v["winner"] == "approve"


def test_dojo_settle_rounds_size_to_prime():
    # size 4 → 5 judges; need all five votes.
    r = client.post("/coach/api/dojo/settle", json={
        "motion": "x", "size": 4,
        "votes": {f"judge-{i}": ("approve" if i < 3 else "reject") for i in range(5)},
    })
    assert r.status_code == 200, r.text
    v = r.json()
    assert len(v["judges"]) == 5
    assert v["winner"] == "approve"


def test_dojo_settle_missing_vote_422():
    r = client.post("/coach/api/dojo/settle", json={
        "motion": "x", "size": 3, "votes": {"judge-0": "approve"},
    })
    assert r.status_code == 422


def test_coach_index_has_boardroom_card():
    r = client.get("/coach/")
    assert r.status_code == 200
    assert 'id="step-boardroom"' in r.text
    assert 'id="br-convene"' in r.text
    assert 'id="br-settle"' in r.text


def test_coach_js_wires_boardroom():
    r = client.get("/coach/static/coach.js")
    assert r.status_code == 200
    assert "wireBoardroom" in r.text
    assert "/coach/api/boardroom/convene" in r.text
    assert "/coach/api/dojo/settle" in r.text
