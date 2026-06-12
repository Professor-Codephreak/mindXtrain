"""dcoach page routes + decentralized panel + prompt-eval + streamed proof loop."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


def test_dcoach_and_prompts_pages_serve():
    for path in ("/coach/dcoach", "/coach/prompts"):
        r = client.get(path)
        assert r.status_code == 200, r.text
        assert "text/html" in r.headers["content-type"]


def test_decentralized_panel_data():
    r = client.get("/coach/api/decentralized")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["thesis"]
    names = {n["name"] for n in data["networks"]}
    # every 2026 network the deep-dive covers is represented
    assert {"Templar · Bittensor SN3", "Gensyn", "Pluralis · Node0"} <= names
    for n in data["networks"]:
        assert {"name", "what", "hardware", "token", "fit"} <= set(n)
    assert any("Verde" in f["maps_to"] for f in data["fit"])


def test_eval_prompt_similarity_only():
    # no judge → only the free semantic-similarity evaluator runs
    r = client.post("/coach/api/eval/prompt", json={
        "query": "who are you?",
        "response": "i am codephreak, augmentic intelligence.",
        "reference": "i am codephreak, augmentic intelligence.",
        "use_judge": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert "semantic_similarity" in body["scores"]
    assert "correctness" not in body["scores"]
    assert body["scores"]["semantic_similarity"]["score"] > 0.9
    assert body["overall"] > 0.9 and body["advantageous"] is True


def test_dcoach_run_streams_phases(monkeypatch):
    """Patch the heavy loop with a fast fake; assert the SSE bridge streams phases
    + the terminal result + [DONE]."""
    from mindxtrain.governance import classroom as _cr
    from mindxtrain.governance import proof_loop as _pl

    report = _cr.ClassroomReport(
        inquiries=["who?"], before=["an AI"], after=["codephreak"],
        before_recall=0.1, recall=0.6, imprint_delta=0.5,
        pairwise_after_better=1.0, persona_maintained=True, passed=True,
    )

    def _fake(*, run_id, on_event=None, **kw):
        if on_event:
            on_event("dataset", "authored 5 rows")
            on_event("train", "imprinting…")
        return _pl.ProofResult(
            run_id=run_id, dataset_path="/tmp/s.jsonl", rows=5,
            train_params={"epochs": 12, "grad_accum": 1, "per_device": 1},
            classroom=report, boardroom_outcome="approved",
            boardroom_rationale="approved 3-0", passed=True,
            next_params={"epochs": 12, "grad_accum": 1, "per_device": 1},
        )

    monkeypatch.setattr(_pl, "run_proof_loop", _fake)

    with client.stream("POST", "/coach/api/dcoach/run",
                       json={"persona": "codephreak", "run_id": "t1"}) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    events.append({"phase": "done"})
                else:
                    events.append(json.loads(payload))

    phases = [e["phase"] for e in events]
    assert phases[0] == "start"
    assert "dataset" in phases and "train" in phases
    assert "result" in phases and phases[-1] == "done"
    result = next(e["result"] for e in events if e["phase"] == "result")
    assert result["passed"] is True
    assert result["boardroom_outcome"] == "approved"
