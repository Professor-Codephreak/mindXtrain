"""Model-backed deliberation — boardroom/dojo ballots over a mocked chat backend."""

from __future__ import annotations

import httpx

from mindxtrain.governance import Boardroom, Dojo, Member
from mindxtrain.governance import panel as P


def test_parse_vote_verdict_line():
    assert P.parse_vote("looks good.\nVERDICT: APPROVE") == "approve"
    assert P.parse_vote("risky.\nverdict - reject") == "reject"
    assert P.parse_vote("unsure\nVERDICT: ABSTAIN") == "abstain"


def test_parse_vote_keyword_fallback_and_abstain_coercion():
    assert P.parse_vote("I would approve this") == "approve"
    assert P.parse_vote("definitely reject") == "reject"
    # ambiguous → abstain, but a dojo judge can't abstain → reject.
    assert P.parse_vote("hmm, both sides", allow_abstain=True) == "abstain"
    assert P.parse_vote("hmm, both sides", allow_abstain=False) == "reject"


def test_resolve_base_url_precedence(monkeypatch):
    for e in ("MINDXTRAIN_OPENAI_BASE_URL", "MINDXTRAIN_VLLM_BASE_URL", "MINDXTRAIN_OLLAMA_BASE_URL"):
        monkeypatch.delenv(e, raising=False)
    assert P.resolve_chat_base_url() == "http://localhost:11434/v1"
    monkeypatch.setenv("MINDXTRAIN_VLLM_BASE_URL", "http://vllm:8000/v1/")
    assert P.resolve_chat_base_url() == "http://vllm:8000/v1"
    assert P.resolve_chat_base_url("http://explicit/v1/") == "http://explicit/v1"


def test_deliberate_parses_model_output(monkeypatch):
    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "Strong case.\nVERDICT: APPROVE")
    d = P.deliberate(Member(id="a", role="advocate", model="m"), "promote actor x")
    assert d.vote == "approve"
    assert d.rationale
    assert d.error == ""


def test_deliberate_handles_backend_error(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("backend down")
    monkeypatch.setattr(P, "chat_once", _boom)
    d = P.deliberate(Member(id="a", role="critic"), "x")
    assert d.vote == "abstain"
    assert d.error


def test_model_ballot_drives_boardroom(monkeypatch):
    # advocate approves, everyone else rejects.
    def _chat(model, messages, **k):
        sys = messages[0]["content"].lower()
        return "VERDICT: APPROVE" if "advocate" in sys else "VERDICT: REJECT"
    monkeypatch.setattr(P, "chat_once", _chat)

    board = Boardroom(members=[
        Member(id="a", role="advocate"), Member(id="b", role="critic"),
        Member(id="c", role="analyst"),
    ])
    decision = board.convene("promote actor x", P.model_ballot())
    assert decision.approvals == 1 and decision.rejections == 2
    assert decision.outcome == "rejected"


def test_model_judge_ballot_drives_dojo(monkeypatch):
    calls = {"n": 0}
    def _chat(model, messages, **k):
        calls["n"] += 1
        return "VERDICT: APPROVE" if calls["n"] <= 2 else "VERDICT: REJECT"
    monkeypatch.setattr(P, "chat_once", _chat)

    dojo = Dojo.sized(3)
    verdict = dojo.settle("promote actor x", P.model_judge_ballot())
    assert verdict.settled is True
    assert verdict.approvals == 2 and verdict.rejections == 1
    assert verdict.winner == "approve"


def test_model_judge_never_abstains(monkeypatch):
    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "I'm not sure either way")
    dojo = Dojo.sized(3)
    verdict = dojo.settle("x", P.model_judge_ballot())
    # ambiguous → reject for every judge; still settles (3-0).
    assert verdict.rejections == 3
    assert verdict.winner == "reject"
