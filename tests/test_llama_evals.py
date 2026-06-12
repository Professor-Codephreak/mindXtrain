"""Clean-room llama-style evaluators (judge mocked)."""

from __future__ import annotations

from mindxtrain.eval import llama_evals as LE
from mindxtrain.governance import panel as P


def test_parse_judge_score():
    assert LE._parse_judge_score("good\nSCORE: 5")[0] == 1.0
    assert LE._parse_judge_score("bad\nSCORE: 1")[0] == 0.0
    assert LE._parse_judge_score("mid\nSCORE: 3")[0] == 0.5
    assert 0.0 <= LE._parse_judge_score("no score here")[0] <= 1.0


def test_semantic_similarity_lexical():
    ev = LE.SemanticSimilarityEvaluator(threshold=0.7)
    same = ev.evaluate("i am codephreak", "i am codephreak")
    assert same.score == 1.0 and same.passing is True
    diff = ev.evaluate("abc def", "xyz qrs")
    assert diff.score == 0.0 and diff.passing is False


def test_correctness_evaluator(monkeypatch):
    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "Matches well.\nSCORE: 5")
    s = LE.CorrectnessEvaluator(model="m").evaluate("who?", "i am codephreak", "i am codephreak")
    assert s.score == 1.0 and s.passing is True
    assert s.method == "llm-judge:m"

    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "Wrong.\nSCORE: 1")
    s2 = LE.CorrectnessEvaluator().evaluate("who?", "i am a bot", "i am codephreak")
    assert s2.score == 0.0 and s2.passing is False


def test_pairwise_evaluator(monkeypatch):
    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "B is closer to the voice.\nB")
    s = LE.PairwiseEvaluator().evaluate("who?", "I am an AI.", "i am codephreak.", reference="codephreak voice")
    assert s.score == 1.0 and s.passing is True  # B (after) wins

    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "A is better.\nA")
    assert LE.PairwiseEvaluator().evaluate("q", "x", "y").score == 0.0

    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "Even.\nTIE")
    assert LE.PairwiseEvaluator().evaluate("q", "x", "y").score == 0.5


def test_guideline_evaluator(monkeypatch):
    monkeypatch.setattr(P, "chat_once", lambda *a, **k: "Mostly.\nSCORE: 4")
    s = LE.GuidelineEvaluator(threshold=0.6).evaluate("lowercase reply", "must be lowercase")
    assert s.score == 0.75 and s.passing is True


def test_judge_error_is_graceful(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("backend down")
    monkeypatch.setattr(P, "chat_once", _boom)
    s = LE.CorrectnessEvaluator().evaluate("q", "r", "ref")
    assert s.score == 0.5
    assert "judge error" in s.reasoning
