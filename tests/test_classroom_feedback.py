"""Classroom before/after evaluation + autotune feedback loop."""

from __future__ import annotations

from mindxtrain.autotune import feedback as FB
from mindxtrain.governance.classroom import evaluate_classroom


def test_classroom_passes_when_after_recalls_persona():
    inquiries = ["who are you?", "what do you do?"]
    baseline = ["i am codephreak, augmentic intelligence.", "i orchestrate autonomous agents."]
    before = ["I am an AI assistant.", "I help with tasks."]
    after = baseline  # perfect recall after imprint
    rep = evaluate_classroom(inquiries, before, after, baseline)
    assert rep.recall > rep.before_recall
    assert rep.imprint_delta > 0
    assert rep.persona_maintained is True
    assert rep.pairwise_after_better == 1.0
    assert rep.passed is True


def test_classroom_fails_with_no_movement():
    inquiries = ["who?"]
    baseline = ["i am codephreak."]
    same = ["I am an AI."]
    rep = evaluate_classroom(inquiries, same, same, baseline)
    assert rep.passed is False
    assert rep.persona_maintained is False


def test_feedback_record_and_read(tmp_path):
    p = tmp_path / "feedback.jsonl"
    FB.record(run_id="r1", params={"epochs": 12, "grad_accum": 1, "per_device": 1},
              classroom_score=0.04, passed=True, boardroom_outcome="approved",
              timestamp="2026-06-12T00:00:00+00:00", path=p)
    rows = FB.read_all(p)
    assert len(rows) == 1
    assert rows[0].run_id == "r1" and rows[0].boardroom_outcome == "approved"


def test_suggest_next_params_trains_harder_on_failure():
    nxt = FB.suggest_next_params({"epochs": 12, "grad_accum": 4, "per_device": 1},
                                 passed=False, classroom_score=0.0)
    assert nxt["epochs"] > 12          # more epochs
    assert nxt["grad_accum"] == 1      # forced to 1


def test_suggest_next_params_nudges_weak_imprint():
    nxt = FB.suggest_next_params({"epochs": 12, "grad_accum": 1, "per_device": 1},
                                 passed=True, classroom_score=0.02)
    assert nxt["epochs"] == 16


def test_suggest_next_params_keeps_on_good_pass():
    base = {"epochs": 12, "grad_accum": 1, "per_device": 1}
    assert FB.suggest_next_params(base, passed=True, classroom_score=0.4) == base


def test_suggest_from_history(tmp_path):
    p = tmp_path / "f.jsonl"
    assert FB.suggest_from_history({"epochs": 10, "grad_accum": 2, "per_device": 1}, path=p) == \
        {"epochs": 10, "grad_accum": 2, "per_device": 1}  # empty → defaults
    FB.record(run_id="r", params={"epochs": 12, "grad_accum": 2, "per_device": 1},
              classroom_score=0.0, passed=False, path=p)
    nxt = FB.suggest_from_history({"epochs": 10, "grad_accum": 2, "per_device": 1}, path=p)
    assert nxt["epochs"] > 12 and nxt["grad_accum"] == 1  # nudged from the failed run
