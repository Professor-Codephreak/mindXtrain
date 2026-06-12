"""Imprint measurement — before/after recall scoring (lexical fallback path)."""

from __future__ import annotations

from mindxtrain.eval import imprint as I


def test_default_inquiries_nonempty():
    qs = I.default_inquiries("Codephreak")
    assert len(qs) >= 3
    assert any("Codephreak" in q for q in qs)


def test_lexical_similarity_bounds():
    assert I._lexical_similarity("hello world", "hello world") == 1.0
    assert I._lexical_similarity("abc", "xyz") == 0.0
    mid = I._lexical_similarity("the quick fox", "the slow fox")
    assert 0.0 < mid < 1.0


def test_score_imprint_detects_movement_toward_persona():
    inquiries = ["Who are you?", "What do you do?"]
    baseline = ["i am codephreak, augmentic intelligence orchestrator.",
                "i orchestrate autonomous agents."]
    # Before: generic; After: persona voice. After should score closer to baseline.
    before = ["I am an AI assistant.", "I help with tasks."]
    after = ["i am codephreak, augmentic intelligence.", "i orchestrate autonomous agents."]

    rep = I.score_imprint(inquiries, before, after, baseline)
    assert rep.after_voice > rep.before_voice
    assert rep.imprint_delta > 0.0
    assert rep.shift > 0.0
    assert rep.imprinted is True
    assert rep.method in ("lexical", "sentence-transformers")


def test_score_imprint_no_movement_not_imprinted():
    inquiries = ["Who are you?"]
    baseline = ["i am codephreak."]
    same = ["I am an AI assistant."]
    rep = I.score_imprint(inquiries, same, same, baseline)
    # Identical before/after → no shift → not imprinted.
    assert rep.shift == 0.0
    assert rep.imprinted is False


def test_score_imprint_report_is_frozen_and_jsonable():
    import pytest
    from pydantic import ValidationError

    rep = I.score_imprint(["q"], ["a"], ["b"], ["c"])
    blob = rep.model_dump_json()
    assert '"imprint_delta"' in blob
    with pytest.raises(ValidationError):
        rep.imprint_delta = 1.0  # frozen
