"""MEI history ledger — append-only JSONL with promotion lookup."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mindxtrain.eval.mei.history import (
    HistoryEntry,
    append,
    currently_promoted,
    read_all,
    trajectory,
)
from mindxtrain.eval.mei.score import MEIScore


def _score(composite: float = 0.55) -> MEIScore:
    return MEIScore(
        composite=composite, quality=0.55, decode_throughput=0.50,
        prefill_throughput=0.45, memory=0.55, energy=0.50,
        quality_bands={"instruction": 0.6}, mab_provisional=True, notes=[],
    )


def test_append_creates_file_and_writes_entry(tmp_path):
    target = tmp_path / "history.jsonl"
    append(_score(), run_id="r1", model_id="m", model_sha256="abcd", path=target)
    assert target.exists()
    assert target.read_text().count("\n") == 1


def test_append_creates_parent_directory(tmp_path):
    """Auto-mkdir so callers don't have to pre-create out/mei/."""
    target = tmp_path / "deep" / "nested" / "history.jsonl"
    append(_score(), run_id="r", model_id="m", model_sha256="x", path=target)
    assert target.exists()


def test_read_all_yields_entries_in_file_order(tmp_path):
    target = tmp_path / "history.jsonl"
    append(_score(0.40), run_id="r1", model_id="m1", model_sha256="aa", path=target)
    append(_score(0.55), run_id="r2", model_id="m2", model_sha256="bb", path=target)
    append(_score(0.62), run_id="r3", model_id="m3", model_sha256="cc", path=target)
    rows = read_all(path=target)
    assert [r.run_id for r in rows] == ["r1", "r2", "r3"]
    assert [r.score.composite for r in rows] == [0.40, 0.55, 0.62]


def test_read_all_skips_malformed_lines(tmp_path):
    target = tmp_path / "history.jsonl"
    append(_score(), run_id="ok", model_id="m", model_sha256="x", path=target)
    # Corrupt the file with an invalid line in the middle.
    with target.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")  # blank line, should be skipped
    append(_score(0.6), run_id="ok2", model_id="m", model_sha256="x", path=target)
    rows = read_all(path=target)
    assert [r.run_id for r in rows] == ["ok", "ok2"]


def test_read_all_returns_empty_for_missing_file(tmp_path):
    assert read_all(path=tmp_path / "nope.jsonl") == []


def test_currently_promoted_returns_most_recent_promoted(tmp_path):
    target = tmp_path / "history.jsonl"
    append(_score(0.4), run_id="r1", model_id="m1", model_sha256="x", promoted=True, path=target)
    append(_score(0.5), run_id="r2", model_id="m2", model_sha256="x", promoted=False, path=target)
    append(_score(0.6), run_id="r3", model_id="m3", model_sha256="x", promoted=True, path=target)
    append(_score(0.7), run_id="r4", model_id="m4", model_sha256="x", promoted=False, path=target)
    promoted = currently_promoted(path=target)
    assert promoted is not None
    assert promoted.run_id == "r3"


def test_currently_promoted_returns_none_when_nothing_promoted(tmp_path):
    target = tmp_path / "history.jsonl"
    append(_score(), run_id="r", model_id="m", model_sha256="x", promoted=False, path=target)
    assert currently_promoted(path=target) is None


def test_trajectory_returns_newest_first(tmp_path):
    target = tmp_path / "history.jsonl"
    for i in range(5):
        append(_score(0.5 + i * 0.01), run_id=f"r{i}", model_id="m",
               model_sha256="x", path=target)
    out = trajectory(last_n=3, path=target)
    assert [r.run_id for r in out] == ["r4", "r3", "r2"]


def test_entry_is_frozen():
    """Ledger entries are immutable — same hygiene as MEIRecord."""
    entry = HistoryEntry(
        timestamp="2026-05-15T00:00:00+00:00",
        run_id="r", model_id="m", model_sha256="x", promoted=False, score=_score(),
    )
    with pytest.raises(ValidationError):
        entry.run_id = "tampered"  # type: ignore[misc]


def test_entry_extra_keys_forbidden():
    """A typo in a logged field should fail loudly, not silently persist."""
    with pytest.raises(ValidationError):
        HistoryEntry(
            timestamp="t", run_id="r", model_id="m", model_sha256="x",
            score=_score(), unexpected="nope",  # type: ignore[call-arg]
        )


def test_history_round_trips_through_json(tmp_path):
    """Bytes written by `append` must parse cleanly via `read_all`."""
    target = tmp_path / "history.jsonl"
    sc = _score(0.62)
    append(sc, run_id="r", model_id="m", model_sha256="x", path=target)
    [back] = read_all(path=target)
    assert back.run_id == "r"
    assert back.model_id == "m"
    assert back.score.composite == 0.62
    assert back.score.notes == sc.notes
