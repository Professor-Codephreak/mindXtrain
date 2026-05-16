"""Coach API endpoints for the MEI surface.

Three routes:
- GET  /coach/api/mei/history?last=N — recent scores, newest-first.
- GET  /coach/api/mei/score/{run_id} — full score + promotion preview.
- POST /coach/api/mei/promote/{run_id} — gated promotion (idempotent).

The history file is per-test (monkeypatched) so the suite is hermetic.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mindxtrain.eval.mei import history as _hist
from mindxtrain.eval.mei.score import MEIScore
from mindxtrain.operator.app import app

client = TestClient(app)


def _score(composite: float, *, q: float = 0.5, dt: float = 0.5,
           pp: float = 0.5, m: float = 0.5, e: float = 0.5) -> MEIScore:
    return MEIScore(
        composite=composite, quality=q, decode_throughput=dt,
        prefill_throughput=pp, memory=m, energy=e,
        quality_bands={"instruction": q, "reasoning": q, "knowledge_code": q},
        mab_provisional=True, notes=[],
    )


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch):
    """Point the history module at a tmp ledger for every test."""
    ledger = tmp_path / "history.jsonl"
    monkeypatch.setattr(_hist, "DEFAULT_HISTORY_PATH", ledger)
    return ledger


# ---- /mei/history --------------------------------------------------------


def test_history_empty_returns_empty_list():
    r = client.get("/coach/api/mei/history")
    assert r.status_code == 200
    assert r.json() == []


def test_history_returns_entries_newest_first():
    _hist.append(_score(0.40), run_id="r1", model_id="m1", model_sha256="x")
    _hist.append(_score(0.55), run_id="r2", model_id="m2", model_sha256="x")
    _hist.append(_score(0.62), run_id="r3", model_id="m3", model_sha256="x")
    rows = client.get("/coach/api/mei/history").json()
    assert [r["run_id"] for r in rows] == ["r3", "r2", "r1"]
    assert rows[0]["composite"] == 0.62


def test_history_last_n_cap():
    for i in range(5):
        _hist.append(_score(0.5 + 0.01 * i), run_id=f"r{i}",
                     model_id="m", model_sha256="x")
    rows = client.get("/coach/api/mei/history?last=2").json()
    assert len(rows) == 2
    assert [r["run_id"] for r in rows] == ["r4", "r3"]


def test_history_row_has_required_flat_fields():
    _hist.append(_score(0.55), run_id="r1", model_id="m1", model_sha256="x")
    [row] = client.get("/coach/api/mei/history").json()
    for key in (
        "timestamp", "run_id", "model_id", "promoted",
        "composite", "quality", "decode_throughput", "prefill_throughput",
        "memory", "energy", "mab_provisional",
    ):
        assert key in row, f"missing {key}"


# ---- /mei/score/{run_id} -------------------------------------------------


def test_score_returns_full_view_with_promotion_preview():
    """A passing score → promotable=True."""
    sc = _score(0.62, q=0.60, dt=0.60, pp=0.55, m=0.65, e=0.60)
    _hist.append(sc, run_id="r1", model_id="m1", model_sha256="x")
    body = client.get("/coach/api/mei/score/r1").json()
    assert body["run_id"] == "r1"
    assert body["composite"] == 0.62
    assert body["promotable"] is True
    assert body["promotion_reasons"] == []
    assert body["mab_provisional"] is True


def test_score_below_threshold_marks_not_promotable():
    sc = _score(0.40)  # below 0.55 gate
    _hist.append(sc, run_id="r1", model_id="m1", model_sha256="x")
    body = client.get("/coach/api/mei/score/r1").json()
    assert body["promotable"] is False
    assert any("composite" in r for r in body["promotion_reasons"])


def test_score_unknown_run_returns_404():
    r = client.get("/coach/api/mei/score/nope")
    assert r.status_code == 404


def test_score_returns_most_recent_for_same_run():
    """A run can be scored multiple times — the latest wins."""
    _hist.append(_score(0.40), run_id="r1", model_id="m", model_sha256="x")
    _hist.append(_score(0.62), run_id="r1", model_id="m", model_sha256="x")
    body = client.get("/coach/api/mei/score/r1").json()
    assert body["composite"] == 0.62


# ---- /mei/promote/{run_id} -----------------------------------------------


def test_promote_succeeds_when_gates_pass():
    sc = _score(0.62, q=0.6, dt=0.6, pp=0.55, m=0.65, e=0.60)
    _hist.append(sc, run_id="r1", model_id="m1", model_sha256="x")
    r = client.post("/coach/api/mei/promote/r1")
    assert r.status_code == 200
    body = r.json()
    assert body["promoted"] is True
    assert body["reasons"] == []
    # Ledger now records a promoted entry.
    promoted = _hist.currently_promoted()
    assert promoted is not None
    assert promoted.run_id == "r1"


def test_promote_refused_below_threshold():
    _hist.append(_score(0.40), run_id="r1", model_id="m", model_sha256="x")
    r = client.post("/coach/api/mei/promote/r1")
    assert r.status_code == 200  # the gate result is a body, not a 4xx
    body = r.json()
    assert body["promoted"] is False
    assert any("composite" in r for r in body["reasons"])
    # No promotion entry appended.
    assert _hist.currently_promoted() is None


def test_promote_unknown_run_404():
    r = client.post("/coach/api/mei/promote/nope")
    assert r.status_code == 404


def test_promote_refused_when_not_better_than_currently_promoted():
    """Spec §8: 'MEI strictly higher … than the currently-promoted'."""
    _hist.append(_score(0.65, q=0.6, dt=0.6, pp=0.55, m=0.65, e=0.60),
                 run_id="prior", model_id="m", model_sha256="x", promoted=True)
    _hist.append(_score(0.60, q=0.6, dt=0.6, pp=0.55, m=0.65, e=0.60),
                 run_id="new_run", model_id="m", model_sha256="x")
    r = client.post("/coach/api/mei/promote/new_run").json()
    assert r["promoted"] is False
    assert any("currently-promoted" in s for s in r["reasons"])


def test_promote_is_idempotent_for_already_promoted_run():
    """Promoting the same run twice should each time pass (no overlap
    with prior since we exclude same-run from the comparison)."""
    sc = _score(0.62, q=0.6, dt=0.6, pp=0.55, m=0.65, e=0.60)
    _hist.append(sc, run_id="r1", model_id="m", model_sha256="x")
    r1 = client.post("/coach/api/mei/promote/r1").json()
    assert r1["promoted"] is True
    r2 = client.post("/coach/api/mei/promote/r1").json()
    assert r2["promoted"] is True
