"""Autotune feedback loop — close the dcoach proof loop.

After the classroom tests a trained actor and the boardroom decides, the outcome is
recorded here and used to **improve the next run's training params** (the autotune
feedback loop). Append-only JSONL ledger (mirrors `eval/mei/history.py`); the suggester
nudges epochs / grad_accum when the imprint was weak or the board rejected.

Pure stdlib + pydantic; base-install importable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_FEEDBACK_PATH = Path("./out/autotune/feedback.jsonl")
Outcome = Literal["approved", "rejected", "disputed", "unknown"]
_MAX_EPOCHS = 40


class FeedbackEntry(BaseModel):
    """One recorded training outcome → params used + how it scored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str
    run_id: str = Field(min_length=1)
    params: dict[str, int]
    classroom_score: float = Field(description="imprint delta / recall the classroom measured")
    passed: bool
    boardroom_outcome: Outcome = "unknown"


def record(
    *,
    run_id: str,
    params: dict[str, int],
    classroom_score: float,
    passed: bool,
    boardroom_outcome: Outcome = "unknown",
    timestamp: str | None = None,
    path: Path | None = None,
) -> Path:
    """Append one feedback row; creates `out/autotune/` if needed."""
    target = path or DEFAULT_FEEDBACK_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = FeedbackEntry(
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        run_id=run_id,
        params={k: int(v) for k, v in params.items()},
        classroom_score=round(float(classroom_score), 4),
        passed=passed,
        boardroom_outcome=boardroom_outcome,
    )
    with target.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")
    return target


def read_all(path: Path | None = None) -> list[FeedbackEntry]:
    """Read every feedback row, oldest-first; missing file → []."""
    target = path or DEFAULT_FEEDBACK_PATH
    if not target.exists():
        return []
    out: list[FeedbackEntry] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(FeedbackEntry.model_validate_json(line))
        except ValueError:
            continue
    return out


def suggest_next_params(
    last_params: dict[str, int],
    *,
    passed: bool,
    classroom_score: float,
) -> dict[str, int]:
    """Nudge training params from the last outcome.

    Not imprinted / rejected → train harder (more epochs, grad_accum=1, so a few-row
    script does many steps). Weak imprint (small positive delta) → slightly more epochs.
    Passed cleanly → keep. `per_device` stays (CPU lane). Returns the same shape as
    `data/scripts.derive_training_params`.
    """
    epochs = int(last_params.get("epochs", 12))
    grad_accum = int(last_params.get("grad_accum", 1))
    per_device = int(last_params.get("per_device", 1))

    if not passed:
        epochs = min(_MAX_EPOCHS, int(epochs * 1.5) + 2)
        grad_accum = 1
    elif classroom_score < 0.05:
        epochs = min(_MAX_EPOCHS, epochs + 4)
    # passed + good score → keep as-is.
    return {"epochs": epochs, "grad_accum": grad_accum, "per_device": per_device}


def suggest_from_history(
    default_params: dict[str, int],
    *,
    path: Path | None = None,
) -> dict[str, int]:
    """Suggest next params from the most-recent feedback row, else the defaults."""
    history = read_all(path)
    if not history:
        return dict(default_params)
    last = history[-1]
    return suggest_next_params(
        last.params, passed=last.passed, classroom_score=last.classroom_score,
    )


__all__ = [
    "FeedbackEntry",
    "Outcome",
    "read_all",
    "record",
    "suggest_from_history",
    "suggest_next_params",
]
