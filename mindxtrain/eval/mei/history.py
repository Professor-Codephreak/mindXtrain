"""Append-only JSONL history of MEI scores (spec §7 top layer).

Every scored checkpoint appends one row to `out/mei/history.jsonl`. The
historical-comparison database is what lets mindXtrain rank a new
checkpoint against the entire alpha history — needed for the §8
promotion gate ("MEI strictly higher … than the currently-promoted").

Pure stdlib; no DB. The append-only file is BLAKE3-resistant to
out-of-order writes (each row carries its own timestamp + run_id), and
the read paths sort/filter in memory — fine for the volumes the alpha
will produce (≤ 200 checkpoints per phase × a handful of phases is
still kilobytes).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mindxtrain.eval.mei.score import MEIScore

# Default path for the history file. Callers can override per-run via
# the function `path` arg if they want a per-experiment ledger.
DEFAULT_HISTORY_PATH = Path("./out/mei/history.jsonl")


class HistoryEntry(BaseModel):
    """One row in the MEI history. Frozen + extra=forbid for ledger hygiene."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str = Field(description="ISO-8601 UTC of the score event.")
    run_id: str = Field(min_length=1, description="The mindXtrain run that produced this score.")
    model_id: str = Field(min_length=1, description="HF Hub repo or local checkpoint name.")
    model_sha256: str = Field(min_length=1, description="Hash of the resolved weights.")
    promoted: bool = Field(
        default=False,
        description="True iff this score is the currently-promoted checkpoint.",
    )
    score: MEIScore


def append(
    score: MEIScore,
    *,
    run_id: str,
    model_id: str,
    model_sha256: str,
    promoted: bool = False,
    path: Path | None = None,
) -> Path:
    """Append one history entry. Creates the file (and out/mei/) if needed.

    Returns the path written. Idempotent in the sense that two appends with
    identical content produce two physical rows — the caller is responsible
    for dedup if it matters. For the promotion-gate use case, we want every
    score event recorded, even repeats of the same checkpoint scored under
    a different anchor version.
    """
    target = path or DEFAULT_HISTORY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = HistoryEntry(
        timestamp=datetime.now(UTC).isoformat(),
        run_id=run_id,
        model_id=model_id,
        model_sha256=model_sha256,
        promoted=promoted,
        score=score,
    )
    with target.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")
    return target


def read_all(path: Path | None = None) -> list[HistoryEntry]:
    """Read every entry from the history file, oldest-first by file order.

    Bad / malformed lines are skipped silently (the ledger is append-only,
    so corrupted writes from process crashes shouldn't break readers).
    Returns an empty list if the file doesn't exist.
    """
    target = path or DEFAULT_HISTORY_PATH
    if not target.exists():
        return []
    out: list[HistoryEntry] = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(HistoryEntry.model_validate_json(line))
            except Exception:
                continue
    return out


def currently_promoted(path: Path | None = None) -> HistoryEntry | None:
    """Return the most-recently-promoted entry, or None.

    Promotion is monotonic in our model: once a checkpoint is promoted,
    the only way a new one beats it is via `is_promotable`. The
    "currently-promoted" view is therefore the most-recent row with
    `promoted=True`.
    """
    for entry in reversed(read_all(path=path)):
        if entry.promoted:
            return entry
    return None


def trajectory(*, last_n: int = 10, path: Path | None = None) -> list[HistoryEntry]:
    """Return the last-N entries, newest-first.

    Used by the promotion-gate trajectory analysis (§8): if MEI declines
    for two consecutive checkpoints, the run is paused; if it plateaus
    for three, the schedule is reviewed.
    """
    rows = read_all(path=path)
    return rows[-last_n:][::-1]


__all__ = [
    "DEFAULT_HISTORY_PATH",
    "HistoryEntry",
    "append",
    "currently_promoted",
    "read_all",
    "trajectory",
]
