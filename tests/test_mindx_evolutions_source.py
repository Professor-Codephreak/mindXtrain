"""mindX evolution-proposal adapter — same shape as the consolidation
adapter, different glob. Tests build a synthetic LTM tree mirroring
`<agent>_evolutions.jsonl` and verify glob, dedup, and max_samples.

The evolution-proposal output is a new dream-cycle phase that emits
JSONL alongside the existing `<agent>_training.jsonl`. mindXtrain
consumes it via `iter_mindx_evolutions` when a recipe opts in via
`data.include_evolutions: true`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindxtrain.data.sources.mindx_dreams import (
    count_mindx_evolutions,
    iter_mindx_evolutions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _evolution_row(content: str) -> dict:
    """Mirror the shape mindX's phase 5c writes."""
    return {
        "messages": [
            {"role": "system", "content": "evolution-proposal engine for agent_a"},
            {"role": "user", "content": f"Top insights: {content}"},
            {
                "role": "assistant",
                "content": json.dumps({
                    "type": "strategy",
                    "target_agent": "agent_a",
                    "proposal": content,
                    "rationale": f"refs:[{content}]",
                    "expected_outcome": "tbd",
                    "confidence": 0.7,
                }),
            },
        ],
    }


def test_iter_evolutions_yields_rows_from_glob(tmp_path):
    root = tmp_path
    _write_jsonl(
        root / "ltm" / "agent_a" / "20260514_010000_evolutions.jsonl",
        [_evolution_row("alpha"), _evolution_row("beta")],
    )
    _write_jsonl(
        root / "ltm" / "agent_b" / "20260514_020000_evolutions.jsonl",
        [_evolution_row("gamma")],
    )
    out = list(iter_mindx_evolutions(root))
    assert len(out) == 3
    assert all("messages" in r for r in out)


def test_iter_evolutions_ignores_training_jsonl(tmp_path):
    """The evolutions glob must not pick up consolidation files."""
    root = tmp_path
    _write_jsonl(
        root / "ltm" / "agent_a" / "20260514_010000_training.jsonl",
        [_evolution_row("should-not-appear")],
    )
    _write_jsonl(
        root / "ltm" / "agent_a" / "20260514_010001_evolutions.jsonl",
        [_evolution_row("real-evolution")],
    )
    out = list(iter_mindx_evolutions(root))
    assert len(out) == 1
    assert "real-evolution" in out[0]["messages"][1]["content"]


def test_evolutions_dedup_across_cycles(tmp_path):
    root = tmp_path
    _write_jsonl(
        root / "ltm" / "agent_a" / "20260514_010000_evolutions.jsonl",
        [_evolution_row("alpha"), _evolution_row("beta")],
    )
    _write_jsonl(
        root / "ltm" / "agent_a" / "20260514_020000_evolutions.jsonl",
        [_evolution_row("alpha"), _evolution_row("gamma")],
    )
    out = list(iter_mindx_evolutions(root))
    contents = [r["messages"][1]["content"] for r in out]
    assert sum(1 for c in contents if "alpha" in c) == 1
    assert len(out) == 3


def test_evolutions_max_samples_cap(tmp_path):
    root = tmp_path
    rows = [_evolution_row(str(i)) for i in range(20)]
    _write_jsonl(root / "ltm" / "agent_a" / "20260514_010000_evolutions.jsonl", rows)
    out = list(iter_mindx_evolutions(root, max_samples=5))
    assert len(out) == 5


def test_evolutions_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_mindx_evolutions(tmp_path / "nowhere"))


def test_count_evolutions_helper(tmp_path):
    root = tmp_path
    _write_jsonl(
        root / "ltm" / "agent_a" / "20260514_010000_evolutions.jsonl",
        [_evolution_row("alpha"), _evolution_row("beta")],
    )
    _write_jsonl(
        root / "ltm" / "agent_b" / "20260514_020000_evolutions.jsonl",
        [_evolution_row("alpha")],
    )
    stats = count_mindx_evolutions(root)
    assert stats == {"files": 2, "raw_lines": 3, "unique_rows": 2}


def test_count_evolutions_returns_zero_for_empty_tree(tmp_path):
    (tmp_path / "ltm").mkdir(parents=True)
    stats = count_mindx_evolutions(tmp_path)
    assert stats == {"files": 0, "raw_lines": 0, "unique_rows": 0}
