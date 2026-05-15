"""mindX dream-corpus adapter — unit tests against a synthetic LTM tree.

The real corpus at /home/hacker/mindX/data/memory is not assumed; tests
build a tiny fixture that mirrors the real shape (chat-format JSONL under
ltm/<agent>/<timestamp>_training.jsonl) and verify glob, dedup, and
max_samples behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

from mindxtrain.data.sources.mindx_dreams import (
    count_mindx_dreams,
    iter_mindx_dreams,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _msg_row(content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "dream-consolidation engine"},
            {"role": "user", "content": f"STM sample: {content}"},
            {"role": "assistant", "content": f"insight: {content}"},
        ],
    }


def test_iter_yields_rows_from_glob(tmp_path):
    root = tmp_path
    _write_jsonl(root / "ltm" / "agent_a" / "20260513_010000_training.jsonl",
                 [_msg_row("alpha"), _msg_row("beta")])
    _write_jsonl(root / "ltm" / "agent_b" / "20260513_020000_training.jsonl",
                 [_msg_row("gamma")])
    out = list(iter_mindx_dreams(root))
    assert len(out) == 3
    assert all("messages" in r for r in out)


def test_dedup_across_files(tmp_path):
    root = tmp_path
    _write_jsonl(root / "ltm" / "agent_a" / "20260513_010000_training.jsonl",
                 [_msg_row("alpha"), _msg_row("beta")])
    _write_jsonl(root / "ltm" / "agent_b" / "20260513_020000_training.jsonl",
                 [_msg_row("alpha"), _msg_row("gamma")])  # alpha is duplicate
    out = list(iter_mindx_dreams(root))
    contents = [r["messages"][-1]["content"] for r in out]
    assert contents.count("insight: alpha") == 1
    assert len(out) == 3


def test_max_samples_cap(tmp_path):
    root = tmp_path
    rows = [_msg_row(str(i)) for i in range(20)]
    _write_jsonl(root / "ltm" / "agent_a" / "20260513_010000_training.jsonl", rows)
    out = list(iter_mindx_dreams(root, max_samples=5))
    assert len(out) == 5


def test_skips_bad_lines(tmp_path):
    root = tmp_path
    path = root / "ltm" / "agent_a" / "20260513_010000_training.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_msg_row("ok")) + "\n"
        + "{not valid json\n"
        + "\n"
        + json.dumps({"no_messages_field": True}) + "\n"
        + json.dumps(_msg_row("ok2")) + "\n",
        encoding="utf-8",
    )
    out = list(iter_mindx_dreams(root))
    assert len(out) == 2


def test_missing_root_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        list(iter_mindx_dreams(tmp_path / "nowhere"))


def test_count_helper(tmp_path):
    root = tmp_path
    _write_jsonl(root / "ltm" / "agent_a" / "20260513_010000_training.jsonl",
                 [_msg_row("alpha"), _msg_row("beta")])
    _write_jsonl(root / "ltm" / "agent_b" / "20260513_020000_training.jsonl",
                 [_msg_row("alpha")])  # cross-file duplicate
    stats = count_mindx_dreams(root)
    assert stats == {"files": 2, "raw_lines": 3, "unique_rows": 2}
