"""Combined consolidation + evolution stream via DataCfg.include_evolutions.

mindXtrain's `load_streaming_dataset` for source=`mindx_dreams` yields
consolidation rows first, then evolution-proposal rows when the recipe
opts in. The shared `max_samples` budget caps the *combined* total.

These tests build a synthetic LTM tree with both filename types and
exercise the cap + ordering + backward compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from mindxtrain.config.schema import DataCfg, XTrainConfig
from mindxtrain.data.curate import load_streaming_dataset


def _row(content: str, kind: str = "consolidation") -> dict:
    return {
        "messages": [
            {"role": "system", "content": f"{kind} engine"},
            {"role": "user", "content": f"stm: {content}"},
            {"role": "assistant", "content": json.dumps({"k": kind, "v": content})},
        ],
    }


def _seed_both(tmp_path: Path, n_train: int, n_evo: int) -> Path:
    """Create an LTM tree with both *_training.jsonl and *_evolutions.jsonl."""
    root = tmp_path
    train_path = root / "ltm" / "agent_a" / "20260514_010000_training.jsonl"
    evo_path = root / "ltm" / "agent_a" / "20260514_010001_evolutions.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("w") as fh:
        for i in range(n_train):
            fh.write(json.dumps(_row(f"t{i}", "consolidation")) + "\n")
    with evo_path.open("w") as fh:
        for i in range(n_evo):
            fh.write(json.dumps(_row(f"e{i}", "evolution")) + "\n")
    return root


def _cfg(path: Path, *, include_evolutions: bool, max_samples: int | None = None) -> DataCfg:
    return DataCfg(
        source="mindx_dreams",
        path=path,
        include_evolutions=include_evolutions,
        max_samples=max_samples,
    )


def test_default_excludes_evolutions(tmp_path):
    root = _seed_both(tmp_path, n_train=5, n_evo=3)
    rows = list(load_streaming_dataset(_cfg(root, include_evolutions=False)))
    assert len(rows) == 5
    assert all("consolidation" in r["messages"][0]["content"] for r in rows)


def test_include_evolutions_yields_both_streams_in_order(tmp_path):
    root = _seed_both(tmp_path, n_train=5, n_evo=3)
    rows = list(load_streaming_dataset(_cfg(root, include_evolutions=True)))
    assert len(rows) == 8
    # First 5 should be consolidation, last 3 evolutions
    assert all("consolidation" in r["messages"][0]["content"] for r in rows[:5])
    assert all("evolution" in r["messages"][0]["content"] for r in rows[5:])


def test_max_samples_caps_combined_total(tmp_path):
    """Cap is over the combined stream, not per-source."""
    root = _seed_both(tmp_path, n_train=5, n_evo=3)
    # Cap below consolidation count → only consolidation yields.
    rows = list(load_streaming_dataset(_cfg(root, include_evolutions=True, max_samples=3)))
    assert len(rows) == 3
    assert all("consolidation" in r["messages"][0]["content"] for r in rows)


def test_max_samples_spans_both_streams(tmp_path):
    """Cap above consolidation count → drains consolidation then dips into evolutions."""
    root = _seed_both(tmp_path, n_train=5, n_evo=3)
    rows = list(load_streaming_dataset(_cfg(root, include_evolutions=True, max_samples=6)))
    assert len(rows) == 6
    assert sum(1 for r in rows if "consolidation" in r["messages"][0]["content"]) == 5
    assert sum(1 for r in rows if "evolution" in r["messages"][0]["content"]) == 1


def test_include_evolutions_round_trips_through_xtrainconfig(tmp_path):
    """The flag must validate cleanly inside a full XTrainConfig YAML."""
    cfg_text = yaml.safe_dump({
        "meta": {"project": "p", "run_name": "r"},
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "data": {
            "source": "mindx_dreams",
            "path": str(tmp_path),
            "include_evolutions": True,
        },
    })
    cfg = XTrainConfig.model_validate(yaml.safe_load(cfg_text))
    assert cfg.data.include_evolutions is True


def test_include_evolutions_defaults_to_false(tmp_path):
    cfg = DataCfg(source="mindx_dreams", path=tmp_path)
    assert cfg.include_evolutions is False
