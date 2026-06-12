"""emit_receipt_for_run — tolerant run-completion manifest emission (CPU lane)."""

from __future__ import annotations

import json

import pytest

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.loader import list_recipes, load_config, render_recipe
from mindxtrain.provenance import manifest as _m
from mindxtrain.provenance.manifest import (
    emit_receipt_for_run,
    write_run_manifest,
)
from mindxtrain.provenance.verify import verify_receipt


@pytest.fixture(autouse=True)
def _no_network_attestation(monkeypatch):
    """Keep emit_receipt_for_run offline + fast in unit tests."""
    monkeypatch.setattr(_m, "_fetch_time_attestation", lambda: _m.TimeAttestation())


def _a_cpu_config():
    """Load the first available CPU smoke recipe as a real XTrainConfig."""
    name = next(
        (r for r in list_recipes() if "cpu" in r and "smoke" in r),
        "mindx_fallback_qwen3_1_5b_cpu_smoke",
    )
    return load_config_from_recipe(name)


def load_config_from_recipe(name: str):
    import tempfile
    from pathlib import Path

    text = render_recipe(name)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"{name}.yaml"
        p.write_text(text)
        return load_config(p)


def test_emit_receipt_for_run_cpu_only_checkpoint(tmp_path):
    cfg = _a_cpu_config()
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 128)

    plan = AutotunePlan()
    m = emit_receipt_for_run(cfg, "cpu-smoke-1", run_dir=tmp_path, plan=plan)

    # Snapshots written into the run dir.
    assert (tmp_path / "config.snapshot.yaml").is_file()
    assert (tmp_path / "autotune_plan.json").is_file()

    # Always-present hashes are set; absent artifacts stay empty.
    assert m.blake3.config_yaml
    assert m.blake3.checkpoint
    assert m.blake3.autotune_plan
    assert m.blake3.dataset == ""
    assert m.blake3.eval_json == ""

    # write_run_manifest round-trips on disk.
    out = write_run_manifest(m, tmp_path)
    assert out == tmp_path / "manifest.json"
    restored = _m.Manifest.model_validate(json.loads(out.read_text()))
    assert restored.run_id == "cpu-smoke-1"

    # Full receipt verifies against the persisted artifacts.
    plan_bytes = (tmp_path / "autotune_plan.json").read_bytes()
    res = verify_receipt(
        m,
        config_yaml_path=tmp_path / "config.snapshot.yaml",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        checkpoint_dir=ckpt,
        eval_json_path=tmp_path / "eval" / "lm_eval.json",
        plan_json=plan_bytes,
    )
    assert all(res.values())


def test_emit_receipt_for_run_detects_checkpoint_tamper(tmp_path):
    cfg = _a_cpu_config()
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 128)

    plan = AutotunePlan()
    m = emit_receipt_for_run(cfg, "cpu-smoke-2", run_dir=tmp_path, plan=plan)

    # Tamper the checkpoint after the receipt is sealed.
    (ckpt / "adapter_model.safetensors").write_bytes(b"\xff" * 128)

    plan_bytes = (tmp_path / "autotune_plan.json").read_bytes()
    res = verify_receipt(
        m,
        config_yaml_path=tmp_path / "config.snapshot.yaml",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        checkpoint_dir=ckpt,
        eval_json_path=tmp_path / "eval" / "lm_eval.json",
        plan_json=plan_bytes,
    )
    assert res["checkpoint"] is False
    assert res["config_yaml"] is True
    assert res["autotune_plan"] is True
