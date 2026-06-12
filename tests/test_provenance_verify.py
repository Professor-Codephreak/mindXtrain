"""Provenance Manifest verify happy + tamper paths."""

from __future__ import annotations

import json

from mindxtrain.provenance.hashing import blake3_dir, blake3_file
from mindxtrain.provenance.manifest import Manifest, ProvenanceHashes
from mindxtrain.provenance.verify import verify_receipt


def test_verify_passes_with_matching_hashes(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("meta: {}")
    ds = tmp_path / "dataset.json"
    ds.write_text("{}")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "weights.safetensors").write_bytes(b"\x00" * 64)
    eval_ = tmp_path / "eval.json"
    eval_.write_text("{}")

    m = Manifest(
        run_id="r1",
        base_model="x",
        blake3=ProvenanceHashes(
            config_yaml=blake3_file(cfg),
            dataset=blake3_file(ds),
            checkpoint=blake3_dir(ckpt),
            eval_json=blake3_file(eval_),
        ),
    )
    res = verify_receipt(
        m,
        config_yaml_path=cfg,
        dataset_manifest_path=ds,
        checkpoint_dir=ckpt,
        eval_json_path=eval_,
    )
    assert all(res.values())


def test_verify_detects_checkpoint_tamper(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a")
    ds = tmp_path / "ds.json"
    ds.write_text("a")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "w.bin").write_bytes(b"a")
    eval_ = tmp_path / "e.json"
    eval_.write_text("a")

    m = Manifest(
        run_id="r1",
        base_model="x",
        blake3=ProvenanceHashes(
            config_yaml=blake3_file(cfg),
            dataset=blake3_file(ds),
            checkpoint=blake3_dir(ckpt),
            eval_json=blake3_file(eval_),
        ),
    )
    # Tamper the checkpoint after the manifest is sealed.
    (ckpt / "w.bin").write_bytes(b"b")

    res = verify_receipt(
        m,
        config_yaml_path=cfg,
        dataset_manifest_path=ds,
        checkpoint_dir=ckpt,
        eval_json_path=eval_,
    )
    assert res["checkpoint"] is False
    assert res["config_yaml"] is True


def test_verify_passes_when_dataset_and_eval_absent(tmp_path):
    """A CPU run writes only a checkpoint; empty dataset/eval hashes must pass."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("meta: {}")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter.safetensors").write_bytes(b"\x01" * 32)

    m = Manifest(
        run_id="cpu-run",
        base_model="x",
        blake3=ProvenanceHashes(
            config_yaml=blake3_file(cfg),
            checkpoint=blake3_dir(ckpt),
            # dataset / eval_json / autotune_plan left at "" default
        ),
    )
    res = verify_receipt(
        m,
        config_yaml_path=cfg,
        dataset_manifest_path=tmp_path / "missing_dataset.json",
        checkpoint_dir=ckpt,
        eval_json_path=tmp_path / "missing_eval.json",
    )
    assert all(res.values())
    assert res["dataset"] is True
    assert res["eval_json"] is True
    assert res["autotune_plan"] is True


def test_verify_autotune_plan_round_trip_and_tamper(tmp_path):
    """The bound AutotunePlan hash verifies against its exact persisted bytes."""
    from mindxtrain.autotune.plan import AutotunePlan
    from mindxtrain.provenance.hashing import blake3_bytes

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("meta: {}")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter.safetensors").write_bytes(b"\x02" * 32)

    plan = AutotunePlan(attention_backend="ck", gemm_heuristic="hipblaslt_default")
    plan_bytes = plan.model_dump_json(indent=2).encode()

    m = Manifest(
        run_id="plan-run",
        base_model="x",
        blake3=ProvenanceHashes(
            config_yaml=blake3_file(cfg),
            checkpoint=blake3_dir(ckpt),
            autotune_plan=blake3_bytes(plan_bytes),
        ),
    )

    good = verify_receipt(
        m,
        config_yaml_path=cfg,
        dataset_manifest_path=tmp_path / "nope.json",
        checkpoint_dir=ckpt,
        eval_json_path=tmp_path / "nope.json",
        plan_json=plan_bytes,
    )
    assert good["autotune_plan"] is True

    # A different plan (or missing bytes) fails the plan check.
    tampered = plan.model_copy(update={"attention_backend": "triton"})
    bad = verify_receipt(
        m,
        config_yaml_path=cfg,
        dataset_manifest_path=tmp_path / "nope.json",
        checkpoint_dir=ckpt,
        eval_json_path=tmp_path / "nope.json",
        plan_json=tampered.model_dump_json(indent=2).encode(),
    )
    assert bad["autotune_plan"] is False

    missing = verify_receipt(
        m,
        config_yaml_path=cfg,
        dataset_manifest_path=tmp_path / "nope.json",
        checkpoint_dir=ckpt,
        eval_json_path=tmp_path / "nope.json",
        plan_json=None,
    )
    assert missing["autotune_plan"] is False


def test_manifest_round_trip_through_json():
    m = Manifest(
        run_id="r-2",
        base_model="qwen3.5",
        blake3=ProvenanceHashes(
            config_yaml="0" * 64, dataset="1" * 64, checkpoint="2" * 64, eval_json="3" * 64
        ),
    )
    blob = m.model_dump_json()
    restored = Manifest.model_validate(json.loads(blob))
    assert restored.run_id == "r-2"
