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
