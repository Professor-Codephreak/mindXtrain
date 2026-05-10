"""custmodel Manifest + BLAKE3 hashing determinism."""

from __future__ import annotations

import json

from mindxtrain.provenance.hashing import blake3_dir, blake3_file
from mindxtrain.provenance.manifest import Manifest, ProvenanceHashes


def test_blake3_file_is_deterministic(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello mindXtrain")
    assert blake3_file(p) == blake3_file(p)
    assert len(blake3_file(p)) == 64  # hex digest of 32-byte blake3 output


def test_blake3_dir_walks_sorted(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "z.txt").write_text("z")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "m.txt").write_text("m")
    h1 = blake3_dir(tmp_path)
    h2 = blake3_dir(tmp_path)
    assert h1 == h2
    assert len(h1) == 64


def test_blake3_dir_changes_with_content(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    h1 = blake3_dir(tmp_path)
    (tmp_path / "a.txt").write_text("b")
    h2 = blake3_dir(tmp_path)
    assert h1 != h2


def test_manifest_round_trip():
    hashes = ProvenanceHashes(
        config_yaml="0" * 64,
        dataset="1" * 64,
        checkpoint="2" * 64,
        eval_json="3" * 64,
    )
    m = Manifest(
        run_id="instella-3b-alpaca-lora-demo",
        base_model="amd/Instella-3B-Instruct",
        blake3=hashes,
    )
    blob = m.model_dump_json()
    restored = Manifest.model_validate(json.loads(blob))
    assert restored.run_id == m.run_id
    assert restored.blake3.config_yaml == "0" * 64
    assert restored.on_chain.inft.chain == "base_sepolia"


def test_manifest_schema_dump():
    schema = Manifest.model_json_schema()
    assert schema["title"] == "Manifest"
    assert "blake3" in schema["properties"]
