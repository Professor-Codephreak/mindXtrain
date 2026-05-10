"""Pure-Python data pipeline pieces (filter, verify, synth interleave)."""

from __future__ import annotations

import json

from mindxtrain.data.filter import quality_filter
from mindxtrain.data.synth import merge_synth_with_real
from mindxtrain.data.verify import verify_dataset
from mindxtrain.provenance.hashing import blake3_file


def test_quality_filter_drops_short_doc():
    docs = ["short", "this is a much longer doc with several words and meaning okay yeah more here"]
    out = list(quality_filter(docs, min_words=8))
    assert len(out) == 1


def test_quality_filter_drops_high_repeat():
    repeat = "the quick brown fox " * 100
    out = list(quality_filter([repeat]))
    assert out == []


def test_synth_interleave_at_30pct():
    real = (f"r{i}" for i in range(10))
    synth = (f"s{i}" for i in range(10))
    blended = list(merge_synth_with_real(synth, real, ratio=0.3))[:10]
    # At ratio=0.3, ~3/10 should be synth.
    n_synth = sum(1 for x in blended if str(x).startswith("s"))
    assert 2 <= n_synth <= 5


def test_synth_ratio_zero_returns_real_only():
    real = ["r1", "r2"]
    synth = ["s1", "s2"]
    out = list(merge_synth_with_real(iter(synth), iter(real), ratio=0.0))
    assert out == ["r1", "r2"]


def test_verify_dataset_detects_mismatch(tmp_path):
    f = tmp_path / "shard-00000.tar"
    f.write_bytes(b"hello")
    actual = blake3_file(f)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shards": [{"path": "shard-00000.tar", "blake3": actual}]}))
    res = verify_dataset(manifest, tmp_path)
    assert res.matched == 1
    assert res.mismatched == []

    # Now tamper.
    f.write_bytes(b"tampered")
    res2 = verify_dataset(manifest, tmp_path)
    assert res2.matched == 0
    assert res2.mismatched == ["shard-00000.tar"]
