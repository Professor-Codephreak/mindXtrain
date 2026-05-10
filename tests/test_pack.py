"""Sequence packing + shard emission invariants."""

from __future__ import annotations

import io
import json
import tarfile

from mindxtrain.data.pack import emit_shards, pack_sequences


def test_pack_emits_full_seq_len_buckets():
    streams = [[1, 2, 3], [4, 5, 6, 7]]
    seq_len = 8
    packed = list(pack_sequences(streams, seq_len, eos_id=0))
    for bucket in packed:
        assert len(bucket) == seq_len
    flat = [t for bucket in packed for t in bucket if t != 0]
    # Original tokens preserved in order.
    assert flat[: len([1, 2, 3, 4, 5, 6, 7])] == [1, 2, 3, 4, 5, 6, 7]


def test_pack_oversize_sequence_chunks():
    seq_len = 4
    long = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    out = list(pack_sequences([long], seq_len, eos_id=0))
    assert all(len(b) == seq_len for b in out)
    # Last bucket padded with EOS.
    assert out[-1][-1] == 0


def test_emit_shards_writes_tarballs(tmp_path):
    streams = [[i for i in range(20)] for _ in range(3)]
    out = emit_shards(streams, tmp_path / "shards", samples_per_shard=2)
    tars = list(out.glob("*.tar"))
    assert len(tars) >= 1
    with tarfile.open(tars[0]) as tf:
        members = tf.getmembers()
        assert len(members) == 2
        payload = tf.extractfile(members[0])
        assert payload is not None
        rec = json.loads(io.BytesIO(payload.read()).read())
        assert "input_ids" in rec
