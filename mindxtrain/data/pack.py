"""Sequence packing + tar-shard emission. Pure stdlib.

`pack_sequences`: greedy first-fit-decreasing into `seq_len` buckets, each
sequence emitted with its EOS-bounded segment boundaries.

`emit_shards`: writes `.tar` shards under `out_dir/shard-{n:05d}.tar` with
JSONL members; returns `out_dir`. Caller pins the resulting tars via
`mindxtrain.storage.lighthouse` or `mindxtrain.storage.ipfs`.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path


def pack_sequences(
    token_streams: Iterable[list[int]],
    seq_len: int,
    *,
    eos_id: int = 0,
) -> Iterator[list[int]]:
    """Greedy first-fit pack of `token_streams` into `seq_len`-sized buckets.

    Each input sequence is appended verbatim followed by `eos_id`; if the
    next sequence would exceed `seq_len`, the current bucket is emitted
    (padded to `seq_len` with `eos_id`).
    """
    bucket: list[int] = []
    for stream in token_streams:
        s = [*list(stream), eos_id]
        if len(s) > seq_len:
            # Sequence is longer than the bucket — split into seq_len chunks.
            for i in range(0, len(s), seq_len):
                chunk = s[i : i + seq_len]
                if len(chunk) < seq_len:
                    chunk = chunk + [eos_id] * (seq_len - len(chunk))
                yield chunk
            continue
        if len(bucket) + len(s) > seq_len:
            # Pad and emit current bucket.
            yield bucket + [eos_id] * (seq_len - len(bucket))
            bucket = []
        bucket.extend(s)
    if bucket:
        yield bucket + [eos_id] * (seq_len - len(bucket))


def emit_shards(
    sequences: Iterable[list[int]],
    out_dir: Path,
    *,
    samples_per_shard: int = 1024,
) -> Path:
    """Write `.tar` shards under `out_dir`, each containing JSONL members."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    in_shard: list[bytes] = []

    def _flush() -> None:
        nonlocal shard_idx, in_shard
        if not in_shard:
            return
        path = out_dir / f"shard-{shard_idx:05d}.tar"
        with tarfile.open(path, "w") as tf:
            for j, payload in enumerate(in_shard):
                info = tarfile.TarInfo(name=f"sample-{shard_idx:05d}-{j:06d}.json")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
        shard_idx += 1
        in_shard = []

    for i, seq in enumerate(sequences):
        in_shard.append(json.dumps({"input_ids": seq}).encode("utf-8"))
        if (i + 1) % samples_per_shard == 0:
            _flush()
    _flush()

    return out_dir


__all__ = ["emit_shards", "pack_sequences"]
