"""Dataset curation — dispatch on `DataCfg.source` to the right adapter.

Three sources today:

- `hf` (default) — `datasets.load_dataset(streaming=True)`, requires `--extra ml`.
- `local` — JSONL files under `cfg.path`, pure stdlib.
- `mindx_dreams` — mindX dream-cycle JSONL corpus under `cfg.path`, pure stdlib.
  See `mindxtrain.data.sources.mindx_dreams`.

The HF path stays lazy-import so consumers without `--extra ml` can still
read configs that target other sources.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mindxtrain.config.schema import DataCfg


def load_streaming_dataset(cfg: DataCfg) -> Iterator[dict[str, Any]]:
    """Yield rows for the configured `DataCfg.source` one at a time."""
    if cfg.source == "mindx_dreams":
        from mindxtrain.data.sources.mindx_dreams import (
            iter_mindx_dreams,
            iter_mindx_evolutions,
        )

        assert cfg.path is not None  # validated by DataCfg
        # Two-stream yield with a SHARED budget so max_samples caps the
        # combined corpus, not each stream individually. Consolidation rows
        # come first (richer base signal), evolutions follow when opted in.
        cap = cfg.max_samples
        emitted = 0
        remaining = (cap - emitted) if cap is not None else None
        for row in iter_mindx_dreams(cfg.path, max_samples=remaining):
            yield row
            emitted += 1
            if cap is not None and emitted >= cap:
                return
        if cfg.include_evolutions:
            remaining = (cap - emitted) if cap is not None else None
            try:
                for row in iter_mindx_evolutions(cfg.path, max_samples=remaining):
                    yield row
                    emitted += 1
                    if cap is not None and emitted >= cap:
                        return
            except FileNotFoundError:
                # mindx_dreams already succeeded above, so the root exists —
                # this would only fire on a path race. Swallow silently.
                pass
        return

    if cfg.source == "local":
        assert cfg.path is not None  # validated by DataCfg
        yield from _iter_local_jsonl(cfg.path, max_samples=cfg.max_samples)
        return

    if cfg.source == "lighthouse":
        msg = (
            "DataCfg.source='lighthouse' is reserved for shard-tar inputs; "
            "use `mindxtrain.storage.lighthouse.fetch` to materialize them locally first, "
            "then point a `local` source at the resulting directory."
        )
        raise NotImplementedError(msg)

    # source == "hf"
    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "datasets not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    kwargs: dict[str, Any] = {"streaming": cfg.streaming}
    revision = getattr(cfg, "revision", None)
    if revision:
        kwargs["revision"] = revision

    ds = load_dataset(cfg.hf_id, split=cfg.split or "train", **kwargs)
    emitted = 0
    for row in ds:
        yield row
        emitted += 1
        if cfg.max_samples is not None and emitted >= cfg.max_samples:
            return


def _iter_local_jsonl(
    path: Path,
    *,
    max_samples: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Walk *.jsonl under `path` and yield parsed rows.

    Skips lines that fail to parse — local datasets are often hand-assembled
    and a single bad line shouldn't fail the run.
    """
    path = Path(path).expanduser()
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.jsonl"))
    emitted = 0
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield row
                emitted += 1
                if max_samples is not None and emitted >= max_samples:
                    return


__all__ = ["load_streaming_dataset"]
