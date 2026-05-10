"""Dataset curation — load streaming HF datasets per `DataCfg`.

Lazy `import datasets` so users without `--extra ml` can still import this
module. Returns an iterator of `dict[str, Any]` rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from mindxtrain.config.schema import DataCfg


def load_streaming_dataset(cfg: DataCfg) -> Iterator[dict[str, Any]]:
    """Load a streaming HF dataset per DataCfg.

    Honors `cfg.hf_id`, `cfg.split`, optional `cfg.revision`, and yields
    one row at a time so downstream packing/dedupe never materializes the
    full corpus.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = "datasets not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    kwargs: dict[str, Any] = {
        "streaming": True,
    }
    split = getattr(cfg, "split", None) or "train"
    revision = getattr(cfg, "revision", None)
    if revision:
        kwargs["revision"] = revision

    ds = load_dataset(cfg.hf_id, split=split, **kwargs)
    yield from ds


__all__ = ["load_streaming_dataset"]
