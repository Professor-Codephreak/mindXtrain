"""Tokenization with the base model's tokenizer (cached per-model).

Lazy `import transformers` — users without `--extra ml` get a clean error.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        msg = "transformers not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def tokenize_stream(
    docs: Iterable[str],
    model_name: str,
    *,
    add_special_tokens: bool = False,
) -> Iterator[list[int]]:
    """Yield list[int] token ids per doc, using the model's fast tokenizer."""
    tok = _get_tokenizer(model_name)
    for doc in docs:
        if not isinstance(doc, str) or not doc:
            continue
        ids = tok.encode(doc, add_special_tokens=add_special_tokens)
        yield list(ids)


__all__ = ["tokenize_stream"]
