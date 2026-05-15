"""Canonical tokenizer wrapper (spec Rule 3.1).

The model's own `tokenizer.json` loaded via the HuggingFace `tokenizers`
(Rust) library is the authoritative token counter. Every reported token
figure carries the tokenizer revision hash, so cross-run comparisons
remain honest when tokenizer vocabularies differ (Qwen3.5's 248k vs
Qwen2.5's 152k vs Llama 3's 128k vocabulary).

The two cross-checks that matter operationally:

1. **Scaffold accounting (Rule 3.2)** — the ChatML wrapping adds 10-25
   tokens per turn. `count_with_scaffold` reports both inclusive and
   content-only counts so a "tok/s" figure can be disambiguated.
2. **Bytes-per-second (Rule 3.3)** — Qwen3.5 compresses English to ~3.5
   chars/tok where Llama 3 sits at ~4.0; tokenizer-relative throughput
   alone is misleading. `bytes_per_decoded_token` is the cross-vocabulary
   honest answer.

Lazy import: this module loads without `tokenizers` installed; callers
get a clear RuntimeError pointing at `uv sync --extra ml`.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any


def _require_tokenizers() -> Any:
    try:
        import tokenizers  # type: ignore
    except ImportError as exc:
        msg = (
            "MEI canonical tokenizer requires the `tokenizers` library — "
            "run `uv sync --extra ml`."
        )
        raise RuntimeError(msg) from exc
    return tokenizers


@lru_cache(maxsize=8)
def _load_tokenizer(model_id_or_path: str) -> Any:
    """Load Tokenizer.from_file(<dir>/tokenizer.json) or from_pretrained.

    Cached so repeated count calls don't re-parse the JSON. The cache key
    is the input string — callers can either pass a local path or an HF
    Hub repo ID; both resolve through the same code path.
    """
    tok_mod = _require_tokenizers()
    path = Path(model_id_or_path).expanduser()
    if path.is_dir() and (path / "tokenizer.json").exists():
        return tok_mod.Tokenizer.from_file(str(path / "tokenizer.json"))
    if path.is_file() and path.name == "tokenizer.json":
        return tok_mod.Tokenizer.from_file(str(path))
    # Fall back to HF Hub lookup via the transformers convenience layer if
    # available; otherwise raise so the caller knows the lookup failed.
    try:
        return tok_mod.Tokenizer.from_pretrained(model_id_or_path)
    except Exception as exc:
        msg = (
            f"could not load tokenizer for {model_id_or_path!r}: "
            f"neither a local tokenizer.json nor an HF Hub repo could be "
            f"resolved ({type(exc).__name__}: {exc})"
        )
        raise RuntimeError(msg) from exc


def tokenizer_revision(model_id_or_path: str) -> str:
    """Stable hash over the tokenizer.json bytes (spec Rule 3.1).

    Every reported token figure carries this hash so a measurement run
    can be replayed and cross-checked against the exact tokenizer
    revision it consumed.
    """
    path = Path(model_id_or_path).expanduser()
    tj = path / "tokenizer.json" if path.is_dir() else path
    if tj.exists() and tj.is_file():
        digest = hashlib.blake2b(tj.read_bytes(), digest_size=16).hexdigest()
        return f"local:{digest}"
    # For HF Hub references we can't BLAKE the local file (it isn't
    # materialized yet at this layer). Surface the repo identity instead.
    return f"hub:{model_id_or_path}"


def encode_with_revision(
    text: str, model_id_or_path: str, *, add_special_tokens: bool = False,
) -> tuple[list[int], str]:
    """Return (token_ids, tokenizer_revision_hash). Rule 3.1."""
    tok = _load_tokenizer(model_id_or_path)
    encoding = tok.encode(text, add_special_tokens=add_special_tokens)
    return list(encoding.ids), tokenizer_revision(model_id_or_path)


def _chatml_render(messages: list[dict[str, str]]) -> str:
    """Render a ChatML conversation the way Qwen tokenizers expect.

    The MEI counts every token the model processes, *including* scaffold
    (Rule 3.2). We render explicitly here rather than delegating to a
    tokenizer's `apply_chat_template` because we want byte-stable output
    regardless of template revisions — the count is what matters.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    # Trailing assistant prompt — the model's generation prefix.
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def count_with_scaffold(
    messages: list[dict[str, str]],
    model_id_or_path: str,
    *,
    include_scaffold: bool = True,
) -> int:
    """Token count for a chat-style payload (Rule 3.2).

    When `include_scaffold=True` (the operationally-honest default), the
    ChatML `<|im_start|>`, `<|im_end|>` markers and role headers are
    counted — matches what the model actually processes. When False, the
    count strips structural overhead — suitable for user-facing
    reporting only, never for tok/s denominators.
    """
    if include_scaffold:
        rendered = _chatml_render(messages)
    else:
        rendered = "\n".join(m.get("content", "") for m in messages)
    ids, _ = encode_with_revision(rendered, model_id_or_path)
    return len(ids)


def bytes_per_decoded_token(
    token_ids: list[int], model_id_or_path: str,
) -> float:
    """UTF-8 bytes per decoded token (Rule 3.3).

    Cross-tokenizer-invariant cross-check. A Qwen 100 tok/s and a Llama
    100 tok/s produce different amounts of useful output; this number is
    how MEI keeps cross-vocab comparisons honest. Returns 0.0 for empty
    token lists.
    """
    if not token_ids:
        return 0.0
    tok = _load_tokenizer(model_id_or_path)
    decoded = tok.decode(token_ids)
    return len(decoded.encode("utf-8")) / len(token_ids)


__all__ = [
    "bytes_per_decoded_token",
    "count_with_scaffold",
    "encode_with_revision",
    "tokenizer_revision",
]
