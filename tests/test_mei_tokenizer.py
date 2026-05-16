"""Canonical tokenizer (spec Rule 3.1 — 3.3).

The pure-stdlib pieces (revision hash, scaffold rendering) run on a base
install. Encoding-through-Rust pieces are skipped when `tokenizers` isn't
available — they need `uv sync --extra ml`.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from mindxtrain.eval.mei.tokenizer import (
    _chatml_render,
    tokenizer_revision,
)

_HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


# ---- tokenizer_revision (pure stdlib) ------------------------------------


def test_revision_uses_blake_of_local_tokenizer_json(tmp_path):
    """Per Rule 3.1, every report carries the tokenizer revision."""
    tjson = tmp_path / "tokenizer.json"
    tjson.write_text(json.dumps({"version": "1.0", "vocab": []}))
    rev = tokenizer_revision(str(tmp_path))
    assert rev.startswith("local:")
    # 16-byte BLAKE2b → 32 hex chars.
    assert len(rev.split(":", 1)[1]) == 32


def test_revision_handles_direct_tokenizer_json_path(tmp_path):
    tjson = tmp_path / "tokenizer.json"
    tjson.write_text("{}")
    rev = tokenizer_revision(str(tjson))
    assert rev.startswith("local:")


def test_revision_falls_back_to_hub_marker_for_unknown_path():
    """HF Hub repo IDs that aren't materialized locally surface as `hub:`."""
    rev = tokenizer_revision("Qwen/Qwen3-1.5B-NotMaterialized")
    assert rev == "hub:Qwen/Qwen3-1.5B-NotMaterialized"


def test_revision_changes_when_tokenizer_content_changes(tmp_path):
    """Different tokenizer.json bytes → different revision hash."""
    tjson = tmp_path / "tokenizer.json"
    tjson.write_text('{"version":"1.0"}')
    rev_a = tokenizer_revision(str(tmp_path))
    tjson.write_text('{"version":"1.1"}')
    rev_b = tokenizer_revision(str(tmp_path))
    assert rev_a != rev_b


# ---- ChatML scaffold rendering (pure stdlib) -----------------------------


def test_chatml_render_basic_three_turn():
    text = _chatml_render([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ])
    assert "<|im_start|>system\nS<|im_end|>\n" in text
    assert "<|im_start|>user\nU<|im_end|>\n" in text
    # Trailing assistant prompt — the model's generation prefix.
    assert text.endswith("<|im_start|>assistant\n")


def test_chatml_render_handles_empty_content():
    text = _chatml_render([{"role": "user", "content": ""}])
    assert "<|im_start|>user\n<|im_end|>\n" in text


def test_chatml_render_defaults_role_when_missing():
    """Robust to upstream callers that forget the role key."""
    text = _chatml_render([{"content": "hello"}])
    assert "<|im_start|>user\nhello<|im_end|>\n" in text


# ---- Encoding through the Rust tokenizer (requires --extra ml) ----------


def _build_minimal_bpe_tokenizer(tmp_path):
    """Build a tiny but valid BPE tokenizer with [UNK] registered in the
    vocab so Rust's encode() won't reject inputs. The vocab is a single
    [UNK] token plus pre-tokenizer byte-level — enough for the smoke
    tests to drive a real encode/decode without an HF download.
    """
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import BPE

    tok = Tokenizer(BPE(vocab={"[UNK]": 0}, merges=[], unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tj = tmp_path / "tokenizer.json"
    tok.save(str(tj))
    # Bust the lru_cache so the test gets a fresh load.
    from mindxtrain.eval.mei import tokenizer as tok_mod
    tok_mod._load_tokenizer.cache_clear()
    return tmp_path


@pytest.mark.skipif(not _HAS_TOKENIZERS, reason="tokenizers (Rust) not installed")
def test_encode_with_revision_returns_ids_and_hash(tmp_path):
    """End-to-end Rule 3.1: encode + revision hash for a synthetic
    tokenizer.json. Uses a minimal-but-valid BPE so the Rust encoder
    can actually run."""
    _build_minimal_bpe_tokenizer(tmp_path)
    from mindxtrain.eval.mei.tokenizer import encode_with_revision
    ids, rev = encode_with_revision("hello", str(tmp_path))
    assert rev.startswith("local:")
    # A vocab-of-one BPE collapses every input to [UNK] tokens — id 0.
    assert all(i == 0 for i in ids)


@pytest.mark.skipif(not _HAS_TOKENIZERS, reason="tokenizers (Rust) not installed")
def test_count_with_scaffold_inclusive_exceeds_content_only(tmp_path):
    """Rule 3.2: scaffold-inclusive count is at least content-only count.

    On a real Qwen / Llama tokenizer it would be strictly greater (~10-25
    extra tokens per turn). On this single-token vocab everything maps to
    [UNK]s; the assertion is bound to character-count delta instead."""
    _build_minimal_bpe_tokenizer(tmp_path)
    from mindxtrain.eval.mei.tokenizer import count_with_scaffold
    messages = [{"role": "system", "content": "hi"}, {"role": "user", "content": "ok"}]
    inclusive = count_with_scaffold(messages, str(tmp_path), include_scaffold=True)
    content = count_with_scaffold(messages, str(tmp_path), include_scaffold=False)
    assert inclusive >= content
    # The chatml scaffold adds *characters*, which on byte-level become
    # more tokens. Sanity check: inclusive should not equal content for
    # non-empty messages on this byte-level pre-tokenizer.
    assert inclusive > content


def test_unknown_path_raises_clear_runtime_error_when_no_tokenizers(monkeypatch, tmp_path):
    """The lazy-import contract: importing the module always succeeds,
    but calling an encode function without `tokenizers` installed raises
    a clear RuntimeError pointing at the install command.

    Simulate the missing dep by monkey-patching the importer.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tokenizers":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Bust the lru_cache + drop the module so the next call re-imports.
    from mindxtrain.eval.mei import tokenizer as tok_mod
    tok_mod._load_tokenizer.cache_clear()

    from mindxtrain.eval.mei.tokenizer import encode_with_revision
    with pytest.raises(RuntimeError, match="uv sync --extra ml"):
        encode_with_revision("hi", str(tmp_path))
