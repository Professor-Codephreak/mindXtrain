"""Quality filtering — pure-Python heuristics + optional KenLM perplexity.

Cheap heuristics first (length, repetition, language-ish character class).
KenLM is optional; if not installed, the perplexity gate is skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _ngram_repeat_ratio(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are duplicates of an earlier n-gram."""
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) < n:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def _alpha_ratio(text: str) -> float:
    """Fraction of characters that are alphabetic (rough language gate)."""
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def quality_filter(
    docs: Iterable[str],
    *,
    min_words: int = 16,
    max_words: int = 32_768,
    max_repeat_ratio: float = 0.3,
    min_alpha_ratio: float = 0.5,
    max_perplexity: float | None = None,
) -> Iterator[str]:
    """Yield docs that pass the heuristic gate.

    `max_perplexity` triggers a KenLM check if the lib is installed; ignored
    silently otherwise.
    """
    kenlm_model = None
    if max_perplexity is not None:
        try:
            # Caller can override which model to load by setting MINDXTRAIN_KENLM_PATH;
            # fall back to skipping the perplexity gate if no model is configured.
            import os

            import kenlm

            path = os.environ.get("MINDXTRAIN_KENLM_PATH")
            if path:
                kenlm_model = kenlm.Model(path)
        except ImportError:
            kenlm_model = None

    for doc in docs:
        if not isinstance(doc, str) or not doc:
            continue
        words = _WORD_RE.findall(doc)
        if len(words) < min_words or len(words) > max_words:
            continue
        if _ngram_repeat_ratio(doc) > max_repeat_ratio:
            continue
        if _alpha_ratio(doc) < min_alpha_ratio:
            continue
        if kenlm_model is not None and max_perplexity is not None:
            ppl = kenlm_model.perplexity(doc)
            if ppl > max_perplexity:
                continue
        yield doc


__all__ = ["quality_filter"]
