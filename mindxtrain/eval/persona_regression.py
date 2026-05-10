"""Persona regression — does the post-training model still sound like Codephreak?

Compares a sample of model generations against a baseline JSONL of
held-out persona examples using sentence-transformer cosine similarity.
"""

from __future__ import annotations

import json
from pathlib import Path


def regression_score(
    samples: list[str],
    baseline_jsonl: Path,
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Return persona-similarity score in [0, 1]; 1.0 = identical voice.

    `samples` is a list of generated strings (caller is responsible for
    producing them via inference). `baseline_jsonl` is one persona example
    per line (each line a JSON object with a `text` field).
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        msg = "sentence-transformers + numpy not installed; run `uv sync --extra data`."
        raise RuntimeError(msg) from exc

    baseline_texts: list[str] = []
    with Path(baseline_jsonl).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("content") or ""
            if text:
                baseline_texts.append(text)

    if not samples or not baseline_texts:
        return 0.0

    enc = SentenceTransformer(model)
    sample_embs = enc.encode(samples, normalize_embeddings=True)
    baseline_embs = enc.encode(baseline_texts, normalize_embeddings=True)

    # Per-sample max cosine vs. any baseline; mean across samples.
    sims = sample_embs @ baseline_embs.T
    per_sample_max = sims.max(axis=1)
    return float(np.mean(per_sample_max))


__all__ = ["regression_score"]
