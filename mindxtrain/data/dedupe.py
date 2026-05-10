"""Deduplication — MinHash near-duplicate detection + SemDeDup semantic similarity.

Two complementary passes:

- `dedupe_minhash`: datasketch.MinHashLSH on 5-gram char shingles. Cheap,
  syntactic, removes verbatim near-duplicates.
- `dedupe_semdedup`: sentence-transformer embeddings + FAISS cosine. Drops
  docs within `threshold` similarity of an earlier doc.

Both lazy-import the heavyweight deps; `--extra data` enables them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def _shingle(text: str, k: int = 5) -> list[str]:
    text = text.lower()
    if len(text) < k:
        return [text] if text else []
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def dedupe_minhash(
    docs: Iterable[str],
    threshold: float = 0.85,
    *,
    num_perm: int = 128,
    shingle_k: int = 5,
) -> Iterator[str]:
    """Yield docs that are not near-duplicates of an earlier doc."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        msg = "datasketch not installed; run `uv sync --extra data`."
        raise RuntimeError(msg) from exc

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, doc in enumerate(docs):
        if not isinstance(doc, str) or not doc:
            continue
        m = MinHash(num_perm=num_perm)
        for s in _shingle(doc, shingle_k):
            m.update(s.encode("utf-8"))
        if lsh.query(m):
            continue
        lsh.insert(f"doc-{idx}", m)
        yield doc


def dedupe_semdedup(
    docs: Iterable[str],
    threshold: float = 0.95,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    batch_size: int = 64,
) -> Iterator[str]:
    """Yield docs not within `threshold` cosine similarity of an earlier doc."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        msg = "sentence-transformers + numpy not installed; run `uv sync --extra data`."
        raise RuntimeError(msg) from exc

    encoder = SentenceTransformer(model)
    seen_embeddings: list[np.ndarray] = []  # type: ignore[type-arg]

    buffer: list[str] = []

    def _flush() -> Iterator[str]:
        nonlocal buffer
        if not buffer:
            return
        embs = encoder.encode(buffer, normalize_embeddings=True, batch_size=batch_size)
        for txt, emb in zip(buffer, embs, strict=True):
            if seen_embeddings:
                stack = np.stack(seen_embeddings)
                sims = stack @ emb
                if float(sims.max()) >= threshold:
                    continue
            seen_embeddings.append(emb)
            yield txt
        buffer = []

    for doc in docs:
        if not isinstance(doc, str) or not doc:
            continue
        buffer.append(doc)
        if len(buffer) >= batch_size:
            yield from _flush()
    yield from _flush()


__all__ = ["dedupe_minhash", "dedupe_semdedup"]
