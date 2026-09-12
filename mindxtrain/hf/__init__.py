"""mindxtrain.hf — the Hugging Face extension.

The framework already knew how to *push a folder* (`mindxtrain.storage.hf_hub`). This package is
the rest of the Hub as mindXtrain needs it, and nothing more:

- `account` — who the token is, which namespaces it can actually write (membership is not write scope)
- `publish_generation` — a trained run as a model repo: merged weights + adapter + train.log + a
  Modelfile carrying the persona, and a card written from the run's own numbers
- `pull_base` / `warm` — fetch a base before training so a run does not fail three hours in
- `lineage` — what is on the Hub for a project, reconciled against local runs
- `spaces` — push a Gradio UI as a Space (the free ZeroGPU slot rules are stated, not guessed)
- `datasets` — push/pull a training corpus with its manifest

Every function returns a plain dict: `{"ok": bool, …}`. Nothing here raises at import time, and
`huggingface_hub` is imported lazily so a CPU-only install without `--extra chain` still loads.
"""
from .extension import (  # noqa: F401
    account,
    lineage,
    publish_generation,
    pull_base,
    push_dataset,
    push_space,
    warm,
)

__all__ = ["account", "lineage", "publish_generation", "pull_base", "push_dataset", "push_space", "warm"]
