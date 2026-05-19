"""Held-out perplexity / cross-entropy of a LoRA-adapted checkpoint.

Answers the question Coach can't otherwise answer: did the adapter
actually move the base model toward the training distribution? The
trainer's `train_loss` field is averaged over the seen rows; only a
*held-out* slice tells us whether the adapter learned something
transferable, or just memorised the 32 rows we showed it.

Public entry point: `score_checkpoint(...)`. Loads the base model,
optionally wraps with the adapter, runs each chat row through with
`labels = input_ids`, returns mean CE loss for both. `adapter_loss <
base_loss` on the held-out slice is the signal that the adapter is
doing something useful.

Lazy-imports `torch` + `transformers` + `peft` so `import mindxtrain.eval`
stays cheap on the CPU-only base install (the `ml` extras gate
everything heavy). Mirrors the pattern in
`mindxtrain.deploy.ollama_push` so failures point at the same
`uv sync --extra ml` install hint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HeldOutScore:
    """Per-row + summary breakdown of held-out CE loss."""

    base_model: str
    adapter_dir: str
    n: int
    base_loss: float
    adapter_loss: float
    delta: float  # adapter_loss - base_loss; negative = adapter improved

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_jsonl_messages(path: Path, max_samples: int | None) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file shaped `{"messages": [...]}` (OpenAI chat)."""
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or "messages" not in row:
                continue
            yield row
            n += 1
            if max_samples is not None and n >= max_samples:
                return


def _row_loss(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> float:
    """Compute mean CE loss for `messages` rendered through the chat template."""
    import torch  # type: ignore

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"]
    # Causal LM loss = next-token CE averaged across the sequence. Pass
    # labels = input_ids so the model auto-shifts internally.
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return float(out.loss.detach().cpu().item())


def score_checkpoint(
    adapter_dir: Path,
    base_model: str,
    jsonl_path: Path,
    *,
    max_samples: int | None = 32,
    sink: Callable[[str], None] | None = None,
) -> HeldOutScore:
    """Mean held-out CE loss of (base) vs (base + adapter).

    `jsonl_path` is a `*_training.jsonl` file produced by mindX's
    machine_dreaming phase 5b — same shape `iter_mindx_dreams` yields.
    Loads the rows lazily, evaluates each under both model variants,
    returns the summary.

    Caller is responsible for picking rows the adapter *didn't* see at
    training time — `mindxtrain eval-checkpoint` defaults to the
    deterministic held-out slice when `data.eval_split` is set in the
    config; otherwise it samples random rows from the same path the
    recipe pointed at, which is a weaker (overlapping) signal but still
    catches catastrophic forgetting.
    """
    _emit = sink or (lambda _line: None)

    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        msg = (
            "held-out scoring needs `peft` + `transformers`. install with: "
            "uv sync --extra ml"
        )
        raise ImportError(msg) from exc

    rows = list(_iter_jsonl_messages(Path(jsonl_path), max_samples))
    if not rows:
        msg = f"no readable JSONL rows in {jsonl_path}"
        raise ValueError(msg)
    _emit(f"[eval] {len(rows)} held-out rows from {jsonl_path}")

    _emit(f"[eval] loading tokenizer + base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(base_model)
    base.eval()

    base_total = 0.0
    for row in rows:
        base_total += _row_loss(base, tokenizer, row["messages"])
    base_loss = base_total / len(rows)
    _emit(f"[eval] base mean loss = {base_loss:.4f}")

    _emit(f"[eval] applying adapter from {adapter_dir}")
    adapter_model = PeftModel.from_pretrained(base, str(adapter_dir))
    adapter_model.eval()

    adapter_total = 0.0
    for row in rows:
        adapter_total += _row_loss(adapter_model, tokenizer, row["messages"])
    adapter_loss = adapter_total / len(rows)
    _emit(f"[eval] adapter mean loss = {adapter_loss:.4f}")

    delta = adapter_loss - base_loss
    _emit(
        f"[eval] delta = {delta:+.4f} ({'improved' if delta < 0 else 'regressed'})",
    )
    return HeldOutScore(
        base_model=base_model,
        adapter_dir=str(adapter_dir),
        n=len(rows),
        base_loss=base_loss,
        adapter_loss=adapter_loss,
        delta=delta,
    )


__all__ = ["HeldOutScore", "score_checkpoint"]
