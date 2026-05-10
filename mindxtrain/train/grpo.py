"""GRPO trainer — TRL GRPOTrainer wrapper.

Lazy `import trl` so users without `--extra ml` get a clean error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mindxtrain.config.schema import XTrainConfig


def run_grpo(cfg: XTrainConfig, out_dir: Path) -> Path:
    """Run a GRPO fine-tune; return the checkpoint directory."""
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        msg = "TRL stack not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    method: Any = cfg.train.method
    if getattr(method, "kind", "") != "grpo":
        msg = f"run_grpo expects train.method.kind == 'grpo'; got {getattr(method, 'kind', None)!r}"
        raise ValueError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    train_ds = load_dataset(cfg.data.hf_id, split=getattr(cfg.data, "split", "train"))

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=cfg.train.optim.learning_rate,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_epochs,
        max_completion_length=cfg.data.seq_len,
        num_generations=getattr(method, "num_generations", 4),
        logging_steps=10,
    )
    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[lambda **_: 0.0],  # caller wires real reward
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


__all__ = ["run_grpo"]
