"""DPO trainer — TRL DPOTrainer wrapper.

Lazy `import trl` so users without `--extra ml` get a clean error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mindxtrain.config.schema import XTrainConfig


def run_dpo(cfg: XTrainConfig, out_dir: Path) -> Path:
    """Run a DPO fine-tune; return the checkpoint directory."""
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        msg = "TRL stack not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    method: Any = cfg.train.method
    if getattr(method, "kind", "") != "dpo":
        msg = f"run_dpo expects train.method.kind == 'dpo'; got {getattr(method, 'kind', None)!r}"
        raise ValueError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    train_ds = load_dataset(cfg.data.hf_id, split=getattr(cfg.data, "split", "train"))

    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        beta=getattr(method, "beta", 0.1),
        learning_rate=cfg.train.optim.learning_rate,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_epochs,
        max_length=cfg.data.seq_len,
        logging_steps=10,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


__all__ = ["run_dpo"]
