"""RLHF (PPO) trainer — TRL PPOTrainer wrapper.

Online preference optimization with a learned reward model. Less efficient
than DPO for offline pairs; kept here as the canonical surface for online
RLHF runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mindxtrain.config.schema import XTrainConfig


def run_rlhf(cfg: XTrainConfig, out_dir: Path) -> Path:
    """Run a PPO/RLHF fine-tune; return the checkpoint directory."""
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import PPOConfig, PPOTrainer
    except ImportError as exc:
        msg = "TRL stack not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    method: Any = cfg.train.method
    if getattr(method, "kind", "") != "rlhf":
        msg = f"run_rlhf expects train.method.kind == 'rlhf'; got {getattr(method, 'kind', None)!r}"
        raise ValueError(msg)

    reward_model_path = getattr(method, "reward_model_path", "")
    if not reward_model_path:
        msg = "rlhf requires train.method.reward_model_path"
        raise ValueError(msg)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    policy = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    ref_policy = AutoModelForCausalLM.from_pretrained(cfg.model.name)
    reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)

    train_ds = load_dataset(cfg.data.hf_id, split=getattr(cfg.data, "split", "train"))

    ppo_cfg = PPOConfig(
        output_dir=str(out_dir),
        learning_rate=cfg.train.optim.learning_rate,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_epochs,
    )
    trainer = PPOTrainer(
        args=ppo_cfg,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


__all__ = ["run_rlhf"]
