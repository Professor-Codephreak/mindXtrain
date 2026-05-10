"""BFCL-style tool-trajectory training — supervised on multi-turn tool calls.

Uses TRL's SFTTrainer with a tool-call dataset format compatible with the
Berkeley Function-Calling Leaderboard schema (single-turn function calls,
multi-turn trajectories, error-recovery turns).
"""

from __future__ import annotations

from pathlib import Path

from mindxtrain.config.schema import XTrainConfig


def run_tool_use(cfg: XTrainConfig, out_dir: Path) -> Path:
    """Run a tool-use SFT pass; return the checkpoint directory."""
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        msg = "TRL + transformers + datasets not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    train_ds = load_dataset(cfg.data.hf_id, split=getattr(cfg.data, "split", "train"))

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        learning_rate=cfg.train.optim.learning_rate,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        num_train_epochs=cfg.train.num_epochs,
        max_seq_length=cfg.data.seq_len,
        packing=cfg.data.packing,
        logging_steps=10,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    return out_dir


__all__ = ["run_tool_use"]
