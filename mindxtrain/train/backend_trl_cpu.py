"""CPU training backend — real SFT/LoRA via TRL, no GPU required.

The closed-loop case: mindX produces a small JSONL dataset (dream cycle),
mindXtrain needs to fine-tune a tiny base model on it locally without
provisioning a MI300X droplet. This backend exists so that mindX agents can
trigger self-training on commodity hardware; it is also the smoke lane for
any new recipe before burning AMD credits.

Slow but produces a *real* checkpoint and is compatible with the rest of the
pipeline (`mindxtrain quantize`, `mindxtrain receipt`, `publish` — all
expect a HF-format checkpoint directory).

This module follows the project lazy-import contract: importing the module
must succeed on a base install, but calling `run_trl_cpu` requires
`uv sync --extra ml`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.schema import LoraMethod, QLoraMethod, XTrainConfig


def _require_ml_deps() -> dict[str, Any]:
    """Import TRL + transformers + peft + datasets eagerly; surface a single message."""
    missing: list[str] = []
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        missing.append("datasets")
        Dataset = None  # type: ignore
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        missing.append("transformers")
        AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError:
        missing.append("trl")
        SFTConfig = SFTTrainer = None  # type: ignore
    try:
        from peft import LoraConfig  # type: ignore
    except ImportError:
        # peft only required for LoRA/QLoRA — kept optional here
        LoraConfig = None  # type: ignore

    if missing:
        msg = (
            f"CPU training backend requires {', '.join(missing)} — "
            "run `uv sync --extra ml`."
        )
        raise RuntimeError(msg)

    return {
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "LoraConfig": LoraConfig,
    }


def _stream_dataset_rows(cfg: XTrainConfig) -> Iterator[dict[str, Any]]:
    """Yield raw dataset rows for the configured DataCfg.source."""
    from mindxtrain.data.curate import load_streaming_dataset

    yield from load_streaming_dataset(cfg.data)


def _materialize_dataset(cfg: XTrainConfig, Dataset: Any) -> Any:
    """Pull the stream into an in-memory `datasets.Dataset` for TRL.

    CPU corpora are small by construction; we don't try to stream into TRL
    here since `SFTTrainer` wants a `Dataset` with `__len__`. Cap at
    `cfg.data.max_samples or 50000` to keep memory bounded.
    """
    cap = cfg.data.max_samples if cfg.data.max_samples is not None else 50_000
    rows: list[dict[str, Any]] = []
    for row in _stream_dataset_rows(cfg):
        rows.append(row)
        if len(rows) >= cap:
            break
    if not rows:
        msg = (
            f"data.source={cfg.data.source!r} yielded zero examples — "
            "check the `path` / `hf_id` and that the source has data."
        )
        raise RuntimeError(msg)
    return Dataset.from_list(rows)


def _build_lora_config(cfg: XTrainConfig, LoraConfig: Any) -> Any | None:
    method = cfg.train.method
    if isinstance(method, LoraMethod) or isinstance(method, QLoraMethod):
        if LoraConfig is None:
            msg = "peft not installed; `uv sync --extra ml` or drop method.kind to 'full'."
            raise RuntimeError(msg)
        return LoraConfig(
            r=method.r,
            lora_alpha=method.alpha,
            lora_dropout=method.dropout,
            target_modules=list(method.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
    return None


def run_trl_cpu(
    cfg: XTrainConfig,
    plan: AutotunePlan,
    out_dir: Path,
    *,
    on_line: Callable[[str], None] | None = None,
) -> Path:
    """Run a TRL SFT job on CPU; return the produced checkpoint directory.

    `on_line` mirrors the axolotl backend signature so the Coach UI can
    stream log lines uniformly across lanes. TRL doesn't emit one-line-per-
    step by default, but we forward `transformers` log records via a tiny
    handler so the streaming surface stays consistent.
    """
    deps = _require_ml_deps()
    Dataset = deps["Dataset"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    SFTConfig = deps["SFTConfig"]
    SFTTrainer = deps["SFTTrainer"]
    LoraConfig = deps["LoraConfig"]

    import torch  # type: ignore

    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sink = on_line if on_line is not None else (lambda _line: None)
    sink(f"[trl_cpu] base={cfg.model.name} method={cfg.train.method.kind}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sink("[trl_cpu] materializing dataset (in-memory)")
    dataset = _materialize_dataset(cfg, Dataset)
    sink(f"[trl_cpu] dataset size={len(dataset)}")

    sink("[trl_cpu] loading base model on CPU (float32)")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        attn_implementation="eager",
    )

    peft_config = _build_lora_config(cfg, LoraConfig)

    sft_args = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.train.schedule.epochs,
        per_device_train_batch_size=max(1, min(cfg.train.batch.per_device, 2)),
        gradient_accumulation_steps=cfg.train.batch.grad_accum,
        learning_rate=cfg.train.optimizer.lr,
        warmup_ratio=cfg.train.schedule.warmup_ratio,
        max_length=cfg.data.seq_len,
        packing=cfg.data.packing,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,  # CPU + checkpointing is pathologically slow
        seed=cfg.meta.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    sink("[trl_cpu] starting trainer.train()")
    trainer.train()
    sink("[trl_cpu] training complete, saving checkpoint")
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    sink(f"[trl_cpu] checkpoint at {checkpoint_dir}")
    return checkpoint_dir


__all__ = ["run_trl_cpu"]
