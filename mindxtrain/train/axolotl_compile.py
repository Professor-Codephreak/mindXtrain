"""Compile XTrainConfig + AutotunePlan into an Axolotl YAML dict.

Pure Python — no GPU, no torch. The output is what gets written to
`out/<run_id>/axolotl.yaml` and consumed by `accelerate launch -m axolotl.cli.train`.

Apply order (later overrides earlier):
    1. base from XTrainConfig (model / data / train / eval).
    2. AutotunePlan overrides (attention_backend, suggested_micro_batch_size, etc.).
    3. cfg.train.env merged with plan-derived env additions.
"""

from __future__ import annotations

from typing import Any

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.schema import (
    CptMethod,
    DpoMethod,
    FullMethod,
    GrpoMethod,
    GspoMethod,
    KtoMethod,
    LoraMethod,
    OrpoMethod,
    QLoraMethod,
    XTrainConfig,
)


def _method_block(cfg: XTrainConfig) -> dict[str, Any]:
    """Translate cfg.train.method into Axolotl-flavored fields."""
    m = cfg.train.method
    if isinstance(m, FullMethod):
        return {"adapter": None}
    if isinstance(m, LoraMethod):
        return {
            "adapter": "lora",
            "lora_r": m.r,
            "lora_alpha": m.alpha,
            "lora_dropout": m.dropout,
            "lora_target_modules": list(m.target_modules),
        }
    if isinstance(m, QLoraMethod):
        return {
            "adapter": "qlora",
            "lora_r": m.r,
            "lora_alpha": m.alpha,
            "lora_dropout": m.dropout,
            "lora_target_modules": list(m.target_modules),
            "load_in_4bit": m.quant_bits == 4,
            "load_in_8bit": m.quant_bits == 8,
        }
    if isinstance(m, DpoMethod):
        return {"rl": "dpo", "rl_beta": m.beta}
    if isinstance(m, OrpoMethod):
        return {"rl": "orpo", "rl_beta": m.beta}
    if isinstance(m, GrpoMethod):
        return {"rl": "grpo", "rl_num_generations": m.num_generations, "rl_kl_coef": m.kl_coef}
    if isinstance(m, GspoMethod):
        return {"rl": "gspo", "rl_num_generations": m.num_generations}
    if isinstance(m, KtoMethod):
        return {"rl": "kto", "rl_beta": m.beta}
    if isinstance(m, CptMethod):
        return {"adapter": None, "pretraining": True}
    msg = f"unhandled method kind: {m!r}"
    raise ValueError(msg)


def _attention_field(plan_backend: str) -> dict[str, Any]:
    """Map autotune plan's attention_backend onto Axolotl's flash_attention flags."""
    if plan_backend == "ck":
        return {"flash_attention": True, "flash_attn_backend": "ck"}
    if plan_backend == "triton":
        return {"flash_attention": True, "flash_attn_backend": "triton"}
    if plan_backend == "aiter":
        return {"flash_attention": True, "flash_attn_backend": "aiter"}
    msg = f"unknown attention backend in plan: {plan_backend!r}"
    raise ValueError(msg)


def _plan_env(cfg: XTrainConfig, plan: AutotunePlan) -> dict[str, str]:
    """Merge cfg.train.env with plan-derived additions; plan wins on conflict."""
    env = dict(cfg.train.env)
    env["PYTORCH_ROCM_ARCH"] = plan.gpu_arch
    if plan.rccl_config == "8gpu_xgmi":
        env["NCCL_MIN_NCHANNELS"] = "112"
        env["GPU_MAX_HW_QUEUES"] = "1"
    if plan.attention_backend == "triton":
        # Force Triton path; otherwise CK is default.
        env["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
    return env


def compile_axolotl_yaml(cfg: XTrainConfig, plan: AutotunePlan) -> dict[str, Any]:
    """Return an Axolotl-compatible YAML dict.

    The caller writes this to disk via `yaml.safe_dump`.
    """
    micro_batch = min(cfg.train.batch.per_device, plan.suggested_micro_batch_size)

    out: dict[str, Any] = {
        # --- base / model ----------------------------------------------------
        "base_model": cfg.model.name,
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "trust_remote_code": cfg.model.trust_remote_code,
        "torch_dtype": cfg.model.torch_dtype,

        # --- precision -------------------------------------------------------
        "bf16": cfg.train.precision == "bfloat16",
        "fp16": cfg.train.precision == "float16",
        "gradient_checkpointing": cfg.train.gradient_checkpointing,

        # --- data ------------------------------------------------------------
        "datasets": [
            {
                "path": cfg.data.hf_id,
                "type": "alpaca",  # Axolotl format hint; recipe-overridable.
                "split": cfg.data.split,
            },
        ],
        "sequence_len": cfg.data.seq_len,
        "sample_packing": cfg.data.packing,
        "streaming": cfg.data.streaming,

        # --- batch -----------------------------------------------------------
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": cfg.train.batch.grad_accum,

        # --- optimizer / schedule -------------------------------------------
        "optimizer": cfg.train.optimizer.name,
        "learning_rate": cfg.train.optimizer.lr,
        "adam_beta1": cfg.train.optimizer.betas[0],
        "adam_beta2": cfg.train.optimizer.betas[1],
        "weight_decay": cfg.train.optimizer.weight_decay,
        "max_grad_norm": cfg.train.optimizer.grad_clip,
        "lr_scheduler": cfg.train.schedule.type,
        "warmup_ratio": cfg.train.schedule.warmup_ratio,
        "num_epochs": cfg.train.schedule.epochs,

        # --- run identity ---------------------------------------------------
        "seed": cfg.meta.seed,
        "wandb_project": cfg.meta.project,
        "wandb_run_id": cfg.meta.run_name,
        "output_dir": f"./runs/{cfg.meta.run_name}/checkpoint",
    }

    # --- method (LoRA / QLoRA / DPO / GRPO / etc.) --------------------------
    out.update(_method_block(cfg))

    # --- attention backend (autotune plan overrides cfg) --------------------
    out.update(_attention_field(plan.attention_backend))

    # --- FSDP ---------------------------------------------------------------
    if cfg.train.fsdp.enabled:
        out["fsdp"] = "full_shard auto_wrap" if cfg.train.fsdp.auto_wrap else "full_shard"
        out["fsdp_config"] = {
            "fsdp_offload_params": False,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
        }

    # --- env (set in subprocess before launching) ---------------------------
    out["env"] = _plan_env(cfg, plan)

    # --- max_samples cap (truncate streaming dataset) -----------------------
    if cfg.data.max_samples is not None:
        out["max_samples"] = cfg.data.max_samples

    return out


def autotune_overrides_summary(plan: AutotunePlan) -> list[str]:
    """Human-readable summary of how the plan modified the cfg.

    Used by `mindxtrain train` to print why the run is different from the
    raw YAML the user wrote.
    """
    summary: list[str] = []
    summary.append(f"attention_backend={plan.attention_backend} (autotune-selected)")
    summary.append(f"gemm_heuristic={plan.gemm_heuristic}")
    summary.append(f"rccl_config={plan.rccl_config}")
    if plan.suggested_micro_batch_size:
        summary.append(f"suggested_micro_batch_size={plan.suggested_micro_batch_size}")
    if plan.suggested_lora_rank:
        summary.append(f"suggested_lora_rank={plan.suggested_lora_rank}")
    return summary
