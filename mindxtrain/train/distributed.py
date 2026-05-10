"""Distributed-training config builders — Accelerate FSDP + DeepSpeed ZeRO.

Pure-Python: returns plain dicts that the caller writes to YAML/JSON for
`accelerate launch --config_file <path>` or
`deepspeed --deepspeed_config <path>`.

Hard invariant: MI300X xGMI permits only 1- or 8-GPU FSDP. The 2/4-GPU
configurations have a known bandwidth bug; this module rejects them.
"""

from __future__ import annotations

from typing import Literal


def build_fsdp_config(
    num_gpus: Literal[1, 8],
    *,
    shard_size: Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"] = "FULL_SHARD",
    transformer_layer_class: str = "Qwen3DecoderLayer",
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16",
) -> dict[str, object]:
    """Return an Accelerate FSDP config dict for `num_gpus`."""
    if num_gpus not in (1, 8):
        msg = f"MI300X xGMI permits only 1 or 8 GPUs; got {num_gpus}."
        raise ValueError(msg)
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP" if num_gpus > 1 else "NO",
        "downcast_bf16": "no",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": mixed_precision,
        "num_machines": 1,
        "num_processes": num_gpus,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
        "fsdp_config": {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
            "fsdp_forward_prefetch": False,
            "fsdp_offload_params": False,
            "fsdp_sharding_strategy": shard_size,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_sync_module_states": True,
            "fsdp_transformer_layer_cls_to_wrap": transformer_layer_class,
            "fsdp_use_orig_params": True,
        },
    }


def build_deepspeed_config(
    *,
    zero_stage: Literal[1, 2, 3] = 3,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    overlap_comm: bool = True,
) -> dict[str, object]:
    """Return a DeepSpeed ZeRO config dict at the given stage."""
    if zero_stage not in (1, 2, 3):
        msg = f"zero_stage must be 1/2/3; got {zero_stage}"
        raise ValueError(msg)
    return {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {"device": "cpu" if offload_optimizer else "none"},
            "offload_param": {"device": "cpu" if offload_param else "none"},
            "overlap_comm": overlap_comm,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }


__all__ = ["build_deepspeed_config", "build_fsdp_config"]
