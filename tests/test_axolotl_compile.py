"""Axolotl YAML compiler — XTrainConfig + AutotunePlan -> dict."""

from __future__ import annotations

import pytest
import yaml

from mindxtrain.autotune.benchmark import run_autotune
from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.loader import render_recipe
from mindxtrain.config.schema import XTrainConfig
from mindxtrain.train import autotune_overrides_summary, compile_axolotl_yaml


def _cfg(name: str = "qwen3_8b_sft_lora") -> XTrainConfig:
    return XTrainConfig.model_validate(yaml.safe_load(render_recipe(name)))


def _plan(**overrides) -> AutotunePlan:
    p = run_autotune(dry_run=True)
    if overrides:
        return p.model_copy(update=overrides)
    return p


def test_lora_recipe_compiles_with_adapter_block():
    out = compile_axolotl_yaml(_cfg(), _plan())
    assert out["base_model"] == "Qwen/Qwen3-8B"
    assert out["adapter"] == "lora"
    assert out["lora_r"] == 16
    assert out["lora_alpha"] == 32
    assert "q_proj" in out["lora_target_modules"]
    assert out["bf16"] is True


def test_full_recipe_compiles_without_adapter():
    out = compile_axolotl_yaml(_cfg("qwen3_8b_sft_full"), _plan())
    assert out["adapter"] is None
    assert "lora_r" not in out


def test_dpo_recipe_compiles_with_rl_block():
    out = compile_axolotl_yaml(_cfg("qwen3_32b_dpo"), _plan())
    assert out["rl"] == "dpo"
    assert out["rl_beta"] == pytest.approx(0.1)


def test_grpo_recipe_compiles_with_rl_block():
    out = compile_axolotl_yaml(_cfg("qwen3_32b_grpo"), _plan())
    assert out["rl"] == "grpo"
    assert out["rl_num_generations"] == 4
    assert out["rl_kl_coef"] == pytest.approx(0.04)


def test_attention_backend_picked_from_plan():
    out = compile_axolotl_yaml(_cfg(), _plan(attention_backend="triton"))
    assert out["flash_attention"] is True
    assert out["flash_attn_backend"] == "triton"
    assert out["env"]["VLLM_USE_TRITON_FLASH_ATTN"] == "1"


def test_micro_batch_capped_by_plan_suggestion():
    cfg = _cfg("qwen3_8b_sft_lora")  # cfg says per_device=8
    plan = _plan(suggested_micro_batch_size=4)
    out = compile_axolotl_yaml(cfg, plan)
    assert out["micro_batch_size"] == 4


def test_rccl_8gpu_sets_env_overrides():
    cfg = _cfg("qwen3_32b_full_fsdp")  # gpus=8
    out = compile_axolotl_yaml(cfg, _plan(rccl_config="8gpu_xgmi"))
    assert out["env"]["NCCL_MIN_NCHANNELS"] == "112"
    assert out["env"]["GPU_MAX_HW_QUEUES"] == "1"


def test_fsdp_block_only_when_enabled():
    cfg_8b = _cfg("qwen3_8b_sft_lora")
    out_8b = compile_axolotl_yaml(cfg_8b, _plan())
    assert "fsdp" not in out_8b

    cfg_32b = _cfg("qwen3_32b_full_fsdp")
    out_32b = compile_axolotl_yaml(cfg_32b, _plan())
    assert out_32b["fsdp"] == "full_shard auto_wrap"


def test_autotune_overrides_summary_lists_decisions():
    summary = autotune_overrides_summary(_plan())
    assert any("attention_backend" in s for s in summary)
    assert any("gemm_heuristic" in s for s in summary)


def test_max_samples_cap_only_when_set():
    out = compile_axolotl_yaml(_cfg("instella_3b_lora"), _plan())
    assert out.get("max_samples") == 5000

    # qwen3_8b_sft_lora has no max_samples
    out2 = compile_axolotl_yaml(_cfg("qwen3_8b_sft_lora"), _plan())
    assert "max_samples" not in out2


def test_pytorch_rocm_arch_always_in_env():
    out = compile_axolotl_yaml(_cfg(), _plan())
    assert out["env"]["PYTORCH_ROCM_ARCH"] == "gfx942"
