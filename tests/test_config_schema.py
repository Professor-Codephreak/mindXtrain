"""XTrainConfig YAML round-trip and validation against the canonical 10-section schema."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from mindxtrain.config.loader import list_recipes, load_config, render_recipe
from mindxtrain.config.schema import LoraMethod, XTrainConfig


def test_qwen3_8b_sft_lora_validates(tmp_path):
    yaml_text = render_recipe("qwen3_8b_sft_lora")
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_config(cfg_path)
    assert cfg.meta.run_name == "qwen3_8b_sft_lora"
    assert cfg.meta.seed == 2048
    assert cfg.model.name == "Qwen/Qwen3-8B"
    assert cfg.hardware.gfx_arch == "gfx942"
    assert cfg.hardware.gpus == 1
    assert cfg.autotune.policy == "aot_only"
    assert cfg.train.backend == "axolotl"
    assert isinstance(cfg.train.method, LoraMethod)
    assert cfg.train.method.r == 16
    assert cfg.train.flash_attention.backend == "ck"
    assert cfg.train.env["PYTORCH_ROCM_ARCH"] == "gfx942"
    assert "mmlu" in cfg.eval.harness.tasks
    assert cfg.quantize.scheme == "quark_fp8"
    assert cfg.quantize.ptpc is True
    assert cfg.serve.backend == "vllm-rocm"
    assert cfg.serve.reasoning_parser == "qwen3"


def test_instella_template_validates(tmp_path):
    yaml_text = render_recipe("instella_3b_lora")
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_config(cfg_path)
    assert cfg.model.name == "amd/Instella-3B-Instruct"
    assert cfg.data.max_samples == 5000


def test_demo_qwen3_8b_sft_example_validates():
    """The hero config copied verbatim from the production blueprint must load."""
    cfg = load_config("examples/demo_qwen3_8b_sft.yaml")
    assert cfg.meta.run_name == "qwen3_8b_sft_demo"
    assert isinstance(cfg.train.method, LoraMethod)
    assert cfg.train.method.alpha == 32
    assert cfg.publish.enabled is True
    assert cfg.publish.billing.x402.network == "algorand"
    assert cfg.publish.billing.x402.asset == "USDC"


def test_all_recipes_validate():
    """Every shipped recipe must round-trip against XTrainConfig."""
    for name in list_recipes():
        text = render_recipe(name)
        cfg = XTrainConfig.model_validate(yaml.safe_load(text))
        assert cfg.meta.run_name


def test_unknown_template_raises():
    with pytest.raises(FileNotFoundError):
        render_recipe("does_not_exist")


def test_xgmi_2gpu_rejected(tmp_path):
    """hardware.gpus must be 1 or 8 — xGMI asymmetry on 2/4-GPU MI300X."""
    bad = yaml.safe_load(render_recipe("qwen3_32b_full_fsdp"))
    bad["hardware"]["gpus"] = 2
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_extra_field_forbidden(tmp_path):
    bad = yaml.safe_load(render_recipe("qwen3_8b_sft_lora"))
    bad["meta"]["nonexistent"] = "foo"
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_method_discriminator_rejects_unknown_kind(tmp_path):
    bad = yaml.safe_load(render_recipe("qwen3_8b_sft_lora"))
    bad["train"]["method"] = {"kind": "not_a_real_method"}
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_dpo_method_round_trip():
    text = render_recipe("qwen3_32b_dpo")
    cfg = XTrainConfig.model_validate(yaml.safe_load(text))
    assert cfg.train.method.kind == "dpo"
    assert cfg.train.method.beta == 0.1
    assert cfg.hardware.gpus == 8


def test_grpo_method_round_trip():
    text = render_recipe("qwen3_32b_grpo")
    cfg = XTrainConfig.model_validate(yaml.safe_load(text))
    assert cfg.train.method.kind == "grpo"
    assert cfg.train.method.num_generations == 4


def test_round_trip_via_model_dump(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(render_recipe("qwen3_8b_sft_lora"))
    cfg = load_config(cfg_path)
    redumped = yaml.safe_dump(cfg.model_dump(mode="json"))
    cfg2 = XTrainConfig.model_validate(yaml.safe_load(redumped))
    assert cfg2 == cfg
