"""DeepSeek V3.2 base-model preset.

Reasoning-tuned checkpoint with `<think>...</think>` framing. Auto-registers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DeepSeekV32Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str = "deepseek-ai/DeepSeek-V3.2"
    chat_template: str = "deepseek_r1"
    max_seq_len: int = 65536
    fsdp_layer_class: str = "DeepseekV3DecoderLayer"
    moe: bool = True


_PRESET = DeepSeekV32Preset()


def preset() -> DeepSeekV32Preset:
    return _PRESET


def _register() -> None:
    from mindxtrain.models.registry import register_preset

    register_preset(_PRESET.hf_id, _PRESET)
    register_preset("deepseek_v32", _PRESET)


_register()


__all__ = ["DeepSeekV32Preset", "preset"]
