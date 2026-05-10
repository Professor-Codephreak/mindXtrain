"""Qwen3.5 base-model preset (Alibaba, Apache 2.0).

Recommended primary base per mindxtrain2.md (Qwen3.5-122B-A10B). Auto-registers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Qwen35Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str = "Qwen/Qwen3.5-8B"
    chat_template: str = "qwen3_reasoning"
    max_seq_len: int = 32768
    fsdp_layer_class: str = "Qwen3DecoderLayer"
    moe: bool = False


_PRESET = Qwen35Preset()


def preset() -> Qwen35Preset:
    return _PRESET


def _register() -> None:
    from mindxtrain.models.registry import register_preset

    register_preset(_PRESET.hf_id, _PRESET)
    register_preset("qwen35", _PRESET)


_register()


__all__ = ["Qwen35Preset", "preset"]
