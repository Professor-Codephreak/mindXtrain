"""Mistral Large 3 base-model preset. Auto-registers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Mistral3Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str = "mistralai/Mistral-Large-3"
    chat_template: str = "hermes"
    max_seq_len: int = 32768
    fsdp_layer_class: str = "MistralDecoderLayer"
    moe: bool = False


_PRESET = Mistral3Preset()


def preset() -> Mistral3Preset:
    return _PRESET


def _register() -> None:
    from mindxtrain.models.registry import register_preset

    register_preset(_PRESET.hf_id, _PRESET)
    register_preset("mistral3", _PRESET)


_register()


__all__ = ["Mistral3Preset", "preset"]
