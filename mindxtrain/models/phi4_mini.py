"""Phi-4-mini base-model preset (Microsoft). Auto-registers.

Small-footprint default for laptop / single-GPU local fine-tuning.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Phi4MiniPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str = "microsoft/Phi-4-mini-instruct"
    chat_template: str = "hermes"
    max_seq_len: int = 16384
    fsdp_layer_class: str = "Phi3DecoderLayer"
    moe: bool = False


_PRESET = Phi4MiniPreset()


def preset() -> Phi4MiniPreset:
    return _PRESET


def _register() -> None:
    from mindxtrain.models.registry import register_preset

    register_preset(_PRESET.hf_id, _PRESET)
    register_preset("phi4_mini", _PRESET)


_register()


__all__ = ["Phi4MiniPreset", "preset"]
