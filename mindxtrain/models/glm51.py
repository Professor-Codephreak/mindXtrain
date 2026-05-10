"""GLM-5.1 base-model preset (Z.ai, 754B/40B-A MoE, MIT-licensed).

Specialist track only — see LICENSE-MIT-upstream-glm51 for upstream license
terms. Auto-registers with `ModelRegistry` at import time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Glm51Preset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str = "zai-org/GLM-5.1-Base"
    chat_template: str = "qwen3_reasoning"
    max_seq_len: int = 32768
    fsdp_layer_class: str = "GLMDecoderLayer"
    moe: bool = True


_PRESET = Glm51Preset()


def preset() -> Glm51Preset:
    return _PRESET


# Auto-register so `ModelRegistry.presets()` knows about us.
def _register() -> None:
    from mindxtrain.models.registry import register_preset

    register_preset(_PRESET.hf_id, _PRESET)
    register_preset("glm51", _PRESET)


_register()


__all__ = ["Glm51Preset", "preset"]
