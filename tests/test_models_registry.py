"""ModelRegistry preset + chat-template lookup."""

from __future__ import annotations

import pytest

from mindxtrain.models.registry import ModelRegistry, chat_template_for


def test_canonical_presets_registered():
    presets = ModelRegistry.presets()
    for required in ("qwen35", "glm51", "deepseek_v32", "mistral3", "phi4_mini"):
        assert required in presets


def test_chat_template_for_qwen35_is_reasoning():
    assert chat_template_for("qwen35") == "qwen3_reasoning"


def test_chat_template_for_phi4_is_hermes():
    assert chat_template_for("phi4_mini") == "hermes"


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        ModelRegistry.preset("not-a-real-model")


def test_backends_registered():
    backends = ModelRegistry.names()
    for required in ("vllm", "openai_compat"):
        assert required in backends
