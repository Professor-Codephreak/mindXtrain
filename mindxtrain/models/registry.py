"""ModelRegistry — backend protocol + factory dispatch.

Canonical home per mindxtrain2.md §Part 4 `models.registry`. Merges the previous
`automindx.models.base` (Backend ABC + Pydantic chat schemas) and
`automindx.models.factory` (decorator-style register/build).

`Backend` is the plug interface; concrete backends live under
`mindxtrain.operator.backends.*` (vllm, openai_compat, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=131072)
    stream: bool = False


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    content: str
    finish_reason: Literal["stop", "length", "error"] = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Backend(ABC):
    """Pluggable inference backend; one of vLLM, OpenAI-compatible, HF Transformers, etc."""

    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse: ...

    @abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]: ...


# Back-compat alias — old code imported `ModelBackend`.
ModelBackend = Backend


# ---- registry -------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Backend]] = {}


def register_backend(name: str) -> Callable[[Callable[..., Backend]], Callable[..., Backend]]:
    """Decorator: register a Backend constructor under `name`."""

    def _wrap(ctor: Callable[..., Backend]) -> Callable[..., Backend]:
        if name in _REGISTRY:
            msg = f"backend {name!r} already registered"
            raise ValueError(msg)
        _REGISTRY[name] = ctor
        return ctor

    return _wrap


def build_backend(name: str, **kwargs: object) -> Backend:
    """Instantiate the backend registered under `name`."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        msg = f"unknown backend {name!r}. registered: {available}"
        raise KeyError(msg)
    return _REGISTRY[name](**kwargs)


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


# ---- per-base-model preset registry --------------------------------------

_PRESETS: dict[str, BaseModel] = {}


def register_preset(name: str, preset: BaseModel) -> None:
    """Register a per-base-model preset (Pydantic model)."""
    _PRESETS[name] = preset


def get_preset(name: str) -> BaseModel:
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS)) or "(none registered)"
        msg = f"unknown base model {name!r}. registered: {available}"
        raise KeyError(msg)
    return _PRESETS[name]


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def chat_template_for(name: str) -> str:
    """Return the canonical chat-template name for a registered base model."""
    preset = get_preset(name)
    return str(getattr(preset, "chat_template", "hermes"))


class ModelRegistry:
    """Class facade over the module-level registry — preferred API per mindxtrain2.md."""

    @staticmethod
    def register(name: str) -> Callable[[Callable[..., Backend]], Callable[..., Backend]]:
        return register_backend(name)

    @staticmethod
    def build(name: str, **kwargs: object) -> Backend:
        return build_backend(name, **kwargs)

    @staticmethod
    def names() -> list[str]:
        return list_backends()

    @staticmethod
    def register_preset(name: str, preset: BaseModel) -> None:
        register_preset(name, preset)

    @staticmethod
    def preset(name: str) -> BaseModel:
        return get_preset(name)

    @staticmethod
    def presets() -> list[str]:
        return list_presets()


# Side-effect imports — register the built-in backends.
# Side-effect imports — register the built-in base-model presets.
from mindxtrain.models import deepseek_v32 as _deepseek_v32  # noqa: E402, F401
from mindxtrain.models import glm51 as _glm51  # noqa: E402, F401
from mindxtrain.models import mistral3 as _mistral3  # noqa: E402, F401
from mindxtrain.models import phi4_mini as _phi4_mini  # noqa: E402, F401
from mindxtrain.models import qwen35 as _qwen35  # noqa: E402, F401
from mindxtrain.operator.backends import openai_compat as _openai_compat  # noqa: E402, F401
from mindxtrain.operator.backends import vllm as _vllm  # noqa: E402, F401
