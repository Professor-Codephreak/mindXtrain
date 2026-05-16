"""Ollama backend — the laptop dev counterpart of vLLM-rocm.

Ollama serves an OpenAI-compatible `/v1/chat/completions` at port 11434
by default. The wire protocol is identical to vLLM's, so this backend
is a thin alias of `OpenAICompatBackend` with sane defaults — no env vars
needed when ollama is the only thing running locally.

The same OpenAI wire protocol runs on three deployment targets in the
mindXtrain ecosystem:

- **Laptop dev:** ollama (this backend), serving local GGUF quantized models.
- **mindx.pythai.net VPS:** vLLM-rocm + ollama-as-fallback (per mindX's
  `models/ollama.yaml` + `models/vllm.yaml`).
- **AMD Dev Cloud MI300X droplet:** vLLM-rocm post-train serving.

`MINDXTRAIN_OLLAMA_BASE_URL` overrides the URL if ollama is on a non-default
host or port.
"""

from __future__ import annotations

import os

from mindxtrain.models.registry import register_backend
from mindxtrain.operator.backends.openai_compat import OpenAICompatBackend


@register_backend("ollama")
class OllamaBackend(OpenAICompatBackend):
    name = "ollama"

    def __init__(self, base_url: str | None = None, timeout_s: float = 120.0) -> None:
        super().__init__(
            base_url=base_url
            or os.environ.get("MINDXTRAIN_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=None,
            timeout_s=timeout_s,
        )


__all__ = ["OllamaBackend"]
