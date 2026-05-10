"""OpenAI-compatible backend — works against OpenAI, Anthropic-via-proxy,
Groq, ZAI, or any HTTP endpoint that follows the OpenAI Chat Completions
protocol.

Reads `MINDXTRAIN_OPENAI_BASE_URL` (default `https://api.openai.com/v1`) and
`MINDXTRAIN_OPENAI_API_KEY`.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx

from mindxtrain.models.registry import ChatRequest, ChatResponse, ModelBackend, register_backend


@register_backend("openai_compat")
class OpenAICompatBackend(ModelBackend):
    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MINDXTRAIN_OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("MINDXTRAIN_OPENAI_API_KEY", "")
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, object]:
        return {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }

    async def chat(self, request: ChatRequest) -> ChatResponse:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(request, stream=False),
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return ChatResponse(
            model=data.get("model", request.model),
            content=(choice.get("message") or {}).get("content", "") or "",
            finish_reason=choice.get("finish_reason", "stop") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=self._payload(request, stream=True),
                    headers=self._headers(),
                ) as resp:
                    resp.raise_for_status()
                    async for raw in resp.aiter_lines():
                        if not raw or not raw.startswith("data:"):
                            continue
                        data = raw[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield token

        return _gen()


__all__ = ["OpenAICompatBackend"]
