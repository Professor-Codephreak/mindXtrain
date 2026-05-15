"""External-API clients — register with mindx.pythai.net + list on AgenticPlace.

Real httpx POSTs against configurable base URLs (env-overridable).

These endpoints are part of the mindX cognitive ecosystem; if your `*.pythai.net`
endpoints aren't deployed yet, set `MINDXTRAIN_API_BASE_URL` /
`MINDXTRAIN_AGENTICPLACE_URL` to your own service.
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, ConfigDict, Field


class MindXAgentRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    hf_url: str
    cid: str
    capability: str = "chat"


class MindXFallbackSwap(BaseModel):
    """Payload for the mindX runtime fallback-swap endpoint."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="vllm", description="LLM provider in mindX (vllm, ollama, ...).")
    model: str = Field(..., min_length=1, description="HF Hub repo or provider-local model name.")


class AgenticPlaceListing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    hf_url: str
    title: str = ""
    price_usdc_per_million_tokens: float = 1.0


def register_with_mindx(
    *,
    run_id: str,
    hf_url: str,
    cid: str,
    api_url: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """POST /v1/agents on the mindX cognitive API; return the registration receipt."""
    api_url = (api_url or os.environ.get("MINDXTRAIN_API_BASE_URL", "https://mindx.pythai.net")).rstrip("/")
    body = MindXAgentRegistration(run_id=run_id, hf_url=hf_url, cid=cid).model_dump()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{api_url}/v1/agents", json=body)
        resp.raise_for_status()
        data: dict[str, str] = resp.json()
    return data


def swap_mindx_fallback_model(
    *,
    provider: str = "vllm",
    model: str,
    api_url: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """PATCH /v1/config/fallback-model on mindX; return {previous, current, ...}.

    Called by the `publish` step after the trained checkpoint lands on HF Hub
    so subsequent LLM handler creations in mindX resolve the new default.

    `api_url` defaults to `MINDXTRAIN_API_BASE_URL` env (or `https://mindx.pythai.net`).
    `api_key`, if provided or read from `MINDXTRAIN_API_KEY`, is sent as
    `Authorization: Bearer <key>` — required when the mindX deployment has
    its bearer-auth secret set.
    """
    api_url = (api_url or os.environ.get("MINDXTRAIN_API_BASE_URL", "https://mindx.pythai.net")).rstrip("/")
    api_key = api_key if api_key is not None else os.environ.get("MINDXTRAIN_API_KEY", "")

    body = MindXFallbackSwap(provider=provider, model=model).model_dump()
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=timeout_s) as client:
        resp = client.patch(f"{api_url}/v1/config/fallback-model", json=body, headers=headers)
        resp.raise_for_status()
        data: dict[str, str] = resp.json()
    return data


def list_on_agenticplace(
    *,
    run_id: str,
    hf_url: str,
    title: str = "",
    price_usdc_per_million_tokens: float = 1.0,
    api_url: str | None = None,
    timeout_s: float = 30.0,
) -> str:
    """POST /v1/listings on AgenticPlace; return the listing slug/url."""
    api_url = (
        api_url
        or os.environ.get("MINDXTRAIN_AGENTICPLACE_URL", "https://agenticplace.pythai.net")
    ).rstrip("/")
    body = AgenticPlaceListing(
        run_id=run_id,
        hf_url=hf_url,
        title=title or run_id,
        price_usdc_per_million_tokens=price_usdc_per_million_tokens,
    ).model_dump()
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{api_url}/v1/listings", json=body)
        resp.raise_for_status()
        data = resp.json()
    return str(data.get("listing_url", data))


__all__ = [
    "AgenticPlaceListing",
    "MindXAgentRegistration",
    "MindXFallbackSwap",
    "list_on_agenticplace",
    "register_with_mindx",
    "swap_mindx_fallback_model",
]
