"""External-API clients — register with mindx.pythai.net + list on AgenticPlace.

Real httpx POSTs against configurable base URLs (env-overridable).

These endpoints are part of the mindX cognitive ecosystem; if your `*.pythai.net`
endpoints aren't deployed yet, set `MINDXTRAIN_API_BASE_URL` /
`MINDXTRAIN_AGENTICPLACE_URL` to your own service.
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, ConfigDict


class MindXAgentRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    hf_url: str
    cid: str
    capability: str = "chat"


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
    "list_on_agenticplace",
    "register_with_mindx",
]
