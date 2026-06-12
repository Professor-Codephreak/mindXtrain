"""External-API clients — register with mindx.pythai.net + list on AgenticPlace.

Real httpx POSTs against configurable base URLs (env-overridable).

These endpoints are part of the mindX cognitive ecosystem; if your `*.pythai.net`
endpoints aren't deployed yet, set `MINDXTRAIN_API_BASE_URL` /
`MINDXTRAIN_AGENTICPLACE_URL` to your own service.
"""

from __future__ import annotations

import json
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


def trigger_dream_ingestion(
    *,
    run_id: str,
    adapter_dir: str,
    base_model: str,
    persona_name: str = "",
    imprint_delta: float | None = None,
    api_url: str | None = None,
    timeout_s: float = 10.0,
) -> dict[str, str]:
    """Hand a freshly-imprinted actor to mindX's `machine.dream` 8hr cycle.

    Clean-room boundary: we never import or run mindX code — we hand off an
    artifact *pointer* (run id + adapter path + base model + imprint delta) so the
    mindX dream cycle (`agents/machine_dreaming.py`) can ingest the trained actor
    on its next pass. Best-effort, with two delivery modes:

    1. HTTP — POST `/v1/dream/ingest` on the mindX API when `MINDXTRAIN_API_BASE_URL`
       is set and reachable.
    2. Inbox drop — write a pointer JSON into
       `$MINDXTRAIN_MINDX_HOME/data/incoming/<run_id>.dream.json` so a filesystem-
       watching dream cycle picks it up.

    Returns `{"mode": ..., "target": ...}`; never raises — a failed trigger reports
    via the return dict rather than failing the training run.
    """
    payload = {
        "run_id": run_id,
        "adapter_dir": adapter_dir,
        "base_model": base_model,
        "persona": persona_name,
        "imprint_delta": "" if imprint_delta is None else f"{imprint_delta:.4f}",
        "source": "mindxtrain.imprint",
    }
    api = (api_url or os.environ.get("MINDXTRAIN_API_BASE_URL", "")).rstrip("/")
    if api:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(f"{api}/v1/dream/ingest", json=payload)
                resp.raise_for_status()
            return {"mode": "http", "target": f"{api}/v1/dream/ingest"}
        except (httpx.HTTPError, OSError) as exc:
            payload["http_error"] = str(exc)

    # Filesystem inbox fallback — the 8hr dream cycle watches data/incoming.
    home = os.environ.get("MINDXTRAIN_MINDX_HOME", "")
    if home:
        from pathlib import Path

        inbox = Path(home).expanduser() / "data" / "incoming"
        try:
            inbox.mkdir(parents=True, exist_ok=True)
            ptr = inbox / f"{run_id}.dream.json"
            ptr.write_text(json.dumps(payload, indent=2))
            return {"mode": "inbox", "target": str(ptr)}
        except OSError as exc:
            return {"mode": "failed", "target": str(inbox), "error": str(exc)}

    return {
        "mode": "skipped",
        "target": "",
        "note": "set MINDXTRAIN_API_BASE_URL or MINDXTRAIN_MINDX_HOME to deliver",
    }


__all__ = [
    "AgenticPlaceListing",
    "MindXAgentRegistration",
    "MindXFallbackSwap",
    "list_on_agenticplace",
    "register_with_mindx",
    "swap_mindx_fallback_model",
    "trigger_dream_ingestion",
]
