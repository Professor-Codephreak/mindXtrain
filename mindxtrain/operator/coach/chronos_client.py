"""Coach client for mindX's chronos.agent HTTP surface.

Wraps `GET /v1/oracle/{time,anchors,drift}` on the mindX backend. The
purpose is to let Coach (and provenance manifests) stamp artefacts with
mindX's *promised time* — a number that comes with a measured
confidence interval instead of raw `time.time()`.

Degrades gracefully: if mindX is unreachable, every method returns a
shape consumers can still render (with `consensus: "unavailable"`) so
the UI never blanks out and manifests can fall back to local time
with `attested: false`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("mindxtrain.chronos_client")

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT_S = 2.0


def _base_url(override: str | None) -> str:
    if override:
        return override.rstrip("/")
    return os.environ.get("MINDX_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


async def now(
    *,
    base_url: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Fetch chronos.agent's `PromisedTime`.

    Always returns a dict with the same keys as the PromisedTime
    contract (`unix_18dp`, `utc`, `consensus`, `confidence_ms`,
    `sources`, `anchor_count_24h`, `promised_by`). On any failure the
    dict carries `consensus: "unavailable"` and the rest of the
    fields populated with safe defaults so callers don't have to
    branch.
    """
    url = f"{_base_url(base_url)}/v1/oracle/time"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug("chronos /time unreachable: %r", exc)
        return _unavailable_promised_time(str(exc))
    # Be defensive about partial responses — mindX upstream might
    # change shape slightly. Fill any missing keys.
    body.setdefault("consensus", "unavailable")
    body.setdefault("confidence_ms", 0.0)
    body.setdefault("sources", {})
    body.setdefault("anchor_count_24h", 0)
    body.setdefault("promised_by", "chronos.agent")
    body.setdefault("unix_18dp", "")
    body.setdefault("utc", "")
    return body


async def anchors(
    *,
    limit: int = 100,
    base_url: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Recent transaction-time anchors. Returns `{anchors: [...], n: int}`."""
    url = f"{_base_url(base_url)}/v1/oracle/anchors"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, params={"limit": int(limit)})
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug("chronos /anchors unreachable: %r", exc)
        return {"anchors": [], "n": 0, "error": str(exc)}


async def drift(
    *,
    hours: int = 24,
    base_url: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Bucketed drift history. Returns the DriftHistory dict shape."""
    url = f"{_base_url(base_url)}/v1/oracle/drift"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, params={"hours": int(hours)})
            resp.raise_for_status()
            return resp.json()
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug("chronos /drift unreachable: %r", exc)
        return {
            "hours": hours, "bucket_count": 0, "buckets": [],
            "drift_std_ms": 0.0, "drift_max_abs_ms": 0.0,
            "anchor_count": 0, "error": str(exc),
        }


def _unavailable_promised_time(error: str) -> dict[str, Any]:
    """The "unavailable" PromisedTime shape — keep keys aligned with the live one."""
    return {
        "unix_18dp": "",
        "utc": "",
        "consensus": "unavailable",
        "confidence_ms": 0.0,
        "sources": {"error": error},
        "anchor_count_24h": 0,
        "promised_by": "chronos.agent",
    }


__all__ = ["anchors", "drift", "now"]
