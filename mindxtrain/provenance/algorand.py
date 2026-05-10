"""BANKON ASA + ENS subname allocation hooks (Algorand chain).

`allocate_ens_subname` POSTs to the BANKON allocation service
(`MINDXTRAIN_BANKON_ENS_URL`) and returns the assigned `<subname>.bankon.eth`.

`asa_info` reads ASA metadata via py-algorand-sdk indexer (lazy import).
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, ConfigDict


class EnsAllocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent: str = "bankon.eth"
    subname: str
    full: str = ""
    tx_id: str = ""


def allocate_ens_subname(
    subname: str,
    *,
    parent: str = "bankon.eth",
    base_url: str | None = None,
    timeout_s: float = 30.0,
) -> EnsAllocation:
    """POST to the BANKON ENS allocator; return the allocation receipt."""
    base_url = (base_url or os.environ.get("MINDXTRAIN_BANKON_ENS_URL", "https://ens.bankon.pythai.net")).rstrip("/")
    body = {"parent": parent, "subname": subname}
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{base_url}/v1/subname", json=body)
        resp.raise_for_status()
        data = resp.json()
    return EnsAllocation(
        parent=data.get("parent", parent),
        subname=data.get("subname", subname),
        full=data.get("full", f"{subname}.{parent}"),
        tx_id=data.get("tx_id", ""),
    )


def asa_info(asset_id: int, *, indexer_url: str | None = None) -> dict[str, object]:
    """Fetch ASA metadata via the Algorand indexer; lazy py-algorand-sdk import."""
    try:
        from algosdk.v2client import indexer
    except ImportError as exc:
        msg = "py-algorand-sdk not installed; run `uv sync --extra chain`."
        raise RuntimeError(msg) from exc
    url = indexer_url or os.environ.get(
        "MINDXTRAIN_ALGORAND_INDEXER_URL",
        "https://mainnet-idx.algonode.cloud",
    )
    client = indexer.IndexerClient("", url)
    info: dict[str, object] = client.asset_info(asset_id)
    return info


__all__ = ["EnsAllocation", "allocate_ens_subname", "asa_info"]
