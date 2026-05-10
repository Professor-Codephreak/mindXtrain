"""x402 invoice issuance + Algorand settlement validation.

`issue_invoice` POSTs to a configurable facilitator URL (`MINDXTRAIN_FACILITATOR_URL`)
and returns an `Invoice` typed by Pydantic. `validate_settlement` lazily imports
`algosdk` to verify a USDC ASA (id=203977300) transaction's amount + receiver.
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, ConfigDict, Field

USDC_ASA_ID = 203977300


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str
    run_id: str
    amount_usdc: float
    receiver: str = Field(description="Algorand address to receive payment")
    asset_id: int = USDC_ASA_ID
    pay_url: str = ""
    expires_at: str | None = None


class Settlement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tx_id: str
    asset_id: int = USDC_ASA_ID
    amount_usdc: float
    sender: str = ""
    receiver: str = ""
    confirmed: bool = False


def issue_invoice(
    *,
    run_id: str,
    price_usdc: float,
    facilitator_url: str | None = None,
    receiver: str | None = None,
    timeout_s: float = 30.0,
) -> Invoice:
    """POST to the x402 facilitator and return the Invoice payload."""
    facilitator_url = (
        facilitator_url
        or os.environ.get("MINDXTRAIN_FACILITATOR_URL", "https://facilitator.bankon.io/x402")
    ).rstrip("/")
    body = {
        "run_id": run_id,
        "amount_usdc": price_usdc,
        "asset_id": USDC_ASA_ID,
    }
    if receiver:
        body["receiver"] = receiver
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{facilitator_url}/invoice", json=body)
        resp.raise_for_status()
        data = resp.json()
    return Invoice.model_validate(data)


def validate_settlement(
    tx_id: str,
    *,
    expected_amount_usdc: float | None = None,
    expected_receiver: str | None = None,
    algod_url: str | None = None,
) -> Settlement:
    """Verify an Algorand USDC tx by id; cross-check amount + receiver if given."""
    try:
        from algosdk.v2client import algod
    except ImportError as exc:
        msg = "py-algorand-sdk not installed; run `uv sync --extra chain`."
        raise RuntimeError(msg) from exc

    algod_url = algod_url or os.environ.get(
        "MINDXTRAIN_ALGORAND_ALGOD_URL",
        "https://mainnet-api.algonode.cloud",
    )
    client = algod.AlgodClient("", algod_url)
    info = client.pending_transaction_info(tx_id)
    if not info or "txn" not in info:
        return Settlement(tx_id=tx_id, amount_usdc=0.0, confirmed=False)

    inner = info["txn"]["txn"]
    asset_id = int(inner.get("xaid", 0))
    amount_micro = int(inner.get("aamt", 0))
    sender = info["txn"].get("snd", "") or inner.get("snd", "")
    receiver = inner.get("arcv", "")

    amount_usdc = amount_micro / 1_000_000.0
    confirmed = info.get("confirmed-round", 0) > 0 and asset_id == USDC_ASA_ID
    if expected_amount_usdc is not None and abs(amount_usdc - expected_amount_usdc) > 1e-6:
        confirmed = False
    if expected_receiver and receiver != expected_receiver:
        confirmed = False

    return Settlement(
        tx_id=tx_id,
        asset_id=asset_id,
        amount_usdc=amount_usdc,
        sender=sender,
        receiver=receiver,
        confirmed=confirmed,
    )


__all__ = ["USDC_ASA_ID", "Invoice", "Settlement", "issue_invoice", "validate_settlement"]
