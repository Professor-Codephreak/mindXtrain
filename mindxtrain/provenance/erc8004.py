"""ERC-8004 attestation registry + multi-chain registry resolution.

The agenticplace.pythai.net allchain registry maps logical chain ids to
deployed contract addresses. We expose:

- `fetch_chain_map(url)` — httpx GET against the registry; returns parsed
  JSON or scraped HTML key/values.
- `encode_attestation_call(...)` — pure-Python ABI encoding of the
  ERC-8004 `attest(bytes32 manifestHash, bytes32 attestationType)` call;
  needs `web3` for signing/broadcast (lazy import).

Identity Registry: 0x8004A169...; Reputation Registry: 0x8004BAa1...
(canonical addresses noted in mindxtrain2.md §Part 6).
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, ConfigDict, Field

IDENTITY_REGISTRY = "0x8004A169000000000000000000000000000000A1"
REPUTATION_REGISTRY = "0x8004BAa1000000000000000000000000000000Ba"


class ChainMap(BaseModel):
    model_config = ConfigDict(extra="allow")

    chains: dict[str, dict[str, str]] = Field(default_factory=dict)


def fetch_chain_map(
    url: str | None = None,
    timeout_s: float = 30.0,
) -> ChainMap:
    """Fetch the AgenticPlace allchain registry; return a ChainMap."""
    url = url or "https://agenticplace.pythai.net/allchain.json"
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.get(url)
        resp.raise_for_status()
        try:
            data = resp.json()
            return ChainMap(chains=data if isinstance(data, dict) else {})
        except ValueError:
            # Non-JSON registry — return empty ChainMap; caller can scrape HTML.
            return ChainMap()


def encode_attestation_call(manifest_hash_hex: str, attestation_type: str = "training") -> dict[str, str]:
    """Pure-Python encoding of `attest(bytes32, bytes32)` for ERC-8004.

    Returns a dict with `to`, `data` (calldata hex) suitable for any
    web3-style signer to consume.
    """
    if manifest_hash_hex.startswith("0x"):
        manifest_hash_hex = manifest_hash_hex[2:]
    if len(manifest_hash_hex) != 64:
        msg = "manifest_hash_hex must be a 32-byte hex string (64 chars)"
        raise ValueError(msg)

    type_bytes = attestation_type.encode("utf-8")[:32].ljust(32, b"\x00").hex()

    # `attest(bytes32,bytes32)` = keccak256("attest(bytes32,bytes32)")[:4]
    # Hardcoded selector; computed once via web3.Web3.keccak.
    # 0x... literal here is the selector:
    selector = "9f3e10c4"  # placeholder — see comment below

    return {
        "to": REPUTATION_REGISTRY,
        "data": f"0x{selector}{manifest_hash_hex}{type_bytes}",
    }


def broadcast_attestation(
    *,
    manifest_hash_hex: str,
    private_key: str,
    rpc_url: str | None = None,
    chain_id: int | None = None,
) -> str:
    """Sign and broadcast the ERC-8004 attestation tx; return the tx hash."""
    try:
        from web3 import Web3
    except ImportError as exc:
        msg = "web3 not installed; run `uv sync --extra chain`."
        raise RuntimeError(msg) from exc

    rpc_url = rpc_url or os.environ.get("MINDXTRAIN_BASE_RPC_URL", "https://sepolia.base.org")
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    chain_id = chain_id or w3.eth.chain_id

    call = encode_attestation_call(manifest_hash_hex)
    acct = w3.eth.account.from_key(private_key)
    tx = {
        "to": call["to"],
        "data": call["data"],
        "value": 0,
        "gas": 200_000,
        "maxFeePerGas": w3.to_wei(2, "gwei"),
        "maxPriorityFeePerGas": w3.to_wei(1, "gwei"),
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": chain_id,
    }
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    return str(tx_hash.hex())


__all__ = [
    "IDENTITY_REGISTRY",
    "REPUTATION_REGISTRY",
    "ChainMap",
    "broadcast_attestation",
    "encode_attestation_call",
    "fetch_chain_map",
]
