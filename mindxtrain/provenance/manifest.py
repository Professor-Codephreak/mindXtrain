"""Provenance manifest — the artifact spec for a trained model.

Produced by `mindxtrain publish` and verified by `mindxtrain receipt`. Single
canonical home per mindxtrain2.md §Part 4 `provenance.manifest`. Merges the
previous `custmodel.manifest` (artifact-side schema) and `xtrain.receipt.manifest`
(run-side emit_receipt helper).

Captures:
    - run identity (run_id, owner, git SHA, ROCm version, gfx arch)
    - BLAKE3 hashes of YAML config, dataset shards, checkpoint dir, eval JSON
    - paths: hf_repo_id, lighthouse_cid, vllm_serve_url
    - on-chain pointers (ERC-7857 INFT, Algorand ASA, ERC-8004 attestation)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mindxtrain.provenance.hashing import blake3_dir, blake3_file


class ProvenanceHashes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_yaml: str = Field(description="BLAKE3 of the XTrainConfig YAML")
    dataset: str = Field(description="BLAKE3 of the dataset shard manifest")
    checkpoint: str = Field(description="BLAKE3 of the checkpoint directory")
    eval_json: str = Field(description="BLAKE3 of the lm-eval-harness output JSON")


class INFTPointer(BaseModel):
    """ERC-7857 INFT reference (Base mainnet)."""

    model_config = ConfigDict(extra="forbid")

    chain: Literal["base", "base_sepolia"] = "base_sepolia"
    contract: str = ""
    token_id: int = 0


class ASAPointer(BaseModel):
    """Algorand ASA for x402 settlement (USDC ASA = 203977300)."""

    model_config = ConfigDict(extra="forbid")

    network: Literal["mainnet", "testnet"] = "mainnet"
    asset_id: int = 0


class OnChainPointers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inft: INFTPointer = Field(default_factory=INFTPointer)
    asa: ASAPointer = Field(default_factory=ASAPointer)
    erc8004_attestation: str = Field(default="", description="tx hash of ERC-8004 attestation")


class Manifest(BaseModel):
    """Trained-model artifact manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    run_id: str
    owner: str = "mindx"
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    base_model: str
    rocm_version: str = "7.2.1"
    gfx_arch: str = "gfx942"
    git_sha: str = ""

    blake3: ProvenanceHashes
    hf_repo_id: str = ""
    lighthouse_cid: str = ""
    vllm_serve_url: str = ""

    on_chain: OnChainPointers = Field(default_factory=OnChainPointers)

    eval_summary: dict[str, float] = Field(default_factory=dict)

    # MEI promotion-gate audit trail. When the publish step bypasses the
    # gate via --force, the manifest records the bypass + the failing
    # reasons so reviewers can see retroactively that promotion was not
    # earned by the §8 thresholds.
    promotion_bypassed: bool = False
    promotion_bypass_reasons: list[str] = Field(default_factory=list)


def emit_receipt(
    cfg: object,
    run_id: str,
    *,
    config_yaml_path: Path,
    dataset_manifest_path: Path,
    checkpoint_dir: Path,
    eval_json_path: Path,
    git_sha: str = "",
    rocm_version: str = "7.2.1",
) -> Manifest:
    """Build a Manifest with BLAKE3 hashes of every artifact.

    `cfg` is duck-typed as an `XTrainConfig` to avoid a circular import; we only
    read `cfg.meta.project`, `cfg.model.name`, `cfg.hardware.gfx_arch`.
    """
    hashes = ProvenanceHashes(
        config_yaml=blake3_file(config_yaml_path),
        dataset=blake3_file(dataset_manifest_path),
        checkpoint=blake3_dir(checkpoint_dir),
        eval_json=blake3_file(eval_json_path),
    )
    return Manifest(
        run_id=run_id,
        owner=cfg.meta.project,  # type: ignore[attr-defined]
        base_model=cfg.model.name,  # type: ignore[attr-defined]
        rocm_version=rocm_version,
        gfx_arch=cfg.hardware.gfx_arch,  # type: ignore[attr-defined]
        git_sha=git_sha,
        blake3=hashes,
    )
