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
    checkpoint: str = Field(description="BLAKE3 of the checkpoint directory")
    # Optional artifacts: a CPU `trl_cpu` run produces only a checkpoint, so the
    # dataset manifest and eval JSON may be absent. An empty hash means "artifact
    # not produced" and is treated as a pass (nothing to verify) by verify_receipt.
    dataset: str = Field(default="", description="BLAKE3 of the dataset shard manifest")
    eval_json: str = Field(default="", description="BLAKE3 of the lm-eval-harness output JSON")
    # BLAKE3 of the frozen AutotunePlan JSON. Binding the plan hash to the
    # checkpoint is what makes a run "natively verifiable" — it proves which
    # AOT-fixed backend/heuristic/RCCL config produced these weights.
    autotune_plan: str = Field(default="", description="BLAKE3 of the frozen AutotunePlan JSON")


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


class TimeAttestation(BaseModel):
    """chronos.agent promised-time stamp.

    Populated when chronos.agent's /v1/oracle/time is reachable at
    manifest-emit time. Otherwise `attested=False` and the unix/utc
    fields fall back to local clock readings — the receipt is still
    valid, just not network-promised.
    """
    model_config = ConfigDict(extra="forbid")

    attested: bool = False
    unix_18dp: str = ""
    utc: str = ""
    consensus: str = "offline"          # correlated | degraded | drifted | offline | unavailable
    confidence_ms: float = 0.0
    anchor_count_24h: int = 0
    promised_by: str = ""


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

    # Promised time from chronos.agent when reachable; falls back to
    # local clock + attested=False otherwise.
    time_attestation: TimeAttestation = Field(default_factory=TimeAttestation)


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
        time_attestation=_fetch_time_attestation(),
    )


CONFIG_SNAPSHOT_NAME = "config.snapshot.yaml"
PLAN_SNAPSHOT_NAME = "autotune_plan.json"


def write_config_snapshot(cfg: object, run_dir: Path) -> Path:
    """Serialize the validated config to `run_dir/config.snapshot.yaml`.

    Deterministic (sorted keys) so re-hashing the same config yields the same
    digest across machines. Returns the snapshot path.
    """
    import yaml

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / CONFIG_SNAPSHOT_NAME
    payload = cfg.model_dump(mode="json")  # type: ignore[attr-defined]
    snapshot.write_text(yaml.safe_dump(payload, sort_keys=True))
    return snapshot


def write_plan_snapshot(plan: object, run_dir: Path) -> Path:
    """Persist the exact AutotunePlan JSON bytes that get hashed into the receipt.

    Re-verification reads these bytes back rather than re-deriving the plan, so
    the receipt proves the plan that actually drove the run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / PLAN_SNAPSHOT_NAME
    snapshot.write_text(plan.model_dump_json(indent=2))  # type: ignore[attr-defined]
    return snapshot


def emit_receipt_for_run(
    cfg: object,
    run_id: str,
    *,
    run_dir: Path,
    plan: object,
    git_sha: str = "",
    rocm_version: str = "7.2.1",
) -> Manifest:
    """Build a Manifest for a completed run, hashing whatever artifacts exist.

    Unlike `emit_receipt`, this is tolerant of the CPU `trl_cpu` lane, which writes
    only `run_dir/checkpoint/`. It snapshots the config and the frozen AutotunePlan
    into `run_dir`, always hashes config + checkpoint + plan, and conditionally
    hashes `dataset_manifest.json` / `eval/lm_eval.json` only when present.
    """
    config_snapshot = write_config_snapshot(cfg, run_dir)
    plan_snapshot = write_plan_snapshot(plan, run_dir)

    dataset_path = run_dir / "dataset_manifest.json"
    eval_path = run_dir / "eval" / "lm_eval.json"
    checkpoint_dir = run_dir / "checkpoint"

    hashes = ProvenanceHashes(
        config_yaml=blake3_file(config_snapshot),
        checkpoint=blake3_dir(checkpoint_dir),
        autotune_plan=blake3_file(plan_snapshot),
        dataset=blake3_file(dataset_path) if dataset_path.is_file() else "",
        eval_json=blake3_file(eval_path) if eval_path.is_file() else "",
    )
    return Manifest(
        run_id=run_id,
        owner=cfg.meta.project,  # type: ignore[attr-defined]
        base_model=cfg.model.name,  # type: ignore[attr-defined]
        rocm_version=rocm_version,
        gfx_arch=cfg.hardware.gfx_arch,  # type: ignore[attr-defined]
        git_sha=git_sha,
        blake3=hashes,
        time_attestation=_fetch_time_attestation(),
    )


def write_run_manifest(manifest: Manifest, run_dir: Path) -> Path:
    """Write `manifest.json` into the run directory and return its path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "manifest.json"
    out.write_text(manifest.model_dump_json(indent=2))
    return out


def _fetch_time_attestation() -> TimeAttestation:
    """Best-effort sync call to chronos.agent's /v1/oracle/time.

    Returns a populated TimeAttestation when mindX is reachable
    (consensus in {correlated, degraded, drifted}); otherwise an
    `attested=False` placeholder so the manifest schema always
    validates and reviewers can tell at a glance whether promotion was
    network-promised time.
    """
    import os
    try:
        import httpx
    except ImportError:
        return TimeAttestation()

    base = os.environ.get("MINDX_BASE_URL", "http://localhost:8000").rstrip("/")
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{base}/v1/oracle/time")
            resp.raise_for_status()
            body = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        return TimeAttestation()

    consensus = body.get("consensus", "offline")
    # Only mark `attested=True` when the upstream reported a real
    # consensus tier — "unavailable" or "offline" stay attested=False.
    attested = consensus in {"correlated", "degraded", "drifted"}
    return TimeAttestation(
        attested=attested,
        unix_18dp=body.get("unix_18dp", ""),
        utc=body.get("utc", ""),
        consensus=consensus,
        confidence_ms=float(body.get("confidence_ms") or 0.0),
        anchor_count_24h=int(body.get("anchor_count_24h") or 0),
        promised_by=body.get("promised_by", ""),
    )
