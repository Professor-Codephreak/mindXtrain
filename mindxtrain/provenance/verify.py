"""Verify a custmodel Manifest by re-hashing on-disk artifacts."""

from __future__ import annotations

from pathlib import Path

from mindxtrain.provenance.hashing import blake3_bytes, blake3_dir, blake3_file
from mindxtrain.provenance.manifest import Manifest


def _verify_optional_file(path: Path, expected: str) -> bool:
    """Pass-by-default for optional artifacts.

    An empty stored hash means the artifact was never produced (e.g. a CPU run
    with no eval JSON) — nothing to verify, so report True. Otherwise the file
    must exist and re-hash to the stored digest.
    """
    if not expected:
        return True
    if not path.is_file():
        return False
    return blake3_file(path) == expected


def verify_receipt(
    manifest: Manifest,
    *,
    config_yaml_path: Path,
    dataset_manifest_path: Path,
    checkpoint_dir: Path,
    eval_json_path: Path,
    plan_json: bytes | None = None,
) -> dict[str, bool]:
    """Re-hash each artifact and report a per-field pass/fail dict.

    Optional artifacts (dataset, eval JSON, autotune plan) pass when their stored
    hash is empty. The autotune plan is verified against `plan_json` — the exact
    bytes persisted as `autotune_plan.json` — when both are present.
    """
    checks = {
        "config_yaml": blake3_file(config_yaml_path) == manifest.blake3.config_yaml,
        "checkpoint": blake3_dir(checkpoint_dir) == manifest.blake3.checkpoint,
        "dataset": _verify_optional_file(dataset_manifest_path, manifest.blake3.dataset),
        "eval_json": _verify_optional_file(eval_json_path, manifest.blake3.eval_json),
    }
    expected_plan = manifest.blake3.autotune_plan
    if not expected_plan:
        checks["autotune_plan"] = True
    elif plan_json is None:
        checks["autotune_plan"] = False
    else:
        checks["autotune_plan"] = blake3_bytes(plan_json) == expected_plan
    return checks
