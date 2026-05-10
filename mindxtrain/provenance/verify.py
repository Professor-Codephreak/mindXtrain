"""Verify a custmodel Manifest by re-hashing on-disk artifacts."""

from __future__ import annotations

from pathlib import Path

from mindxtrain.provenance.hashing import blake3_dir, blake3_file
from mindxtrain.provenance.manifest import Manifest


def verify_receipt(
    manifest: Manifest,
    *,
    config_yaml_path: Path,
    dataset_manifest_path: Path,
    checkpoint_dir: Path,
    eval_json_path: Path,
) -> dict[str, bool]:
    """Re-hash each artifact and report a per-field pass/fail dict."""
    return {
        "config_yaml": blake3_file(config_yaml_path) == manifest.blake3.config_yaml,
        "dataset": blake3_file(dataset_manifest_path) == manifest.blake3.dataset,
        "checkpoint": blake3_dir(checkpoint_dir) == manifest.blake3.checkpoint,
        "eval_json": blake3_file(eval_json_path) == manifest.blake3.eval_json,
    }
