"""Best-effort verifiable-receipt emission at the end of an operator run.

Shared by the Coach UI spawner (`coach/api.py`) and the public training-jobs API
(`training_api.py`). Binding the AutotunePlan hash to the checkpoint at completion
is what makes a run "natively verifiable" — the receipt proves which AOT-fixed
plan produced these weights. Emission is best-effort: a failure logs a line but
never fails the run (mirrors how `publish` tolerates HF/Lighthouse outages).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from mindxtrain.operator import runs as _runs

if TYPE_CHECKING:
    from mindxtrain.autotune.plan import AutotunePlan
    from mindxtrain.config.schema import XTrainConfig


def _git_sha() -> str:
    """Resolve the current commit SHA, empty string if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def emit_run_receipt(
    registry: _runs.RunRegistry,
    run: _runs.Run,
    cfg: XTrainConfig,
    plan: AutotunePlan,
) -> Path | None:
    """Write `manifest.json` + snapshots into the run dir; log the outcome.

    Returns the manifest path on success, None on failure. Publishes a LogEvent
    either way so the Coach train-log tail shows what happened.
    """
    from mindxtrain.provenance.manifest import emit_receipt_for_run, write_run_manifest

    try:
        manifest = emit_receipt_for_run(
            cfg, run.id, run_dir=Path(run.out_dir), plan=plan, git_sha=_git_sha(),
        )
        path = write_run_manifest(manifest, Path(run.out_dir))
    except Exception as exc:  # best-effort: never fail the run on receipt errors
        registry.publish_threadsafe(
            run.id,
            _runs.LogEvent(
                run_id=run.id, line=f"receipt emission skipped: {exc}", level="stderr",
            ),
        )
        return None

    registry.publish_threadsafe(
        run.id,
        _runs.LogEvent(
            run_id=run.id,
            line=f"verifiable receipt written: {path}",
            level="stdout",
        ),
    )
    return path
