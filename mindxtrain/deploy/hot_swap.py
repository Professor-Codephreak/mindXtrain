"""Atomic hot-swap with canary — promote canary -> live, rollback to previous.

Pure registry-state mutation; the actual vLLM router slot rotation happens
in the inference layer (out of scope for the hackathon MVP).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from mindxtrain.deploy.registry import DeployRegistry, Slot


def promote_canary_to_live(registry_path: Path) -> Slot:
    """Promote `canary` -> `live` atomically; retain previous live for rollback.

    Returns the new live Slot. Raises if no canary is staged.
    """
    reg = DeployRegistry(registry_path)
    canary = reg.get("canary")
    if canary is None:
        msg = "no canary slot to promote"
        raise RuntimeError(msg)
    current_live = reg.get("live")
    if current_live is not None:
        reg.set(
            Slot(
                name="previous_live",
                run_id=current_live.run_id,
                blake3=current_live.blake3,
                activated_at=current_live.activated_at,
            )
        )
    new_live = Slot(
        name="live",
        run_id=canary.run_id,
        blake3=canary.blake3,
        activated_at=datetime.now(tz=UTC),
    )
    reg.set(new_live)
    reg.clear("canary")
    return new_live


def rollback_live(registry_path: Path) -> Slot:
    """Roll `live` back to `previous_live`. Raises if no rollback target exists."""
    reg = DeployRegistry(registry_path)
    prev = reg.get("previous_live")
    if prev is None:
        msg = "no previous_live slot — nothing to roll back to"
        raise RuntimeError(msg)
    new_live = Slot(
        name="live",
        run_id=prev.run_id,
        blake3=prev.blake3,
        activated_at=datetime.now(tz=UTC),
    )
    reg.set(new_live)
    reg.clear("previous_live")
    return new_live


__all__ = ["promote_canary_to_live", "rollback_live"]
