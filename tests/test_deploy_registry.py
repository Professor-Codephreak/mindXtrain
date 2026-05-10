"""Deploy registry + hot-swap atomicity."""

from __future__ import annotations

import pytest

from mindxtrain.deploy.hot_swap import promote_canary_to_live, rollback_live
from mindxtrain.deploy.registry import DeployRegistry, Slot


def test_registry_round_trip(tmp_path):
    reg = DeployRegistry(tmp_path / "reg.json")
    assert reg.get("live") is None
    reg.set(Slot(name="live", run_id="r1", blake3="a" * 64))
    got = reg.get("live")
    assert got is not None
    assert got.run_id == "r1"


def test_promote_canary_to_live_retains_previous(tmp_path):
    reg = DeployRegistry(tmp_path / "reg.json")
    reg.set(Slot(name="live", run_id="v1", blake3="0" * 64))
    reg.set(Slot(name="canary", run_id="v2", blake3="1" * 64))
    new_live = promote_canary_to_live(tmp_path / "reg.json")
    assert new_live.run_id == "v2"
    assert reg.get("previous_live") is not None
    assert reg.get("previous_live").run_id == "v1"
    assert reg.get("canary") is None


def test_rollback_live(tmp_path):
    reg = DeployRegistry(tmp_path / "reg.json")
    reg.set(Slot(name="live", run_id="v1", blake3="0" * 64))
    reg.set(Slot(name="canary", run_id="v2", blake3="1" * 64))
    promote_canary_to_live(tmp_path / "reg.json")
    rolled = rollback_live(tmp_path / "reg.json")
    assert rolled.run_id == "v1"


def test_rollback_with_no_previous_raises(tmp_path):
    DeployRegistry(tmp_path / "reg.json")
    with pytest.raises(RuntimeError):
        rollback_live(tmp_path / "reg.json")
