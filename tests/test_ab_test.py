"""A/B traffic Splitter."""

from __future__ import annotations

from mindxtrain.deploy.ab_test import AbConfig, Splitter


def test_zero_canary_pct_always_live():
    s = Splitter(AbConfig(canary_pct=0.0))
    for i in range(100):
        assert s.pick(f"r-{i}") == "live"


def test_full_canary_pct_always_canary():
    s = Splitter(AbConfig(canary_pct=1.0))
    for i in range(100):
        assert s.pick(f"r-{i}") == "canary"


def test_deterministic_per_request_id():
    s = Splitter(AbConfig(canary_pct=0.5))
    a = s.pick("request-42")
    b = s.pick("request-42")
    assert a == b


def test_split_distribution_roughly_canary_pct():
    s = Splitter(AbConfig(canary_pct=0.2))
    n = 2000
    canary = sum(1 for i in range(n) if s.pick(f"req-{i}") == "canary")
    # 20% +/- a generous band.
    assert 0.10 * n <= canary <= 0.30 * n
