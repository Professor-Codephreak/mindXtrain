"""Pricing math for x402 invoice issuance."""

from __future__ import annotations

import pytest

from mindxtrain.budget.pricing import MI300X_USDC_PER_HOUR, gpu_hour_price


def test_baseline_rate_pinned():
    assert MI300X_USDC_PER_HOUR == pytest.approx(1.99)


def test_single_gpu_one_hour_with_safety_margin():
    quote = gpu_hour_price(gpus=1, hours=1.0)
    assert quote == pytest.approx(1.99 * 1.15, rel=1e-9)


def test_eight_gpu_two_hours():
    quote = gpu_hour_price(gpus=8, hours=2.0)
    expected = 8 * 2 * 1.99 * 1.15
    assert quote == pytest.approx(expected, rel=1e-9)


def test_zero_safety_margin_passthrough():
    quote = gpu_hour_price(gpus=1, hours=1.5, safety_margin=1.0)
    assert quote == pytest.approx(1.99 * 1.5, rel=1e-9)


def test_fractional_hour():
    quote = gpu_hour_price(gpus=1, hours=0.25, safety_margin=1.0)
    assert quote == pytest.approx(0.4975, rel=1e-9)
