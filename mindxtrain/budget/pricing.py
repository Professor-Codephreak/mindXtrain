"""Pricing function for x402 invoices.

Per blueprint: per-GPU-hour rate * estimated_steps * safety_margin.
$1.99/hr is the AMD Developer Cloud single-MI300X rate (May 2026).
"""

from __future__ import annotations

MI300X_USDC_PER_HOUR = 1.99
SAFETY_MARGIN = 1.15


def gpu_hour_price(gpus: int, hours: float, *, safety_margin: float = SAFETY_MARGIN) -> float:
    """Quote in USDC for `gpus` MI300X x `hours`."""
    return gpus * hours * MI300X_USDC_PER_HOUR * safety_margin
