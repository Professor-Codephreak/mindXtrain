"""XEI training-side companion (spec §6) — MFU, grad-norm health,
power-law loss fit, convergence slope, cost per quality unit.

Reference points pinned from the spec where applicable: PaLM 540B's
published 46.2% MFU is the canonical sanity check for `compute_mfu`.
"""
from __future__ import annotations

import math

import pytest

from mindxtrain.eval.mei.xei import (
    GradNormSpike,
    PowerLawFit,
    XEIRecord,
    compute_mfu,
    convergence_slope_per_log_tokens,
    cost_per_quality_unit,
    detect_grad_norm_spikes,
    fit_loss_power_law,
    flops_per_token,
    grad_norm_stability,
    power_law_health,
    score_xei,
)

# ---- flops_per_token + compute_mfu ---------------------------------------


def test_flops_per_token_obeys_6N_plus_attention_formula():
    """Spec §6: 6N + 12 L H Q T."""
    fpt = flops_per_token(
        params_nonembed=1_000_000_000,
        num_layers=24, num_heads=32, head_dim=128, seq_len=2048,
    )
    expected = 6 * 1_000_000_000 + 12 * 24 * 32 * 128 * 2048
    assert fpt == pytest.approx(float(expected), abs=1e-6)


def test_flops_per_token_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        flops_per_token(params_nonembed=0, num_layers=1, num_heads=1, head_dim=1, seq_len=1)
    with pytest.raises(ValueError):
        flops_per_token(params_nonembed=1, num_layers=0, num_heads=1, head_dim=1, seq_len=1)


def test_compute_mfu_clean_numbers_reproduce_palm_46p2pct():
    """1000 tok/s * 46.2e9 fpt / (1 * 100e12 peak) = 0.462 exactly.

    Spec §6 — PaLM 540B's published MFU. Confirms the formula direction
    and units."""
    mfu = compute_mfu(
        tokens_per_sec_global=1000.0,
        num_devices=1,
        peak_device_flops=100e12,
        fpt=46.2e9,
    )
    assert mfu == pytest.approx(0.462, abs=1e-9)


def test_compute_mfu_rejects_zero_devices_or_peak():
    with pytest.raises(ValueError):
        compute_mfu(tokens_per_sec_global=1, num_devices=0, peak_device_flops=1e12, fpt=1e9)
    with pytest.raises(ValueError):
        compute_mfu(tokens_per_sec_global=1, num_devices=1, peak_device_flops=0, fpt=1e9)


# ---- grad-norm health ----------------------------------------------------


def test_grad_norm_stability_full_in_band():
    """All values in the [0.5, 2.0] healthy band → score 1.0."""
    assert grad_norm_stability([0.6, 1.0, 1.5, 1.8, 0.9]) == 1.0


def test_grad_norm_stability_mixed():
    """Half in / half out."""
    values = [0.6, 0.7, 5.0, 6.0]  # 2 in-band, 2 out
    assert grad_norm_stability(values) == 0.5


def test_grad_norm_stability_empty_returns_zero():
    assert grad_norm_stability([]) == 0.0


def test_grad_norm_stability_custom_band():
    assert grad_norm_stability([0.3, 0.4, 0.5], target_low=0.3, target_high=0.5) == 1.0


# ---- grad-norm spike detection ------------------------------------------


def test_detect_grad_norm_spikes_finds_clear_spike():
    """A single 100x spike above a stable baseline should be detected."""
    grads = [1.0] * 60 + [50.0]  # baseline 1.0, then a spike at index 60
    spikes = detect_grad_norm_spikes(grads, spike_ratio=10.0, baseline_window=50)
    assert len(spikes) == 1
    assert spikes[0].step == 60
    assert spikes[0].ratio == pytest.approx(50.0, abs=1e-9)


def test_detect_grad_norm_spikes_ignores_early_pre_baseline():
    """Within the baseline window we can't compute a baseline; skip."""
    grads = [100.0] * 5  # huge values but no baseline established
    spikes = detect_grad_norm_spikes(grads, baseline_window=50)
    assert spikes == []


def test_detect_grad_norm_spikes_below_threshold_quiet():
    """A 5x bump on a 10x threshold is ignored."""
    grads = [1.0] * 60 + [5.0]
    spikes = detect_grad_norm_spikes(grads, spike_ratio=10.0, baseline_window=50)
    assert spikes == []


def test_detect_grad_norm_spikes_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        detect_grad_norm_spikes([1.0], spike_ratio=0.5)


# ---- power-law loss fit -------------------------------------------------


def test_fit_loss_power_law_recovers_known_shape():
    """Synthesise a clean L = 1.0 + 5.0/D^0.3 curve and recover α ≈ 0.3."""
    tokens = [10.0 ** i for i in range(2, 12)]
    e_true, a_true, alpha_true = 1.0, 5.0, 0.30
    losses = [e_true + a_true / (d ** alpha_true) for d in tokens]
    fit = fit_loss_power_law(tokens, losses)
    assert fit.alpha == pytest.approx(alpha_true, abs=0.05)
    # The residuals should be near zero for a noise-free curve.
    assert fit.residual_rms < 0.1


def test_fit_loss_power_law_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        fit_loss_power_law([1, 2, 3, 4], [1, 2, 3])


def test_fit_loss_power_law_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_loss_power_law([1, 2], [1, 2])


def test_fit_loss_power_law_rejects_non_positive_tokens():
    with pytest.raises(ValueError):
        fit_loss_power_law([0, 1, 2, 3], [1, 1, 1, 1])


def test_power_law_health_in_band():
    """α inside [0.20, 0.40] → 1.0."""
    fit = PowerLawFit(e=1.0, a=5.0, alpha=0.30, residual_rms=0.01)
    assert power_law_health(fit) == 1.0


def test_power_law_health_outside_ramps_down():
    """α just outside the band ramps linearly toward 0 over another
    half-width, then clamps."""
    # band=[0.20, 0.40], centre=0.30, half_width=0.10. α=0.45 sits
    # halfway across the ramp (distance 0.15, ramp 0.5). α=0.50 is at
    # the ramp's end (distance 0.20, ramp 0.0).
    fit_mid_ramp = PowerLawFit(e=0, a=1, alpha=0.45, residual_rms=0.01)
    fit_at_end = PowerLawFit(e=0, a=1, alpha=0.50, residual_rms=0.01)
    fit_far = PowerLawFit(e=0, a=1, alpha=1.00, residual_rms=0.01)
    assert power_law_health(fit_mid_ramp) == pytest.approx(0.5, abs=1e-9)
    assert power_law_health(fit_at_end) == pytest.approx(0.0, abs=1e-9)
    assert power_law_health(fit_far) == 0.0


# ---- convergence slope ---------------------------------------------------


def test_convergence_slope_negative_for_falling_loss():
    """Loss decreasing as tokens grow → negative slope (the healthy case)."""
    tokens = [1e6, 1e7, 1e8, 1e9, 1e10]
    losses = [4.0, 3.5, 3.0, 2.5, 2.0]
    slope = convergence_slope_per_log_tokens(tokens, losses)
    assert slope < 0
    assert slope == pytest.approx(-0.5, abs=1e-6)


def test_convergence_slope_zero_for_flat_loss():
    """Constant loss → zero slope."""
    tokens = [1e6, 1e7, 1e8]
    losses = [3.0, 3.0, 3.0]
    assert convergence_slope_per_log_tokens(tokens, losses) == 0.0


def test_convergence_slope_handles_single_point():
    assert convergence_slope_per_log_tokens([1e6], [3.0]) == 0.0


def test_convergence_slope_drops_non_positive_tokens():
    """A 0-token point would log-explode; the function should skip it."""
    tokens = [0.0, 1e6, 1e7]
    losses = [99.0, 3.0, 2.0]
    slope = convergence_slope_per_log_tokens(tokens, losses)
    # Only the 2 valid points used → slope of those.
    assert slope == pytest.approx(-1.0, abs=1e-6)


# ---- cost per quality unit ----------------------------------------------


def test_cost_per_quality_unit_basic():
    """100 GPU-hours to advance MAB by 0.05 → 2000 GPU-hr/quality-point."""
    cpqu = cost_per_quality_unit(gpu_hours=100.0, mab_delta=0.05)
    assert cpqu == pytest.approx(2000.0, abs=1e-9)


def test_cost_per_quality_unit_infinite_when_no_improvement():
    """No MAB gain = infinite waste; calling code should terminate the run."""
    assert math.isinf(cost_per_quality_unit(gpu_hours=100.0, mab_delta=0.0))
    assert math.isinf(cost_per_quality_unit(gpu_hours=100.0, mab_delta=-0.01))


def test_cost_per_quality_unit_rejects_negative_hours():
    with pytest.raises(ValueError):
        cost_per_quality_unit(gpu_hours=-1.0, mab_delta=0.1)


# ---- score_xei end-to-end ------------------------------------------------


def test_score_xei_minimum_inputs():
    """The smallest valid call — MFU + grad-norm series only."""
    r = score_xei(
        tokens_per_sec_global=1000.0,
        num_devices=1,
        peak_device_flops=100e12,
        fpt=46.2e9,
        grad_norms=[1.0] * 60,
    )
    assert isinstance(r, XEIRecord)
    assert r.mfu == pytest.approx(0.462, abs=1e-9)
    assert r.grad_norm_stability == 1.0
    assert r.grad_norm_spikes == 0
    assert r.loss_power_law_alpha is None
    assert r.convergence_slope == 0.0
    assert r.cost_per_quality_unit is None


def test_score_xei_full_inputs_populates_every_field():
    tokens = [10.0 ** i for i in range(2, 12)]
    losses = [1.0 + 5.0 / (d ** 0.30) for d in tokens]
    r = score_xei(
        tokens_per_sec_global=1000.0,
        num_devices=1,
        peak_device_flops=100e12,
        fpt=46.2e9,
        grad_norms=[1.0] * 60,
        loss_history=losses,
        tokens_history=tokens,
        val_losses=[2.0, 1.5, 1.0],
        val_tokens=[1e6, 1e7, 1e8],
        gpu_hours=100.0,
        mab_delta=0.05,
        hfu=0.52,
    )
    assert r.mfu == pytest.approx(0.462, abs=1e-9)
    assert r.hfu == 0.52
    assert r.loss_power_law_alpha is not None
    assert r.loss_power_law_alpha == pytest.approx(0.30, abs=0.05)
    assert r.convergence_slope < 0
    assert r.cost_per_quality_unit == pytest.approx(2000.0, abs=1e-9)


def test_score_xei_handles_bad_loss_history_gracefully():
    """When the power-law fit can't converge (e.g., constant loss → log(0)),
    score_xei still returns a record with alpha=None rather than crashing."""
    r = score_xei(
        tokens_per_sec_global=1000.0,
        num_devices=1,
        peak_device_flops=100e12,
        fpt=46.2e9,
        grad_norms=[1.0] * 60,
        loss_history=[1.0, 1.0, 1.0, 1.0],  # all equal → log(0) inside fit
        tokens_history=[1e6, 1e7, 1e8, 1e9],
    )
    assert r.loss_power_law_alpha is None


def test_xei_record_is_immutable():
    """Frozen dataclass — attempting to mutate raises FrozenInstanceError."""
    import dataclasses
    r = XEIRecord(
        mfu=0.4, hfu=None, grad_norm_stability=0.9, grad_norm_spikes=0,
        loss_power_law_alpha=None, loss_power_law_residual_rms=None,
        convergence_slope=-0.5, cost_per_quality_unit=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.mfu = 0.99  # type: ignore[misc]


def test_grad_norm_spike_record_carries_baseline():
    """The spike record exposes the trailing baseline so callers can log
    diagnostics, not just the trigger value."""
    spike = GradNormSpike(step=42, value=15.0, baseline=1.5, ratio=10.0)
    assert spike.step == 42
    assert spike.ratio == 10.0
    assert spike.baseline == 1.5
