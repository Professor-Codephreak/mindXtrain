"""XEI — mindXtrain Efficiency Index (training-side companion, spec §6).

Where MEI scores inference behaviour, XEI scores the *production process*
producing each checkpoint. Four components:

1. **Training throughput** — tokens/sec/device + global, expressed as
   Model FLOPs Utilization (MFU). PaLM 540B reached 46.2% MFU; the spec's
   alpha target is ≥ 35%.
2. **Optimization health** — gradient-norm stability + power-law loss-
   curve fit `L(D) = E + A/D^α`.
3. **Convergence rate** — slope of validation loss vs log-tokens-trained.
4. **Cost per quality unit** — $ or GPU-hours to advance the held-out
   MAB probe by a fixed delta.

Pure math — no `torch`, no `numpy`. Inputs are plain numbers and lists;
outputs are dataclasses the training callback emits per checkpoint.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# ---- FLOPs per token + MFU --------------------------------------------------

# Spec §6: FLOPs per token follows `6N + 12 L H Q T` decomposition where
# N is non-embedding params, L layers, H attention heads, Q head dim,
# T sequence length. The `6N` covers the forward + backward through the
# linear layers; `12 L H Q T` is the attention computation that scales
# linearly in layers and quadratically (in T) in the sequence dimension.
# (The 12 absorbs the 4 attention matrix multiplies × 3 for fwd+bwd.)


def flops_per_token(
    *, params_nonembed: int, num_layers: int, num_heads: int,
    head_dim: int, seq_len: int,
) -> float:
    """Compute FLOPs per training token per the §6 decomposition.

    Returns the total floating-point operations a single token traverses
    in one forward+backward pass. PaLM-style estimate; matches the
    formula used by Megatron-LM's MFU calculation.
    """
    if params_nonembed <= 0 or num_layers <= 0 or num_heads <= 0:
        msg = "params, layers, and heads must be positive"
        raise ValueError(msg)
    return 6.0 * params_nonembed + 12.0 * num_layers * num_heads * head_dim * seq_len


def compute_mfu(
    *, tokens_per_sec_global: float, num_devices: int,
    peak_device_flops: float, fpt: float,
) -> float:
    """Model FLOPs Utilization.

    `MFU = (observed_tokens_per_sec * FLOPs_per_token) /
           (num_devices * peak_device_FLOPS)`

    Returns a value in [0, 1]. PaLM 540B published 0.462; well-tuned
    Megatron 0.50-0.55; Llama-3.1 0.38-0.43; DeepSeek-V3 on H800 ≈ 0.38.
    The alpha target is ≥ 0.35; below 0.30 flags a config defect.
    """
    if num_devices <= 0 or peak_device_flops <= 0 or fpt <= 0:
        msg = "num_devices, peak_device_flops, and fpt must be positive"
        raise ValueError(msg)
    return (tokens_per_sec_global * fpt) / (num_devices * peak_device_flops)


# ---- Optimization health (gradient norm + loss-curve fit) ------------------


@dataclass(frozen=True)
class GradNormSpike:
    """One detected spike in the gradient-norm series."""

    step: int
    value: float
    baseline: float
    ratio: float  # value / baseline


def detect_grad_norm_spikes(
    grad_norms: list[float],
    *,
    spike_ratio: float = 10.0,
    baseline_window: int = 50,
) -> list[GradNormSpike]:
    """Return every step where grad-norm exceeds spike_ratio × trailing baseline.

    Spec §6: "spikes exceeding ten times the trailing baseline trigger a
    step-skip protocol and are logged as instability events." We treat
    the trailing baseline as the median of the last `baseline_window`
    pre-spike values for robustness against tail outliers.
    """
    if spike_ratio <= 1.0:
        msg = "spike_ratio must exceed 1.0 (baseline is the comparator)"
        raise ValueError(msg)
    spikes: list[GradNormSpike] = []
    for i, g in enumerate(grad_norms):
        if i < baseline_window:
            continue
        window = grad_norms[max(0, i - baseline_window):i]
        if not window:
            continue
        baseline = statistics.median(window)
        if baseline <= 0:
            continue
        ratio = g / baseline
        if ratio >= spike_ratio:
            spikes.append(GradNormSpike(step=i, value=g, baseline=baseline, ratio=ratio))
    return spikes


def grad_norm_stability(
    grad_norms: list[float],
    *,
    target_low: float = 0.5,
    target_high: float = 2.0,
) -> float:
    """Fraction of grad-norm samples in the healthy [0.5, 2.0] band.

    Spec §6: "the gradient norm should remain in the 0.5–2.0 band once
    linear warmup completes." Higher = more stable optimization.
    Returns a value in [0, 1].
    """
    if not grad_norms:
        return 0.0
    in_band = sum(1 for g in grad_norms if target_low <= g <= target_high)
    return in_band / len(grad_norms)


@dataclass(frozen=True)
class PowerLawFit:
    """Result of a `L(D) = E + A / D^α` fit to a loss curve."""

    e: float
    a: float
    alpha: float
    residual_rms: float  # root-mean-square residual after fit


def fit_loss_power_law(
    tokens_trained: list[float],
    losses: list[float],
    *,
    alpha_initial: float = 0.30,
) -> PowerLawFit:
    """Fit a simple power-law `L = E + A / D^α` to (tokens, loss) points.

    Spec §6 expects α ≈ 0.28-0.34 on the token axis. We use a coarse
    iterative refinement (no scipy dep): linearise `log(L - E_guess) =
    log(A) - α · log(D)` and least-squares fit slope and intercept for a
    few candidate `E` floors, picking the fit with the smallest residual.
    Good enough for "is this curve power-law-shaped" rather than a fully
    principled non-linear regression.
    """
    if len(tokens_trained) != len(losses):
        msg = "tokens_trained and losses must align"
        raise ValueError(msg)
    if len(tokens_trained) < 4:
        msg = "need at least 4 (tokens, loss) points for a meaningful fit"
        raise ValueError(msg)
    if min(tokens_trained) <= 0:
        msg = "tokens_trained must all be > 0"
        raise ValueError(msg)

    min_loss = min(losses)
    best: PowerLawFit | None = None
    # Sweep E candidates below min_loss in a multiplicative grid — the
    # true asymptote can be anywhere from 0 (no irreducible loss) to
    # arbitrarily close to min_loss (the curve is near its floor). A
    # coarse absolute grid misses the close-to-floor case for curves
    # whose true E is near min_loss.
    multiplicative = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
    e_grid: list[float] = [0.0] + [
        max(0.0, min_loss * (1.0 - frac)) for frac in multiplicative
    ]
    for e_guess in e_grid:
        residuals_squared: list[float] = []
        # Skip points where loss <= e_guess (log is undefined).
        log_d: list[float] = []
        log_y: list[float] = []
        for d, lo in zip(tokens_trained, losses, strict=True):
            if lo <= e_guess:
                continue
            log_d.append(math.log(d))
            log_y.append(math.log(lo - e_guess))
        if len(log_d) < 4:
            continue
        # Least-squares slope of log_y vs log_d. slope = -α.
        n = len(log_d)
        mean_x = sum(log_d) / n
        mean_y = sum(log_y) / n
        var_x = sum((x - mean_x) ** 2 for x in log_d)
        if var_x == 0:
            continue
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_d, log_y, strict=True))
        # Degenerate fit: flat loss curve (constant losses) has zero
        # covariance and would yield α ≈ 0, which is not a power law.
        # Skip this candidate; a power-law-shaped curve must show some
        # log-log slope.
        if abs(cov_xy) < 1e-12:
            continue
        slope = cov_xy / var_x
        alpha = -slope
        # A power law needs a positive exponent — α ≤ 0 means the loss
        # rises (or stays flat) with more tokens, the opposite of what
        # the §6 fit expects.
        if alpha <= 1e-6:
            continue
        log_a = mean_y - slope * mean_x
        a = math.exp(log_a)
        # Residual in original (not log) space.
        for d, lo in zip(tokens_trained, losses, strict=True):
            predicted = e_guess + a / (d ** alpha) if alpha != 0 else e_guess + a
            residuals_squared.append((lo - predicted) ** 2)
        residual_rms = math.sqrt(sum(residuals_squared) / len(residuals_squared))
        fit = PowerLawFit(e=e_guess, a=a, alpha=alpha, residual_rms=residual_rms)
        if best is None or fit.residual_rms < best.residual_rms:
            best = fit
    if best is None:
        msg = "no valid power-law fit found (all loss values below every E candidate)"
        raise ValueError(msg)
    return best


def power_law_health(fit: PowerLawFit, *, expected_alpha_low: float = 0.20,
                     expected_alpha_high: float = 0.40) -> float:
    """Score how close a fit's α is to the spec's expected band [0.20, 0.40].

    Returns 1.0 when α is squarely in the band, ramping linearly to 0.0
    outside it. The spec's "expected α ≈ 0.28-0.34" is the centre; we
    widen the acceptance to [0.20, 0.40] because exact-band fits are
    sensitive to noise.
    """
    a = fit.alpha
    centre = 0.5 * (expected_alpha_low + expected_alpha_high)
    half_width = 0.5 * (expected_alpha_high - expected_alpha_low)
    if half_width <= 0:
        return 0.0
    distance = abs(a - centre)
    if distance <= half_width:
        return 1.0
    # Ramp down linearly over another half-width; clamp at 0.
    return max(0.0, 1.0 - (distance - half_width) / half_width)


# ---- Convergence rate ------------------------------------------------------


def convergence_slope_per_log_tokens(
    tokens_trained: list[float],
    val_losses: list[float],
) -> float:
    """Slope of validation loss against log10(tokens_trained).

    Negative is good — loss falling as tokens accumulate. Magnitude
    indicates pace. Returns 0.0 for ill-conditioned input (single point,
    flat loss, zero tokens).
    """
    if len(tokens_trained) != len(val_losses):
        msg = "tokens_trained and val_losses must align"
        raise ValueError(msg)
    if len(tokens_trained) < 2:
        return 0.0
    # Drop any non-positive token counts (can't log).
    pairs = [(d, lo) for d, lo in zip(tokens_trained, val_losses, strict=True) if d > 0]
    if len(pairs) < 2:
        return 0.0
    log_d = [math.log10(d) for d, _ in pairs]
    y = [lo for _, lo in pairs]
    n = len(log_d)
    mean_x = sum(log_d) / n
    mean_y = sum(y) / n
    var_x = sum((x - mean_x) ** 2 for x in log_d)
    if var_x == 0:
        return 0.0
    cov_xy = sum((x - mean_x) * (yi - mean_y) for x, yi in zip(log_d, y, strict=True))
    return cov_xy / var_x


# ---- Cost per quality unit -------------------------------------------------


def cost_per_quality_unit(*, gpu_hours: float, mab_delta: float) -> float:
    """GPU-hours required to advance the held-out MAB probe by 1.0 of score.

    Returns ∞ when `mab_delta` is non-positive (no improvement — every
    dollar is wasted; the run should stop). The metric is directly the
    spec's "analogue of intelligence per dollar applied to training."
    """
    if mab_delta <= 0:
        return float("inf")
    if gpu_hours < 0:
        msg = "gpu_hours cannot be negative"
        raise ValueError(msg)
    return gpu_hours / mab_delta


# ---- XEIRecord — emitted per checkpoint ------------------------------------


@dataclass(frozen=True)
class XEIRecord:
    """One XEI score per checkpoint interval.

    Carried in the training callback so each checkpoint emits an
    `xei.jsonl` row alongside the existing manifest pipeline.
    """

    mfu: float
    hfu: float | None
    grad_norm_stability: float
    grad_norm_spikes: int
    loss_power_law_alpha: float | None
    loss_power_law_residual_rms: float | None
    convergence_slope: float
    cost_per_quality_unit: float | None


def score_xei(
    *,
    tokens_per_sec_global: float,
    num_devices: int,
    peak_device_flops: float,
    fpt: float,
    grad_norms: list[float],
    loss_history: list[float] | None = None,
    tokens_history: list[float] | None = None,
    val_losses: list[float] | None = None,
    val_tokens: list[float] | None = None,
    gpu_hours: float | None = None,
    mab_delta: float | None = None,
    hfu: float | None = None,
) -> XEIRecord:
    """End-to-end XEI scoring for one checkpoint snapshot.

    All series arguments are lists of numbers — the caller is responsible
    for sampling them from the training loop. No torch import here.
    """
    mfu = compute_mfu(
        tokens_per_sec_global=tokens_per_sec_global,
        num_devices=num_devices,
        peak_device_flops=peak_device_flops,
        fpt=fpt,
    )
    stability = grad_norm_stability(grad_norms)
    spikes = detect_grad_norm_spikes(grad_norms)

    fit: PowerLawFit | None = None
    if loss_history and tokens_history and len(loss_history) >= 4:
        try:
            fit = fit_loss_power_law(tokens_history, loss_history)
        except (ValueError, ZeroDivisionError):
            fit = None

    slope = 0.0
    if val_losses and val_tokens:
        slope = convergence_slope_per_log_tokens(val_tokens, val_losses)

    cpqu: float | None = None
    if gpu_hours is not None and mab_delta is not None:
        cpqu = cost_per_quality_unit(gpu_hours=gpu_hours, mab_delta=mab_delta)

    return XEIRecord(
        mfu=mfu,
        hfu=hfu,
        grad_norm_stability=stability,
        grad_norm_spikes=len(spikes),
        loss_power_law_alpha=fit.alpha if fit else None,
        loss_power_law_residual_rms=fit.residual_rms if fit else None,
        convergence_slope=slope,
        cost_per_quality_unit=cpqu,
    )


__all__ = [
    "GradNormSpike",
    "PowerLawFit",
    "XEIRecord",
    "compute_mfu",
    "convergence_slope_per_log_tokens",
    "cost_per_quality_unit",
    "detect_grad_norm_spikes",
    "fit_loss_power_law",
    "flops_per_token",
    "grad_norm_stability",
    "power_law_health",
    "score_xei",
]
