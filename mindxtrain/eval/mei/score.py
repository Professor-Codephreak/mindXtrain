"""MEI composite computation — pure functions over MEIRecord (spec §5).

Each sub-index lives on [0, 1] with explicit anchor calibration. The
composite is a weighted geometric mean (spec §5.6) so a weak sub-index
collapses the score — exactly the property the promotion gate relies on.

Worked-example anchors used as test ground truth:
- 30 tok/s decode → Dt ≈ 0.50 (spec §5.2: log10(30/3) / log10(300/3))
- 100 tok/s decode → Dt ≈ 0.77 (spec §5.2)
- 8B Q4_K_M at ~5 GB → M ≈ 0.61 (spec §5.4)
- 70B Q4_K_M at ~40 GB → M ≈ 0.26 (spec §5.4)
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from mindxtrain.eval.mei.anchors import (
    ANCHORS_V01,
    PROMOTION_COMPOSITE_THRESHOLD,
    PROMOTION_QUALITY_BAND_FLOOR,
    PROMOTION_SUBINDEX_FLOOR,
    QUALITY_BAND_EVALS,
    MEIAnchors,
    quality_band_weights,
)
from mindxtrain.eval.mei.record import MEIRecord


class ReferencePool(BaseModel):
    """Random-baseline floors + Qwen3.5-flagship ceilings per spec §5.1.

    Min-max normalization (Open LLM Leaderboard v2 convention):
        s_i = clamp((raw_i - random_i) / (Q3.5_flag_i - random_i), 0, 1)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "v0.1"
    random_baselines: dict[str, float] = Field(
        default_factory=lambda: {
            # MMLU-Pro ten-choice: 10% random floor per Wang et al.
            "mmlu_pro": 0.10,
            # GPQA-Diamond four-choice: 25%.
            "gpqa_diamond": 0.25,
            # IFEval strict-prompt: 0 baseline (binary follow).
            "ifeval_strict_prompt": 0.0,
            # LiveBench-Reasoning: 0 baseline.
            "livebench_reasoning": 0.0,
            # BigCodeBench-Hard pass@1: 0 baseline.
            "bigcodebench_hard_pass1": 0.0,
            # MT-Bench-2-turn: 0 baseline (normalized 0-1).
            "mt_bench_2turn": 0.0,
            # mindX Agentic Battery — frozen reference; 0 baseline.
            "mab": 0.0,
        },
    )
    qwen35_flagship_ceilings: dict[str, float] = Field(
        default_factory=lambda: {
            # Initial calibration — refresh in Phase 6 with actual measurements.
            "mmlu_pro": 0.75,
            "gpqa_diamond": 0.55,
            "ifeval_strict_prompt": 0.85,
            "livebench_reasoning": 0.65,
            "bigcodebench_hard_pass1": 0.45,
            "mt_bench_2turn": 0.85,
            "mab": 0.75,
        },
    )


REFERENCE_POOL_V01: ReferencePool = ReferencePool()


class MEIScore(BaseModel):
    """Headline MEI score plus the disclosed sub-indices (spec §5.6).

    Per the spec, a composite without its five components is non-
    conformant and must be rejected at intake — so this object always
    carries both.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    composite: float = Field(ge=0.0, le=1.0)
    quality: float = Field(ge=0.0, le=1.0)
    decode_throughput: float = Field(ge=0.0, le=1.0)
    prefill_throughput: float = Field(ge=0.0, le=1.0)
    memory: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    quality_bands: dict[str, float] = Field(default_factory=dict)
    mab_provisional: bool = True
    anchors_version: str = "0.1"
    notes: list[str] = Field(default_factory=list)


def _log_compress(value: float, floor: float, ceiling: float) -> float:
    """Logarithmic [0, 1] compression of a value between floor and ceiling.

    `(log10(value) - log10(floor)) / (log10(ceiling) - log10(floor))`
    clamped to [0, 1]. Values <= 0 collapse to 0 (the floor) to keep the
    composite geometric mean well-defined.
    """
    if value <= 0.0:
        return 0.0
    if floor <= 0.0 or ceiling <= floor:
        msg = f"degenerate anchors: floor={floor}, ceiling={ceiling}"
        raise ValueError(msg)
    raw = (math.log10(value) - math.log10(floor)) / (math.log10(ceiling) - math.log10(floor))
    return max(0.0, min(1.0, raw))


def _log_compress_inverse(value: float, floor: float, ceiling: float) -> float:
    """Inverse-log compression for sub-indices where smaller is better
    (memory footprint, energy per token). At `floor` → 1.0, at `ceiling` → 0.0.
    """
    if value <= 0.0:
        return 1.0
    if floor <= 0.0 or ceiling <= floor:
        msg = f"degenerate anchors: floor={floor}, ceiling={ceiling}"
        raise ValueError(msg)
    raw = (math.log10(ceiling) - math.log10(value)) / (math.log10(ceiling) - math.log10(floor))
    return max(0.0, min(1.0, raw))


def _geometric_mean(values: list[float], weights: list[float] | None = None) -> float:
    """Weighted geometric mean. Returns 0.0 if any value is 0.0 (which is
    the spec's intended behaviour — a single weak sub-index collapses the
    composite)."""
    if not values:
        return 0.0
    if any(v <= 0.0 for v in values):
        return 0.0
    if weights is None:
        return math.exp(sum(math.log(v) for v in values) / len(values))
    if len(weights) != len(values):
        msg = f"weight length {len(weights)} != values length {len(values)}"
        raise ValueError(msg)
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return math.exp(
        sum(w * math.log(v) for v, w in zip(values, weights, strict=True)) / total_w,
    )


def _normalize_eval(name: str, raw: float, pool: ReferencePool) -> float:
    """Min-max against the reference pool (Open LLM Leaderboard v2 form)."""
    floor = pool.random_baselines.get(name, 0.0)
    ceiling = pool.qwen35_flagship_ceilings.get(name, 1.0)
    if ceiling <= floor:
        return 0.0
    norm = (raw - floor) / (ceiling - floor)
    return max(0.0, min(1.0, norm))


def quality_subindex(
    quality_raw: dict[str, float],
    pool: ReferencePool,
    *,
    mab_provisional: bool,
) -> tuple[float, dict[str, float]]:
    """Compute Q as the weighted geometric mean across the four bands.

    Within each band, evaluations are arithmetic-meaned; bands themselves
    are geometric-meaned (penalizes catastrophic weakness in any one band
    per spec §5.1). Returns (Q, per-band scores) so callers can disclose
    the band breakdown.

    When `mab_provisional` is True, the Agentic band weight redistributes
    across the other three bands (spec §9).
    """
    weights = quality_band_weights(mab_provisional=mab_provisional)
    band_scores: dict[str, float] = {}
    for band, evals in QUALITY_BAND_EVALS.items():
        # If the band is the Agentic band under provisional, skip
        # entirely — its weight is 0.0 and any non-zero in `band_scores`
        # would carry through with weight 0, polluting the geo-mean.
        if mab_provisional and band == "agentic":
            continue
        normalized = [
            _normalize_eval(name, quality_raw[name], pool)
            for name in evals
            if name in quality_raw
        ]
        if not normalized:
            band_scores[band] = 0.0
            continue
        band_scores[band] = sum(normalized) / len(normalized)

    # Build the weighted geo-mean inputs in stable order.
    used_bands = [b for b in weights if weights[b] > 0.0]
    values = [band_scores.get(b, 0.0) for b in used_bands]
    band_weights_in_order = [weights[b] for b in used_bands]
    q = _geometric_mean(values, weights=band_weights_in_order)
    return q, band_scores


def decode_throughput_subindex(record: MEIRecord, anchors: MEIAnchors = ANCHORS_V01) -> float:
    """`Dt`: geometric mean of Ttg across context tiers, log-compressed.

    Spec §5.2: penalizes catastrophic long-context degradation more than
    an arithmetic mean would.
    """
    ttg_values = [t.ttg_tok_s for t in record.tier_measurements if t.ttg_tok_s > 0]
    if not ttg_values:
        return 0.0
    geo_ttg = _geometric_mean(ttg_values)
    return _log_compress(geo_ttg, anchors.t_floor_decode, anchors.t_ceiling_decode)


def prefill_throughput_subindex(record: MEIRecord, anchors: MEIAnchors = ANCHORS_V01) -> float:
    """`Pp`: log-compressed prefill rate. We use the geometric mean of
    Tpp across context tiers for symmetry with Dt, even though prefill
    typically scales differently — the four-tier picture captures both.
    """
    tpp_values = [t.tpp_tok_s for t in record.tier_measurements if t.tpp_tok_s > 0]
    if not tpp_values:
        return 0.0
    geo_tpp = _geometric_mean(tpp_values)
    return _log_compress(geo_tpp, anchors.p_floor_prefill, anchors.p_ceiling_prefill)


def memory_subindex(record: MEIRecord, anchors: MEIAnchors = ANCHORS_V01) -> float:
    """`M`: inverse-log of peak memory at the 32K working point.

    Spec §5.4: rewards small, dense memory footprints. A Qwen3-8B Q4_K_M
    at ~5 GB scores ≈ 0.61 per the spec example.
    """
    return _log_compress_inverse(record.peak_memory_gb, anchors.m_floor_gb, anchors.m_ceiling_gb)


def energy_subindex(record: MEIRecord, anchors: MEIAnchors = ANCHORS_V01) -> float:
    """`E`: inverse-log of joules per useful decoded token.

    When energy was not measured (and not estimated), returns 0.5 with a
    note flag — the spec's "where direct measurement is unavailable"
    fallback. The score uses the bandwidth-bound proxy when
    `energy_estimated=True` and the value is set.
    """
    j = record.energy_j_per_useful_token
    if j is None:
        # Neutral midpoint when truly unmeasured. The CI will widen
        # accordingly via the bootstrap step downstream.
        return 0.5
    return _log_compress_inverse(j, anchors.j_floor_per_tok, anchors.j_ceiling_per_tok)


def composite_mei(
    q: float, dt: float, pp: float, m: float, e: float, anchors: MEIAnchors = ANCHORS_V01,
) -> float:
    """`MEI = Q^wQ · Dt^wDt · Pp^wPp · M^wM · E^wE` (spec §5.6)."""
    values = [q, dt, pp, m, e]
    weights = [anchors.w_q, anchors.w_dt, anchors.w_pp, anchors.w_m, anchors.w_e]
    return _geometric_mean(values, weights=weights)


def score_record(
    record: MEIRecord,
    pool: ReferencePool = REFERENCE_POOL_V01,
    anchors: MEIAnchors = ANCHORS_V01,
) -> MEIScore:
    """End-to-end score function — the public entry point."""
    q, band_scores = quality_subindex(
        record.quality_raw, pool, mab_provisional=record.mab_provisional,
    )
    dt = decode_throughput_subindex(record, anchors)
    pp = prefill_throughput_subindex(record, anchors)
    m = memory_subindex(record, anchors)
    e = energy_subindex(record, anchors)
    composite = composite_mei(q, dt, pp, m, e, anchors)

    notes: list[str] = []
    if record.mab_provisional:
        notes.append(
            "Agentic Battery v1.0 not yet sealed — 0.35 band weight "
            "redistributed equally across the other three Q bands.",
        )
    if record.energy_j_per_useful_token is None:
        notes.append("Energy not measured; E sub-index defaulted to 0.5 neutral.")
    elif record.energy_estimated:
        notes.append("Energy derived from bandwidth-bound proxy, not direct measurement.")

    return MEIScore(
        composite=composite,
        quality=q,
        decode_throughput=dt,
        prefill_throughput=pp,
        memory=m,
        energy=e,
        quality_bands=dict(band_scores),
        mab_provisional=record.mab_provisional,
        anchors_version=anchors.version,
        notes=notes,
    )


def is_promotable(
    score: MEIScore,
    prior_promoted: MEIScore | None = None,
) -> tuple[bool, list[str]]:
    """Three-gate promotion logic per spec §8.

    Returns (promotable, reasons). When `promotable=False`, the reasons
    list enumerates every failing gate (not short-circuited) so callers
    can surface all problems at once.
    """
    reasons: list[str] = []
    if score.composite < PROMOTION_COMPOSITE_THRESHOLD:
        reasons.append(
            f"composite {score.composite:.3f} < {PROMOTION_COMPOSITE_THRESHOLD} threshold",
        )
    subindices = {
        "quality": score.quality,
        "decode_throughput": score.decode_throughput,
        "prefill_throughput": score.prefill_throughput,
        "memory": score.memory,
        "energy": score.energy,
    }
    for name, value in subindices.items():
        if value < PROMOTION_SUBINDEX_FLOOR:
            reasons.append(
                f"sub-index {name}={value:.3f} below {PROMOTION_SUBINDEX_FLOOR} floor",
            )
    # The Agentic band must clear its own floor independently — but only
    # when the MAB is sealed. While provisional, the 0.35 weight already
    # redistributed and the composite-level check carries the load.
    if not score.mab_provisional:
        agentic = score.quality_bands.get("agentic", 0.0)
        if agentic < PROMOTION_QUALITY_BAND_FLOOR:
            reasons.append(
                f"Agentic band {agentic:.3f} below {PROMOTION_QUALITY_BAND_FLOOR} floor",
            )
    if prior_promoted is not None and score.composite <= prior_promoted.composite:
        reasons.append(
            f"composite {score.composite:.3f} does not exceed currently-promoted "
            f"{prior_promoted.composite:.3f}",
        )
    return (not reasons, reasons)


__all__ = [
    "REFERENCE_POOL_V01",
    "MEIScore",
    "ReferencePool",
    "composite_mei",
    "decode_throughput_subindex",
    "energy_subindex",
    "is_promotable",
    "memory_subindex",
    "prefill_throughput_subindex",
    "quality_subindex",
    "score_record",
]
