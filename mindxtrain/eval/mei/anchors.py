"""MEI v0.1 anchors — the floors and ceilings that calibrate the
logarithmic compression of each sub-index (spec §5.2-§5.5).

These anchors are frozen at module load. Calibration to mid-2026
hardware per the spec; bump to v1.0 when the It's FOSS reference and
the Apple Silicon / Blackwell-Ultra envelopes drift. Cross-era
comparisons must be made via raw sub-index values, not composite
scores, when anchors differ.

The seven evaluation names are canonical — every harness that emits a
quality score uses these keys verbatim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Spec §5.1 — the four quality bands, their assigned evaluations, and the
# weights they receive in the geometric mean. The Agentic band gets 0.35
# (deliberately higher than any public scoreboard, because mindX is an
# autonomous agent and agentic failure dominates cost).
QualityBand = Literal["agentic", "instruction", "reasoning", "knowledge_code"]

QUALITY_BAND_EVALS: dict[QualityBand, tuple[str, ...]] = {
    "agentic": ("mab",),
    "instruction": ("ifeval_strict_prompt", "mt_bench_2turn"),
    "reasoning": ("livebench_reasoning", "gpqa_diamond"),
    "knowledge_code": ("mmlu_pro", "bigcodebench_hard_pass1"),
}

QUALITY_BAND_WEIGHTS_SEALED: dict[QualityBand, float] = {
    "agentic": 0.35,
    "instruction": 0.25,
    "reasoning": 0.20,
    "knowledge_code": 0.20,
}


def quality_band_weights(*, mab_provisional: bool) -> dict[QualityBand, float]:
    """Return the band weights to apply for this run.

    When `mab_provisional` is True (spec §9), the 0.35 Agentic weight is
    redistributed equally across the other three bands until the MAB v1.0
    seals. The other three bands gain ≈0.117 each (their original 0.25 /
    0.20 / 0.20 stretches to ≈0.367 / ≈0.317 / ≈0.317).
    """
    if not mab_provisional:
        return dict(QUALITY_BAND_WEIGHTS_SEALED)
    sealed = QUALITY_BAND_WEIGHTS_SEALED
    agentic_w = sealed["agentic"]
    redistributed = agentic_w / 3.0
    return {
        "agentic": 0.0,
        "instruction": sealed["instruction"] + redistributed,
        "reasoning": sealed["reasoning"] + redistributed,
        "knowledge_code": sealed["knowledge_code"] + redistributed,
    }


class MEIAnchors(BaseModel):
    """Frozen calibration constants for MEI v0.1 (spec §5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = "0.1"

    # §5.2 decode throughput — log-compressed, floor=3 tok/s (It's FOSS
    # "painfully slow"), ceiling=300 tok/s (Apple M3 Ultra / B200 class).
    t_floor_decode: float = Field(default=3.0, gt=0.0)
    t_ceiling_decode: float = Field(default=300.0, gt=0.0)

    # §5.3 prefill — floor=20 tok/s (Phi 4 Mini on CPU), ceiling=5000
    # (FlashAttention-3 on H100).
    p_floor_prefill: float = Field(default=20.0, gt=0.0)
    p_ceiling_prefill: float = Field(default=5000.0, gt=0.0)

    # §5.4 memory — floor=0.5 GB (Qwen 3.0.6B Q4_K_M), ceiling=200 GB
    # (405B-class at FP8).
    m_floor_gb: float = Field(default=0.5, gt=0.0)
    m_ceiling_gb: float = Field(default=200.0, gt=0.0)

    # §5.5 energy — floor=0.1 J/useful-token (efficient Apple Silicon),
    # ceiling=100 J/token (405B on H100 without batching).
    j_floor_per_tok: float = Field(default=0.1, gt=0.0)
    j_ceiling_per_tok: float = Field(default=100.0, gt=0.0)

    # §5.6 composite weights. Must sum to 1.0.
    w_q: float = Field(default=0.40, ge=0.0, le=1.0)
    w_dt: float = Field(default=0.20, ge=0.0, le=1.0)
    w_pp: float = Field(default=0.10, ge=0.0, le=1.0)
    w_m: float = Field(default=0.15, ge=0.0, le=1.0)
    w_e: float = Field(default=0.15, ge=0.0, le=1.0)


ANCHORS_V01: MEIAnchors = MEIAnchors()
"""Frozen v0.1 anchors. Any consumer using this constant is implicitly
declaring its results MEI v0.1-comparable."""

# §8 promotion gate constants.
PROMOTION_COMPOSITE_THRESHOLD: float = 0.55
PROMOTION_SUBINDEX_FLOOR: float = 0.30
PROMOTION_QUALITY_BAND_FLOOR: float = 0.50  # Agentic band specifically (§8).


__all__ = [
    "ANCHORS_V01",
    "PROMOTION_COMPOSITE_THRESHOLD",
    "PROMOTION_QUALITY_BAND_FLOOR",
    "PROMOTION_SUBINDEX_FLOOR",
    "QUALITY_BAND_EVALS",
    "QUALITY_BAND_WEIGHTS_SEALED",
    "MEIAnchors",
    "QualityBand",
    "quality_band_weights",
]
