"""MEI scoring against the spec's worked examples and gate logic.

Every sub-index has at least one ground-truth point pinned from the
spec — the goal is that if anyone ever tweaks the anchors without
re-running calibration (spec Phase 6), these tests immediately fail.
"""
from __future__ import annotations

import math

import pytest

from mindxtrain.eval.mei.anchors import (
    ANCHORS_V01,
    PROMOTION_COMPOSITE_THRESHOLD,
    QUALITY_BAND_WEIGHTS_SEALED,
    quality_band_weights,
)
from mindxtrain.eval.mei.record import (
    ConcurrencyPoint,
    ContextTierMeasurement,
    HardwareIdent,
    InferenceEngineIdent,
    LatencyPercentiles,
    MEIRecord,
    QuantizationTuple,
)
from mindxtrain.eval.mei.score import (
    REFERENCE_POOL_V01,
    MEIScore,
    composite_mei,
    decode_throughput_subindex,
    is_promotable,
    memory_subindex,
    quality_subindex,
    score_record,
)


def _latency(p50=10.0, p95=20.0, p99=40.0):
    return LatencyPercentiles(
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        ci_low_p95=p95 - 1.0, ci_high_p95=p95 + 1.0, sample_n=120,
    )


def _tier(c, *, tpp=400.0, ttg=30.0):
    return ContextTierMeasurement(
        context_tokens=c, tpp_tok_s=tpp, ttg_tok_s=ttg,
        bytes_per_sec=ttg * 4.0,
        ttft=_latency(), tpot=_latency(), itl=_latency(),
    )


def _record(**overrides):
    base = dict(
        model_id="test/model",
        model_sha256="abcd1234",
        tokenizer_revision="rev",
        quantization=QuantizationTuple(scheme="Q4_K_M", bpw=4.85),
        hardware=HardwareIdent(cpu_sku="test"),
        engine=InferenceEngineIdent(name="llama.cpp", commit_sha="0"),
        seed=0,
        tier_measurements=[_tier(c) for c in (32, 512, 8192, 32768)],
        concurrency=[ConcurrencyPoint(concurrency=1, aggregate_throughput_tok_s=30.0, p99_ttft_ms=200.0)],
        peak_memory_gb=5.0,
        kv_cache_gb_at_32k=1.0,
        quality_raw={
            "mmlu_pro": 0.40, "gpqa_diamond": 0.30,
            "ifeval_strict_prompt": 0.60, "livebench_reasoning": 0.40,
            "bigcodebench_hard_pass1": 0.20, "mt_bench_2turn": 0.55, "mab": 0.30,
        },
    )
    base.update(overrides)
    return MEIRecord(**base)


# ---- spec §5.2 worked examples for Dt -----------------------------------


def test_dt_30_toks_per_sec_is_approx_half():
    """Spec §5.2: 30 tok/s → log10(30/3) / log10(300/3) = 1/2 = 0.50."""
    record = _record(tier_measurements=[_tier(c, ttg=30.0) for c in (32, 512, 8192, 32768)])
    dt = decode_throughput_subindex(record)
    assert dt == pytest.approx(0.5, abs=1e-9)


def test_dt_100_toks_per_sec_is_approx_0p77():
    """Spec §5.2: 100 tok/s → log10(100/3) / log10(300/3) ≈ 0.7616."""
    record = _record(tier_measurements=[_tier(c, ttg=100.0) for c in (32, 512, 8192, 32768)])
    dt = decode_throughput_subindex(record)
    expected = math.log10(100 / 3) / math.log10(300 / 3)
    assert dt == pytest.approx(expected, abs=1e-9)
    assert 0.76 < dt < 0.77


def test_dt_at_or_below_floor_collapses_to_zero():
    record = _record(tier_measurements=[_tier(c, ttg=3.0) for c in (32, 512, 8192, 32768)])
    # log10(3/3) = 0 → exactly at the floor; clamp produces 0 modulo FP noise.
    assert decode_throughput_subindex(record) == pytest.approx(0.0, abs=1e-9)
    # And explicitly below the floor.
    record_below = _record(
        tier_measurements=[_tier(c, ttg=1.0) for c in (32, 512, 8192, 32768)],
    )
    assert decode_throughput_subindex(record_below) == 0.0


def test_dt_at_or_above_ceiling_clamps_to_one():
    record = _record(tier_measurements=[_tier(c, ttg=300.0) for c in (32, 512, 8192, 32768)])
    # log10(300/3) / log10(300/3) = 1 modulo FP noise.
    assert decode_throughput_subindex(record) == pytest.approx(1.0, abs=1e-9)
    record_above = _record(
        tier_measurements=[_tier(c, ttg=900.0) for c in (32, 512, 8192, 32768)],
    )
    assert decode_throughput_subindex(record_above) == 1.0


def test_dt_uses_geometric_mean_across_tiers():
    """Spec §5.2 — penalises catastrophic long-context degradation."""
    # 100 / 100 / 100 / 10 — arithmetic mean is 77.5, geometric is ≈ 56.2.
    record = _record(tier_measurements=[
        _tier(32, ttg=100.0), _tier(512, ttg=100.0),
        _tier(8192, ttg=100.0), _tier(32768, ttg=10.0),
    ])
    dt = decode_throughput_subindex(record)
    geo = math.pow(100 * 100 * 100 * 10, 0.25)
    expected = math.log10(geo / 3) / math.log10(300 / 3)
    assert dt == pytest.approx(expected, abs=1e-6)


# ---- spec §5.4 worked example for M -------------------------------------


def test_memory_8b_q4km_5gb_is_approx_0p61():
    """Spec §5.4: an 8B Q4_K_M derivative at ~5 GB scores ≈ 0.61."""
    record = _record(peak_memory_gb=5.0)
    m = memory_subindex(record)
    expected = (math.log10(200) - math.log10(5)) / (math.log10(200) - math.log10(0.5))
    assert m == pytest.approx(expected, abs=1e-9)
    assert 0.60 < m < 0.62


def test_memory_70b_q4km_40gb_is_approx_0p26():
    """Spec §5.4: 70B Q4_K_M at ~40 GB scores ≈ 0.26."""
    record = _record(peak_memory_gb=40.0)
    m = memory_subindex(record)
    expected = (math.log10(200) - math.log10(40)) / (math.log10(200) - math.log10(0.5))
    assert m == pytest.approx(expected, abs=1e-9)
    assert 0.26 < m < 0.27


# ---- composite + geometric collapse property ----------------------------


def test_composite_collapses_when_one_subindex_is_zero():
    """Geometric form: a single weak axis kills the composite. This is the
    spec's intended behaviour — promotion gating relies on it."""
    assert composite_mei(0.9, 0.0, 0.9, 0.9, 0.9) == 0.0
    assert composite_mei(0.9, 0.9, 0.0, 0.9, 0.9) == 0.0
    assert composite_mei(0.0, 1.0, 1.0, 1.0, 1.0) == 0.0


def test_composite_all_equal_subindices_returns_same():
    """Geometric mean of N equal values is that value."""
    assert composite_mei(0.5, 0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-9)


def test_composite_weights_match_spec():
    """Spec §5.6 table: 0.40 / 0.20 / 0.10 / 0.15 / 0.15."""
    assert ANCHORS_V01.w_q == 0.40
    assert ANCHORS_V01.w_dt == 0.20
    assert ANCHORS_V01.w_pp == 0.10
    assert ANCHORS_V01.w_m == 0.15
    assert ANCHORS_V01.w_e == 0.15
    total = ANCHORS_V01.w_q + ANCHORS_V01.w_dt + ANCHORS_V01.w_pp + ANCHORS_V01.w_m + ANCHORS_V01.w_e
    assert total == pytest.approx(1.0, abs=1e-9)


# ---- provisional-Agentic redistribution ---------------------------------


def test_provisional_redistribution_sums_to_one():
    """Spec §9: redistribute the 0.35 Agentic weight equally across the
    other three bands until MAB v1.0 seals."""
    weights = quality_band_weights(mab_provisional=True)
    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    assert weights["agentic"] == 0.0
    # The non-Agentic weights should each have gained ~0.117.
    for band in ("instruction", "reasoning", "knowledge_code"):
        sealed = QUALITY_BAND_WEIGHTS_SEALED[band]
        assert weights[band] > sealed


def test_sealed_weights_match_spec_table():
    """When sealed, the weights are the §5.1 table verbatim."""
    weights = quality_band_weights(mab_provisional=False)
    assert weights == QUALITY_BAND_WEIGHTS_SEALED


def test_quality_subindex_provisional_ignores_mab():
    """Under provisional, the `mab` eval contributes nothing to Q."""
    raw_no_mab = {
        "mmlu_pro": 0.40, "gpqa_diamond": 0.30,
        "ifeval_strict_prompt": 0.60, "livebench_reasoning": 0.40,
        "bigcodebench_hard_pass1": 0.20, "mt_bench_2turn": 0.55,
    }
    raw_with_mab = {**raw_no_mab, "mab": 0.99}
    q_a, _ = quality_subindex(raw_no_mab, REFERENCE_POOL_V01, mab_provisional=True)
    q_b, _ = quality_subindex(raw_with_mab, REFERENCE_POOL_V01, mab_provisional=True)
    assert q_a == pytest.approx(q_b, abs=1e-9)


# ---- score_record end-to-end -------------------------------------------


def test_score_record_emits_all_five_subindices():
    record = _record()
    score = score_record(record)
    assert isinstance(score, MEIScore)
    assert 0.0 <= score.composite <= 1.0
    for v in (score.quality, score.decode_throughput, score.prefill_throughput, score.memory, score.energy):
        assert 0.0 <= v <= 1.0


def test_score_record_attaches_provisional_note():
    record = _record()  # mab_provisional default True
    score = score_record(record)
    assert any("Agentic Battery" in n for n in score.notes)
    assert score.mab_provisional is True


def test_score_record_handles_unmeasured_energy():
    record = _record(energy_j_per_useful_token=None, energy_estimated=False)
    score = score_record(record)
    assert score.energy == 0.5
    assert any("Energy not measured" in n for n in score.notes)


def test_score_record_handles_estimated_energy():
    record = _record(energy_j_per_useful_token=2.0, energy_estimated=True)
    score = score_record(record)
    assert any("bandwidth-bound proxy" in n for n in score.notes)


# ---- promotion gate matrix -----------------------------------------------


def _passing_score():
    return MEIScore(
        composite=0.60, quality=0.55, decode_throughput=0.50,
        prefill_throughput=0.45, memory=0.55, energy=0.50,
        quality_bands={"instruction": 0.6, "reasoning": 0.5, "knowledge_code": 0.55},
        mab_provisional=True, notes=[],
    )


def test_promotion_passes_when_all_gates_clear():
    promotable, reasons = is_promotable(_passing_score())
    assert promotable is True, reasons
    assert reasons == []


def test_promotion_fails_below_composite_threshold():
    score = _passing_score().model_copy(update={"composite": 0.40})
    promotable, reasons = is_promotable(score)
    assert promotable is False
    assert any("composite" in r for r in reasons)


def test_promotion_fails_when_any_subindex_below_floor():
    """Spec §8 — `no sub-index below 0.30`."""
    score = _passing_score().model_copy(update={"memory": 0.25})
    promotable, reasons = is_promotable(score)
    assert promotable is False
    # The reason string formats the floor as `0.3` (Python's default
    # float repr for round numbers). Just look for the sub-index name +
    # the word "floor" to keep the test resilient to formatting tweaks.
    assert any("memory" in r and "floor" in r for r in reasons)


def test_promotion_fails_when_not_better_than_prior():
    """Spec §8: 'MEI strictly higher … than the currently-promoted'."""
    s = _passing_score()
    prior = s.model_copy(update={"composite": 0.65})
    promotable, reasons = is_promotable(s, prior_promoted=prior)
    assert promotable is False
    assert any("currently-promoted" in r for r in reasons)


def test_promotion_agentic_floor_only_when_sealed():
    """Under provisional, the Agentic band is 0.0 (redistributed); the
    band-floor check must not fire. Once sealed, it does fire."""
    provisional = _passing_score().model_copy(update={
        "mab_provisional": True,
        "quality_bands": {"agentic": 0.0, "instruction": 0.6, "reasoning": 0.5},
    })
    ok, _ = is_promotable(provisional)
    assert ok is True

    sealed = provisional.model_copy(update={
        "mab_provisional": False,
        "quality_bands": {"agentic": 0.10, "instruction": 0.6, "reasoning": 0.5},
    })
    ok2, reasons = is_promotable(sealed)
    assert ok2 is False
    assert any("Agentic band" in r for r in reasons)


def test_promotion_threshold_constant_matches_spec():
    """Spec §8: 0.55 absolute."""
    assert PROMOTION_COMPOSITE_THRESHOLD == 0.55
