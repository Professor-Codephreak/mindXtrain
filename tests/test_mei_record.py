"""MEIRecord schema validators — the integration boundary for MEI v0.1.

These tests pin every constraint that the rest of the MEI pipeline relies
on. Breaking any of them is a contract violation, not a minor schema bump.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mindxtrain.eval.mei.record import (
    EXPECTED_CONTEXT_TIERS,
    ConcurrencyPoint,
    ContextTierMeasurement,
    HardwareIdent,
    InferenceEngineIdent,
    LatencyPercentiles,
    MEIRecord,
    QuantizationTuple,
    TokenSeries,
)


def _latency(p50=10.0, p95=20.0, p99=40.0, *, ci_low=18.0, ci_high=22.0, n=120):
    return LatencyPercentiles(
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        ci_low_p95=ci_low, ci_high_p95=ci_high, sample_n=n,
    )


def _tier(context, *, tpp=400.0, ttg=30.0, bytes_s=120.0):
    return ContextTierMeasurement(
        context_tokens=context,
        tpp_tok_s=tpp,
        ttg_tok_s=ttg,
        bytes_per_sec=bytes_s,
        ttft=_latency(50.0, 100.0, 200.0, ci_low=95.0, ci_high=105.0),
        tpot=_latency(),
        itl=_latency(),
    )


def _full_record(**overrides):
    base = dict(
        model_id="pythai/mindx-fallback-qwen3-1.5b",
        model_sha256="abcdef0123456789",
        tokenizer_revision="rev-abcd",
        quantization=QuantizationTuple(scheme="Q4_K_M", bpw=4.85, calibration_corpus=""),
        hardware=HardwareIdent(cpu_sku="AMD EPYC 9654", simd_class="AVX-512"),
        engine=InferenceEngineIdent(name="llama.cpp", commit_sha="b3041"),
        seed=2048,
        tier_measurements=[_tier(c) for c in EXPECTED_CONTEXT_TIERS],
        concurrency=[
            ConcurrencyPoint(concurrency=c, aggregate_throughput_tok_s=20.0 * c, p99_ttft_ms=200.0)
            for c in (1, 4, 16, 64)
        ],
        peak_memory_gb=5.2,
        kv_cache_gb_at_32k=1.75,
        quality_raw={"mmlu_pro": 0.42, "ifeval_strict_prompt": 0.55},
    )
    base.update(overrides)
    return MEIRecord(**base)


# ---- LatencyPercentiles --------------------------------------------------


def test_latency_percentiles_must_be_monotone():
    with pytest.raises(ValidationError):
        LatencyPercentiles(p50_ms=20, p95_ms=10, p99_ms=30,
                           ci_low_p95=8, ci_high_p95=12, sample_n=100)


def test_latency_ci_must_bracket_p95():
    """CI bounds outside [ci_low, ci_high] of p95 are nonsensical."""
    with pytest.raises(ValidationError):
        LatencyPercentiles(p50_ms=10, p95_ms=20, p99_ms=30,
                           ci_low_p95=25, ci_high_p95=30, sample_n=100)
    with pytest.raises(ValidationError):
        LatencyPercentiles(p50_ms=10, p95_ms=20, p99_ms=30,
                           ci_low_p95=10, ci_high_p95=15, sample_n=100)


def test_latency_sample_n_minimum():
    """Spec §4 requires ≥100 samples for percentile reporting; schema
    enforces ≥1 (the spec's 100 is operational guidance, not a hard
    floor — but anything less than 1 is meaningless)."""
    with pytest.raises(ValidationError):
        LatencyPercentiles(p50_ms=1, p95_ms=2, p99_ms=3,
                           ci_low_p95=1.5, ci_high_p95=2.5, sample_n=0)


# ---- TokenSeries ---------------------------------------------------------


def test_token_series_useful_cannot_exceed_decode():
    """Spec §3: N_useful_decode is a SUBSET of N_decode after scaffold strip."""
    with pytest.raises(ValidationError):
        TokenSeries(n_prefill=100, n_decode=50, b_decode=200, n_useful_decode=51)


def test_token_series_allows_zero_useful():
    """Empty useful set is valid (model produced only scaffold tokens)."""
    ts = TokenSeries(n_prefill=10, n_decode=5, b_decode=10, n_useful_decode=0)
    assert ts.n_useful_decode == 0


# ---- ContextTierMeasurement ---------------------------------------------


def test_context_tier_must_be_canonical():
    """Spec §4: only 32 / 512 / 8192 / 32768 are conformant tiers."""
    with pytest.raises(ValidationError):
        _tier(64)  # not a canonical tier
    with pytest.raises(ValidationError):
        _tier(1024)


def test_context_tier_accepts_each_canonical_value():
    for c in EXPECTED_CONTEXT_TIERS:
        t = _tier(c)
        assert t.context_tokens == c


# ---- QuantizationTuple ---------------------------------------------------


def test_quantization_requires_positive_bpw():
    with pytest.raises(ValidationError):
        QuantizationTuple(scheme="Q4_K_M", bpw=0.0)


def test_quantization_carries_calibration():
    """An imatrix run is a different measurement than a no-imatrix one (§4)."""
    q = QuantizationTuple(scheme="Q4_K_M", bpw=4.85, calibration_corpus="wiki-en-1M")
    assert q.calibration_corpus == "wiki-en-1M"


# ---- MEIRecord top-level -------------------------------------------------


def test_record_requires_exactly_four_tiers():
    """Spec §4 mandates the four-prompt calibration battery — no fewer, no more."""
    with pytest.raises(ValidationError):
        _full_record(tier_measurements=[_tier(c) for c in (32, 512, 8192)])
    with pytest.raises(ValidationError):
        _full_record(tier_measurements=[_tier(c) for c in (32, 512, 8192, 32768, 131072)])


def test_record_rejects_duplicate_tiers():
    with pytest.raises(ValidationError):
        _full_record(tier_measurements=[_tier(32), _tier(32), _tier(8192), _tier(32768)])


def test_record_energy_estimated_requires_value():
    with pytest.raises(ValidationError):
        _full_record(energy_j_per_useful_token=None, energy_estimated=True)


def test_record_energy_none_with_estimated_false_is_valid():
    """Energy is optional; flag default-false. Both unset = direct-measure
    unavailable on this run."""
    r = _full_record(energy_j_per_useful_token=None, energy_estimated=False)
    assert r.energy_j_per_useful_token is None
    assert r.energy_estimated is False


def test_record_energy_measured_marks_unestimated():
    r = _full_record(energy_j_per_useful_token=2.5, energy_estimated=False)
    assert r.energy_j_per_useful_token == 2.5


def test_record_energy_estimated_with_value():
    r = _full_record(energy_j_per_useful_token=4.0, energy_estimated=True)
    assert r.energy_estimated is True


def test_record_mab_provisional_default():
    """Until the Agentic Battery seals, all records are provisional."""
    r = _full_record()
    assert r.mab_provisional is True


def test_record_is_frozen():
    """Records must be immutable — frozen=True on every Pydantic model."""
    r = _full_record()
    with pytest.raises(ValidationError):
        r.seed = 9999  # type: ignore[misc]


def test_record_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        MEIRecord(
            model_id="x",
            model_sha256="0123456789ab",
            tokenizer_revision="r",
            quantization=QuantizationTuple(scheme="Q4_K_M", bpw=4.85),
            hardware=HardwareIdent(cpu_sku="x"),
            engine=InferenceEngineIdent(name="llama.cpp", commit_sha="x"),
            seed=0,
            tier_measurements=[_tier(c) for c in EXPECTED_CONTEXT_TIERS],
            concurrency=[],
            peak_memory_gb=1.0,
            kv_cache_gb_at_32k=0.1,
            unexpected="nope",  # type: ignore[call-arg]
        )


def test_record_roundtrips_through_json():
    """Records must survive serialization for the historical-comparison DB."""
    import json
    r = _full_record()
    blob = r.model_dump_json()
    r2 = MEIRecord.model_validate_json(blob)
    assert r2 == r
    # And the JSON parses to a plain dict.
    parsed = json.loads(blob)
    assert parsed["model_id"] == r.model_id
