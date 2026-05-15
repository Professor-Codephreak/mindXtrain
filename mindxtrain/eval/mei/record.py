"""Pydantic schemas for the canonical MEI measurement record (spec §7).

The structured record is the integration boundary between inference-engine
wrappers (`throughput.py`), the measurement orchestrator (`orchestrator.py`),
and MEI scoring (`score.py`). Once written, every harness conforms to it.

Design constraints from the spec:

- Every reported figure carries its measurement context: hardware SKU,
  engine commit SHA, tokenizer revision, quantization tuple, seed,
  warmup count, sample size.
- Latency is reported at p50 / p95 / p99 with bootstrap 95% CIs —
  *never* as a mean alone (spec §4, "production latency distributions
  are routinely 15x heavier at p99 than at the mean under contention").
- Throughput is reported per the four-tier context battery (32, 512,
  8192, 32768 tokens) so context-scaling degradation is visible.
- Concurrency sweep at (1, 4, 16, 64, max) captures the Pareto curve;
  a single batch-1 number is non-conformant per spec §4.
- Energy is optional but carries an `energy_estimated` flag when the
  bandwidth-bound proxy was used in lieu of direct measurement.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Exactly four context tiers per spec §4. The headline Ttg used in the
# composite is the geometric mean across these tiers.
EXPECTED_CONTEXT_TIERS: tuple[int, ...] = (32, 512, 8192, 32768)

# Concurrency points per spec §4. `max` is the hardware's sustained max
# under a 10s p99 TTFT SLO — recorded as an integer for the actual point
# achieved by the run.
EXPECTED_CONCURRENCY: tuple[int, ...] = (1, 4, 16, 64)


class HardwareIdent(BaseModel):
    """Hardware reproducibility envelope (spec §4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_sku: str = Field(min_length=1, description="e.g. 'AMD EPYC 9654' or 'Apple M3 Ultra'.")
    simd_class: str = Field(
        default="",
        description="AVX2 | AVX-512 | AVX-512-VNNI | AMX | NEON | SVE2 | SME | Metal | gfx942 …",
    )
    dram_channel_count: int = Field(default=0, ge=0)
    dram_clock_mhz: int = Field(default=0, ge=0)
    gpu_sku: str = Field(default="", description="e.g. 'AMD MI300X', 'NVIDIA H100', or empty for CPU-only.")
    gpu_clock_mhz: int = Field(default=0, ge=0)
    os_kernel: str = Field(default="", description="e.g. 'Linux 6.8.0-110-generic'.")


class InferenceEngineIdent(BaseModel):
    """Identify exactly which engine produced the measurement (spec §4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["llama.cpp", "ollama", "vllm", "sglang", "transformers"] = Field(
        description="Which serving engine collected the timings.",
    )
    commit_sha: str = Field(min_length=1, description="Engine binary commit SHA or version string.")
    config: dict[str, str] = Field(
        default_factory=dict,
        description="Engine-specific config knobs (cache_type_k, --n_threads, --gpu_layers, etc.).",
    )


class QuantizationTuple(BaseModel):
    """Per spec §4: every reported figure carries an explicit quantization tuple.

    A Q4_K_M-with-imatrix run is distinguishable from Q4_K_M-without-imatrix.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheme: str = Field(
        min_length=1,
        description=(
            "Quantization name: Q2_K, Q4_K_M, Q5_K_M, Q8_0, IQ2_XXS, IQ4_NL, "
            "GPTQ, AWQ, EXL2, NF4, FP8, NVFP4, BF16, FP16, FP32."
        ),
    )
    bpw: float = Field(gt=0.0, description="Bits per weight (effective).")
    calibration_corpus: str = Field(
        default="",
        description="Imatrix / GPTQ / AWQ calibration set identifier; empty if none.",
    )


class TokenSeries(BaseModel):
    """The four primary token counts per request (spec §3, §4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_prefill: int = Field(ge=0, description="Prompt tokens consumed, including chat-template scaffolding.")
    n_decode: int = Field(ge=0, description="Output tokens generated (incl. stop tokens before EOS).")
    b_decode: int = Field(ge=0, description="UTF-8 byte length of decoded output.")
    n_useful_decode: int = Field(
        ge=0,
        description="Content tokens after structured-output parsing — excludes ChatML scaffolding.",
    )

    @model_validator(mode="after")
    def _useful_does_not_exceed_decode(self) -> TokenSeries:
        if self.n_useful_decode > self.n_decode:
            msg = (
                f"n_useful_decode ({self.n_useful_decode}) cannot exceed "
                f"n_decode ({self.n_decode}) — useful tokens are a subset."
            )
            raise ValueError(msg)
        return self


class LatencyPercentiles(BaseModel):
    """p50/p95/p99 with bootstrap 95% CI (spec §4).

    Never report a mean alone — distributions are heavy-tailed. The
    `ci_low_p95` / `ci_high_p95` pair is the bootstrap envelope around
    the p95 specifically; the spec privileges p95 as the headline
    latency point.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)
    ci_low_p95: float = Field(ge=0.0, description="Bootstrap 95% CI lower bound on p95.")
    ci_high_p95: float = Field(ge=0.0, description="Bootstrap 95% CI upper bound on p95.")
    sample_n: int = Field(ge=1, description="Samples behind these percentiles (≥ 100 per spec §4).")

    @model_validator(mode="after")
    def _percentile_ordering(self) -> LatencyPercentiles:
        if not (self.p50_ms <= self.p95_ms <= self.p99_ms):
            msg = (
                f"latency percentiles must be monotone: "
                f"p50={self.p50_ms} p95={self.p95_ms} p99={self.p99_ms}"
            )
            raise ValueError(msg)
        if self.ci_low_p95 > self.p95_ms or self.ci_high_p95 < self.p95_ms:
            msg = (
                f"bootstrap CI [{self.ci_low_p95}, {self.ci_high_p95}] "
                f"does not bracket p95 {self.p95_ms}"
            )
            raise ValueError(msg)
        return self


class ContextTierMeasurement(BaseModel):
    """One row of the four-tier context battery (spec §4).

    `Tpp` is the prefill throughput, `Ttg` is the decode throughput, both
    in tokens/sec. The geometric mean of Ttg across tiers is the headline
    decode rate used by `Dt`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_tokens: int = Field(
        description="One of EXPECTED_CONTEXT_TIERS — 32, 512, 8192, 32768.",
    )
    tpp_tok_s: float = Field(ge=0.0, description="Prefill (prompt-processing) throughput.")
    ttg_tok_s: float = Field(ge=0.0, description="Decode (generation) throughput.")
    bytes_per_sec: float = Field(
        ge=0.0,
        description="Auxiliary tokenizer-invariant rate (UTF-8 bytes of decoded output / sec).",
    )
    ttft: LatencyPercentiles = Field(description="Time-to-first-token percentiles.")
    tpot: LatencyPercentiles = Field(description="Time-per-output-token percentiles.")
    itl: LatencyPercentiles = Field(
        description="Inter-token latency (vLLM convention: excludes TTFT).",
    )

    @model_validator(mode="after")
    def _context_is_canonical_tier(self) -> ContextTierMeasurement:
        if self.context_tokens not in EXPECTED_CONTEXT_TIERS:
            msg = (
                f"context_tokens={self.context_tokens} not one of the "
                f"canonical tiers {EXPECTED_CONTEXT_TIERS}; non-conformant"
            )
            raise ValueError(msg)
        return self


class ConcurrencyPoint(BaseModel):
    """One point on the concurrency-vs-throughput-vs-latency Pareto curve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = Field(ge=1, description="Number of in-flight requests.")
    aggregate_throughput_tok_s: float = Field(
        ge=0.0,
        description="Total tokens/sec across all concurrent requests.",
    )
    p99_ttft_ms: float = Field(
        ge=0.0,
        description="p99 TTFT at this concurrency — gates against the 10s SLO.",
    )
    goodput_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of requests meeting the (ttft:500ms, tpot:50ms) SLO. "
            "Cheng et al. smooth-goodput preferred where instrumentable."
        ),
    )


class MEIRecord(BaseModel):
    """The canonical structured record (spec §7).

    One per (model, hardware, engine, quantization, run) tuple. Every
    headline metric the MEI computes is downstream of this record.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ---- identity ----
    model_id: str = Field(min_length=1, description="HF Hub repo ID or local checkpoint name.")
    model_sha256: str = Field(
        min_length=8,
        description="BLAKE3 or SHA-256 of the resolved model weights file(s).",
    )
    tokenizer_revision: str = Field(
        min_length=1,
        description="Tokenizer revision hash per spec Rule 3.1.",
    )
    quantization: QuantizationTuple

    # ---- environment ----
    hardware: HardwareIdent
    engine: InferenceEngineIdent
    seed: int = Field(ge=0)
    warmup_requests: int = Field(default=5, ge=0, description="Discarded warmups per spec §4.")
    sample_size: int = Field(
        default=100,
        ge=1,
        description="Steady-state requests behind every latency percentile.",
    )

    # ---- measurements ----
    tier_measurements: list[ContextTierMeasurement] = Field(
        description="Exactly 4 entries — one per canonical context tier.",
    )
    concurrency: list[ConcurrencyPoint] = Field(
        description="Concurrency sweep — typically 5 points.",
    )
    peak_memory_gb: float = Field(
        ge=0.0,
        description="Resident set at the 32K context working point — weights + KV + runtime.",
    )
    kv_cache_gb_at_32k: float = Field(
        ge=0.0,
        description="KV cache footprint at 32K context — reported separately per spec §4.",
    )
    energy_j_per_useful_token: float | None = Field(
        default=None,
        description="Energy per useful decoded token (J). None if not measured.",
    )
    energy_estimated: bool = Field(
        default=False,
        description="True when E was derived from the bandwidth-bound proxy.",
    )

    # ---- quality ----
    quality_raw: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-evaluation raw scores keyed by canonical name "
            "(mmlu_pro, gpqa_diamond, ifeval_strict_prompt, livebench_reasoning, "
            "bigcodebench_hard_pass1, mt_bench_2turn, mab)."
        ),
    )
    quality_pool_version: str = Field(
        default="v0.1",
        description="Reference pool version (random baseline + Qwen3.5-flagship ceilings).",
    )
    mab_provisional: bool = Field(
        default=True,
        description=(
            "True until the mindX Agentic Battery v1.0 is sealed. While "
            "provisional, the 0.35 Agentic-band weight redistributes "
            "across the other three Q bands."
        ),
    )

    @model_validator(mode="after")
    def _exactly_four_tiers(self) -> MEIRecord:
        contexts = sorted(t.context_tokens for t in self.tier_measurements)
        if contexts != sorted(EXPECTED_CONTEXT_TIERS):
            msg = (
                f"tier_measurements must cover exactly {EXPECTED_CONTEXT_TIERS}; "
                f"got {tuple(contexts)}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _energy_estimated_flag_consistent(self) -> MEIRecord:
        if self.energy_j_per_useful_token is None and self.energy_estimated:
            msg = "energy_estimated=True requires a non-null energy value"
            raise ValueError(msg)
        return self


__all__ = [
    "EXPECTED_CONCURRENCY",
    "EXPECTED_CONTEXT_TIERS",
    "ConcurrencyPoint",
    "ContextTierMeasurement",
    "HardwareIdent",
    "InferenceEngineIdent",
    "LatencyPercentiles",
    "MEIRecord",
    "QuantizationTuple",
    "TokenSeries",
]
