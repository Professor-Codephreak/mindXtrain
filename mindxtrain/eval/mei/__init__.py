"""mindX Efficiency Index (MEI) — composite model-evaluation metric.

Implements the v0.1 specification at
`/home/hacker/mindX/docs/operations/mindX Efficiency Index_ … .md`.

The MEI is a single scalar in [0, 1] derived from five log-compressed
sub-indices — quality (Q), decode throughput (Dt), prefill throughput
(Pp), memory footprint (M), and energy per useful token (E) — combined
via weighted geometric mean. It is the in-house metric mindXtrain uses
to gate checkpoint promotion to AgenticPlace.

Layout (mirrors the spec's §7 three-layer instrumentation stack):

- `record.py` — Pydantic schemas for the canonical measurement record.
- `tokenizer.py` — canonical tokenizer wrapper (spec Rule 3.1).
- `throughput.py` — engine harness wrappers (llama.cpp, Ollama, vLLM, HF).
- `energy.py` — power sampling (NVML, RAPL, powermetrics, ROCm-SMI).
- `orchestrator.py` — measurement protocol: warmup → battery → sweep → CI.
- `score.py` — pure functions computing MEI from a record.
- `xei.py` — training-side companion (MFU, optimization health).
- `promotion.py` — three-gate AgenticPlace promotion logic.
- `history.py` — append-only JSONL of historical scores.

Anchor calibration (§5.2-§5.5) is frozen at module load for v0.1; the
ANCHORS object is the authoritative source for floors and ceilings.
"""

from mindxtrain.eval.mei.record import (
    ConcurrencyPoint,
    ContextTierMeasurement,
    HardwareIdent,
    InferenceEngineIdent,
    LatencyPercentiles,
    MEIRecord,
    QuantizationTuple,
    TokenSeries,
)

__all__ = [
    "ConcurrencyPoint",
    "ContextTierMeasurement",
    "HardwareIdent",
    "InferenceEngineIdent",
    "LatencyPercentiles",
    "MEIRecord",
    "QuantizationTuple",
    "TokenSeries",
]
