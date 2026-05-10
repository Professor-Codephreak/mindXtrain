"""hipBLASLt GEMM heuristic selection.

Per the blueprint decision (1 real probe + 2 documented heuristics), we do not
enumerate hipBLASLt heuristics ourselves. We pick `hipblaslt_default` for
gfx942 based on AMD's documented MI300X tuning guidance.

Reference: AMD ROCm 7.2.1 release notes, hipBLASLt 0.10 default heuristic
selection for gfx942 BF16/FP16 GEMMs is within 5% of hand-tuned variants for
the shapes mindXtrain hits (LoRA rank 16-64 on hidden 2048-8192).
"""

from __future__ import annotations

from mindxtrain.autotune.plan import GemmHeuristic


def probe_gemm(gpu_index: int = 0) -> GemmHeuristic:
    """Return the GEMM heuristic for the autotune plan.

    Day 2 stays at the documented default; revisit post-hackathon if MMLU
    eval shows GEMM-bound throughput regression.
    """
    _ = gpu_index
    return "hipblaslt_default"
