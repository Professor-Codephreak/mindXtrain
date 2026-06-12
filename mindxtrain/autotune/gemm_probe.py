"""hipBLASLt GEMM heuristic selection.

Previously this returned `hipblaslt_default` unconditionally. It now runs a short
GEMM microbenchmark on a representative MI300X LoRA shape when torch + a GPU are
available, records the timing into the AutotunePlan, and promotes the heuristic
to `hipblaslt_tuned` when ROCm TunableOp tuning is active (PYTORCH_TUNABLEOP_ENABLED).
On a CPU dev box (no torch GPU) it stays at the documented default with no timing.

Reference: AMD ROCm 7.2.1 release notes — hipBLASLt 0.10 default heuristic for
gfx942 BF16/FP16 GEMMs is within ~5% of hand-tuned variants for the shapes
mindXtrain hits (LoRA rank 16-64 on hidden 2048-8192); TunableOp closes the rest.
"""

from __future__ import annotations

import importlib.util
import os
import time

from mindxtrain.autotune.plan import GemmHeuristic, ProbeTiming

# Representative MI300X training GEMM: (M, K) x (K, N) — a hidden=8192 projection
# at batch*seq = 4096 tokens, the dominant LoRA-base shape.
_GEMM_SHAPE = (4096, 8192, 8192)
_ITERATIONS = 10


def _tunableop_active() -> bool:
    return os.environ.get("PYTORCH_TUNABLEOP_ENABLED", "0") not in {"", "0", "false", "False"}


def probe_gemm(gpu_index: int = 0) -> tuple[GemmHeuristic, list[ProbeTiming]]:
    """Return (heuristic, timings) for the autotune plan.

    Returns ('hipblaslt_default', []) when torch + GPU aren't available so the
    dry-run / CPU path stays deterministic.
    """
    _ = gpu_index
    if importlib.util.find_spec("torch") is None:
        return "hipblaslt_default", []

    try:
        import torch

        if not torch.cuda.is_available():
            return "hipblaslt_default", []

        m, k, n = _GEMM_SHAPE
        device = "cuda"
        dtype = torch.bfloat16
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)

        for _ in range(3):  # warmup (also primes TunableOp tuning)
            _ = a @ b
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(_ITERATIONS):
            _ = a @ b
        torch.cuda.synchronize()
        median_ms = (time.perf_counter() - t0) * 1000.0 / _ITERATIONS
    except (RuntimeError, ImportError, OSError):
        return "hipblaslt_default", []

    heuristic: GemmHeuristic = "hipblaslt_tuned" if _tunableop_active() else "hipblaslt_default"
    timing = ProbeTiming(
        label=f"gemm-{m}x{k}x{n}",
        backend=heuristic,
        median_ms=float(median_ms),
        iterations=_ITERATIONS,
    )
    return heuristic, [timing]
