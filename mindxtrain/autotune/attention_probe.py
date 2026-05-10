"""CK-vs-Triton SDPA microbenchmark.

If torch + ROCm are available, runs ~30 s of timed `scaled_dot_product_attention`
across 4 representative shapes and picks the faster backend. If torch isn't
available (typical CPU dev box), returns the canonical default ('ck', []) so
the AutotunePlan dry-run path stays consistent.
"""

from __future__ import annotations

import importlib.util
import time

from mindxtrain.autotune.plan import AttentionBackend, ProbeTiming

_SHAPES = [
    # (batch, seqlen, num_heads, head_dim)
    (1, 2048, 32, 128),
    (1, 4096, 32, 128),
    (1, 8192, 16, 128),
    (1, 16384, 8, 128),
]


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _time_backend(
    *,
    backend_name: str,
    enable_flash: bool,
    enable_math: bool,
    enable_mem_efficient: bool,
    iterations: int = 5,
) -> ProbeTiming:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = []
    if enable_flash:
        backends.append(SDPBackend.FLASH_ATTENTION)
    if enable_math:
        backends.append(SDPBackend.MATH)
    if enable_mem_efficient:
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    if not backends:
        backends = [SDPBackend.MATH]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    timings: list[float] = []
    for batch, seqlen, heads, head_dim in _SHAPES:
        q = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch, heads, seqlen, head_dim, device=device, dtype=dtype)

        # Warmup.
        with sdpa_kernel(backends=backends):
            for _ in range(2):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with sdpa_kernel(backends=backends):
            for _ in range(iterations):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0 / iterations)

    median_ms = sorted(timings)[len(timings) // 2]
    return ProbeTiming(
        label=f"sdpa-{backend_name}",
        backend=backend_name,
        median_ms=float(median_ms),
        iterations=iterations,
    )


def probe_attention(
    gpu_index: int = 0,
) -> tuple[AttentionBackend, list[ProbeTiming]]:
    """Pick the faster SDPA backend for MI300X.

    Returns ('ck', []) if torch isn't installed (typical CPU dev box).
    """
    _ = gpu_index
    if not _torch_available():
        return "ck", []

    try:
        ck_timing = _time_backend(
            backend_name="ck",
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        )
        triton_timing = _time_backend(
            backend_name="triton",
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )
    except (RuntimeError, ImportError):
        return "ck", []

    timings = [ck_timing, triton_timing]
    winner: AttentionBackend = "ck" if ck_timing.median_ms <= triton_timing.median_ms else "triton"
    return winner, timings


__all__ = ["probe_attention"]
