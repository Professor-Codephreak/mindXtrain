"""GPTQ quantization on ROCm — alternative to Quark FP8.

Subprocess wrapper around `python -m auto_gptq` (ROCm wheels available at
https://huggingface.github.io/autogptq-index/whl/rocm573/). Prefer
`mindxtrain.deploy.quark.quark_fp8` for MI300X — keep this around as a
fallback for older ROCm versions that don't have FP8 support.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _autogptq_available() -> bool:
    return importlib.util.find_spec("auto_gptq") is not None


def gptq_rocm(in_dir: Path, out_dir: Path, bits: int = 4) -> Path:
    """Quantize an HF checkpoint to GPTQ-`bits` weights; return out_dir."""
    if not _autogptq_available():
        msg = (
            "auto-gptq not installed; install the ROCm wheel from "
            "https://huggingface.github.io/autogptq-index/whl/rocm573/auto_gptq/ "
            "or use mindxtrain.deploy.quark.quark_fp8 instead."
        )
        raise RuntimeError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "auto_gptq",
        "--model_name_or_path",
        str(in_dir),
        "--output_dir",
        str(out_dir),
        "--bits",
        str(bits),
    ]
    subprocess.run(cmd, check=True)
    return out_dir


__all__ = ["gptq_rocm"]
