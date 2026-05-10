"""AMD Quark quantization — FP8 (E4M3) for MI300X, MXFP4 for MI350X+.

Subprocess wrapper around `python -m amd_quark.quantize`. The actual
amd-quark Python module ships in the rocm/primus container (or the AMD
Developer Cloud image); on a CPU-only dev box it's not installed and the
function returns a clear error pointing the user to the container.

Outputs a vLLM-loadable directory: `config.json`, `*.safetensors` with FP8
scales, tokenizer files, `generation_config.json`.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Literal


def _quark_available() -> bool:
    return importlib.util.find_spec("amd_quark") is not None or shutil.which("quark") is not None


def _run_quark(in_dir: Path, out_dir: Path, scheme: Literal["fp8_e4m3", "mxfp4"]) -> Path:
    if not _quark_available():
        msg = (
            "amd-quark not installed; run inside the rocm/primus:v26.2 "
            "container or install per "
            "https://quark.docs.amd.com/. (scheme=" + scheme + ")"
        )
        raise RuntimeError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "amd_quark.quantize",
        "--model_dir",
        str(in_dir),
        "--output_dir",
        str(out_dir),
        "--scheme",
        scheme,
    ]
    subprocess.run(cmd, check=True)
    return out_dir


def quark_fp8(in_dir: Path, out_dir: Path) -> Path:
    """Quantize an HF checkpoint to FP8 E4M3; return out_dir (vLLM-loadable)."""
    return _run_quark(in_dir, out_dir, "fp8_e4m3")


def quark_mxfp4(in_dir: Path, out_dir: Path) -> Path:
    """Quantize an HF checkpoint to MXFP4 (CDNA 4 / MI350X+); return out_dir."""
    return _run_quark(in_dir, out_dir, "mxfp4")


__all__ = ["quark_fp8", "quark_mxfp4"]
