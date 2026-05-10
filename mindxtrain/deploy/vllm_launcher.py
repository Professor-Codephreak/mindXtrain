"""Build the `vllm serve` command line from a ServeCfg + QuantizeCfg.

Real implementation (Day 5 wiring is just os.execvp on this list).
"""

from __future__ import annotations

from pathlib import Path

from mindxtrain.config.schema import QuantizeCfg, ServeCfg


def build_vllm_command(
    cfg: ServeCfg,
    model_dir: Path,
    quantize: QuantizeCfg | None = None,
) -> list[str]:
    """Return argv for booting vLLM-ROCm against a quantized checkpoint.

    The serving-time quantization flag mirrors `quantize.scheme` from the
    full XTrainConfig. Pass `quantize=None` to omit the flag (treat the
    checkpoint as the dtype on disk).
    """
    cmd = [
        "vllm",
        "serve",
        str(model_dir),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel),
        "--max-model-len",
        str(cfg.max_model_len),
        "--port",
        str(cfg.port),
    ]
    if quantize is not None and quantize.scheme != "none":
        if quantize.scheme == "quark_fp8":
            cmd += ["--quantization", "fp8"]
        elif quantize.scheme == "quark_mxfp4":
            cmd += ["--quantization", "mxfp4"]
        elif quantize.scheme == "gptq_rocm":
            cmd += ["--quantization", "gptq"]
    return cmd
