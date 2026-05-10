"""vLLM serve command builder."""

from __future__ import annotations

from pathlib import Path

from mindxtrain.config.schema import QuantizeCfg, ServeCfg
from mindxtrain.deploy import build_sglang_command, build_vllm_command


def test_vllm_basic_no_quantize_arg():
    cfg = ServeCfg()
    cmd = build_vllm_command(cfg, Path("/runs/x/quantized"))
    assert cmd[:2] == ["vllm", "serve"]
    assert "/runs/x/quantized" in cmd
    assert "--port" in cmd and "8000" in cmd
    assert "--max-model-len" in cmd and "8192" in cmd
    assert "--tensor-parallel-size" in cmd and "1" in cmd
    # No QuantizeCfg passed → no --quantization flag.
    assert "--quantization" not in cmd


def test_vllm_quark_fp8_flag():
    cfg = ServeCfg()
    q = QuantizeCfg(scheme="quark_fp8")
    cmd = build_vllm_command(cfg, Path("/runs/x/quantized"), q)
    qi = cmd.index("--quantization")
    assert cmd[qi + 1] == "fp8"


def test_vllm_quark_mxfp4_flag():
    cfg = ServeCfg()
    q = QuantizeCfg(scheme="quark_mxfp4")
    cmd = build_vllm_command(cfg, Path("/m"), q)
    qi = cmd.index("--quantization")
    assert cmd[qi + 1] == "mxfp4"


def test_vllm_gptq_flag():
    cfg = ServeCfg()
    q = QuantizeCfg(scheme="gptq_rocm")
    cmd = build_vllm_command(cfg, Path("/m"), q)
    qi = cmd.index("--quantization")
    assert cmd[qi + 1] == "gptq"


def test_vllm_no_flag_for_scheme_none():
    cfg = ServeCfg()
    q = QuantizeCfg(scheme="none")
    cmd = build_vllm_command(cfg, Path("/m"), q)
    assert "--quantization" not in cmd


def test_vllm_custom_tensor_parallel_and_port():
    cfg = ServeCfg(tensor_parallel=8, port=9999, max_model_len=131072)
    cmd = build_vllm_command(cfg, Path("/m"))
    assert "8" in cmd
    assert "9999" in cmd
    assert "131072" in cmd


def test_sglang_basic():
    cfg = ServeCfg(backend="sglang")
    cmd = build_sglang_command(cfg, Path("/runs/x/quantized"))
    assert cmd[0] == "python"
    assert "sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert "/runs/x/quantized" in cmd
