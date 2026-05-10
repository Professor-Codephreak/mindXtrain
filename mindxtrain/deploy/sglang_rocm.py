"""SGLang-ROCm launcher — alternative to vLLM for serving.

Day 5+ stub. SGLang ships explicit `rocm/sgl-dev:v0.5.8.post1-rocm720-mi30x` images.
Pairs with the Mooncake distributed-KV-cache plugin used by AMD/Xiaomi MiMo-V2.5-Pro.
"""

from __future__ import annotations

from pathlib import Path

from mindxtrain.config.schema import ServeCfg


def build_sglang_command(cfg: ServeCfg, model_dir: Path) -> list[str]:
    """Return argv for booting SGLang-ROCm against a quantized checkpoint."""
    return [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_dir),
        "--port",
        str(cfg.port),
        "--mem-fraction-static",
        "0.85",
        "--tp",
        str(cfg.tensor_parallel),
    ]
