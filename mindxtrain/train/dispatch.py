"""Training backend dispatch.

Day 1: 4-way switch routes to backend stubs that all raise NotImplementedError.
Day 3: backend_axolotl gets a real implementation; the other three remain
stubs until post-hackathon.
"""

from __future__ import annotations

from pathlib import Path

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.schema import XTrainConfig


def dispatch_training(
    cfg: XTrainConfig,
    plan: AutotunePlan,
    out_dir: Path,
) -> Path:
    """Dispatch a training run to the configured backend.

    Returns the path to the produced checkpoint directory.
    """
    backend = cfg.train.backend
    if backend == "axolotl":
        from mindxtrain.train.sft import run_axolotl

        return run_axolotl(cfg, plan, out_dir)
    if backend == "unsloth":
        from mindxtrain.train.backend_unsloth import run_unsloth

        return run_unsloth(cfg, plan, out_dir)
    if backend == "torchtune":
        from mindxtrain.train.backend_torchtune import run_torchtune

        return run_torchtune(cfg, plan, out_dir)
    if backend == "primus":
        from mindxtrain.train.backend_primus import run_primus

        return run_primus(cfg, plan, out_dir)
    msg = f"unknown backend {backend!r}"
    raise ValueError(msg)
