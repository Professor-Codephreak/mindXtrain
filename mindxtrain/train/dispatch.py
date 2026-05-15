"""Training backend dispatch.

Lane selection happens at `cfg.train.backend`:

- `axolotl` — GPU SFT/LoRA via Axolotl subprocess (default for MI300X recipes).
- `unsloth` — GPU SFT via Unsloth.
- `torchtune` — GPU SFT via torchtune.
- `primus` — AMD's training stack.
- `trl_cpu` — CPU SFT/LoRA via TRL in-process. Real checkpoints, slow.
  Use for: mindX self-training, smoke-testing a recipe before burning AMD
  credits, anywhere a MI300X droplet isn't available.

The CPU lane is paired with `hardware.gpus: 0` in the recipe. The dispatcher
itself does not enforce that pairing — the recipe is the source of truth —
but the schema's `Literal[0, 1, 8]` constrains the GPU count.
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
    if backend == "trl_cpu":
        from mindxtrain.train.backend_trl_cpu import run_trl_cpu

        return run_trl_cpu(cfg, plan, out_dir)
    msg = f"unknown backend {backend!r}"
    raise ValueError(msg)
