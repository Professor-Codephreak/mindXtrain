"""YAML <-> XTrainConfig with recipe-name resolution.

Recipes live in `mindxtrain/train/recipes/*.yaml`.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from mindxtrain.config.schema import XTrainConfig


def load_config(path: str | Path) -> XTrainConfig:
    """Load a YAML config file from disk and validate against XTrainConfig."""
    raw = yaml.safe_load(Path(path).read_text())
    return XTrainConfig.model_validate(raw)


def render_recipe(name: str) -> str:
    """Return the YAML text for a named recipe (e.g. `qwen3_8b_sft_lora`)."""
    pkg = resources.files("mindxtrain.train.recipes")
    candidate = pkg / f"{name}.yaml"
    if not candidate.is_file():
        available = sorted(
            p.name.removesuffix(".yaml")
            for p in pkg.iterdir()
            if p.name.endswith(".yaml")
        )
        msg = f"unknown recipe {name!r}. available: {', '.join(available)}"
        raise FileNotFoundError(msg)
    return candidate.read_text()


def list_recipes() -> list[str]:
    pkg = resources.files("mindxtrain.train.recipes")
    return sorted(
        p.name.removesuffix(".yaml")
        for p in pkg.iterdir()
        if p.name.endswith(".yaml")
    )
