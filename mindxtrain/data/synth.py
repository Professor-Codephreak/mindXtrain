"""Synthetic data generation via teacher models.

Drives bulk-prompt rollouts through an OpenAI-compatible teacher endpoint
(default vLLM-ROCm at `MINDXTRAIN_TEACHER_BASE_URL`). Useful for filling
gaps in the curated corpus (style transfer, persona conditioning, tool-call
trajectories).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class SynthRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teacher: str = Field(default="Qwen/Qwen3.5-8B", description="HF id of the teacher model")
    seeds: list[str] = Field(default_factory=list)
    n_per_seed: int = Field(default=8, ge=1)
    style: Literal["sft", "dpo", "tool_use"] = "sft"
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


def _post_completion(
    base_url: str,
    teacher: str,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    body = {
        "model": teacher,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{base_url}/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""


def synthesize(recipe: SynthRecipe, *, base_url: str | None = None, timeout_s: float = 120.0) -> Iterator[dict[str, str]]:
    """Yield synthetic samples per `recipe`. POSTs to a vLLM-compatible endpoint."""
    base_url = (
        base_url or os.environ.get("MINDXTRAIN_TEACHER_BASE_URL", "http://localhost:8000/v1")
    ).rstrip("/")
    teacher = os.environ.get("MINDXTRAIN_TEACHER_MODEL", recipe.teacher)
    for seed in recipe.seeds:
        for i in range(recipe.n_per_seed):
            response = _post_completion(
                base_url,
                teacher,
                seed,
                temperature=recipe.temperature,
                max_tokens=recipe.max_tokens,
                timeout_s=timeout_s,
            )
            yield {
                "seed": seed,
                "rollout_index": str(i),
                "response": response,
                "style": recipe.style,
            }


def merge_synth_with_real(
    synth: Iterable[dict[str, str]],
    real: Iterable[dict[str, str]],
    ratio: float = 0.3,
) -> Iterator[dict[str, str]]:
    """Interleave synth and real samples at the given synth-share ratio.

    Deterministic round-robin: every k-th sample is synth where k=1/ratio.
    """
    if not (0.0 <= ratio <= 1.0):
        msg = f"ratio must be in [0,1]; got {ratio}"
        raise ValueError(msg)
    if ratio == 0.0:
        yield from real
        return
    if ratio == 1.0:
        yield from synth
        return

    real_iter = iter(real)
    synth_iter = iter(synth)
    counter = 0.0
    for _ in range(10**9):  # effectively infinite; consumer breaks
        counter += ratio
        if counter >= 1.0:
            counter -= 1.0
            try:
                yield next(synth_iter)
            except StopIteration:
                # synth exhausted; fall through to real-only
                yield from real_iter
                return
        else:
            try:
                yield next(real_iter)
            except StopIteration:
                yield from synth_iter
                return


__all__ = ["SynthRecipe", "merge_synth_with_real", "synthesize"]
