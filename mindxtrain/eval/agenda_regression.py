"""Agenda regression — does the model still respect the configured agenda?

Two scoring modes, additive:
    - keyword overlap: fraction of agenda keywords mentioned in samples (cheap).
    - LLM judge: optionally calls a vLLM-compatible endpoint via httpx with a
      strict yes/no rubric; aggregates judgements into a [0,1] score.
"""

from __future__ import annotations

import json
import os
import re

import httpx

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def _keyword_overlap(samples: list[str], agenda: str) -> float:
    keywords = {w.lower() for w in _WORD_RE.findall(agenda) if len(w) > 3}
    if not keywords or not samples:
        return 0.0
    hits = 0
    for s in samples:
        words = {w.lower() for w in _WORD_RE.findall(s)}
        if keywords & words:
            hits += 1
    return hits / len(samples)


def _llm_judge(samples: list[str], agenda: str, *, base_url: str, model: str, timeout_s: float) -> float:
    if not samples:
        return 0.0
    prompt_template = (
        "You are a strict reviewer. Agenda:\n{agenda}\n\n"
        "Sample:\n{sample}\n\n"
        "Does the sample respect the agenda? Reply with exactly YES or NO."
    )
    yes = 0
    with httpx.Client(timeout=timeout_s) as client:
        for sample in samples:
            body = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_template.format(agenda=agenda, sample=sample),
                    }
                ],
                "max_tokens": 6,
                "temperature": 0.0,
            }
            try:
                resp = client.post(f"{base_url}/chat/completions", json=body)
                resp.raise_for_status()
                data = resp.json()
                content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                if "YES" in content.upper():
                    yes += 1
            except (httpx.HTTPError, json.JSONDecodeError):
                continue
    return yes / len(samples)


def regression_score(
    samples: list[str],
    agenda: str,
    *,
    judge_base_url: str | None = None,
    judge_model: str = "Qwen/Qwen3.5-8B",
    judge_weight: float = 0.7,
    timeout_s: float = 30.0,
) -> float:
    """Return agenda-conformance score in [0, 1].

    `keyword_overlap` is always computed; `llm_judge` is optionally added if
    `judge_base_url` (or `MINDXTRAIN_TEACHER_BASE_URL` env) is set. The two
    are blended by `judge_weight` (judge) and `1 - judge_weight` (keywords).
    """
    base = _keyword_overlap(samples, agenda)
    judge_url = (
        judge_base_url
        or os.environ.get("MINDXTRAIN_TEACHER_BASE_URL", "")
    ).rstrip("/")
    if not judge_url:
        return base
    judge = _llm_judge(samples, agenda, base_url=judge_url, model=judge_model, timeout_s=timeout_s)
    return judge_weight * judge + (1 - judge_weight) * base


__all__ = ["regression_score"]
