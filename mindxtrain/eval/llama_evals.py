"""Simple evaluators — clean-room reimplementation of LlamaIndex-style evals.

Reimplemented from the *behavior* of `llama_index.core.evaluation` (MIT) — never copied.
A minimal set for the "did the trained model recall the training / maintain the persona"
question:

- `SemanticSimilarityEvaluator` — embedding/lexical cosine (reuses `eval.imprint`).
- `CorrectnessEvaluator` — LLM-as-judge of a response vs a reference.
- `PairwiseEvaluator` — which of two responses (before vs after) better matches the persona.
- `GuidelineEvaluator` — does a response comply with a rubric / guidelines.

The LLM-judge evaluators call the existing chat backend via `governance.panel.chat_once`
(ollama/vLLM). Pure stdlib + pydantic + httpx; base-install importable.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

_SCORE_RE = re.compile(r"score\s*[:=]?\s*([0-5](?:\.\d+)?)", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\b(A|B|TIE)\b", re.IGNORECASE)
_DEFAULT_JUDGE = "llama3.2"


class EvalScore(BaseModel):
    """A single evaluation result, score normalized to [0, 1]."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    passing: bool
    reasoning: str = ""
    method: str


def _parse_judge_score(text: str) -> tuple[float, str]:
    """Parse a 1-5 `SCORE: N` from judge output → ([0,1], one-line reasoning)."""
    m = _SCORE_RE.search(text)
    raw = float(m.group(1)) if m else 2.5  # neutral when unparseable
    raw = max(1.0, min(5.0, raw))
    score01 = (raw - 1.0) / 4.0
    reasoning = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())[:240]
    return round(score01, 4), reasoning


def _judge(model: str, system: str, user: str, *, base_url: str | None) -> tuple[float, str]:
    """Run one LLM-judge call; returns ([0,1], reasoning). Best-effort (0.5 on error)."""
    from mindxtrain.governance.panel import chat_once

    try:
        text = chat_once(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            base_url=base_url, max_tokens=300, temperature=0.0,
        )
    except Exception as exc:  # never let an eval crash a workflow
        return 0.5, f"judge error: {exc}"
    return _parse_judge_score(text)


class SemanticSimilarityEvaluator:
    """Embedding/lexical similarity between two texts (reuses `eval.imprint`)."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def evaluate(self, text_a: str, text_b: str) -> EvalScore:
        from mindxtrain.eval.imprint import _voice_similarity

        score, method = _voice_similarity([text_a], [text_b])
        return EvalScore(
            score=round(score, 4), passing=score >= self.threshold,
            reasoning=f"similarity {score:.3f} vs threshold {self.threshold}", method=method,
        )


class CorrectnessEvaluator:
    """LLM-as-judge: is RESPONSE correct/faithful vs REFERENCE for the QUERY?"""

    def __init__(self, model: str = _DEFAULT_JUDGE, base_url: str | None = None, threshold: float = 0.6) -> None:
        self.model, self.base_url, self.threshold = model, base_url, threshold

    def evaluate(self, query: str, response: str, reference: str) -> EvalScore:
        system = (
            "You are a strict evaluator. Score from 1 (wrong) to 5 (perfect) how well the "
            "RESPONSE matches the REFERENCE answer for the QUERY. Give one sentence of "
            "reasoning, then a final line 'SCORE: N'."
        )
        user = f"QUERY: {query}\nREFERENCE: {reference}\nRESPONSE: {response}"
        score, reasoning = _judge(self.model, system, user, base_url=self.base_url)
        return EvalScore(score=score, passing=score >= self.threshold, reasoning=reasoning,
                         method=f"llm-judge:{self.model}")


class PairwiseEvaluator:
    """Which of two responses better matches the persona/reference? B (after) vs A (before)."""

    def __init__(self, model: str = _DEFAULT_JUDGE, base_url: str | None = None) -> None:
        self.model, self.base_url = model, base_url

    def evaluate(self, query: str, response_a: str, response_b: str, *, reference: str = "") -> EvalScore:
        ref = f" The target voice/reference is: {reference}." if reference else ""
        system = (
            "You compare two assistant responses (A and B) to a query." + ref +
            " Decide which better matches the target voice. Give one sentence, then a final "
            "line with exactly 'A', 'B', or 'TIE'."
        )
        user = f"QUERY: {query}\nA: {response_a}\nB: {response_b}"
        from mindxtrain.governance.panel import chat_once

        try:
            text = chat_once(
                self.model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                base_url=self.base_url, max_tokens=200, temperature=0.0,
            )
        except Exception as exc:
            return EvalScore(score=0.5, passing=False, reasoning=f"judge error: {exc}",
                             method=f"llm-judge:{self.model}")
        # Read the LAST A/B/TIE token (the verdict line).
        choices = _CHOICE_RE.findall(text)
        verdict = (choices[-1].upper() if choices else "TIE")
        score = {"B": 1.0, "TIE": 0.5, "A": 0.0}[verdict]
        reasoning = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())[:240]
        return EvalScore(score=score, passing=verdict == "B", reasoning=f"verdict {verdict}: {reasoning}",
                         method=f"llm-judge:{self.model}")


class GuidelineEvaluator:
    """LLM-as-judge: does RESPONSE comply with the GUIDELINES/rubric?"""

    def __init__(self, model: str = _DEFAULT_JUDGE, base_url: str | None = None, threshold: float = 0.6) -> None:
        self.model, self.base_url, self.threshold = model, base_url, threshold

    def evaluate(self, response: str, guidelines: str) -> EvalScore:
        system = (
            "You check compliance with guidelines. Score from 1 (violates) to 5 (fully "
            "complies) how well the RESPONSE follows the GUIDELINES. One sentence, then "
            "'SCORE: N'."
        )
        user = f"GUIDELINES: {guidelines}\nRESPONSE: {response}"
        score, reasoning = _judge(self.model, system, user, base_url=self.base_url)
        return EvalScore(score=score, passing=score >= self.threshold, reasoning=reasoning,
                         method=f"llm-judge:{self.model}")


__all__ = [
    "CorrectnessEvaluator",
    "EvalScore",
    "GuidelineEvaluator",
    "PairwiseEvaluator",
    "SemanticSimilarityEvaluator",
]
