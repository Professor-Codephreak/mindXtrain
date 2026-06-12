"""Classroom — where an actor trains, and the graduation gate.

The classroom is the training environment (the `trl_local`/`trl_cpu` lanes + the imprint
measurement). An actor **graduates** when its imprint took — i.e. recall moved toward the
persona by at least a threshold. Graduation is the motion the boardroom then convenes on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from mindxtrain.eval.imprint import ImprintReport


class Graduation(BaseModel):
    """The classroom's verdict on whether an actor is ready to leave."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    graduated: bool
    imprint_delta: float
    min_delta: float
    reasons: list[str] = Field(default_factory=list)

    @property
    def motion(self) -> str:
        """The promotion motion the boardroom convenes on."""
        return f"promote graduated actor {self.run_id!r} (imprint Δ={self.imprint_delta:+.4f})"


def graduate(
    report: ImprintReport,
    *,
    run_id: str = "",
    min_delta: float = 0.0,
    require_imprinted: bool = True,
) -> Graduation:
    """Decide whether an imprinted actor graduates the classroom.

    Criteria: the imprint took (`report.imprinted`, when `require_imprinted`) and the
    recall gain meets `min_delta`. Failing reasons are recorded for the boardroom.
    """
    reasons: list[str] = []
    if require_imprinted and not report.imprinted:
        reasons.append("imprint did not take (no movement toward persona)")
    if report.imprint_delta < min_delta:
        reasons.append(f"imprint Δ {report.imprint_delta:+.4f} < required {min_delta:+.4f}")

    return Graduation(
        run_id=run_id,
        graduated=not reasons,
        imprint_delta=report.imprint_delta,
        min_delta=min_delta,
        reasons=reasons,
    )


class ClassroomReport(BaseModel):
    """Before-vs-after test of a trained actor against the previous model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inquiries: list[str]
    before: list[str]
    after: list[str]
    before_recall: float = Field(description="similarity of before-utterances to the persona voice")
    recall: float = Field(description="similarity of after-utterances to the persona voice")
    imprint_delta: float
    pairwise_after_better: float = Field(description="[0,1]; >0.5 = after closer to persona than before")
    persona_maintained: bool
    passed: bool
    notes: list[str] = Field(default_factory=list)


def evaluate_classroom(
    inquiries: list[str],
    before: list[str],
    after: list[str],
    baseline: list[str],
    *,
    use_judge: bool = False,
    model: str | None = None,
    base_url: str | None = None,
) -> ClassroomReport:
    """Run the classroom test: does the trained model recall/maintain the persona?

    Compares the previous (before) and trained (after) utterances against the persona
    `baseline` voice using the imprint metric + a semantic-similarity check; optionally
    a pairwise LLM judge (before vs after). `passed` iff the imprint took AND the
    after-utterances are at least as close to the persona as the before-utterances.
    """
    from mindxtrain.eval.imprint import score_imprint

    imprint = score_imprint(inquiries, before, after, baseline)
    notes: list[str] = [f"imprint method={imprint.method}"]

    # Pairwise: judge each inquiry (before vs after) toward the persona, else use the
    # imprint delta sign as the signal.
    if use_judge and model:
        from mindxtrain.eval.llama_evals import PairwiseEvaluator

        ev = PairwiseEvaluator(model=model, base_url=base_url)
        ref = " ".join(baseline)[:400]
        scores = [
            ev.evaluate(q, b, a, reference=ref).score
            for q, b, a in zip(inquiries, before, after, strict=False)
        ]
        pairwise = round(sum(scores) / len(scores), 4) if scores else 0.5
        notes.append(f"pairwise judge={model}")
    else:
        pairwise = 1.0 if imprint.after_voice > imprint.before_voice else (
            0.5 if imprint.after_voice == imprint.before_voice else 0.0)
        notes.append("pairwise from imprint delta")

    persona_maintained = imprint.imprinted and imprint.after_voice >= imprint.before_voice
    passed = persona_maintained and pairwise >= 0.5
    return ClassroomReport(
        inquiries=inquiries, before=before, after=after,
        before_recall=imprint.before_voice, recall=imprint.after_voice,
        imprint_delta=imprint.imprint_delta, pairwise_after_better=pairwise,
        persona_maintained=persona_maintained, passed=passed, notes=notes,
    )


__all__ = ["ClassroomReport", "Graduation", "evaluate_classroom", "graduate"]
