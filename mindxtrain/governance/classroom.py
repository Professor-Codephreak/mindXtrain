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


__all__ = ["Graduation", "graduate"]
