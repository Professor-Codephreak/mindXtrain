"""Boardroom — any-N role-based consensus over a motion.

A boardroom is a panel of members, each in an advisor role, that convenes on a motion
(e.g. "promote the graduated actor") and casts a vote. The decision aggregates the votes
into approved / rejected / disputed. A boardroom can be **any number** of members; when it
is *disputed* (a tie, or below quorum) the dispute escalates to a prime-sized dojo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["advocate", "critic", "analyst", "devils_advocate", "expert", "generalist"]
Vote = Literal["approve", "reject", "abstain"]
Outcome = Literal["approved", "rejected", "disputed"]

ROLES: tuple[Role, ...] = (
    "advocate", "critic", "analyst", "devils_advocate", "expert", "generalist",
)

# Preset boards mirror OpenMind's named boards (reimplemented, not copied).
PRESET_BOARDS: dict[str, tuple[Role, ...]] = {
    "classic_triad": ("advocate", "critic", "analyst"),
    "devils_court": ("advocate", "devils_advocate", "critic", "analyst"),
    "full_board": ROLES,
    "peer_review": ("expert", "expert", "generalist"),
}


class Member(BaseModel):
    """One boardroom advisor. `model` optionally names the backing LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: Role = "generalist"
    model: str = ""


class BoardroomDecision(BaseModel):
    """Aggregated outcome of a convened motion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion: str
    members: list[str]
    votes: dict[str, Vote]
    approvals: int
    rejections: int
    abstentions: int
    outcome: Outcome
    rationale: str
    disputed: bool = Field(description="True when the boardroom could not settle (tie / no quorum)")


def board_from_preset(name: str, *, model: str = "") -> list[Member]:
    """Build a list of Members from a preset board name (roles get indexed ids)."""
    roles = PRESET_BOARDS.get(name)
    if roles is None:
        msg = f"unknown preset board {name!r}; available: {', '.join(sorted(PRESET_BOARDS))}"
        raise KeyError(msg)
    return [Member(id=f"{r}-{i}", role=r, model=model) for i, r in enumerate(roles)]


class Boardroom(BaseModel):
    """An any-N consensus panel governing the classroom."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    members: list[Member]
    quorum: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="min fraction of members that must cast a decisive (non-abstain) vote",
    )

    def convene(
        self,
        motion: str,
        ballot: dict[str, Vote] | Callable[[Member, str], Vote],
    ) -> BoardroomDecision:
        """Convene the board on a motion and tally the vote.

        `ballot` is either an explicit `{member_id: vote}` map or a callable
        `(member, motion) -> vote`. Missing members abstain.
        """
        if not self.members:
            msg = "a boardroom needs at least one member to convene"
            raise ValueError(msg)

        votes: dict[str, Vote] = {}
        for m in self.members:
            if callable(ballot):
                votes[m.id] = ballot(m, motion)
            else:
                votes[m.id] = ballot.get(m.id, "abstain")

        approvals = sum(1 for v in votes.values() if v == "approve")
        rejections = sum(1 for v in votes.values() if v == "reject")
        abstentions = sum(1 for v in votes.values() if v == "abstain")
        decisive = approvals + rejections
        quorum_needed = math.ceil(self.quorum * len(self.members))

        if decisive < quorum_needed:
            outcome: Outcome = "disputed"
            rationale = (
                f"no quorum: {decisive} decisive vote(s) < {quorum_needed} needed "
                f"({abstentions} abstained)"
            )
            disputed = True
        elif approvals > rejections:
            outcome = "approved"
            rationale = f"approved {approvals}-{rejections} ({abstentions} abstained)"
            disputed = False
        elif rejections > approvals:
            outcome = "rejected"
            rationale = f"rejected {rejections}-{approvals} ({abstentions} abstained)"
            disputed = False
        else:
            outcome = "disputed"
            rationale = f"tie {approvals}-{rejections}; escalate to the dojo to settle"
            disputed = True

        return BoardroomDecision(
            motion=motion,
            members=[m.id for m in self.members],
            votes=votes,
            approvals=approvals,
            rejections=rejections,
            abstentions=abstentions,
            outcome=outcome,
            rationale=rationale,
            disputed=disputed,
        )


__all__ = [
    "PRESET_BOARDS",
    "ROLES",
    "Boardroom",
    "BoardroomDecision",
    "Member",
    "Outcome",
    "Role",
    "Vote",
    "board_from_preset",
]
