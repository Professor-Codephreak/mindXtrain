"""Dojo — prime-N dispute settlement for the boardroom.

When a boardroom is *disputed* (a tie or no quorum), a dojo settles it. A dojo's panel
size is **always a prime** — specifically an odd prime (>= 3), because an odd number of
decisive judges cannot tie, so the dispute always resolves. The dojo votes approve/reject
on the contested motion; the majority verdict is final.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from mindxtrain.governance.boardroom import BoardroomDecision, Member
from mindxtrain.governance.primes import is_prime, nearest_prime, next_prime

JudgeVote = Literal["approve", "reject"]


def prime_dojo_size(requested: int) -> int:
    """Round `requested` to a dispute-settling dojo size: the nearest odd prime >= 3.

    2 is prime but even (can tie), so it is excluded — the smallest dojo is 3.
    """
    base = max(3, requested)
    n = nearest_prime(base)
    while n < 3 or not is_prime(n):
        n = next_prime(n)
    return n


class DojoVerdict(BaseModel):
    """The dojo's final ruling on a disputed motion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    motion: str
    judges: list[str]
    votes: dict[str, JudgeVote]
    approvals: int
    rejections: int
    winner: JudgeVote
    settled: bool = True


class Dojo(BaseModel):
    """A prime panel of judges that settles a boardroom dispute."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    judges: list[Member]

    @field_validator("judges")
    @classmethod
    def _panel_is_odd_prime(cls, v: list[Member]) -> list[Member]:
        n = len(v)
        if not is_prime(n) or n < 3:
            msg = (
                f"a dojo panel must be an odd prime (>=3) so disputes never tie; "
                f"got {n}. Use Dojo.sized({n}) to round to the nearest valid size."
            )
            raise ValueError(msg)
        return v

    @classmethod
    def sized(cls, requested: int, *, model: str = "", role: str = "expert") -> Dojo:
        """Build a dojo whose panel is the nearest odd prime >= 3 to `requested`."""
        n = prime_dojo_size(requested)
        judges = [Member(id=f"judge-{i}", role="expert", model=model) for i in range(n)]
        _ = role  # judges are experts by construction; kept for call symmetry
        return cls(judges=judges)

    def settle(
        self,
        motion: str,
        ballot: dict[str, JudgeVote] | Callable[[Member, str], JudgeVote],
    ) -> DojoVerdict:
        """Settle a motion by majority of the prime judge panel.

        Every judge must cast approve/reject (no abstention in a dojo). With an odd
        prime panel the majority is strict — the verdict always settles.
        """
        votes: dict[str, JudgeVote] = {}
        for j in self.judges:
            if callable(ballot):
                votes[j.id] = ballot(j, motion)
            else:
                if j.id not in ballot:
                    msg = f"dojo judge {j.id!r} did not vote; every judge must rule"
                    raise ValueError(msg)
                votes[j.id] = ballot[j.id]

        approvals = sum(1 for v in votes.values() if v == "approve")
        rejections = sum(1 for v in votes.values() if v == "reject")
        winner: JudgeVote = "approve" if approvals > rejections else "reject"
        return DojoVerdict(
            motion=motion,
            judges=[j.id for j in self.judges],
            votes=votes,
            approvals=approvals,
            rejections=rejections,
            winner=winner,
            settled=True,
        )


def settle_dispute(
    decision: BoardroomDecision,
    dojo: Dojo,
    ballot: dict[str, JudgeVote] | Callable[[Member, str], JudgeVote],
) -> DojoVerdict:
    """Settle a disputed boardroom decision with a prime dojo.

    Raises if the decision was not actually disputed (the dojo only settles ties /
    no-quorum outcomes — a clean approve/reject stands on its own).
    """
    if not decision.disputed:
        msg = (
            f"boardroom decision on {decision.motion!r} was {decision.outcome!r}, "
            "not disputed — nothing for the dojo to settle"
        )
        raise ValueError(msg)
    return dojo.settle(decision.motion, ballot)


__all__ = [
    "Dojo",
    "DojoVerdict",
    "JudgeVote",
    "prime_dojo_size",
    "settle_dispute",
]
