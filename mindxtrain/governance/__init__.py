"""Governance — classroom / boardroom / dojo.

A clean-room reimplementation (from the behaviour of github.com/openmindx/openmind —
Boardroom multi-model consensus + Dojo head-to-head evaluation) of the decision layer
that governs training:

- **Classroom** — where an actor trains; it *graduates* when imprint/quality criteria pass.
- **Boardroom** — a panel of **any number** of role-based members that convenes on a motion
  (e.g. "promote the graduated actor") and produces a consensus. Governs the classroom.
- **Dojo** — the boardroom's dispute-settlement extension. When the boardroom is split, a dojo
  of a **prime number** of judges settles it head-to-head; prime guarantees a strict majority,
  so a dispute always resolves with no tie.

Pure stdlib + pydantic; importable on a base install. Members vote via supplied callables /
scores, so the whole layer is testable without any LLM, and can later be backed by real models.
"""

from __future__ import annotations

from mindxtrain.governance.boardroom import (
    PRESET_BOARDS,
    ROLES,
    Boardroom,
    BoardroomDecision,
    Member,
    Vote,
)
from mindxtrain.governance.classroom import Graduation, graduate
from mindxtrain.governance.dojo import Dojo, DojoVerdict, settle_dispute
from mindxtrain.governance.primes import is_prime, nearest_prime, next_prime

__all__ = [
    "PRESET_BOARDS",
    "ROLES",
    "Boardroom",
    "BoardroomDecision",
    "Dojo",
    "DojoVerdict",
    "Graduation",
    "Member",
    "Vote",
    "graduate",
    "is_prime",
    "nearest_prime",
    "next_prime",
    "settle_dispute",
]
