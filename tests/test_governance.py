"""Governance: classroom (graduate) → boardroom (any-N) → dojo (prime-N) settlement."""

from __future__ import annotations

import pytest

from mindxtrain.governance import (
    Boardroom,
    Dojo,
    Member,
    graduate,
    is_prime,
    nearest_prime,
    next_prime,
    settle_dispute,
)
from mindxtrain.governance.boardroom import board_from_preset
from mindxtrain.governance.dojo import prime_dojo_size

# ---- primes -----------------------------------------------------------------

@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, False), (1, False), (2, True), (3, True), (4, False), (5, True),
     (9, False), (11, True), (15, False), (17, True), (97, True), (100, False)],
)
def test_is_prime(n, expected):
    assert is_prime(n) is expected


def test_next_prime():
    assert next_prime(2) == 3
    assert next_prime(8) == 11
    assert next_prime(13) == 17


def test_nearest_prime_rounds_up_on_tie():
    assert nearest_prime(4) == 5   # 3 and 5 equidistant → up
    assert nearest_prime(3) == 3
    assert nearest_prime(1) == 2


def test_prime_dojo_size_is_odd_prime_ge3():
    for req in range(0, 12):
        n = prime_dojo_size(req)
        assert n >= 3 and is_prime(n)
    assert prime_dojo_size(2) == 3   # 2 excluded (even, can tie)
    assert prime_dojo_size(4) == 5


# ---- boardroom (any N) ------------------------------------------------------

def _board(n):
    return Boardroom(members=[Member(id=f"m{i}", role="generalist") for i in range(n)])


def test_boardroom_any_number_of_members():
    for n in (1, 2, 4, 6, 10):  # any N, not just prime
        b = _board(n)
        assert len(b.members) == n


def test_boardroom_approves_on_majority():
    b = _board(4)
    ballot = {"m0": "approve", "m1": "approve", "m2": "approve", "m3": "reject"}
    d = b.convene("promote actor x", ballot)
    assert d.outcome == "approved"
    assert d.disputed is False
    assert d.approvals == 3 and d.rejections == 1


def test_boardroom_tie_is_disputed():
    b = _board(4)
    ballot = {"m0": "approve", "m1": "approve", "m2": "reject", "m3": "reject"}
    d = b.convene("promote actor x", ballot)
    assert d.outcome == "disputed"
    assert d.disputed is True


def test_boardroom_no_quorum_is_disputed():
    b = Boardroom(members=[Member(id=f"m{i}") for i in range(4)], quorum=0.75)
    # Only 2 decisive votes of 4; quorum needs ceil(0.75*4)=3.
    ballot = {"m0": "approve", "m1": "reject"}  # m2, m3 abstain
    d = b.convene("x", ballot)
    assert d.disputed is True
    assert "no quorum" in d.rationale


def test_boardroom_callable_ballot_and_preset():
    members = board_from_preset("classic_triad")
    b = Boardroom(members=members)
    # advocate approves, others reject → rejected.
    d = b.convene("x", lambda m, _motion: "approve" if m.role == "advocate" else "reject")
    assert d.outcome == "rejected"
    assert len(b.members) == 3


# ---- dojo (prime N) ---------------------------------------------------------

def test_dojo_rejects_non_prime_panel():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Dojo(judges=[Member(id=f"j{i}") for i in range(4)])  # 4 not prime
    with pytest.raises(ValidationError):
        Dojo(judges=[Member(id="j0"), Member(id="j1")])      # 2 excluded (even)


def test_dojo_accepts_odd_prime_panel():
    d = Dojo(judges=[Member(id=f"j{i}") for i in range(5)])
    assert len(d.judges) == 5


def test_dojo_sized_rounds_to_prime():
    d = Dojo.sized(4)
    assert len(d.judges) == 5 and is_prime(len(d.judges))


def test_dojo_settles_without_tie():
    d = Dojo.sized(3)
    verdict = d.settle("promote actor x",
                       {"judge-0": "approve", "judge-1": "approve", "judge-2": "reject"})
    assert verdict.settled is True
    assert verdict.winner == "approve"
    assert verdict.approvals + verdict.rejections == 3  # no abstentions


def test_dojo_requires_every_judge_to_vote():
    d = Dojo.sized(3)
    with pytest.raises(ValueError, match="did not vote"):
        d.settle("x", {"judge-0": "approve"})


# ---- end-to-end flow: classroom → boardroom → dojo --------------------------

class _Report:
    """Minimal ImprintReport stand-in for the graduation gate."""
    imprinted = True
    imprint_delta = 0.04


def test_graduation_then_disputed_board_settled_by_dojo():
    grad = graduate(_Report(), run_id="actor-1", min_delta=0.0)
    assert grad.graduated is True
    assert "actor-1" in grad.motion

    board = _board(4)
    tie = {"m0": "approve", "m1": "approve", "m2": "reject", "m3": "reject"}
    decision = board.convene(grad.motion, tie)
    assert decision.disputed is True

    dojo = Dojo.sized(len(board.members))  # 4 → 5 judges (odd prime)
    verdict = settle_dispute(
        decision, dojo,
        {"judge-0": "approve", "judge-1": "approve", "judge-2": "approve",
         "judge-3": "reject", "judge-4": "reject"},
    )
    assert verdict.settled is True
    assert verdict.winner == "approve"  # 3-2, no tie possible


def test_settle_dispute_refuses_undisputed_decision():
    board = _board(3)
    decision = board.convene("x", {"m0": "approve", "m1": "approve", "m2": "reject"})
    assert decision.disputed is False
    with pytest.raises(ValueError, match="not disputed"):
        settle_dispute(decision, Dojo.sized(3), {})


def test_graduation_fails_when_not_imprinted():
    class _Weak:
        imprinted = False
        imprint_delta = 0.0
    grad = graduate(_Weak(), run_id="actor-2")
    assert grad.graduated is False
    assert grad.reasons
