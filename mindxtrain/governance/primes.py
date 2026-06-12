"""Prime helpers — the dojo's panel size is always prime.

A dojo settles a dispute by majority of its judges. A *prime* panel size (which for
p > 2 is odd) guarantees no tie, so a dispute always resolves. These helpers size and
validate a dojo panel.
"""

from __future__ import annotations


def is_prime(n: int) -> bool:
    """True if `n` is a prime number (n < 2 is not prime)."""
    if n < 2:
        return False
    if n < 4:
        return True  # 2, 3
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def next_prime(n: int) -> int:
    """Smallest prime strictly greater than `n`."""
    candidate = max(n + 1, 2)
    while not is_prime(candidate):
        candidate += 1
    return candidate


def prev_prime(n: int) -> int | None:
    """Largest prime strictly less than `n`, or None if none exists."""
    candidate = n - 1
    while candidate >= 2:
        if is_prime(candidate):
            return candidate
        candidate -= 1
    return None


def nearest_prime(n: int) -> int:
    """The prime nearest to `n` (rounds up on a tie). Used to size a dojo.

    `nearest_prime(4) == 5` (3 and 5 are equidistant; round up). Never returns < 2.
    """
    if n < 2:
        return 2
    if is_prime(n):
        return n
    lo = prev_prime(n)
    hi = next_prime(n)
    if lo is None:
        return hi
    # Equidistant → prefer the larger prime (a bigger panel is more robust).
    return hi if (hi - n) <= (n - lo) else lo
