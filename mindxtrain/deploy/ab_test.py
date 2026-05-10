"""A/B traffic split — canary vs live.

Pure-Python `Splitter`: deterministic per-request seed → canary or live based
on `cfg.canary_pct`. The actual traffic injection happens in the operator
FastAPI app's chat handler (which calls `Splitter.pick(req)` to decide which
slot's backend to route to).
"""

from __future__ import annotations

import hashlib
import random

from pydantic import BaseModel, ConfigDict, Field


class AbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canary_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    auto_rollback_threshold: float = Field(default=0.02, ge=0.0)


class Splitter:
    """Decides per-request which slot serves traffic."""

    def __init__(self, cfg: AbConfig | None = None) -> None:
        self.cfg = cfg or AbConfig()

    def pick(self, request_id: str | None = None) -> str:
        """Return `'canary'` or `'live'`.

        If `request_id` is provided we hash it (stable across retries); else
        we use random.random() (stateless).
        """
        if self.cfg.canary_pct <= 0.0:
            return "live"
        if request_id is None:
            r = random.random()
        else:
            digest = hashlib.blake2s(request_id.encode("utf-8"), digest_size=8).digest()
            r = int.from_bytes(digest, "big") / float(2**64)
        return "canary" if r < self.cfg.canary_pct else "live"


def serve_split(cfg: AbConfig) -> Splitter:
    """Construct the Splitter for an active A/B run."""
    return Splitter(cfg)


__all__ = ["AbConfig", "Splitter", "serve_split"]
