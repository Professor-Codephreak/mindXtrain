"""Model-backed deliberation — back boardroom members + dojo judges with real LLMs.

`Boardroom.convene` and `Dojo.settle` take a ballot callable. This module provides ballots
that query a real model (any OpenAI-compatible backend — the same ollama / vLLM the operator
serves) so a boardroom literally deliberates and a dojo literally judges. Each member is
prompted from its role's stance and must end with `VERDICT: APPROVE|REJECT|ABSTAIN`.

Lazy + best-effort: a model that errors or returns no parseable verdict abstains (boardroom)
or is recorded as a reject (dojo, which forbids abstention). Pure stdlib + httpx (a base dep)
+ pydantic, so the module imports without `--extra ml`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

import httpx
from pydantic import BaseModel, ConfigDict

from mindxtrain.governance.boardroom import Member, Role, Vote
from mindxtrain.governance.dojo import JudgeVote

# Role → the stance the member argues from when deliberating.
ROLE_STANCE: dict[Role, str] = {
    "advocate": "Argue in favour of the motion; make the strongest case for approval.",
    "critic": "Argue against the motion; surface flaws, risks, and reasons to reject.",
    "analyst": "Weigh the evidence neutrally and reason to a balanced judgement.",
    "devils_advocate": "Challenge the prevailing view; stress-test the motion adversarially.",
    "expert": "Judge on technical merit and correctness.",
    "generalist": "Use plain common sense and practical judgement.",
}

_DEFAULT_MODEL = "llama3.2"
_VERDICT_RE = re.compile(r"verdict\s*[:\-]?\s*(approve|reject|abstain)", re.IGNORECASE)


def resolve_chat_base_url(base_url: str | None = None) -> str:
    """Resolve the OpenAI-compatible chat base URL the operator/backends use."""
    if base_url:
        return base_url.rstrip("/")
    for env in ("MINDXTRAIN_OPENAI_BASE_URL", "MINDXTRAIN_VLLM_BASE_URL", "MINDXTRAIN_OLLAMA_BASE_URL"):
        val = os.environ.get(env)
        if val:
            return val.rstrip("/")
    return "http://localhost:11434/v1"


def chat_once(
    model: str,
    messages: list[dict[str, str]],
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout_s: float = 60.0,
) -> str:
    """One non-streaming OpenAI-compatible chat completion; returns the content."""
    base = resolve_chat_base_url(base_url)
    key = api_key or os.environ.get("MINDXTRAIN_OPENAI_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(
            f"{base}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content", "") or ""


def parse_vote(text: str, *, allow_abstain: bool = True) -> Vote:
    """Parse a vote from model output: explicit `VERDICT:` line, else keyword scan."""
    m = _VERDICT_RE.search(text)
    if m:
        v = m.group(1).lower()
    else:
        low = text.lower()
        if "approve" in low and "reject" not in low:
            v = "approve"
        elif "reject" in low and "approve" not in low:
            v = "reject"
        else:
            v = "abstain"
    if v == "abstain" and not allow_abstain:
        return "reject"  # a dojo judge must decide; default the unclear to reject
    return v  # type: ignore[return-value]


class Deliberation(BaseModel):
    """One member's model-backed deliberation on a motion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    member_id: str
    role: Role
    vote: Vote
    rationale: str
    error: str = ""


def deliberate(
    member: Member,
    motion: str,
    *,
    base_url: str | None = None,
    allow_abstain: bool = True,
    default_model: str = _DEFAULT_MODEL,
    **chat_kw: object,
) -> Deliberation:
    """Query `member`'s model from its role stance and parse a vote + rationale."""
    stance = ROLE_STANCE.get(member.role, ROLE_STANCE["generalist"])
    verdicts = "APPROVE, REJECT, or ABSTAIN" if allow_abstain else "APPROVE or REJECT"
    system = (
        f"You are the {member.role.replace('_', ' ')} on a review board deliberating a "
        f"motion. {stance} Give a one-sentence rationale, then on the final line write "
        f"exactly 'VERDICT: <{verdicts}>'."
    )
    model = member.model or default_model
    try:
        text = chat_once(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": motion}],
            base_url=base_url,
            **chat_kw,  # type: ignore[arg-type]
        )
    except (httpx.HTTPError, OSError, ValueError) as exc:
        return Deliberation(
            member_id=member.id, role=member.role,
            vote="reject" if not allow_abstain else "abstain",
            rationale="", error=str(exc),
        )
    vote = parse_vote(text, allow_abstain=allow_abstain)
    rationale = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())[:280]
    return Deliberation(member_id=member.id, role=member.role, vote=vote, rationale=rationale)


def model_ballot(
    *, base_url: str | None = None, default_model: str = _DEFAULT_MODEL, **chat_kw: object,
) -> Callable[[Member, str], Vote]:
    """A boardroom ballot backed by real models (members may abstain)."""

    def _ballot(member: Member, motion: str) -> Vote:
        return deliberate(
            member, motion, base_url=base_url, allow_abstain=True,
            default_model=default_model, **chat_kw,
        ).vote

    return _ballot


def model_judge_ballot(
    *, base_url: str | None = None, default_model: str = _DEFAULT_MODEL, **chat_kw: object,
) -> Callable[[Member, str], JudgeVote]:
    """A dojo ballot backed by real models (judges must approve/reject)."""

    def _ballot(judge: Member, motion: str) -> JudgeVote:
        v = deliberate(
            judge, motion, base_url=base_url, allow_abstain=False,
            default_model=default_model, **chat_kw,
        ).vote
        return "approve" if v == "approve" else "reject"

    return _ballot


__all__ = [
    "ROLE_STANCE",
    "Deliberation",
    "chat_once",
    "deliberate",
    "model_ballot",
    "model_judge_ballot",
    "parse_vote",
    "resolve_chat_base_url",
]
