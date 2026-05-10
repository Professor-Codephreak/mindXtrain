"""Approval flow — gate destructive tool calls behind explicit confirmation.

Three transports:
    - CLITransport: prompts on stdin (sync wrapped in async).
    - WebTransport: registers a pending approval, served via the operator
      FastAPI app's /approval/{id} endpoint (caller must wire that route).
    - SlackTransport: posts an interactive message; replies via httpx-driven
      polling against a configured webhook.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    reason: str = ""


class ApprovalTransport(Protocol):
    async def request(self, req: ApprovalRequest) -> bool: ...


class CLITransport:
    """Read y/N from stdin via asyncio.to_thread."""

    name = "cli"

    async def request(self, req: ApprovalRequest) -> bool:
        prompt = (
            f"\n[approval] run={req.run_id} tool={req.tool_name} "
            f"reason={req.reason or '<none>'}\n"
            f"  arguments: {req.arguments}\n"
            f"  approve? [y/N] "
        )
        resp = await asyncio.to_thread(input, prompt)
        return resp.strip().lower() in {"y", "yes"}


class WebTransport:
    """In-memory pending-approvals dict; the operator FastAPI app polls/resolves."""

    name = "web"

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def request(self, req: ApprovalRequest) -> bool:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[req.request_id] = fut
        try:
            return await fut
        finally:
            self._pending.pop(req.request_id, None)

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Called from the FastAPI route handler when the user clicks approve/deny."""
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True

    def pending(self) -> list[str]:
        return list(self._pending)


class SlackTransport:
    """Post an interactive message; resolve via webhook POST back to us."""

    name = "slack"

    def __init__(self, webhook_url: str | None = None, timeout_s: float = 300.0) -> None:
        self.webhook_url = webhook_url or os.environ.get("MINDXTRAIN_SLACK_WEBHOOK", "")
        self.timeout_s = timeout_s
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def request(self, req: ApprovalRequest) -> bool:
        if not self.webhook_url:
            msg = "SlackTransport requires MINDXTRAIN_SLACK_WEBHOOK to be set"
            raise RuntimeError(msg)
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                self.webhook_url,
                json={
                    "text": f"approve {req.tool_name} on run {req.run_id}? ({req.reason})",
                    "request_id": req.request_id,
                },
            )
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[req.request_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=self.timeout_s)
        except TimeoutError:
            return False
        finally:
            self._pending.pop(req.request_id, None)

    def resolve(self, request_id: str, approved: bool) -> bool:
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True


def get_transport(name: Literal["cli", "web", "slack"] = "cli") -> ApprovalTransport:
    if name == "cli":
        return CLITransport()
    if name == "web":
        return WebTransport()
    if name == "slack":
        return SlackTransport()
    msg = f"unknown approval transport: {name}"
    raise ValueError(msg)


__all__ = [
    "ApprovalRequest",
    "ApprovalTransport",
    "CLITransport",
    "SlackTransport",
    "WebTransport",
    "get_transport",
]
