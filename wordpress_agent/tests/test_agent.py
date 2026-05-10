# SPDX-License-Identifier: Apache-2.0
# (c) 2026 BANKON — all rights reserved.
"""Tests for WordpressAgent."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from wordpress_agent.agent import (
    AuthenticationError,
    PublishError,
    WordpressAgent,
)
from wordpress_agent.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        base_url="https://rage.example.test",
        user="codephreak",
        app_password="test-pass-1234-5678",
        retry_count=1,
        retry_backoff=0.0,
    )


@pytest.fixture
async def agent(settings: Settings):
    async with WordpressAgent(settings) as a:
        yield a


@pytest.mark.asyncio
async def test_publish_success(httpx_mock, agent: WordpressAgent) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        json={
            "id": 42,
            "link": "https://rage.example.test/?p=42",
            "status": "publish",
            "slug": "hello-world",
            "date_gmt": "2026-05-09T22:00:00",
        },
        status_code=201,
    )
    result = await agent.publish(title="Hello", content="<p>World</p>")
    assert result.post_id == 42
    assert result.url == "https://rage.example.test/?p=42"
    assert result.status == "publish"


@pytest.mark.asyncio
async def test_publish_scheduled_requires_tz_aware_date(agent: WordpressAgent) -> None:
    naive_date = datetime(2026, 6, 1, 9, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        await agent.publish(
            title="Scheduled",
            content="<p>Future post</p>",
            status="future",
            date=naive_date,
        )


@pytest.mark.asyncio
async def test_publish_empty_title_rejected(agent: WordpressAgent) -> None:
    with pytest.raises(ValueError, match="title"):
        await agent.publish(title="   ", content="<p>Body</p>")


@pytest.mark.asyncio
async def test_publish_empty_content_rejected(agent: WordpressAgent) -> None:
    with pytest.raises(ValueError, match="content"):
        await agent.publish(title="Title", content="")


@pytest.mark.asyncio
async def test_publish_authentication_failure(httpx_mock, agent: WordpressAgent) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=401,
        json={"code": "rest_cannot_create"},
    )
    with pytest.raises(AuthenticationError):
        await agent.publish(title="Hello", content="<p>World</p>")


@pytest.mark.asyncio
async def test_publish_retries_on_5xx(httpx_mock, agent: WordpressAgent) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=503,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=201,
        json={
            "id": 7,
            "link": "https://rage.example.test/?p=7",
            "status": "publish",
            "slug": "retry-success",
            "date_gmt": "2026-05-09T22:00:00",
        },
    )
    result = await agent.publish(title="Retry", content="<p>Body</p>")
    assert result.post_id == 7


@pytest.mark.asyncio
async def test_publish_gives_up_after_max_retries(httpx_mock, agent: WordpressAgent) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=503,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=503,
    )
    with pytest.raises(PublishError):
        await agent.publish(title="Down", content="<p>Body</p>")


@pytest.mark.asyncio
async def test_health_check_ok(httpx_mock, agent: WordpressAgent) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://rage.example.test/wp-json/wp/v2/users/me",
        status_code=200,
        json={"id": 1, "name": "codephreak"},
    )
    result = await agent.health_check()
    assert result["ok"] is True
    assert result["wp_user_id"] == 1


@pytest.mark.asyncio
async def test_publish_includes_scheduled_date_in_payload(
    httpx_mock, agent: WordpressAgent
) -> None:
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=201,
        json={
            "id": 99,
            "link": "https://rage.example.test/?p=99",
            "status": "future",
            "slug": "scheduled",
            "date_gmt": "2026-06-01T09:00:00",
        },
    )
    when = datetime(2026, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    await agent.publish(
        title="Scheduled",
        content="<p>Body</p>",
        status="future",
        date=when,
    )
    request = httpx_mock.get_request()
    assert request is not None
    body = request.read().decode()
    assert "date_gmt" in body
    assert "future" in body
