# SPDX-License-Identifier: Apache-2.0
# (c) 2026 BANKON — all rights reserved.
"""Tests for the FastAPI server surface."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wordpress_agent.server import app


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WP_BASE_URL", "https://rage.example.test")
    monkeypatch.setenv("WP_USER", "codephreak")
    monkeypatch.setenv("WP_APP_PASSWORD", "test-pass-1234-5678")
    monkeypatch.setenv("WP_RETRY_COUNT", "0")
    monkeypatch.setenv("WP_RETRY_BACKOFF", "0")


def test_healthz_endpoint_responds(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://rage.example.test/wp-json/wp/v2/users/me",
        status_code=200,
        json={"id": 1, "name": "codephreak"},
    )
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["wp_user_id"] == 1


def test_publish_endpoint_validates_payload() -> None:
    with TestClient(app) as client:
        response = client.post("/publish", json={"title": "", "content": ""})
    assert response.status_code == 422


def test_publish_endpoint_success(httpx_mock) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://rage.example.test/wp-json/wp/v2/users/me",
        status_code=200,
        json={"id": 1, "name": "codephreak"},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://rage.example.test/wp-json/wp/v2/posts",
        status_code=201,
        json={
            "id": 42,
            "link": "https://rage.example.test/?p=42",
            "status": "publish",
            "slug": "hello",
            "date_gmt": "2026-05-09T22:00:00",
        },
    )
    with TestClient(app) as client:
        client.get("/healthz")  # warm lifespan
        response = client.post(
            "/publish",
            json={"title": "Hello", "content": "<p>World</p>"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["post_id"] == 42
    assert body["url"] == "https://rage.example.test/?p=42"
