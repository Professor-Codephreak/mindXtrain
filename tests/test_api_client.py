"""Tests for `mindxtrain.deploy.api_client` — particularly the new
`swap_mindx_fallback_model` which closes the publish-side of the loop with
the mindX PATCH /v1/config/fallback-model endpoint shipped earlier.

We mock httpx with a MockTransport so the tests run offline and assert the
exact request shape (URL, method, body, bearer header) without any network.
"""
from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from mindxtrain.deploy import api_client
from mindxtrain.deploy.api_client import (
    MindXFallbackSwap,
    swap_mindx_fallback_model,
)


def _capturing_transport(*, recorder: dict, response_body: dict, status: int = 200):
    """Build a httpx MockTransport that records the request and returns `response_body`."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorder["method"] = request.method
        recorder["url"] = str(request.url)
        recorder["headers"] = dict(request.headers)
        recorder["body"] = json.loads(request.content.decode("utf-8")) if request.content else None
        return httpx.Response(status, json=response_body)

    return httpx.MockTransport(handler)


def test_swap_payload_validates():
    """MindXFallbackSwap is `extra=forbid` and rejects an empty model name."""
    with pytest.raises(ValidationError):
        MindXFallbackSwap(model="")  # min_length=1 violation
    p = MindXFallbackSwap(model="pythai/mindx-fallback-qwen3-1.5b")
    assert p.provider == "vllm"
    assert p.model == "pythai/mindx-fallback-qwen3-1.5b"


def test_swap_sends_patch_with_correct_body(monkeypatch):
    recorder: dict = {}
    transport = _capturing_transport(
        recorder=recorder,
        response_body={
            "success": True,
            "provider": "vllm",
            "previous": "Qwen/Qwen3-0.6B",
            "current": "pythai/mindx-fallback-qwen3-1.5b",
            "config_file": "models/vllm.yaml",
        },
    )

    # Patch httpx.Client so the function picks up our transport.
    orig_client = httpx.Client

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _client)
    monkeypatch.setenv("MINDXTRAIN_API_BASE_URL", "https://mindx.pythai.net")
    monkeypatch.delenv("MINDXTRAIN_API_KEY", raising=False)

    out = swap_mindx_fallback_model(
        provider="vllm",
        model="pythai/mindx-fallback-qwen3-1.5b",
    )

    assert recorder["method"] == "PATCH"
    assert recorder["url"] == "https://mindx.pythai.net/v1/config/fallback-model"
    assert recorder["body"] == {"provider": "vllm", "model": "pythai/mindx-fallback-qwen3-1.5b"}
    # No bearer header when MINDXTRAIN_API_KEY is unset.
    assert "authorization" not in {k.lower() for k in recorder["headers"]}
    assert out["current"] == "pythai/mindx-fallback-qwen3-1.5b"


def test_swap_sends_bearer_when_key_set(monkeypatch):
    recorder: dict = {}
    transport = _capturing_transport(
        recorder=recorder,
        response_body={"success": True, "previous": "x", "current": "y"},
    )
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    monkeypatch.setenv("MINDXTRAIN_API_KEY", "secret-token")

    swap_mindx_fallback_model(provider="vllm", model="x/y")

    assert recorder["headers"].get("authorization") == "Bearer secret-token"


def test_swap_honours_api_url_override(monkeypatch):
    recorder: dict = {}
    transport = _capturing_transport(
        recorder=recorder,
        response_body={"success": True, "previous": "a", "current": "b"},
    )
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    monkeypatch.delenv("MINDXTRAIN_API_KEY", raising=False)

    swap_mindx_fallback_model(
        provider="vllm",
        model="x/y",
        api_url="http://localhost:8080/",  # trailing slash should be stripped
    )

    assert recorder["url"] == "http://localhost:8080/v1/config/fallback-model"


def test_swap_raises_on_http_error(monkeypatch):
    recorder: dict = {}
    transport = _capturing_transport(
        recorder=recorder,
        response_body={"detail": "unknown model"},
        status=422,
    )
    orig_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig_client(*a, **{**kw, "transport": transport}),
    )
    monkeypatch.delenv("MINDXTRAIN_API_KEY", raising=False)

    with pytest.raises(httpx.HTTPStatusError):
        swap_mindx_fallback_model(provider="vllm", model="bogus/model")


def test_swap_exposed_in_module_all():
    """Function and payload model must be in __all__ so `from … import *` users see them."""
    assert "swap_mindx_fallback_model" in api_client.__all__
    assert "MindXFallbackSwap" in api_client.__all__
