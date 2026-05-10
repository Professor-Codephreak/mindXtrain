"""AMD Dev Cloud REST client — stubs the httpx transport, no network.

`httpx.MockTransport` is built into httpx (no `respx` dep needed).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mindxtrain.deploy.amd_dev_cloud import (
    DEFAULT_BASE,
    AmdDevCloudAuthError,
    AmdDevCloudClient,
    AmdDevCloudConfig,
    AmdDevCloudError,
    build_create_payload,
    extract_public_ip,
    from_env,
    missing_env,
)


def _cfg(**ov: Any) -> AmdDevCloudConfig:
    base = {"token": "dop_v1_TEST", "ssh_key_id": 56216059}
    base.update(ov)
    return AmdDevCloudConfig(**base)  # type: ignore[arg-type]


def _client_with_handler(handler, *, cfg: AmdDevCloudConfig | None = None) -> AmdDevCloudClient:
    cfg = cfg or _cfg()
    transport = httpx.MockTransport(handler)
    return AmdDevCloudClient(
        cfg,
        client=httpx.Client(
            base_url=cfg.api_base,
            headers={"Authorization": f"Bearer {cfg.token}"},
            transport=transport,
        ),
    )


def test_missing_env_lists_required() -> None:
    miss = missing_env({})
    assert "AMD_DEV_CLOUD_TOKEN" in miss
    assert "AMD_DEV_CLOUD_SSH_KEY_ID" in miss
    full = missing_env({"AMD_DEV_CLOUD_TOKEN": "x", "AMD_DEV_CLOUD_SSH_KEY_ID": "1"})
    assert full == []


def test_from_env_parses_int_ssh_key_id() -> None:
    cfg = from_env({
        "AMD_DEV_CLOUD_TOKEN": "x",
        "AMD_DEV_CLOUD_SSH_KEY_ID": "56216059",
        "AMD_DEV_CLOUD_TAGS": "a, b ,c",
    })
    assert cfg.ssh_key_id == 56216059
    assert cfg.tags == ("a", "b", "c")
    assert cfg.api_base == DEFAULT_BASE


def test_from_env_rejects_non_integer_ssh_key_id() -> None:
    with pytest.raises(RuntimeError, match="must be an int"):
        from_env({"AMD_DEV_CLOUD_TOKEN": "x", "AMD_DEV_CLOUD_SSH_KEY_ID": "notanint"})


def test_build_create_payload_matches_curl_example() -> None:
    """Pin the exact JSON body the user supplied in the spec."""
    cfg = _cfg(
        region="atl1",
        size="gpu-mi300x8-1536gb-devcloud",
        image="vllm-0-17-1",
        ssh_key_id=56216059,
        tags=("mindx", "train", "aglm", "agenticplace", "pythai"),
    )
    body = build_create_payload(cfg, name="mindxtrain", user_data="")
    # Field-by-field match to the curl example to lock the contract.
    assert body == {
        "name": "mindxtrain",
        "region": "atl1",
        "size": "gpu-mi300x8-1536gb-devcloud",
        "image": "vllm-0-17-1",
        "ssh_keys": [56216059],
        "backups": False,
        "ipv6": True,
        "monitoring": True,
        "tags": ["mindx", "train", "aglm", "agenticplace", "pythai"],
        "user_data": "",
        "vpc_uuid": "",
    }


def test_create_returns_droplet_dict() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/droplets"
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(202, json={"droplet": {"id": 12345, "status": "new", "name": "mindxtrain"}})

    with _client_with_handler(handler) as c:
        droplet = c.create(name="mindxtrain", user_data="#cloud-config\n")
    assert droplet["id"] == 12345
    assert captured["auth"] == "Bearer dop_v1_TEST"
    assert captured["body"]["name"] == "mindxtrain"
    assert captured["body"]["user_data"] == "#cloud-config\n"


def test_create_401_raises_auth_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad token"})

    with _client_with_handler(handler) as c:
        with pytest.raises(AmdDevCloudAuthError, match="auth failed"):
            c.create()


def test_create_500_raises_generic_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    with _client_with_handler(handler) as c:
        with pytest.raises(AmdDevCloudError, match="500"):
            c.create()


def test_poll_until_active_returns_when_status_active() -> None:
    statuses = iter(["new", "new", "active"])

    def handler(_request: httpx.Request) -> httpx.Response:
        s = next(statuses)
        body = {"droplet": {
            "id": 1, "status": s,
            "networks": {"v4": [{"type": "public", "ip_address": "1.2.3.4"}]},
        }}
        return httpx.Response(200, json=body)

    sleeps: list[float] = []
    with _client_with_handler(handler) as c:
        droplet = c.poll_until_active(1, sleep=sleeps.append, interval=5.0, timeout=600)
    assert droplet["status"] == "active"
    # Three GETs → two sleeps between them.
    assert sleeps == [5.0, 5.0]


def test_poll_until_active_raises_on_timeout() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"droplet": {"id": 1, "status": "new"}})

    # Fake a clock that's always past the deadline after one tick.
    times = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0])

    with _client_with_handler(handler) as c:
        with pytest.raises(TimeoutError, match="did not reach 'active'"):
            c.poll_until_active(1, timeout=10.0, interval=1.0, sleep=lambda _s: None, now=lambda: next(times))


def test_poll_until_active_raises_on_terminal_state() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"droplet": {"id": 1, "status": "errored"}})

    with _client_with_handler(handler) as c:
        with pytest.raises(AmdDevCloudError, match="terminal state"):
            c.poll_until_active(1, sleep=lambda _s: None, now=lambda: 0.0, timeout=60)


def test_extract_public_ip_handles_no_v4() -> None:
    assert extract_public_ip({"networks": {"v4": []}}) is None
    assert extract_public_ip({"networks": {}}) is None
    assert extract_public_ip({}) is None


def test_extract_public_ip_prefers_public_type() -> None:
    droplet = {"networks": {"v4": [
        {"type": "private", "ip_address": "10.0.0.1"},
        {"type": "public", "ip_address": "1.2.3.4"},
    ]}}
    assert extract_public_ip(droplet) == "1.2.3.4"


def test_list_filters_by_name() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"droplets": [
            {"id": 1, "name": "mindxtrain"},
            {"id": 2, "name": "other"},
        ]})

    with _client_with_handler(handler) as c:
        all_ = c.list()
        only = c.list(name="mindxtrain")
    assert len(all_) == 2
    assert len(only) == 1
    assert only[0]["id"] == 1
