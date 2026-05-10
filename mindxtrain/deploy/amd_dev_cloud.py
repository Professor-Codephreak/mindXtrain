"""AMD Developer Cloud REST client.

The schema follows DigitalOcean's `/v2/droplets` shape (the AMD Dev Cloud is
a DO-derived control plane). Documented at:
  https://docs.digitalocean.com/reference/api/reference/

We only need a thin slice: create, poll-until-active, list (filtered by
name), destroy. Auth is `Bearer <AMD_DEV_CLOUD_TOKEN>`.

Pure httpx — no SDK, no new dependencies.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_BASE = "https://api.devcloud.amd.com"

# Sane non-zero defaults so a misconfigured droplet doesn't sit in a poll
# loop forever. 20 minutes covers a cold cloud-init bootstrap with apt-get
# update + container pull on a slow link.
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_POLL_TIMEOUT = 1200.0


class AmdDevCloudError(RuntimeError):
    """Generic Dev Cloud REST error."""


class AmdDevCloudAuthError(AmdDevCloudError):
    """401/403 from the Dev Cloud control plane."""


@dataclass(frozen=True)
class AmdDevCloudConfig:
    token: str
    api_base: str = DEFAULT_BASE
    region: str = "atl1"
    size: str = "gpu-mi300x8-1536gb-devcloud"
    image: str = "vllm-0-17-1"
    ssh_key_id: int = 0
    tags: tuple[str, ...] = ("mindx", "train")


def required_env() -> tuple[str, ...]:
    return ("AMD_DEV_CLOUD_TOKEN", "AMD_DEV_CLOUD_SSH_KEY_ID")


def missing_env(env: dict[str, str] | None = None) -> list[str]:
    src = env if env is not None else os.environ
    out = [k for k in required_env() if not src.get(k)]
    return out


def status_target(env: dict[str, str] | None = None) -> str:
    src = env if env is not None else os.environ
    region = src.get("AMD_DEV_CLOUD_REGION", "atl1")
    size = src.get("AMD_DEV_CLOUD_SIZE", "gpu-mi300x8-1536gb-devcloud")
    return f"amd-dev-cloud:{region}:{size}"


def from_env(env: dict[str, str] | None = None) -> AmdDevCloudConfig:
    src = env if env is not None else os.environ
    missing = missing_env(env)
    if missing:
        msg = f"AMD Dev Cloud config missing env: {', '.join(missing)}"
        raise RuntimeError(msg)
    tags_raw = src.get("AMD_DEV_CLOUD_TAGS", "mindx,train")
    tags = tuple(t.strip() for t in tags_raw.split(",") if t.strip())
    try:
        ssh_key_id = int(src["AMD_DEV_CLOUD_SSH_KEY_ID"])
    except (KeyError, ValueError) as exc:
        msg = f"AMD_DEV_CLOUD_SSH_KEY_ID must be an int (got {src.get('AMD_DEV_CLOUD_SSH_KEY_ID')!r})"
        raise RuntimeError(msg) from exc
    return AmdDevCloudConfig(
        token=src["AMD_DEV_CLOUD_TOKEN"],
        api_base=src.get("AMD_DEV_CLOUD_API_BASE", DEFAULT_BASE),
        region=src.get("AMD_DEV_CLOUD_REGION", "atl1"),
        size=src.get("AMD_DEV_CLOUD_SIZE", "gpu-mi300x8-1536gb-devcloud"),
        image=src.get("AMD_DEV_CLOUD_IMAGE", "vllm-0-17-1"),
        ssh_key_id=ssh_key_id,
        tags=tags,
    )


def build_create_payload(
    cfg: AmdDevCloudConfig,
    *,
    name: str = "mindxtrain",
    user_data: str = "",
) -> dict[str, Any]:
    """Mirror the JSON shape from the user-supplied curl example."""
    return {
        "name": name,
        "region": cfg.region,
        "size": cfg.size,
        "image": cfg.image,
        "ssh_keys": [cfg.ssh_key_id],
        "backups": False,
        "ipv6": True,
        "monitoring": True,
        "tags": list(cfg.tags),
        "user_data": user_data,
        "vpc_uuid": "",
    }


# ---- client --------------------------------------------------------------

LogFn = Callable[[str], None]


def _noop(_line: str) -> None:
    return None


class AmdDevCloudClient:
    """Thin httpx wrapper around the AMD Dev Cloud REST surface.

    Methods accept an optional `log` callback; when provided, each REST call
    emits a single human-readable line through it. The orchestrator passes a
    callback that publishes `LogEvent`s via `RunRegistry.publish_threadsafe`,
    so the user sees the provision pipeline progress in real time.
    """

    def __init__(
        self,
        cfg: AmdDevCloudConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.cfg = cfg
        self._owned = client is None
        self._client = client or httpx.Client(
            base_url=cfg.api_base,
            headers={"Authorization": f"Bearer {cfg.token}"},
            timeout=30.0,
        )

    def close(self) -> None:
        if self._owned:
            self._client.close()

    def __enter__(self) -> AmdDevCloudClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    # -- create -----------------------------------------------------------

    def create(
        self,
        *,
        name: str = "mindxtrain",
        user_data: str = "",
        log: LogFn = _noop,
    ) -> dict[str, Any]:
        body = build_create_payload(self.cfg, name=name, user_data=user_data)
        log(f"POST {self.cfg.api_base}/v2/droplets name={name} size={self.cfg.size}")
        r = self._client.post("/v2/droplets", json=body)
        if r.status_code in (401, 403):
            msg = f"AMD Dev Cloud auth failed: {r.status_code} {r.text[:200]}"
            raise AmdDevCloudAuthError(msg)
        if r.status_code >= 400:
            msg = f"create droplet failed: {r.status_code} {r.text[:500]}"
            raise AmdDevCloudError(msg)
        out = r.json()
        droplet = out.get("droplet") or out
        log(f"  → 202 droplet_id={droplet.get('id')} status={droplet.get('status')}")
        return droplet

    # -- poll -------------------------------------------------------------

    def get(self, droplet_id: int, *, log: LogFn = _noop) -> dict[str, Any]:
        r = self._client.get(f"/v2/droplets/{droplet_id}")
        if r.status_code in (401, 403):
            msg = f"AMD Dev Cloud auth failed: {r.status_code}"
            raise AmdDevCloudAuthError(msg)
        if r.status_code >= 400:
            msg = f"get droplet {droplet_id} failed: {r.status_code} {r.text[:500]}"
            raise AmdDevCloudError(msg)
        out = r.json()
        droplet = out.get("droplet") or out
        log(f"  status={droplet.get('status')} ip={_extract_public_ip(droplet) or '-'}")
        return droplet

    def poll_until_active(
        self,
        droplet_id: int,
        *,
        timeout: float = DEFAULT_POLL_TIMEOUT,
        interval: float = DEFAULT_POLL_INTERVAL,
        log: LogFn = _noop,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> dict[str, Any]:
        """Poll until status='active' or timeout.

        `sleep` and `now` are injected for tests.
        """
        deadline = now() + timeout
        log(f"polling droplet {droplet_id} until active (timeout={int(timeout)}s)")
        while True:
            droplet = self.get(droplet_id, log=log)
            status = str(droplet.get("status", ""))
            if status == "active":
                ip = _extract_public_ip(droplet)
                log(f"  → active, public_ip={ip}")
                return droplet
            if status in ("errored", "off", "archive"):
                msg = f"droplet {droplet_id} reached terminal state {status!r} before active"
                raise AmdDevCloudError(msg)
            if now() >= deadline:
                msg = f"droplet {droplet_id} did not reach 'active' within {int(timeout)}s (last={status})"
                raise TimeoutError(msg)
            sleep(interval)

    # -- list / destroy ---------------------------------------------------

    def list(self, *, name: str | None = None) -> list[dict[str, Any]]:
        r = self._client.get("/v2/droplets")
        if r.status_code in (401, 403):
            msg = f"AMD Dev Cloud auth failed: {r.status_code}"
            raise AmdDevCloudAuthError(msg)
        if r.status_code >= 400:
            msg = f"list droplets failed: {r.status_code} {r.text[:500]}"
            raise AmdDevCloudError(msg)
        out = r.json().get("droplets", [])
        if name is None:
            return out
        return [d for d in out if d.get("name") == name]

    def destroy(self, droplet_id: int, *, log: LogFn = _noop) -> None:
        log(f"DELETE /v2/droplets/{droplet_id}")
        r = self._client.delete(f"/v2/droplets/{droplet_id}")
        if r.status_code in (401, 403):
            msg = f"AMD Dev Cloud auth failed: {r.status_code}"
            raise AmdDevCloudAuthError(msg)
        if r.status_code >= 400 and r.status_code != 404:
            msg = f"destroy droplet {droplet_id} failed: {r.status_code} {r.text[:500]}"
            raise AmdDevCloudError(msg)
        log(f"  → {r.status_code}")


def extract_public_ip(droplet: dict[str, Any]) -> str | None:
    nets = droplet.get("networks") or {}
    v4 = nets.get("v4") or []
    for net in v4:
        if str(net.get("type", "")).lower() == "public":
            return str(net.get("ip_address", ""))
    if v4:
        return str(v4[0].get("ip_address", ""))
    return None


# Backwards-compatible private alias for callers that haven't migrated.
_extract_public_ip = extract_public_ip


__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_POLL_TIMEOUT",
    "AmdDevCloudAuthError",
    "AmdDevCloudClient",
    "AmdDevCloudConfig",
    "AmdDevCloudError",
    "build_create_payload",
    "extract_public_ip",
    "from_env",
    "missing_env",
    "required_env",
    "status_target",
]
