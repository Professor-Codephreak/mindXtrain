"""Filecoin-pinned IPFS via Lighthouse Storage.

Direct httpx calls to the Lighthouse REST API — no SDK dependency. Returns a
content-addressed `cid://...` URI suitable for the provenance manifest's
`lighthouse_cid` field.

Falls back to a deterministic local-hash CID stub when `LIGHTHOUSE_API_KEY`
is unset (so dev/CI runs don't need credentials).
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import httpx

from mindxtrain.provenance.hashing import blake3_dir
from mindxtrain.storage.provider import StorageProvider, StorageRef


def _stub_cid(checkpoint_dir: Path) -> str:
    digest = blake3_dir(checkpoint_dir)
    return f"cid://stub-{digest[:32]}"


def publish_to_lighthouse(
    checkpoint_dir: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 600.0,
) -> str:
    """Tar + pin `checkpoint_dir` to Lighthouse; return a `cid://...` URI.

    If `LIGHTHOUSE_API_KEY` is unset we return a deterministic stub CID
    derived from BLAKE3, so dev runs still produce a valid manifest.
    """
    api_key = api_key or os.environ.get("LIGHTHOUSE_API_KEY", "")
    if not api_key:
        return _stub_cid(checkpoint_dir)
    base_url = (base_url or os.environ.get("LIGHTHOUSE_BASE_URL", "https://node.lighthouse.storage")).rstrip("/")

    # Tar the directory to a temp file so we upload one binary blob.
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = Path(tmp.name)
    try:
        with tarfile.open(tar_path, "w") as tf:
            tf.add(str(checkpoint_dir), arcname=checkpoint_dir.name)
        with tar_path.open("rb") as fh, httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{base_url}/api/v0/add",
                files={"file": (f"{checkpoint_dir.name}.tar", fh, "application/x-tar")},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
    finally:
        tar_path.unlink(missing_ok=True)

    cid = data.get("Hash") or data.get("cid")
    if not cid:
        msg = f"Lighthouse response missing CID: {data}"
        raise RuntimeError(msg)
    return f"cid://{cid}"


class LighthouseProvider(StorageProvider):
    """`StorageProvider` adapter over `publish_to_lighthouse`."""

    name = "lighthouse"

    def put_dir(self, src: Path, key: str) -> StorageRef:
        _ = key
        cid = publish_to_lighthouse(src)
        return StorageRef(provider=self.name, uri=cid)

    def get_dir(self, ref: StorageRef, dest: Path) -> Path:
        # Lighthouse fetch goes through any IPFS gateway; not implemented here.
        # The user can `ipfs get <cid>` from a kubo node or use the
        # `mindxtrain.storage.ipfs` provider for downloads.
        msg = "LighthouseProvider.get_dir — fetch via mindxtrain.storage.ipfs or kubo CLI."
        raise NotImplementedError(msg)


__all__ = ["LighthouseProvider", "publish_to_lighthouse"]
