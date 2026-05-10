"""Raw IPFS `StorageProvider` via kubo HTTP API.

For users running their own kubo node (`ipfs daemon`) rather than going
through Lighthouse/Filecoin. Direct httpx — no SDK dependency.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import httpx

from mindxtrain.storage.provider import StorageProvider, StorageRef


class IpfsProvider(StorageProvider):
    name = "ipfs"

    def __init__(self, api_url: str | None = None, timeout_s: float = 600.0) -> None:
        self.api_url = (api_url or os.environ.get("IPFS_API_URL", "http://127.0.0.1:5001")).rstrip("/")
        self.timeout_s = timeout_s

    def put_dir(self, src: Path, key: str) -> StorageRef:
        """Tar + add the directory; return a `cid://<hash>` ref."""
        _ = key
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tar_path = Path(tmp.name)
        try:
            with tarfile.open(tar_path, "w") as tf:
                tf.add(str(src), arcname=src.name)
            with tar_path.open("rb") as fh, httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(
                    f"{self.api_url}/api/v0/add",
                    files={"file": (f"{src.name}.tar", fh, "application/x-tar")},
                )
                resp.raise_for_status()
                data = resp.json()
        finally:
            tar_path.unlink(missing_ok=True)
        cid = data.get("Hash") or data.get("cid")
        if not cid:
            msg = f"kubo /api/v0/add missing Hash: {data}"
            raise RuntimeError(msg)
        return StorageRef(provider=self.name, uri=f"cid://{cid}")

    def get_dir(self, ref: StorageRef, dest: Path) -> Path:
        """Pull the tar via /api/v0/cat and unpack into dest."""
        if ref.provider != self.name:
            msg = f"ref provider {ref.provider!r} != {self.name!r}"
            raise ValueError(msg)
        cid = ref.uri.removeprefix("cid://")
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tar_path = Path(tmp.name)
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream("POST", f"{self.api_url}/api/v0/cat?arg={cid}") as resp:
                    resp.raise_for_status()
                    with tar_path.open("wb") as fh:
                        for chunk in resp.iter_bytes():
                            fh.write(chunk)
            with tarfile.open(tar_path) as tf:
                tf.extractall(dest)
        finally:
            tar_path.unlink(missing_ok=True)
        return dest


__all__ = ["IpfsProvider"]
