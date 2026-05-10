"""Push checkpoint + model card to Hugging Face Hub.

Lazy `import huggingface_hub` so users without `--extra chain` can still
import this module. `HF_TOKEN` is read from env (or the token kwarg).
"""

from __future__ import annotations

import os
from pathlib import Path

from mindxtrain.storage.provider import StorageProvider, StorageRef


def publish_to_hf(
    checkpoint_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    create: bool = True,
) -> str:
    """Upload `checkpoint_dir` to `repo_id`. Return the canonical HF URL."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        msg = "huggingface_hub not installed; run `uv sync --extra chain`."
        raise RuntimeError(msg) from exc

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    if create:
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{repo_id}"


class HfHubProvider(StorageProvider):
    """`StorageProvider` adapter over `publish_to_hf` for canonical interop."""

    name = "hf_hub"

    def __init__(self, namespace: str | None = None, private: bool = False) -> None:
        self.namespace = namespace or os.environ.get("HF_HUB_USERNAME", "")
        self.private = private

    def put_dir(self, src: Path, key: str) -> StorageRef:
        repo_id = f"{self.namespace}/{key}" if self.namespace else key
        url = publish_to_hf(src, repo_id, private=self.private)
        return StorageRef(provider=self.name, uri=url)

    def get_dir(self, ref: StorageRef, dest: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            msg = "huggingface_hub not installed; run `uv sync --extra chain`."
            raise RuntimeError(msg) from exc
        # ref.uri is `https://huggingface.co/<repo_id>` — extract repo_id.
        repo_id = ref.uri.removeprefix("https://huggingface.co/")
        path = snapshot_download(repo_id=repo_id, local_dir=str(dest))
        return Path(path)


__all__ = ["HfHubProvider", "publish_to_hf"]
