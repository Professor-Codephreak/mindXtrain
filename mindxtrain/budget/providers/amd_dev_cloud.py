"""AMD Developer Cloud (DigitalOcean-hosted MI300X) — superseded.

The real implementation lives at `mindxtrain.deploy.amd_dev_cloud`. This
file is kept as a back-compat re-export so older budget-pricing callers
keep working. New code should import from `mindxtrain.deploy.amd_dev_cloud`
directly.
"""

from __future__ import annotations

from mindxtrain.deploy.amd_dev_cloud import (
    AmdDevCloudClient,
    AmdDevCloudConfig,
    AmdDevCloudError,
)


def provision_mi300x(gpu_count: int = 1, region: str = "atl1") -> dict[str, str]:
    """Back-compat entry point. Real provisioning lives in the operator.

    Use `mindxtrain droplet provision` (CLI) or the Coach UI's "Provision
    MI300X droplet" button instead.
    """
    msg = (
        "provision_mi300x has moved: use `mindxtrain droplet provision` or the "
        "Coach UI's deploy panel. Programmatic API: "
        "mindxtrain.deploy.amd_dev_cloud.AmdDevCloudClient.create()."
    )
    raise NotImplementedError(msg)


__all__ = [
    "AmdDevCloudClient",
    "AmdDevCloudConfig",
    "AmdDevCloudError",
    "provision_mi300x",
]
