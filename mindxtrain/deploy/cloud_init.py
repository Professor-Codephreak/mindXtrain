"""Cloud-init `user_data` generator for AMD Dev Cloud MI300X droplets.

The droplet boots, runs this script, then exits to the login prompt. By the
time SSH is reachable, the repo is cloned, the container image is pulled, and
`mindxtrain bench` has produced `plan.json` on disk.

The orchestrator polls for the sentinel file
`/workspace/mindxtrain/.bootstrap-done` over SSH; cloud-init logs land in
`/var/log/cloud-init-output.log`, which the orchestrator tails for live
feedback.

This module is a pure string builder. No execution, no I/O.
"""

from __future__ import annotations

import re

BOOTSTRAP_SENTINEL = "/workspace/mindxtrain/.bootstrap-done"
CLOUD_INIT_LOG = "/var/log/cloud-init-output.log"

# Reject anything that could break out of the YAML or bash context. The repo
# slug, branch, and container image are interpolated into a `runcmd:` shell
# string — they must not contain quotes, backticks, $, ;, &, |, or whitespace.
_SAFE = re.compile(r"^[A-Za-z0-9_./:@\-]+$")
# `extras` (pip extras list) is the one field where commas are valid — it's
# a comma-separated list of extra-group names, each of which must itself be
# safe.
_SAFE_EXTRAS = re.compile(r"^[A-Za-z0-9_,\-]+$")


def _check(field: str, value: str) -> None:
    pattern = _SAFE_EXTRAS if field == "extras" else _SAFE
    if not value or not pattern.match(value):
        allowed = "[A-Za-z0-9_,-]" if field == "extras" else "[A-Za-z0-9_./:@-]"
        msg = f"cloud-init: refusing unsafe {field}={value!r} (allowed: {allowed})"
        raise ValueError(msg)


def render(
    *,
    repo: str = "professor-codephreak/mindXtrain",
    branch: str = "main",
    container: str = "rocm/primus:v26.2",
    extras: str = "ml,eval,data,obs",
    remote_path: str = "/workspace/mindxtrain",
    run_bench: bool = True,
) -> str:
    """Return a `#cloud-config` YAML payload ready for the `user_data` field.

    The script is idempotent on re-runs: cloud-init only fires `runcmd` on
    first boot, but if a step is re-run manually it short-circuits via the
    sentinel file.
    """
    _check("repo", repo)
    _check("branch", branch)
    _check("container", container)
    _check("extras", extras)
    _check("remote_path", remote_path)

    bench_step = (
        f"  - cd {remote_path} && podman run --rm "
        f"--device /dev/kfd --device /dev/dri "
        f"-v {remote_path}:{remote_path} -w {remote_path} "
        f'{container} bash -lc "pip install -e .[{extras}] && '
        f'mindxtrain bench --gpu 0 --out plan.json"'
    ) if run_bench else (
        f"  - cd {remote_path} && podman run --rm "
        f"--device /dev/kfd --device /dev/dri "
        f"-v {remote_path}:{remote_path} -w {remote_path} "
        f'{container} bash -lc "pip install -e .[{extras}]"'
    )

    return f"""#cloud-config
package_update: true
package_upgrade: false
packages:
  - git
  - podman
  - podman-compose

write_files:
  - path: /etc/profile.d/mindxtrain.sh
    permissions: '0755'
    content: |
      export MINDXTRAIN_HOME={remote_path}

runcmd:
  - mkdir -p /workspace
  - test -d {remote_path}/.git || git clone --depth 1 --branch {branch} https://github.com/{repo}.git {remote_path}
  - cd {remote_path} && podman image exists {container} || podman pull {container}
{bench_step}
  - touch {BOOTSTRAP_SENTINEL}

final_message: "mindXtrain bootstrap complete (sentinel: {BOOTSTRAP_SENTINEL})"
"""


__all__ = ["BOOTSTRAP_SENTINEL", "CLOUD_INIT_LOG", "render"]
