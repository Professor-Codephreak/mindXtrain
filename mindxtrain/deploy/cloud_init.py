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
TRAIN_DONE_SENTINEL = "/workspace/mindxtrain/.train-done"
TRAIN_EXIT_SENTINEL = "/workspace/mindxtrain/.train-exit"
TRAIN_LOG_GLOB = "/workspace/mindxtrain/out/runs/*/train.log"
CLOUD_INIT_LOG = "/var/log/cloud-init-output.log"

# Reject anything that could break out of the YAML or bash context. The repo
# slug, branch, and container image are interpolated into a `runcmd:` shell
# string — they must not contain quotes, backticks, $, ;, &, |, or whitespace.
_SAFE = re.compile(r"^[A-Za-z0-9_./:@\-]+$")
# `extras` (pip extras list) is the one field where commas are valid — it's
# a comma-separated list of extra-group names, each of which must itself be
# safe.
_SAFE_EXTRAS = re.compile(r"^[A-Za-z0-9_,\-]+$")
# `recipe` is a built-in YAML basename — no slashes, dots, or colons. Tight
# regex prevents path traversal (e.g. ../../etc/passwd) and shell tricks
# that the looser _SAFE pattern would let through.
_SAFE_RECIPE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _check(field: str, value: str) -> None:
    if field == "extras":
        pattern = _SAFE_EXTRAS
        allowed = "[A-Za-z0-9_,-]"
    elif field == "recipe":
        pattern = _SAFE_RECIPE
        allowed = "[A-Za-z0-9_-]"
    else:
        pattern = _SAFE
        allowed = "[A-Za-z0-9_./:@-]"
    if not value or not pattern.match(value):
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
    recipe: str | None = None,
) -> str:
    """Return a `#cloud-config` YAML payload ready for the `user_data` field.

    The script is idempotent on re-runs: cloud-init only fires `runcmd` on
    first boot, but if a step is re-run manually it short-circuits via the
    sentinel file.

    When `recipe` is provided, a `mindxtrain train` step runs after bench,
    writes its exit code to `{TRAIN_EXIT_SENTINEL}` and touches
    `{TRAIN_DONE_SENTINEL}` so the operator's SSH-tail bridge knows when to
    stop streaming. Output is tee'd to a stable train.log path globbed by
    the orchestrator (per-recipe run_name lives one directory deeper).
    """
    _check("repo", repo)
    _check("branch", branch)
    _check("container", container)
    _check("extras", extras)
    _check("remote_path", remote_path)
    if recipe is not None:
        _check("recipe", recipe)

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

    if recipe is not None:
        # The train step:
        # 1. Runs `mindxtrain train` inside the container against the recipe
        #    that ships in-tree.
        # 2. Tees output to {remote_path}/out/runs/<run_name>/train.log so
        #    the operator's SSH-tail can glob it.
        # 3. Captures the wrapped exit code and persists both sentinels
        #    atomically. Note the outer shell captures podman's exit, not
        #    the pipeline's, so a pipe-broken tee doesn't mask a train fail.
        train_step = (
            f"  - cd {remote_path} && podman run --rm "
            f"--device /dev/kfd --device /dev/dri "
            f"-v {remote_path}:{remote_path} -w {remote_path} "
            f'{container} bash -lc "mindxtrain train '
            f'mindxtrain/train/recipes/{recipe}.yaml --plan plan.json 2>&1 | '
            f'tee out/runs/_combined_train.log"; '
            f"echo $? > {TRAIN_EXIT_SENTINEL}; touch {TRAIN_DONE_SENTINEL}"
        )
    else:
        train_step = ""

    train_block = f"\n{train_step}" if train_step else ""

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
  - touch {BOOTSTRAP_SENTINEL}{train_block}

final_message: "mindXtrain bootstrap complete (sentinel: {BOOTSTRAP_SENTINEL})"
"""


__all__ = [
    "BOOTSTRAP_SENTINEL",
    "CLOUD_INIT_LOG",
    "TRAIN_DONE_SENTINEL",
    "TRAIN_EXIT_SENTINEL",
    "TRAIN_LOG_GLOB",
    "render",
]
