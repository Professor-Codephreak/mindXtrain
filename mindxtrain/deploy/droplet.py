"""Existing-droplet sync builder (rsync + ssh + scp argv arrays).

Pure builder. The orchestrator runs each step via
`mindxtrain.operator.runs.spawn_subprocess_streaming` so every command's
stdout streams back over the SSE pipeline.

Argv is always list-form — never `shell=True` — so user-supplied env values
can't be turned into shell injection. The remote shell snippets are
single-string command bodies (because `ssh` itself joins them on the wire),
but the *arguments* to ssh are still argv elements.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from mindxtrain.deploy.github_push import Step

DEFAULT_REMOTE_PATH = "/workspace/mindxtrain"
DEFAULT_CONTAINER = "rocm/primus:v26.2"
DEFAULT_SSH_KEY = "~/.ssh/id_ed25519"

# Common ssh hardening flags applied everywhere:
#  - BatchMode=yes        : never prompt for a password (would deadlock)
#  - StrictHostKeyChecking=accept-new : trust on first use, refuse changes
#  - ServerAliveInterval=30 : keep the socket open during long bench runs
_SSH_OPTS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
)

# rsync exclude rules. .git intentionally excluded — the cloud-init path
# clones from GitHub; this rsync path syncs working-tree state for fast
# iteration on a droplet that was provisioned manually.
_DEFAULT_EXCLUDES = (
    ".git",
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "out/",
    "dist/",
    "build/",
    "*.egg-info",
)


@dataclass(frozen=True)
class DropletConfig:
    host: str
    user: str = "root"
    ssh_key: str = DEFAULT_SSH_KEY
    remote_path: str = DEFAULT_REMOTE_PATH
    container: str = DEFAULT_CONTAINER
    extras: str = "ml,eval,data,obs"


def required_env() -> tuple[str, ...]:
    """Env vars that must be set for /api/droplet/sync."""
    return ("DROPLET_HOST", "DROPLET_USER")


def missing_env(env: dict[str, str] | None = None) -> list[str]:
    src = env if env is not None else os.environ
    return [k for k in required_env() if not src.get(k)]


def _which_missing() -> list[str]:
    out = []
    for binary in ("rsync", "ssh", "scp"):
        if shutil.which(binary) is None:
            out.append(binary)
    return out


def status_missing(env: dict[str, str] | None = None) -> list[str]:
    return missing_env(env) + _which_missing()


def status_target(env: dict[str, str] | None = None) -> str:
    src = env if env is not None else os.environ
    user = src.get("DROPLET_USER", "")
    host = src.get("DROPLET_HOST", "")
    path = src.get("DROPLET_REMOTE_PATH", DEFAULT_REMOTE_PATH)
    if not (user and host):
        return ""
    return f"{user}@{host}:{path}"


def from_env(env: dict[str, str] | None = None) -> DropletConfig:
    src = env if env is not None else os.environ
    missing = missing_env(env)
    if missing:
        msg = f"droplet config missing env: {', '.join(missing)}"
        raise RuntimeError(msg)
    return DropletConfig(
        host=src["DROPLET_HOST"],
        user=src.get("DROPLET_USER", "root"),
        ssh_key=src.get("DROPLET_SSH_KEY", DEFAULT_SSH_KEY),
        remote_path=src.get("DROPLET_REMOTE_PATH", DEFAULT_REMOTE_PATH),
        container=src.get("DROPLET_CONTAINER", DEFAULT_CONTAINER),
    )


# ---- argv builders -------------------------------------------------------

def build_rsync(cfg: DropletConfig, repo_root: Path) -> list[str]:
    excludes: list[str] = []
    for pat in _DEFAULT_EXCLUDES:
        excludes += ["--exclude", pat]
    ssh_inline = (
        f"ssh -i {cfg.ssh_key} "
        f"-o BatchMode=yes "
        f"-o StrictHostKeyChecking=accept-new"
    )
    # Trailing slashes matter — copy contents of repo_root into remote_path/.
    return [
        "rsync", "-az", "--delete", "--info=progress2",
        *excludes,
        "-e", ssh_inline,
        f"{repo_root.rstrip('/') if isinstance(repo_root, str) else str(repo_root).rstrip('/')}/",
        f"{cfg.user}@{cfg.host}:{cfg.remote_path}/",
    ]


def build_provision_ssh(cfg: DropletConfig) -> list[str]:
    """Idempotent: install podman + pull the container image, skip if already done."""
    remote = (
        "command -v podman >/dev/null 2>&1 || "
        "(sudo apt-get update && sudo apt-get install -y podman); "
        f"podman image exists {cfg.container} || podman pull {cfg.container}"
    )
    return ["ssh", "-i", cfg.ssh_key, *_SSH_OPTS, f"{cfg.user}@{cfg.host}", remote]


def build_bench_ssh(cfg: DropletConfig) -> list[str]:
    """Run `mindxtrain bench --gpu 0 --out plan.json` inside the container.

    `-tt` forces a remote pty so SIGINT from the local ssh client propagates
    through the SSH channel to `podman` and on to the GPU process. Without
    `-tt`, the existing `_REGISTRY.cancel()` SIGINT/SIGTERM in
    `mindxtrain.operator.runs:331` reaches only the local ssh, leaving the
    remote GPU still spinning.
    """
    remote = (
        f"cd {cfg.remote_path} && podman run --rm "
        f"--device /dev/kfd --device /dev/dri "
        f"-v {cfg.remote_path}:{cfg.remote_path} -w {cfg.remote_path} "
        f"{cfg.container} bash -lc "
        f"'pip install -e .[{cfg.extras}] && "
        f"mindxtrain bench --gpu 0 --out plan.json'"
    )
    return ["ssh", "-tt", "-i", cfg.ssh_key, *_SSH_OPTS, f"{cfg.user}@{cfg.host}", remote]


def build_scp_plan_back(cfg: DropletConfig, local_dest: Path) -> list[str]:
    return [
        "scp", "-i", cfg.ssh_key, *_SSH_OPTS,
        f"{cfg.user}@{cfg.host}:{cfg.remote_path}/plan.json",
        str(local_dest),
    ]


def build_ssh_probe(cfg: DropletConfig) -> list[str]:
    """Quick liveness probe — `ssh user@host true`. Used to wait for the box
    to come up after provisioning."""
    return ["ssh", "-i", cfg.ssh_key, *_SSH_OPTS, "-o", "ConnectTimeout=5",
            f"{cfg.user}@{cfg.host}", "true"]


def build_tail_cloud_init(cfg: DropletConfig) -> list[str]:
    """Tail cloud-init's combined output until the bootstrap sentinel exists.

    Streams the log live; exits 0 once the sentinel appears. The remote shell
    here is a small loop, run via ssh's argv (not /bin/sh -c locally).
    """
    from mindxtrain.deploy.cloud_init import BOOTSTRAP_SENTINEL, CLOUD_INIT_LOG

    remote = (
        f"touch {CLOUD_INIT_LOG}; "
        f"tail -n +1 -F {CLOUD_INIT_LOG} & TAIL_PID=$!; "
        f"while [ ! -f {BOOTSTRAP_SENTINEL} ]; do sleep 5; done; "
        f"sleep 2; kill $TAIL_PID 2>/dev/null; true"
    )
    return ["ssh", "-i", cfg.ssh_key, *_SSH_OPTS, f"{cfg.user}@{cfg.host}", remote]


def build_tail_training_log(cfg: DropletConfig) -> list[str]:
    """Tail the droplet's `mindxtrain train` combined log until done sentinel.

    Streams every stdout line from the in-progress training run; exits with
    the captured training exit code once `.train-done` appears. Used by the
    orchestrator's post-bootstrap step to bridge remote training events into
    the operator's run registry (where Coach's SSE picks them up).

    The training command tees its output to a stable path
    (`out/runs/_combined_train.log`) so we don't have to guess the per-recipe
    run_name from the operator side.
    """
    from mindxtrain.deploy.cloud_init import (
        TRAIN_DONE_SENTINEL,
        TRAIN_EXIT_SENTINEL,
    )

    combined_log = f"{cfg.remote_path}/out/runs/_combined_train.log"
    remote = (
        # Make sure the log exists so tail -F doesn't error before training
        # writes its first line.
        f"mkdir -p {cfg.remote_path}/out/runs; "
        f"touch {combined_log}; "
        f"tail -n +1 -F {combined_log} & TAIL_PID=$!; "
        # Block until training is terminal.
        f"while [ ! -f {TRAIN_DONE_SENTINEL} ]; do sleep 5; done; "
        # Give tail one more pass to flush.
        f"sleep 2; kill $TAIL_PID 2>/dev/null; "
        # Propagate the training exit code so the orchestrator can decide
        # succeeded vs failed. Defaults to 1 if the sentinel is missing.
        f"exit $(cat {TRAIN_EXIT_SENTINEL} 2>/dev/null || echo 1)"
    )
    return ["ssh", "-i", cfg.ssh_key, *_SSH_OPTS, f"{cfg.user}@{cfg.host}", remote]


# ---- pipeline assembly ---------------------------------------------------

def sync_steps(
    cfg: DropletConfig,
    repo_root: Path,
    *,
    run_bench: bool = True,
    fetch_plan: bool = True,
    plan_dest: Path | None = None,
) -> list[Step]:
    steps: list[Step] = [
        Step(label="rsync", cmd=build_rsync(cfg, repo_root)),
        Step(label="provision", cmd=build_provision_ssh(cfg)),
    ]
    if run_bench:
        steps.append(Step(label="bench", cmd=build_bench_ssh(cfg)))
    if fetch_plan and run_bench:
        dest = plan_dest or repo_root / "out" / "plan.remote.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        steps.append(Step(label="scp-plan", cmd=build_scp_plan_back(cfg, dest)))
    return steps


__all__ = [
    "DEFAULT_CONTAINER",
    "DEFAULT_REMOTE_PATH",
    "DEFAULT_SSH_KEY",
    "DropletConfig",
    "build_bench_ssh",
    "build_provision_ssh",
    "build_rsync",
    "build_scp_plan_back",
    "build_ssh_probe",
    "build_tail_cloud_init",
    "from_env",
    "missing_env",
    "required_env",
    "status_missing",
    "status_target",
    "sync_steps",
]
