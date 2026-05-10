"""GitHub push step builder.

Returns an idempotent list of `Step`s for bootstrapping a git repo, creating
the GitHub remote (via `gh`), and pushing the working tree. Pure builder —
no subprocesses are spawned here. The orchestrator in `_orchestrator.py`
runs the steps via `mindxtrain.operator.runs.spawn_subprocess_streaming` so
each command's stdout streams back over the existing SSE pipeline.

Auth: `GITHUB_TOKEN` is forwarded into the subprocess env as `GH_TOKEN` (the
name `gh` looks for) and as `GITHUB_TOKEN` (for `git push` via the credential
helper). The token never lands in argv or `.git/config`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Step:
    """A single shell-out step in a deploy pipeline."""

    label: str
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # If `predicate_step` is set, this step only runs when that step's rc
    # matches `predicate_rc_in`. Used to model conditional steps without
    # building a full DAG.
    predicate_step: str | None = None
    predicate_rc_in: tuple[int, ...] = (0,)
    # If `capture_stdout` is True, the orchestrator stashes the step's stdout
    # under `Step.label` so later steps / callers can read it.
    capture_stdout: bool = False
    # If True, the orchestrator continues even on non-zero rc (used for probes).
    allow_failure: bool = False


# Credential helper inline script — feeds the token to git push without
# touching ~/.gitconfig or .git/config. Single-quoted body so $GH_TOKEN is
# expanded by the spawned shell, not by Python's f-string.
_GIT_CRED_HELPER = (
    "credential.helper="
    "!f() { echo username=x-access-token; echo \"password=$GH_TOKEN\"; }; f"
)


@dataclass(frozen=True)
class GithubConfig:
    token: str
    repo: str  # "owner/name"
    branch: str = "main"
    author_name: str = "mindXtrain bot"
    author_email: str = "noreply@pythai.net"


def _step_env(token: str) -> dict[str, str]:
    """Subprocess env: pass GH_TOKEN + GITHUB_TOKEN, scrub nothing else."""
    base = dict(os.environ)
    base["GH_TOKEN"] = token
    base["GITHUB_TOKEN"] = token
    return base


def required_env() -> tuple[str, ...]:
    """Names of env vars the operator must populate for /api/github/push."""
    return ("GITHUB_TOKEN", "GITHUB_REPO")


def missing_env(env: dict[str, str] | None = None) -> list[str]:
    src = env if env is not None else os.environ
    return [k for k in required_env() if not src.get(k)]


def _which_missing() -> list[str]:
    """Binaries the push pipeline depends on, but only if they're absent."""
    out = []
    for binary in ("git", "gh"):
        if shutil.which(binary) is None:
            out.append(binary)
    return out


def status_target(env: dict[str, str] | None = None) -> str:
    src = env if env is not None else os.environ
    return src.get("GITHUB_REPO", "")


def status_missing(env: dict[str, str] | None = None) -> list[str]:
    """Combine env-missing + binary-missing for the /status endpoint."""
    return missing_env(env) + _which_missing()


def bootstrap_steps(
    cfg: GithubConfig,
    *,
    commit_message: str = "mindXtrain initial push",
    force: bool = False,
) -> list[Step]:
    """Idempotent step list. The orchestrator skips conditional steps by
    inspecting earlier steps' rcs."""
    env = _step_env(cfg.token)
    repo_url = f"https://github.com/{cfg.repo}.git"
    push_cmd = [
        "git",
        "-c", _GIT_CRED_HELPER,
        "push", "-u", "origin", f"HEAD:{cfg.branch}",
    ]
    if force:
        push_cmd.append("--force-with-lease")

    return [
        Step(
            label="probe-git",
            cmd=["git", "rev-parse", "--git-dir"],
            env=env,
            allow_failure=True,
        ),
        Step(
            label="git-init",
            cmd=["git", "init", "-b", cfg.branch],
            env=env,
            predicate_step="probe-git",
            predicate_rc_in=(1, 128),  # not a git repo
        ),
        Step(
            label="probe-repo",
            cmd=["gh", "repo", "view", cfg.repo],
            env=env,
            allow_failure=True,
        ),
        Step(
            label="gh-create",
            cmd=["gh", "repo", "create", cfg.repo, "--public",
                 "--source=.", "--remote=origin"],
            env=env,
            predicate_step="probe-repo",
            predicate_rc_in=(1,),  # repo doesn't exist yet
        ),
        Step(
            label="probe-remote",
            cmd=["git", "remote", "get-url", "origin"],
            env=env,
            allow_failure=True,
            capture_stdout=True,
        ),
        Step(
            # If origin already exists but doesn't match, only rewrite when force=True.
            # The orchestrator emits a guidance LogEvent + bails when force=False
            # and probe-remote stdout doesn't match repo_url.
            label="git-remote-add",
            cmd=["git", "remote", "add", "origin", repo_url],
            env=env,
            predicate_step="probe-remote",
            predicate_rc_in=(1, 2, 128),  # remote not configured
        ),
        Step(label="git-add", cmd=["git", "add", "-A"], env=env),
        Step(
            label="probe-stage",
            cmd=["git", "diff", "--cached", "--quiet"],
            env=env,
            allow_failure=True,
        ),
        Step(
            label="git-commit",
            cmd=[
                "git",
                "-c", f"user.email={cfg.author_email}",
                "-c", f"user.name={cfg.author_name}",
                "commit", "-m", commit_message,
            ],
            env=env,
            predicate_step="probe-stage",
            predicate_rc_in=(1,),  # rc=1 means "differences exist" → there's something to commit
        ),
        Step(label="git-push", cmd=push_cmd, env=env),
        Step(
            label="head-sha",
            cmd=["git", "rev-parse", "HEAD"],
            env=env,
            capture_stdout=True,
        ),
    ]


def remote_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def write_sha_file(out_dir: Path, sha: str) -> Path:
    """Persist the captured HEAD SHA so emit_receipt callers can pin
    `manifest.git_sha` without re-shelling git."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "git_sha.txt"
    target.write_text(sha.strip() + "\n")
    return target


__all__ = [
    "GithubConfig",
    "Step",
    "bootstrap_steps",
    "missing_env",
    "remote_url",
    "required_env",
    "status_missing",
    "status_target",
    "write_sha_file",
]
