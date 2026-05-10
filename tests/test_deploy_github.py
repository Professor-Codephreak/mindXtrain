"""GitHub push pure-builder tests — argv shape, idempotency, secret hygiene."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindxtrain.deploy.github_push import (
    GithubConfig,
    bootstrap_steps,
    missing_env,
    remote_url,
    required_env,
    status_missing,
    status_target,
    write_sha_file,
)


def _cfg(**overrides: str) -> GithubConfig:
    base = {
        "token": "ghp_TESTTOKEN",
        "repo": "professor-codephreak/mindXtrain",
        "branch": "main",
        "author_name": "mindXtrain bot",
        "author_email": "noreply@pythai.net",
    }
    base.update(overrides)
    return GithubConfig(**base)  # type: ignore[arg-type]


def test_required_env_lists_token_and_repo() -> None:
    assert "GITHUB_TOKEN" in required_env()
    assert "GITHUB_REPO" in required_env()


def test_missing_env_returns_unset_keys() -> None:
    miss = missing_env({"GITHUB_TOKEN": "x"})
    assert "GITHUB_REPO" in miss
    assert "GITHUB_TOKEN" not in miss
    full = missing_env({"GITHUB_TOKEN": "x", "GITHUB_REPO": "o/r"})
    assert full == []


def test_status_target_uses_repo_env() -> None:
    assert status_target({"GITHUB_REPO": "owner/repo"}) == "owner/repo"
    assert status_target({}) == ""


def test_remote_url_is_https() -> None:
    assert remote_url("owner/repo") == "https://github.com/owner/repo.git"


def test_bootstrap_steps_emits_all_phases() -> None:
    steps = bootstrap_steps(_cfg())
    labels = [s.label for s in steps]
    # Order matters: probe-git → git-init → probe-repo → gh-create →
    # probe-remote → git-remote-add → git-add → probe-stage → git-commit →
    # git-push → head-sha.
    assert labels == [
        "probe-git",
        "git-init",
        "probe-repo",
        "gh-create",
        "probe-remote",
        "git-remote-add",
        "git-add",
        "probe-stage",
        "git-commit",
        "git-push",
        "head-sha",
    ]


def test_token_never_appears_in_argv() -> None:
    """Secret hygiene: $GH_TOKEN is in env, never in argv. Catches a regression
    where a future refactor inlines the token into a credential URL."""
    steps = bootstrap_steps(_cfg(token="ghp_SUPERSECRET"))
    for step in steps:
        for arg in step.cmd:
            assert "ghp_SUPERSECRET" not in arg, f"token leaked into {step.label}: {arg!r}"


def test_token_is_in_env_for_push_step() -> None:
    steps = bootstrap_steps(_cfg(token="ghp_SUPERSECRET"))
    push = next(s for s in steps if s.label == "git-push")
    assert push.env["GH_TOKEN"] == "ghp_SUPERSECRET"
    assert push.env["GITHUB_TOKEN"] == "ghp_SUPERSECRET"


def test_force_with_lease_only_when_force_true() -> None:
    soft = bootstrap_steps(_cfg(), force=False)
    push_soft = next(s for s in soft if s.label == "git-push")
    assert "--force-with-lease" not in push_soft.cmd

    hard = bootstrap_steps(_cfg(), force=True)
    push_hard = next(s for s in hard if s.label == "git-push")
    assert "--force-with-lease" in push_hard.cmd


def test_commit_uses_dash_c_flags_to_avoid_global_gitconfig() -> None:
    """The commit step injects user.email/name via `git -c …` so we never
    write to ~/.gitconfig (would surprise the user). Pin the contract."""
    steps = bootstrap_steps(_cfg(author_email="ci@x.com", author_name="ci"))
    commit = next(s for s in steps if s.label == "git-commit")
    assert "-c" in commit.cmd
    assert "user.email=ci@x.com" in commit.cmd
    assert "user.name=ci" in commit.cmd


def test_push_uses_credential_helper_not_url_token() -> None:
    """The push command must pass auth via credential.helper, not by
    embedding the token into the remote URL (which would land in `.git/config`
    or `git reflog`)."""
    steps = bootstrap_steps(_cfg(token="ghp_SUPERSECRET"))
    push = next(s for s in steps if s.label == "git-push")
    # No URL with embedded token.
    assert not any("@github.com" in arg and "x-access-token" in arg for arg in push.cmd)
    # Yes credential.helper inline.
    helper = [arg for arg in push.cmd if "credential.helper=" in arg]
    assert helper, "git-push step is missing the credential.helper inline config"


def test_predicate_wiring_for_conditional_steps() -> None:
    steps = {s.label: s for s in bootstrap_steps(_cfg())}
    assert steps["git-init"].predicate_step == "probe-git"
    assert steps["gh-create"].predicate_step == "probe-repo"
    assert steps["gh-create"].predicate_rc_in == (1,)
    assert steps["git-commit"].predicate_step == "probe-stage"
    assert steps["git-commit"].predicate_rc_in == (1,)
    # Probes themselves must allow_failure so a non-zero rc doesn't kill the pipeline.
    for label in ("probe-git", "probe-repo", "probe-remote", "probe-stage"):
        assert steps[label].allow_failure, f"{label} must allow failure"


def test_capture_stdout_on_head_sha_step() -> None:
    steps = bootstrap_steps(_cfg())
    sha = next(s for s in steps if s.label == "head-sha")
    assert sha.capture_stdout, "head-sha must capture stdout for manifest.git_sha"


def test_write_sha_file_persists_to_disk(tmp_path: Path) -> None:
    target = write_sha_file(tmp_path, "abc123\n")
    assert target == tmp_path / "git_sha.txt"
    assert target.read_text().strip() == "abc123"


def test_status_missing_includes_missing_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """If gh is not on PATH, status_missing surfaces it so the UI can disable
    the button + show 'install gh'."""
    monkeypatch.setattr("shutil.which", lambda _b: None)
    miss = status_missing({"GITHUB_TOKEN": "x", "GITHUB_REPO": "o/r"})
    assert "git" in miss
    assert "gh" in miss
