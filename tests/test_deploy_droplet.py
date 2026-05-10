"""Droplet sync builder tests — argv shape, no-shell contract, ssh hardening."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindxtrain.deploy.droplet import (
    DropletConfig,
    build_bench_ssh,
    build_provision_ssh,
    build_rsync,
    build_scp_plan_back,
    build_ssh_probe,
    build_tail_cloud_init,
    from_env,
    missing_env,
    status_missing,
    status_target,
    sync_steps,
)


def _cfg() -> DropletConfig:
    return DropletConfig(host="mi300x.example.com", user="ubuntu")


def test_missing_env_keys() -> None:
    assert "DROPLET_HOST" in missing_env({})
    assert missing_env({"DROPLET_HOST": "h", "DROPLET_USER": "u"}) == []


def test_status_target_formats_user_at_host_path() -> None:
    target = status_target({
        "DROPLET_HOST": "mi300x.example.com",
        "DROPLET_USER": "root",
        "DROPLET_REMOTE_PATH": "/workspace/mindxtrain",
    })
    assert target == "root@mi300x.example.com:/workspace/mindxtrain"


def test_status_target_blank_when_unconfigured() -> None:
    assert status_target({"DROPLET_HOST": "h"}) == ""  # no user
    assert status_target({"DROPLET_USER": "u"}) == ""  # no host


def test_from_env_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="DROPLET_HOST"):
        from_env({})


def test_from_env_picks_defaults() -> None:
    cfg = from_env({"DROPLET_HOST": "h", "DROPLET_USER": "u"})
    assert cfg.host == "h"
    assert cfg.user == "u"
    assert cfg.container == "rocm/primus:v26.2"
    assert cfg.remote_path == "/workspace/mindxtrain"


def test_rsync_argv_has_excludes_and_trailing_slashes(tmp_path: Path) -> None:
    cmd = build_rsync(_cfg(), tmp_path)
    assert cmd[0] == "rsync"
    # Must exclude .git or the cloud-init clone path is the source of truth.
    assert "--exclude" in cmd
    assert ".git" in cmd
    # Source ends with `/`, dest ends with `/` — rsync semantics for "copy contents".
    src = cmd[-2]
    dst = cmd[-1]
    assert src.endswith("/")
    assert dst.endswith("/")
    assert dst == "ubuntu@mi300x.example.com:/workspace/mindxtrain/"


def test_ssh_options_set_batchmode_everywhere() -> None:
    """BatchMode=yes prevents an interactive prompt from silently hanging
    the spawn thread. Pin it for every ssh-flavored command."""
    for cmd in (
        build_provision_ssh(_cfg()),
        build_bench_ssh(_cfg()),
        build_ssh_probe(_cfg()),
        build_tail_cloud_init(_cfg()),
        build_scp_plan_back(_cfg(), Path("/tmp/x")),
    ):
        joined = " ".join(cmd)
        assert "BatchMode=yes" in joined, f"missing BatchMode in {cmd!r}"


def test_bench_ssh_uses_force_pty() -> None:
    """`-tt` is what propagates SIGINT through ssh to the remote podman.
    Without it, registry.cancel() leaves the GPU spinning."""
    cmd = build_bench_ssh(_cfg())
    assert "-tt" in cmd, f"missing -tt: {cmd!r}"


def test_provision_ssh_is_idempotent_shell_body() -> None:
    cmd = build_provision_ssh(_cfg())
    body = cmd[-1]
    assert "command -v podman" in body
    assert "podman image exists" in body


def test_bench_ssh_runs_inside_container() -> None:
    cmd = build_bench_ssh(_cfg())
    body = cmd[-1]
    assert "podman run" in body
    assert "--device /dev/kfd" in body
    assert "--device /dev/dri" in body
    assert "mindxtrain bench --gpu 0 --out plan.json" in body


def test_no_shell_true_contract_via_argv_form() -> None:
    """Hostile-looking host string lands as a single argv element rather
    than getting split by a shell. `subprocess.Popen(cmd_list)` is called
    with `shell=False` (default), so this is a smoke check that the
    builder doesn't accidentally interpolate into a string."""
    evil = "evil; rm -rf /"
    cfg = DropletConfig(host=evil, user="u")
    for cmd in (build_provision_ssh(cfg), build_bench_ssh(cfg), build_ssh_probe(cfg)):
        # The host has to land in one and only one argv element (the user@host bit
        # or in the bench ssh body). Any case where ssh-options leak into shell
        # parsing would split this string across multiple args.
        joined_args = [a for a in cmd if evil in a]
        assert joined_args, f"host string vanished from cmd: {cmd!r}"
        # And no element is bash-looking ("rm -rf" as a standalone arg means
        # the shell already ran it locally — bug).
        for arg in cmd:
            assert arg != "rm", f"shell injection: rm leaked as standalone arg in {cmd!r}"


def test_scp_plan_back_targets_local_path(tmp_path: Path) -> None:
    dest = tmp_path / "plan.remote.json"
    cmd = build_scp_plan_back(_cfg(), dest)
    assert cmd[0] == "scp"
    assert str(dest) in cmd
    src = [a for a in cmd if "@" in a and "plan.json" in a]
    assert src, f"no remote source in {cmd!r}"


def test_sync_steps_with_bench_includes_scp(tmp_path: Path) -> None:
    steps = sync_steps(_cfg(), tmp_path, run_bench=True, fetch_plan=True)
    labels = [s.label for s in steps]
    assert labels == ["rsync", "provision", "bench", "scp-plan"]


def test_sync_steps_without_bench_skips_bench_and_scp(tmp_path: Path) -> None:
    steps = sync_steps(_cfg(), tmp_path, run_bench=False, fetch_plan=True)
    labels = [s.label for s in steps]
    assert labels == ["rsync", "provision"]


def test_status_missing_surfaces_missing_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _b: None)
    miss = status_missing({"DROPLET_HOST": "h", "DROPLET_USER": "u"})
    assert "rsync" in miss
    assert "ssh" in miss
    assert "scp" in miss


def test_tail_cloud_init_waits_for_sentinel() -> None:
    cmd = build_tail_cloud_init(_cfg())
    body = cmd[-1]
    assert "/workspace/mindxtrain/.bootstrap-done" in body
    assert "tail" in body
