"""Multi-step pipeline runner for the deploy/* subcommands.

Each Coach UI deploy action (GitHub push, droplet sync, droplet provision)
expands to a list of `Step`s that must run serially. This module owns the
serial-execution loop; everything else is just a step list.

Design notes:

- We deliberately do NOT reuse `runs.spawn_subprocess_streaming` directly.
  That helper assumes a single Popen per run and emits `StatusEvent('succeeded')`
  + closes subscribers as soon as the Popen exits. Chaining N steps would
  prematurely close the SSE stream after step 1. Instead, this module runs
  its own daemon thread, manually publishes `LogEvent`s line-by-line, and
  emits exactly one terminal `StatusEvent` after all steps complete (or one
  fails).

- Cancellation routes through the existing `RunRegistry.cancel()` API. We
  call `registry.attach_process()` each time a new step starts, so the
  registry's Popen pointer always tracks the live subprocess.

- For non-subprocess work (httpx calls to the Dev Cloud API), the provision
  pipeline does not use `Step` objects — it publishes `LogEvent`s inline
  and then hands off to the step runner for the SSH phases.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from mindxtrain.deploy.amd_dev_cloud import (
    AmdDevCloudClient,
    AmdDevCloudConfig,
    extract_public_ip,
)
from mindxtrain.deploy.cloud_init import BOOTSTRAP_SENTINEL, render
from mindxtrain.deploy.droplet import (
    DropletConfig,
    build_scp_plan_back,
    build_ssh_probe,
    build_tail_cloud_init,
    build_tail_training_log,
    sync_steps,
)
from mindxtrain.deploy.github_push import (
    GithubConfig,
    Step,
    bootstrap_steps,
    remote_url,
    write_sha_file,
)
from mindxtrain.operator.runs import (
    LogEvent,
    RunRegistry,
    StatusEvent,
    default_registry,
    parse_trainer_log_line,
)

# ---- run_pipeline --------------------------------------------------------


def _publish_log(registry: RunRegistry, run_id: str, line: str) -> None:
    registry.publish_threadsafe(run_id, LogEvent(run_id=run_id, line=line))


def _open_log_file(log_path: Path) -> object:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("w", buffering=1)


def _stream_step(
    *,
    step: Step,
    registry: RunRegistry,
    run_id: str,
    log_file: object,
    capture: list[str],
    parse_trainer: bool = False,
) -> int:
    """Run a single step's subprocess, tee stdout to log_file + events.

    By default each line becomes a `LogEvent`. With `parse_trainer=True`,
    lines that match the HF Trainer JSON format are upgraded to `StepEvent`
    (drives Coach's loss chart and metrics table); non-matching lines still
    become `LogEvent`s. This is how remote training output bridges into the
    same SSE channel the in-process trainer would publish to.

    Captures stdout into `capture` if `step.capture_stdout`. Returns the rc.
    """
    proc = subprocess.Popen(
        step.cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=step.env or None,
        text=True,
        bufsize=1,
    )
    registry.attach_process(run_id, proc)
    assert proc.stdout is not None
    step_ctr = 0
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        log_file.write(raw)  # type: ignore[attr-defined]
        log_file.flush()  # type: ignore[attr-defined]
        if step.capture_stdout:
            capture.append(line)
        if parse_trainer:
            step_ev = parse_trainer_log_line(line, fallback_step=step_ctr + 1)
            if step_ev is not None:
                step_ctr = step_ev.step
                registry.publish_threadsafe(
                    run_id, step_ev.model_copy(update={"run_id": run_id}),
                )
                continue
        _publish_log(registry, run_id, line)
    return proc.wait()


def run_pipeline(
    steps: list[Step],
    *,
    run_id: str,
    out_dir: Path,
    registry: RunRegistry | None = None,
    on_done: Callable[[int, dict[str, str]], None] | None = None,
) -> threading.Thread:
    """Run `steps` serially in a daemon thread. Returns the thread immediately.

    Predicate logic: if a step's `predicate_step` was run earlier and that
    step's rc is *not* in `predicate_rc_in`, the step is skipped (with a
    LogEvent) and its rc is recorded as -1 ("skipped").

    Captured stdout: any step with `capture_stdout=True` has its full stdout
    joined into a string and stashed in `captured[step.label]`. The `on_done`
    callback receives (final_rc, captured).

    Final status: emits exactly one `StatusEvent("succeeded"|"failed")` and
    closes subscribers. Cancel via `registry.cancel(run_id)`.
    """
    reg = registry if registry is not None else default_registry()
    log_file = _open_log_file(out_dir / "pipeline.log")

    def _runner() -> None:
        rcs: dict[str, int] = {}
        captured: dict[str, str] = {}
        final_rc = 0
        try:
            for i, step in enumerate(steps, 1):
                # Predicate: skip if the gating step's rc is not in the allowed set.
                if step.predicate_step is not None:
                    gate_rc = rcs.get(step.predicate_step)
                    if gate_rc is None or gate_rc not in step.predicate_rc_in:
                        _publish_log(reg, run_id, f"=== skip {i}/{len(steps)}: {step.label} (predicate {step.predicate_step}={gate_rc}) ===")
                        rcs[step.label] = -1
                        continue

                _publish_log(reg, run_id, f"=== step {i}/{len(steps)}: {step.label} ===")
                buffer: list[str] = []
                rc = _stream_step(
                    step=step,
                    registry=reg,
                    run_id=run_id,
                    log_file=log_file,
                    capture=buffer,
                )
                rcs[step.label] = rc
                if step.capture_stdout:
                    captured[step.label] = "\n".join(buffer)
                if rc != 0 and not step.allow_failure:
                    _publish_log(reg, run_id, f"=== step {step.label} exited rc={rc}; aborting pipeline ===")
                    final_rc = rc
                    break
                if rc != 0:
                    _publish_log(reg, run_id, f"  (probe {step.label} rc={rc} — continuing)")

            status = "succeeded" if final_rc == 0 else "failed"
            reg.publish_threadsafe(
                run_id,
                StatusEvent(run_id=run_id, status=status, message=f"rc={final_rc}"),
            )
        finally:
            with contextlib.suppress(Exception):
                log_file.close()  # type: ignore[attr-defined]
            reg.close_subscribers(run_id)
            if on_done is not None:
                with contextlib.suppress(Exception):
                    on_done(final_rc, captured)

    t = threading.Thread(target=_runner, daemon=True, name=f"deploy-{run_id}")
    t.start()
    return t


# ---- GitHub push pipeline ------------------------------------------------


def github_push_pipeline(
    cfg: GithubConfig,
    *,
    run_id: str,
    out_dir: Path,
    commit_message: str = "mindXtrain initial push",
    force: bool = False,
    registry: RunRegistry | None = None,
    on_done: Callable[[int, dict[str, str]], None] | None = None,
) -> threading.Thread:
    """Drives the github_push step list. Captures HEAD sha to git_sha.txt."""
    reg = registry if registry is not None else default_registry()
    steps = bootstrap_steps(cfg, commit_message=commit_message, force=force)

    # Wrap on_done so we can persist the captured HEAD sha + emit a clean
    # guidance LogEvent if a stale remote was detected without --force.
    def _on_done(rc: int, captured: dict[str, str]) -> None:
        sha = captured.get("head-sha", "").strip()
        existing_remote = captured.get("probe-remote", "").strip()
        if sha:
            target = write_sha_file(out_dir, sha)
            _publish_log(reg, run_id, f"  HEAD sha {sha} written to {target}")
        if existing_remote and existing_remote != remote_url(cfg.repo) and not force:
            _publish_log(
                reg, run_id,
                f"  origin already points at {existing_remote!r}; "
                f"re-run with force=true to switch to {remote_url(cfg.repo)!r}",
            )
        if on_done is not None:
            with contextlib.suppress(Exception):
                on_done(rc, captured)

    return run_pipeline(steps, run_id=run_id, out_dir=out_dir, registry=reg, on_done=_on_done)


# ---- Existing-droplet sync pipeline -------------------------------------


def droplet_sync_pipeline(
    cfg: DropletConfig,
    *,
    repo_root: Path,
    run_id: str,
    out_dir: Path,
    run_bench: bool = True,
    fetch_plan: bool = True,
    registry: RunRegistry | None = None,
    on_done: Callable[[int, dict[str, str]], None] | None = None,
) -> threading.Thread:
    plan_dest = out_dir / "plan.remote.json"
    steps = sync_steps(
        cfg,
        repo_root=repo_root,
        run_bench=run_bench,
        fetch_plan=fetch_plan,
        plan_dest=plan_dest,
    )
    return run_pipeline(steps, run_id=run_id, out_dir=out_dir, registry=registry, on_done=on_done)


# ---- AMD Dev Cloud provision pipeline -----------------------------------


def droplet_provision_pipeline(
    cloud_cfg: AmdDevCloudConfig,
    *,
    name: str,
    repo: str,
    branch: str,
    container: str,
    extras: str,
    run_id: str,
    out_dir: Path,
    wait_for_bootstrap: bool = True,
    recipe: str | None = None,
    registry: RunRegistry | None = None,
    client_factory: Callable[[AmdDevCloudConfig], AmdDevCloudClient] | None = None,
    on_done: Callable[[int, dict[str, str]], None] | None = None,
) -> threading.Thread:
    """Create a droplet, wait for bootstrap, scp plan.json back, optionally
    train + bridge training events into the run's SSE stream.

    When `recipe` is set, cloud-init also runs `mindxtrain train` after
    bench, and the orchestrator adds a fifth step that SSH-tails the
    training log so Coach's loss chart populates live from the droplet.
    Without `recipe`, behaviour is exactly as before (bench only).
    """
    reg = registry if registry is not None else default_registry()
    out_dir.mkdir(parents=True, exist_ok=True)

    user_data = render(
        repo=repo,
        branch=branch,
        container=container,
        extras=extras,
        recipe=recipe,
    )
    factory = client_factory or AmdDevCloudClient

    def _log(line: str) -> None:
        _publish_log(reg, run_id, line)

    def _runner() -> None:
        final_rc = 0
        captured: dict[str, str] = {}
        try:
            with factory(cloud_cfg) as client:
                _publish_log(reg, run_id, "=== step 1/4: create droplet ===")
                droplet = client.create(name=name, user_data=user_data, log=_log)
                droplet_id = int(droplet.get("id", 0))
                captured["droplet_id"] = str(droplet_id)
                (out_dir / "droplet_id.txt").write_text(str(droplet_id) + "\n")

                _publish_log(reg, run_id, f"=== step 2/4: poll until active ({droplet_id}) ===")
                droplet = client.poll_until_active(droplet_id, log=_log)
                ip = extract_public_ip(droplet) or ""
                captured["public_ip"] = ip
                (out_dir / "public_ip.txt").write_text(ip + "\n")
                if not ip:
                    _publish_log(reg, run_id, "  (no public IPv4 returned; bailing out)")
                    final_rc = 2
                    return

            if not wait_for_bootstrap:
                _publish_log(reg, run_id, "wait_for_bootstrap=false — exiting after provision")
                return

            droplet_cfg = DropletConfig(
                host=ip,
                user="root",
                ssh_key=cloud_cfg_ssh_key(cloud_cfg),
                container=container,
                extras=extras,
            )
            _publish_log(reg, run_id, "=== step 3/4: wait for ssh + tail cloud-init ===")
            # Inline ssh-probe with a few retries; bench-stage cloud-init can take a while.
            probe_steps = [
                Step(label=f"ssh-probe-{i}", cmd=build_ssh_probe(droplet_cfg), allow_failure=True)
                for i in range(60)  # 60 * 5s = 5 min
            ]
            log_file = _open_log_file(out_dir / "pipeline.log")
            ready = False
            try:
                for ps in probe_steps:
                    rc = _stream_step(step=ps, registry=reg, run_id=run_id, log_file=log_file, capture=[])
                    if rc == 0:
                        ready = True
                        break
                    import time as _time
                    _time.sleep(5)
            finally:
                with contextlib.suppress(Exception):
                    log_file.close()  # type: ignore[attr-defined]
            if not ready:
                _publish_log(reg, run_id, "  ssh did not come up in 5 minutes; aborting")
                final_rc = 3
                return

            tail_step = Step(label="cloud-init-tail", cmd=build_tail_cloud_init(droplet_cfg))
            scp_step = Step(
                label="scp-plan",
                cmd=build_scp_plan_back(droplet_cfg, out_dir / "plan.remote.json"),
                allow_failure=True,
            )
            total_steps = 5 if recipe else 4
            log_file = _open_log_file(out_dir / "pipeline.log")
            try:
                _publish_log(reg, run_id, f"  (waiting for {BOOTSTRAP_SENTINEL})")
                rc = _stream_step(step=tail_step, registry=reg, run_id=run_id, log_file=log_file, capture=[])
                if rc != 0:
                    final_rc = rc
                    return
                _publish_log(reg, run_id, f"=== step 4/{total_steps}: scp plan.json back ===")
                _stream_step(step=scp_step, registry=reg, run_id=run_id, log_file=log_file, capture=[])

                if recipe:
                    _publish_log(
                        reg, run_id,
                        f"=== step 5/{total_steps}: bridge training log "
                        f"for recipe={recipe} ===",
                    )
                    train_tail = Step(
                        label="train-tail",
                        cmd=build_tail_training_log(droplet_cfg),
                    )
                    train_rc = _stream_step(
                        step=train_tail,
                        registry=reg,
                        run_id=run_id,
                        log_file=log_file,
                        capture=[],
                        parse_trainer=True,
                    )
                    if train_rc != 0:
                        _publish_log(
                            reg, run_id,
                            f"  (remote train exit rc={train_rc})",
                        )
                        final_rc = train_rc
            finally:
                with contextlib.suppress(Exception):
                    log_file.close()  # type: ignore[attr-defined]

        except Exception as exc:
            _publish_log(reg, run_id, f"!! {type(exc).__name__}: {exc}")
            final_rc = 1
        finally:
            status = "succeeded" if final_rc == 0 else "failed"
            reg.publish_threadsafe(
                run_id,
                StatusEvent(run_id=run_id, status=status, message=f"rc={final_rc}"),
            )
            reg.close_subscribers(run_id)
            if on_done is not None:
                with contextlib.suppress(Exception):
                    on_done(final_rc, captured)

    t = threading.Thread(target=_runner, daemon=True, name=f"provision-{run_id}")
    t.start()
    return t


def cloud_cfg_ssh_key(cloud_cfg: AmdDevCloudConfig) -> str:
    """Default ssh key path used to SSH into a freshly-provisioned droplet.

    The Dev Cloud control plane only stores the public key; the matching
    private key must be on the operator host. We assume it lives at
    `~/.ssh/id_ed25519` unless DROPLET_SSH_KEY is set.
    """
    import os as _os
    return _os.environ.get("DROPLET_SSH_KEY", "~/.ssh/id_ed25519")


__all__ = [
    "droplet_provision_pipeline",
    "droplet_sync_pipeline",
    "github_push_pipeline",
    "run_pipeline",
]
