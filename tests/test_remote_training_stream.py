"""Remote MI300X training-feedback bridge — closes the Coach loss-chart gap.

When a user kicks off a real training run via Coach → Deploy → Provision,
the orchestrator now:

1. Renders cloud-init with an optional `recipe=...` so the droplet runs
   `mindxtrain train` after bench.
2. SSH-tails the droplet's combined training log and pipes parsed Trainer
   log lines as `StepEvent`s into the same run-id's SSE stream.

These tests cover the pure-string + provision-API surfaces. The SSH
subprocess + threaded streamer are exercised in integration smokes; here
we keep the harness small and offline.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mindxtrain.deploy.cloud_init import (
    BOOTSTRAP_SENTINEL,
    TRAIN_DONE_SENTINEL,
    TRAIN_EXIT_SENTINEL,
    render,
)
from mindxtrain.deploy.droplet import DropletConfig, build_tail_training_log
from mindxtrain.operator import runs as _runs
from mindxtrain.operator import training_api  # noqa: F401  (registers /v1/training)
from mindxtrain.operator.app import app
from mindxtrain.operator.coach import api as coach_api
from mindxtrain.operator.runs import LogEvent, parse_trainer_log_line

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    """Each provision-API test starts with a fresh run registry so the
    `_busy_deploy_run` gate doesn't see leftover state from prior tests."""
    _runs.reset_default_registry()
    monkeypatch.setattr(coach_api, "_REGISTRY", _runs.default_registry())
    yield


# ---- cloud-init renderer --------------------------------------------------


def test_cloud_init_without_recipe_is_bench_only():
    """Backward compat: render() without recipe omits the train step."""
    out = render()
    assert BOOTSTRAP_SENTINEL in out
    assert "mindxtrain bench" in out
    assert "mindxtrain train" not in out
    assert TRAIN_DONE_SENTINEL not in out
    assert TRAIN_EXIT_SENTINEL not in out


def test_cloud_init_with_recipe_adds_train_and_sentinels():
    out = render(recipe="mindx_fallback_qwen3_1_5b_sft_lora")
    assert "mindxtrain bench" in out
    assert "mindxtrain train mindxtrain/train/recipes/mindx_fallback_qwen3_1_5b_sft_lora.yaml" in out
    assert TRAIN_DONE_SENTINEL in out
    assert TRAIN_EXIT_SENTINEL in out
    # train step must come AFTER the bootstrap sentinel touch (so the
    # operator can switch from cloud-init-tail to train-tail cleanly).
    assert out.index(f"touch {BOOTSTRAP_SENTINEL}") < out.index(TRAIN_DONE_SENTINEL)


def test_cloud_init_rejects_unsafe_recipe_name():
    """Recipe name must pass the strict basename regex (no path traversal)."""
    with pytest.raises(ValueError):
        render(recipe="../../etc/passwd")
    with pytest.raises(ValueError):
        render(recipe="recipe; rm -rf /")
    with pytest.raises(ValueError):
        render(recipe="some/nested/path")


# ---- ssh tail builder ----------------------------------------------------


def test_build_tail_training_log_targets_correct_paths():
    cfg = DropletConfig(
        host="10.0.0.5",
        user="root",
        ssh_key="/tmp/key",
        container="rocm/primus:v26.2",
        extras="ml",
    )
    cmd = build_tail_training_log(cfg)
    assert cmd[0] == "ssh"
    assert "root@10.0.0.5" in cmd
    remote = cmd[-1]
    # Tails the combined log path written by cloud-init.
    assert "out/runs/_combined_train.log" in remote
    # Polls the done sentinel.
    assert TRAIN_DONE_SENTINEL in remote
    # Propagates the captured exit code.
    assert TRAIN_EXIT_SENTINEL in remote
    assert "exit $(cat" in remote


# ---- StepEvent classification from Trainer log lines -------------------


def test_parse_trainer_log_line_extracts_loss_and_lr():
    """The remote tail relies on this parser; smoke a representative line."""
    line = "{'loss': 1.2345, 'learning_rate': 9.5e-05, 'epoch': 0.02}"
    ev = parse_trainer_log_line(line, fallback_step=42)
    assert ev is not None
    assert ev.loss == 1.2345
    assert ev.lr is not None
    assert abs(ev.lr - 9.5e-05) < 1e-9


def test_parse_trainer_log_line_non_match_returns_none():
    assert parse_trainer_log_line("docker pulling layer …", fallback_step=1) is None
    assert parse_trainer_log_line("", fallback_step=1) is None


def test_stream_step_publishes_step_events_when_parse_trainer():
    """The orchestrator's _stream_step (parse_trainer=True) must classify
    Trainer JSON lines as StepEvents and everything else as LogEvents.
    Exercise the classifier directly against a fake registry by simulating
    what the streamer does line-by-line — we don't need to subprocess.
    """
    reg = _runs.RunRegistry()
    run = reg.create("_test", out_dir=__import__("pathlib").Path("/tmp"))
    sample = [
        "{'loss': 0.99, 'learning_rate': 1e-4}",
        "Some axolotl bootstrap chatter",
        "{'loss': 0.88, 'learning_rate': 9.5e-05, 'grad_norm': 0.1}",
    ]
    step_ctr = 0
    log_events = 0
    step_events = 0
    for line in sample:
        ev = parse_trainer_log_line(line, fallback_step=step_ctr + 1)
        if ev is not None:
            step_ctr = ev.step
            reg.publish(run.id, ev.model_copy(update={"run_id": run.id}))
            step_events += 1
        else:
            reg.publish(run.id, LogEvent(run_id=run.id, line=line))
            log_events += 1
    assert step_events == 2
    assert log_events == 1


# ---- coach api: provision request recipe validation --------------------


def test_provision_request_accepts_valid_recipe(monkeypatch):
    """Replace the real provisioner with a no-op so the test stays offline."""
    from mindxtrain.operator.coach import api as coach_api

    captured: dict = {}

    def fake_provision(run, req):
        captured["recipe"] = req.recipe
        captured["run_id"] = run.id

    monkeypatch.setattr(coach_api, "_DROPLET_PROVISION_SPAWN", fake_provision)
    # Pretend AMD env is configured so the 503 gate doesn't fire.
    monkeypatch.setenv("AMD_DEV_CLOUD_TOKEN", "fake")
    monkeypatch.setenv("AMD_DEV_CLOUD_SSH_KEY_ID", "fake")

    r = client.post(
        "/coach/api/droplet/provision",
        json={"recipe": "mindx_fallback_qwen3_1_5b_sft_lora"},
    )
    assert r.status_code == 200, r.text
    assert captured["recipe"] == "mindx_fallback_qwen3_1_5b_sft_lora"


def test_provision_request_rejects_unknown_recipe(monkeypatch):
    monkeypatch.setenv("AMD_DEV_CLOUD_TOKEN", "fake")
    monkeypatch.setenv("AMD_DEV_CLOUD_SSH_KEY_ID", "fake")
    r = client.post(
        "/coach/api/droplet/provision",
        json={"recipe": "does_not_exist"},
    )
    assert r.status_code == 404


def test_provision_request_recipe_is_optional(monkeypatch):
    """Bench-only (no recipe) path stays valid — backward compat."""
    from mindxtrain.operator.coach import api as coach_api

    monkeypatch.setattr(coach_api, "_DROPLET_PROVISION_SPAWN", lambda r, q: None)
    monkeypatch.setenv("AMD_DEV_CLOUD_TOKEN", "fake")
    monkeypatch.setenv("AMD_DEV_CLOUD_SSH_KEY_ID", "fake")
    r = client.post("/coach/api/droplet/provision", json={})
    assert r.status_code == 200, r.text
