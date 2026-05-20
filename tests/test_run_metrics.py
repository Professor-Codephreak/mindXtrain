"""Per-run system-metrics sampler + MetricsEvent + /metrics endpoint."""

from __future__ import annotations

import asyncio
import os
import time

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from mindxtrain.operator import runs as _runs
from mindxtrain.operator.app import app
from mindxtrain.operator.coach import run_metrics as rm
from mindxtrain.operator.coach.api import _REGISTRY

client = TestClient(app)


# ---- MetricsEvent contract ----------------------------------------------


def test_metrics_event_in_train_event_union():
    """MetricsEvent must be a valid member of the discriminated union."""
    from pydantic import TypeAdapter

    ev = _runs.MetricsEvent(
        run_id="r1", ts=1.0, cpu_pct=42.0, ram_pct=18.0,
        load_1m=1.2, proc_rss_mb=512.0, proc_cpu_seconds=10.5,
    )
    ta = TypeAdapter(_runs.TrainEvent)
    restored = ta.validate_python(ev.model_dump())
    assert restored.kind == "metrics"
    assert restored.cpu_pct == 42.0


def test_format_sse_round_trip_for_metrics():
    """format_sse must encode the new kind under `event: metrics`."""
    ev = _runs.MetricsEvent(
        run_id="r1", ts=1.0, cpu_pct=0.0, ram_pct=0.0,
        load_1m=0.0, proc_rss_mb=0.0, proc_cpu_seconds=0.0,
    )
    frame = _runs.format_sse(ev)
    assert frame.startswith("event: metrics\ndata: {")
    assert '"kind":"metrics"' in frame


def test_metrics_event_rejects_unknown_fields():
    """extra='forbid' — typos in publish callers fail loudly."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _runs.MetricsEvent(
            run_id="r1", ts=1.0, cpu_pct=0.0, ram_pct=0.0,
            load_1m=0.0, proc_rss_mb=0.0, proc_cpu_seconds=0.0,
            bogus_field=1.0,  # type: ignore[call-arg]
        )


# ---- Sampler task lifecycle ---------------------------------------------


@pytest_asyncio.fixture
async def fresh_sampler():
    """Each test starts with empty task + buffer registries."""
    for tid in list(rm._TASKS):
        await rm.stop_metrics_sampler(tid)
    rm._BUFFERS.clear()
    yield
    for tid in list(rm._TASKS):
        await rm.stop_metrics_sampler(tid)
    rm._BUFFERS.clear()


@pytest.mark.asyncio
async def test_sampler_publishes_at_requested_cadence(fresh_sampler):
    captured: list[dict] = []

    def _capture(run_id, sample):
        captured.append({"run_id": run_id, **sample})

    rm.start_metrics_sampler(
        "r-cadence", os.getpid(),
        interval_s=0.05, publish=_capture,
    )
    # 0.18 s @ 50 ms cadence → ~3 samples (give a wide grace window).
    await asyncio.sleep(0.18)
    await rm.stop_metrics_sampler("r-cadence")
    assert len(captured) >= 2, captured
    # Each capture conforms to the MetricsEvent shape.
    for s in captured:
        assert {"cpu_pct", "ram_pct", "load_1m",
                "proc_rss_mb", "proc_cpu_seconds", "ts"} <= set(s)


@pytest.mark.asyncio
async def test_sampler_writes_to_buffer(fresh_sampler):
    """get_buffer returns the most recent samples after the task ran."""
    rm.start_metrics_sampler(
        "r-buf", os.getpid(), interval_s=0.05, publish=lambda *_a: None,
    )
    await asyncio.sleep(0.12)
    await rm.stop_metrics_sampler("r-buf")
    buf = rm.get_buffer("r-buf")
    assert len(buf) >= 2
    assert all("ts" in s and "cpu_pct" in s for s in buf)


@pytest.mark.asyncio
async def test_sampler_buffer_filters_by_since(fresh_sampler):
    rm.start_metrics_sampler(
        "r-since", os.getpid(), interval_s=0.05, publish=lambda *_a: None,
    )
    await asyncio.sleep(0.20)
    cutoff = time.time() - 0.05
    await rm.stop_metrics_sampler("r-since")
    recent = rm.get_buffer("r-since", since=cutoff)
    older = rm.get_buffer("r-since", since=0.0)
    assert len(recent) < len(older)


@pytest.mark.asyncio
async def test_sampler_stop_is_idempotent(fresh_sampler):
    rm.start_metrics_sampler(
        "r-stop", os.getpid(), interval_s=0.05, publish=lambda *_a: None,
    )
    await rm.stop_metrics_sampler("r-stop")
    # Second stop must not raise; just be a noop.
    await rm.stop_metrics_sampler("r-stop")
    assert not rm.has_sampler("r-stop")


@pytest.mark.asyncio
async def test_start_is_idempotent_per_run(fresh_sampler):
    """Calling start twice for the same run_id reuses the existing task."""
    t1 = rm.start_metrics_sampler(
        "r-dup", os.getpid(), interval_s=0.05, publish=lambda *_a: None,
    )
    t2 = rm.start_metrics_sampler(
        "r-dup", os.getpid(), interval_s=0.05, publish=lambda *_a: None,
    )
    assert t1 is t2
    await rm.stop_metrics_sampler("r-dup")


@pytest.mark.asyncio
async def test_sampler_handles_dead_pid_gracefully(fresh_sampler):
    """A non-existent PID emits a final zero-sample then exits."""
    captured: list[dict] = []
    # Use a PID that's astronomically unlikely to exist on the test host.
    dead_pid = 2_147_483_640
    rm.start_metrics_sampler(
        "r-dead", dead_pid,
        interval_s=0.05, publish=lambda _r, s: captured.append(s),
    )
    # Give the loop two ticks to discover the missing PID.
    await asyncio.sleep(0.15)
    assert not rm.has_sampler("r-dead"), "sampler should exit on dead pid"
    # The captured records must show zero proc fields.
    assert captured, "at least one sample should have been emitted"
    last = captured[-1]
    assert last["proc_rss_mb"] == 0.0
    assert last["proc_cpu_seconds"] == 0.0


@pytest.mark.asyncio
async def test_publish_failure_does_not_kill_loop(fresh_sampler):
    """publish() raising must not crash the sampler — just log."""
    call_count = {"n": 0}

    def _bad_publish(_r, _s):
        call_count["n"] += 1
        raise RuntimeError("downstream broken")

    rm.start_metrics_sampler(
        "r-bad-pub", os.getpid(), interval_s=0.05, publish=_bad_publish,
    )
    await asyncio.sleep(0.18)
    await rm.stop_metrics_sampler("r-bad-pub")
    # The loop kept ticking through the publish failures.
    assert call_count["n"] >= 2


def test_clear_buffer_drops_the_buffer():
    rm._BUFFERS["r-clear"] = rm._BUFFERS.get("r-clear") or rm.deque(maxlen=10)
    rm.clear_buffer("r-clear")
    assert rm.get_buffer("r-clear") == []


# ---- /coach/api/runs/{id}/metrics endpoint ------------------------------


def test_metrics_endpoint_404_on_unknown_run():
    r = client.get("/coach/api/runs/never-existed/metrics")
    assert r.status_code == 404


def test_metrics_endpoint_returns_buffered_samples(tmp_path):
    """End-to-end: registry knows the run, sampler populated the buffer."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    run = _REGISTRY.create("test-recipe", out_dir)
    # Manually seed the buffer (avoid spinning a real sampler in this sync test).
    rm._BUFFERS[run.id] = rm.deque(maxlen=10)
    for i in range(3):
        rm._BUFFERS[run.id].append({
            "kind": "metrics", "run_id": run.id, "ts": float(i),
            "cpu_pct": 10.0 + i, "ram_pct": 20.0, "load_1m": 0.5,
            "proc_rss_mb": 100.0, "proc_cpu_seconds": float(i),
        })
    r = client.get(f"/coach/api/runs/{run.id}/metrics")
    assert r.status_code == 200
    body = r.json()
    assert len(body["samples"]) == 3
    assert body["samples"][0]["cpu_pct"] == 10.0
    # since=1.5 filters out ts<=1.5 → keeps ts=2 only.
    r = client.get(f"/coach/api/runs/{run.id}/metrics", params={"since": 1.5})
    assert len(r.json()["samples"]) == 1
    rm.clear_buffer(run.id)
