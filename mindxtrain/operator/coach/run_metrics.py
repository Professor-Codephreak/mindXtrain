"""Per-run system-metrics sampler.

While a training run is in `running` status, this module emits a
`MetricsEvent` (cpu_pct, ram_pct, load_1m, trainer-PID rss + cpu_seconds)
into the run's SSE channel once per second. The Coach UI's
`#step-train` panel renders the resulting time-series as d3 sparklines,
so the user can see — without leaving the browser — whether the
configured `cpu_throttle.percent` actually held, whether the trainer
RSS is growing the way the recipe predicted, and how much CPU-time
the run has actually consumed.

Design notes:

- One asyncio task per run_id, registered in `_TASKS`. Stopping a
  run cancels the task and drops the buffer; multiple concurrent
  training runs each get their own buffer + task.
- Each tick publishes via the run registry's `publish_threadsafe`
  + also appends to a rolling deque (last 300 samples = 5 min @ 1 Hz)
  so a tab-switch can re-fetch history via
  `GET /coach/api/runs/{id}/metrics?since=N` without waiting for fresh
  ticks.
- `psutil.NoSuchProcess` (the trainer died or operator-PID disappeared)
  emits a single zero-filled MetricsEvent and exits — never raises
  out of the task.
- Read `/proc/loadavg` directly: cheaper than running `uptime` for a
  number we only need to a couple of decimal places.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from typing import Any

import psutil

logger = logging.getLogger("mindxtrain.run_metrics")

# Rolling buffer of last 300 samples per run (5 min at 1 Hz). The number
# is big enough to feel "live" for a coffee break without growing past
# ~15 KB per run.
_BUFFER_CAP = 300

# Active sampler tasks keyed by run_id; cancelled on stop.
_TASKS: dict[str, asyncio.Task] = {}

# Rolling buffers keyed by run_id. Survive task cancellation so the
# UI can backfill the final frozen state on tab-switch.
_BUFFERS: dict[str, deque[dict[str, Any]]] = {}


def _read_loadavg() -> float:
    """First field of /proc/loadavg — 1-minute load average."""
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def _sample(run_id: str, pid: int) -> dict[str, Any]:
    """Build one MetricsEvent dict. Process-gone → zero-filled record."""
    cpu_pct = float(psutil.cpu_percent(interval=None))
    ram_pct = float(psutil.virtual_memory().percent)
    load_1m = _read_loadavg()
    try:
        proc = psutil.Process(pid)
        rss_bytes = proc.memory_info().rss
        cpu_times = proc.cpu_times()
        proc_rss_mb = rss_bytes / (1024 * 1024)
        proc_cpu_s = float(cpu_times.user + cpu_times.system)
    except psutil.NoSuchProcess:
        proc_rss_mb = 0.0
        proc_cpu_s = 0.0
    return {
        "kind": "metrics",
        "run_id": run_id,
        "ts": time.time(),
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "load_1m": load_1m,
        "proc_rss_mb": proc_rss_mb,
        "proc_cpu_seconds": proc_cpu_s,
    }


async def _sampler_loop(
    run_id: str,
    pid: int,
    interval_s: float,
    publish: Any,
) -> None:
    """Loop body — runs until cancelled or the trainer PID disappears."""
    # Prime psutil.cpu_percent: the first reading-after-prime is 0.0
    # because it has no baseline. Run one priming call before the loop.
    psutil.cpu_percent(interval=None)
    buf = _BUFFERS.setdefault(run_id, deque(maxlen=_BUFFER_CAP))
    try:
        while True:
            sample = _sample(run_id, pid)
            buf.append(sample)
            try:
                publish(run_id, sample)
            except Exception as exc:
                logger.debug("metrics publish failed: %r", exc)
            # Process-gone: emit the zero-sample and exit.
            if sample["proc_rss_mb"] == 0.0 and sample["proc_cpu_seconds"] == 0.0:
                # Only treat as terminal if psutil told us the proc disappeared.
                # A live process with 0 cpu_seconds during startup is normal —
                # check explicitly.
                if not psutil.pid_exists(pid):
                    logger.debug("metrics: pid %d gone, exiting sampler", pid)
                    return
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        # Normal stop path — propagate so the task is properly cancelled.
        raise
    except Exception as exc:
        logger.warning("metrics sampler crashed: %r", exc)


def start_metrics_sampler(
    run_id: str,
    pid: int,
    *,
    interval_s: float = 1.0,
    publish: Any | None = None,
) -> asyncio.Task:
    """Start the per-run sampler. Returns the asyncio.Task.

    `publish(run_id, dict)` is called synchronously inside the loop for
    each sample. Defaults to the live run registry's
    `publish_threadsafe` (which wraps the sample in a MetricsEvent and
    forwards to subscribers). Tests can pass a capture lambda.
    """
    if run_id in _TASKS and not _TASKS[run_id].done():
        return _TASKS[run_id]
    if publish is None:
        publish = _default_publish
    task = asyncio.create_task(
        _sampler_loop(run_id, pid, interval_s, publish),
        name=f"metrics-{run_id}",
    )
    _TASKS[run_id] = task
    return task


def _default_publish(run_id: str, sample: dict[str, Any]) -> None:
    """Publish into the live run registry. Lazy-imported to break a circle."""
    from mindxtrain.operator import runs as _runs

    event = _runs.MetricsEvent(**sample)
    # Use publish (not publish_threadsafe) because the sampler runs on
    # the same event loop the registry was attached to.
    _runs._REGISTRY.publish(run_id, event)


async def stop_metrics_sampler(run_id: str) -> None:
    """Cancel + await the sampler. Buffer is preserved for backfill."""
    task = _TASKS.pop(run_id, None)
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def get_buffer(run_id: str, *, since: float = 0.0) -> list[dict[str, Any]]:
    """Return samples with `ts > since`. Used by the backfill endpoint."""
    buf = _BUFFERS.get(run_id)
    if buf is None:
        return []
    if since <= 0.0:
        return list(buf)
    return [s for s in buf if s["ts"] > since]


def clear_buffer(run_id: str) -> None:
    """Drop the rolling buffer for a run. Called when the run is purged."""
    _BUFFERS.pop(run_id, None)


def has_sampler(run_id: str) -> bool:
    """Test helper: is a sampler currently registered for this run?"""
    task = _TASKS.get(run_id)
    return task is not None and not task.done()


__all__ = [
    "clear_buffer",
    "get_buffer",
    "has_sampler",
    "start_metrics_sampler",
    "stop_metrics_sampler",
]
