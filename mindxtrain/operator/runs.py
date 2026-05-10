"""In-memory training-run registry + SSE event stream.

`RunRegistry` is a singleton keyed by `run_id`. It owns:

- a snapshot `Run` per id (frozen pydantic model — updates produce a new
  snapshot via `model_copy`),
- a ring buffer of the last 200 `TrainEvent` per id (for late-subscriber
  replay),
- a fan-out set of asyncio queues per id (one queue per active subscriber).

The module imports cleanly without `--extra ml`: the only deps are stdlib
(asyncio, signal, subprocess, threading, re, uuid) plus pydantic + httpx
which are base requirements.

The two ingestion paths are covered here:

1. Subprocess stdout regex — `parse_trainer_log_line` extracts HF Trainer
   `{'loss': ..., 'learning_rate': ...}` log lines into `StepEvent`s.
2. In-process `StreamCallback` — POSTs `TrainEvent` JSON to the operator's
   own /coach/api/runs/{id}/ingest endpoint (loopback only). That callback
   lives in `mindxtrain.train.callbacks` to keep the lazy-import boundary.

Event wire format on SSE (per spec):

    event: <kind>
    data: <event.model_dump_json()>
    <blank line>
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import signal
import subprocess
import threading
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---- TrainEvent discriminated union --------------------------------------

RunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StatusEvent(_Event):
    kind: Literal["status"] = "status"
    run_id: str
    status: RunStatus
    message: str = ""


class StepEvent(_Event):
    kind: Literal["step"] = "step"
    run_id: str
    step: int
    loss: float
    lr: float | None = None
    grad_norm: float | None = None
    tokens_per_s: float | None = None


class EvalEvent(_Event):
    kind: Literal["eval"] = "eval"
    run_id: str
    step: int
    suite: str
    metrics: dict[str, float]


class LogEvent(_Event):
    kind: Literal["log"] = "log"
    run_id: str
    line: str
    level: Literal["stdout", "stderr"] = "stdout"


class EnergyEvent(_Event):
    kind: Literal["energy"] = "energy"
    run_id: str
    watts: float
    gpu_index: int = 0


TrainEvent = Annotated[
    StatusEvent | StepEvent | EvalEvent | LogEvent | EnergyEvent,
    Field(discriminator="kind"),
]


def format_sse(event: _Event) -> str:
    """Encode a TrainEvent as a single SSE frame."""
    return f"event: {event.kind}\ndata: {event.model_dump_json()}\n\n"  # type: ignore[attr-defined]


# ---- Run record ----------------------------------------------------------


class Run(BaseModel):
    """Immutable snapshot of a training run; updates produce a new snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    recipe: str
    created_at: datetime
    status: RunStatus = "pending"
    out_dir: Path
    pid: int | None = None
    last_step: int | None = None
    last_loss: float | None = None
    message: str = ""


# ---- Registry ------------------------------------------------------------


_RING_BUFFER_MAX = 200
_RECENT_RUNS_MAX = 20

# HF Trainer logs lines like:
#   {'loss': 2.3145, 'learning_rate': 0.0002, 'epoch': 0.05}
#   {'loss': 1.9831, 'learning_rate': 0.00018, 'epoch': 0.10, 'step': 50}
# We extract loss + lr; step is taken from a running counter when absent.
_LOSS_RE = re.compile(r"'loss'\s*:\s*([0-9.eE+\-]+)")
_LR_RE = re.compile(r"'learning_rate'\s*:\s*([0-9.eE+\-]+)")
_STEP_RE = re.compile(r"'step'\s*:\s*([0-9]+)")
_GRAD_RE = re.compile(r"'grad_norm'\s*:\s*([0-9.eE+\-]+)")


def parse_trainer_log_line(line: str, fallback_step: int) -> StepEvent | None:
    """Extract a StepEvent from an HF-Trainer-style log line, or None.

    `fallback_step` is used when the line lacks an explicit `'step'` key —
    callers maintain a running counter and pass `prev_step + 1`.
    """
    loss_match = _LOSS_RE.search(line)
    if not loss_match:
        return None
    try:
        loss = float(loss_match.group(1))
    except ValueError:
        return None
    lr = None
    lr_match = _LR_RE.search(line)
    if lr_match:
        with contextlib.suppress(ValueError):
            lr = float(lr_match.group(1))
    step = fallback_step
    step_match = _STEP_RE.search(line)
    if step_match:
        with contextlib.suppress(ValueError):
            step = int(step_match.group(1))
    grad = None
    grad_match = _GRAD_RE.search(line)
    if grad_match:
        with contextlib.suppress(ValueError):
            grad = float(grad_match.group(1))
    return StepEvent(run_id="", step=step, loss=loss, lr=lr, grad_norm=grad)


class _RunState:
    """Per-run mutable bookkeeping (process handle, subscribers, ring buffer)."""

    __slots__ = ("buffer", "energy_task", "process", "run", "seen_steps", "step_counter", "subscribers")

    def __init__(self, run: Run) -> None:
        self.run = run
        self.buffer: deque[_Event] = deque(maxlen=_RING_BUFFER_MAX)
        self.subscribers: set[asyncio.Queue[_Event | None]] = set()
        self.process: subprocess.Popen[str] | None = None
        self.energy_task: asyncio.Task[None] | None = None
        self.step_counter: int = 0
        self.seen_steps: set[int] = set()


class RunRegistry:
    """In-memory registry + SSE pub/sub.

    Thread-safe wrt subprocess line-reader threads via
    `loop.call_soon_threadsafe`; otherwise expects to be called from the
    asyncio event loop.
    """

    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}
        self._order: deque[str] = deque(maxlen=_RECENT_RUNS_MAX)
        self._lock = threading.Lock()  # only guards `_runs` insertions
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- lifecycle --------------------------------------------------------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the FastAPI event loop so threaded log readers can publish."""
        self._loop = loop

    def create(self, recipe: str, out_dir: Path) -> Run:
        run = Run(
            id=uuid.uuid4().hex[:12],
            recipe=recipe,
            created_at=datetime.now(UTC),
            out_dir=out_dir,
        )
        with self._lock:
            self._runs[run.id] = _RunState(run)
            self._order.append(run.id)
        return run

    def list_runs(self) -> list[Run]:
        with self._lock:
            return [self._runs[rid].run for rid in list(self._order) if rid in self._runs]

    def get(self, run_id: str) -> Run | None:
        state = self._runs.get(run_id)
        return state.run if state else None

    def attach_process(self, run_id: str, proc: subprocess.Popen[str]) -> None:
        state = self._runs.get(run_id)
        if state is not None:
            state.process = proc
            self._update(run_id, pid=proc.pid)

    def attach_energy_task(self, run_id: str, task: asyncio.Task[None]) -> None:
        state = self._runs.get(run_id)
        if state is not None:
            state.energy_task = task

    # -- publish/subscribe ------------------------------------------------

    def publish(self, run_id: str, event: _Event) -> None:
        """Fan out an event to all subscribers + append to the ring buffer.

        Safe to call from the event loop. Threads must dispatch via
        `loop.call_soon_threadsafe(registry.publish, run_id, event)`.
        """
        state = self._runs.get(run_id)
        if state is None:
            return

        # Dedup step/eval events on (run_id, step) so the subprocess-stdout
        # path and the in-process StreamCallback don't double-emit.
        if isinstance(event, (StepEvent, EvalEvent)):
            if event.step in state.seen_steps:
                return
            state.seen_steps.add(event.step)
            if isinstance(event, StepEvent):
                self._update(run_id, last_step=event.step, last_loss=event.loss)

        if isinstance(event, StatusEvent):
            self._update(run_id, status=event.status, message=event.message)

        state.buffer.append(event)
        for q in list(state.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish_threadsafe(self, run_id: str, event: _Event) -> None:
        """Thread-safe entry point used by the subprocess line-reader thread."""
        loop = self._loop
        if loop is None:
            self.publish(run_id, event)
            return
        loop.call_soon_threadsafe(self.publish, run_id, event)

    async def subscribe(
        self,
        run_id: str,
        kinds: tuple[str, ...] | None = None,
    ) -> AsyncIterator[_Event]:
        """Async iterator yielding events for a run, optionally filtered by kind.

        On connect, replays the last `_RING_BUFFER_MAX` buffered events so a
        late client doesn't miss the first frames. If the run is already in
        a terminal state when we connect (or reaches one mid-stream), the
        iterator returns after draining.
        """
        state = self._runs.get(run_id)
        if state is None:
            return

        terminal: set[RunStatus] = {"succeeded", "failed", "cancelled"}
        queue: asyncio.Queue[_Event | None] = asyncio.Queue()
        state.subscribers.add(queue)
        try:
            for ev in list(state.buffer):
                if kinds is None or ev.kind in kinds:  # type: ignore[attr-defined]
                    yield ev
            # Late subscriber: if the run already terminated, don't block.
            if state.run.status in terminal:
                return
            while True:
                ev = await queue.get()
                if ev is None:  # sentinel: run finished + no more events
                    return
                if kinds is None or ev.kind in kinds:  # type: ignore[attr-defined]
                    yield ev
        finally:
            state.subscribers.discard(queue)

    def close_subscribers(self, run_id: str) -> None:
        """Push a sentinel `None` to every subscriber so they unblock."""
        state = self._runs.get(run_id)
        if state is None:
            return
        for q in list(state.subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(None)

    # -- mutation ---------------------------------------------------------

    def _update(self, run_id: str, **fields: Any) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        state.run = state.run.model_copy(update=fields)

    # -- cancellation -----------------------------------------------------

    async def cancel(self, run_id: str, grace_s: float = 5.0) -> bool:
        """SIGINT, then SIGTERM after `grace_s`. Returns True if a process was signalled."""
        state = self._runs.get(run_id)
        if state is None or state.process is None:
            return False
        proc = state.process
        if proc.poll() is not None:
            return False
        try:
            proc.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError):
            return False
        await asyncio.sleep(grace_s)
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.send_signal(signal.SIGTERM)
        self.publish(run_id, StatusEvent(run_id=run_id, status="cancelled", message="cancel requested"))
        self.close_subscribers(run_id)
        return True


# ---- Subprocess spawn helper --------------------------------------------


def spawn_subprocess_streaming(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    run_id: str,
    registry: RunRegistry,
    on_done: Callable[[int], None] | None = None,
) -> subprocess.Popen[str]:
    """Spawn `cmd` and tee each stdout line to `log_path` + `registry.publish_threadsafe`.

    The line reader runs in a daemon thread so the FastAPI event loop is
    never blocked. HF Trainer log lines are parsed into `StepEvent`s; all
    other lines become `LogEvent`s.

    Returns the `Popen` object for the caller to attach + cancel later.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", buffering=1)
    log_file.write(f"# cmd: {' '.join(cmd)}\n\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    def _reader() -> None:
        step_ctr = 0
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                log_file.write(raw)
                log_file.flush()
                step_ev = parse_trainer_log_line(line, fallback_step=step_ctr + 1)
                if step_ev is not None:
                    step_ctr = step_ev.step
                    registry.publish_threadsafe(
                        run_id, step_ev.model_copy(update={"run_id": run_id})
                    )
                else:
                    registry.publish_threadsafe(
                        run_id, LogEvent(run_id=run_id, line=line, level="stdout")
                    )
        finally:
            rc = proc.wait()
            log_file.close()
            status: RunStatus = "succeeded" if rc == 0 else "failed"
            registry.publish_threadsafe(
                run_id,
                StatusEvent(run_id=run_id, status=status, message=f"exit={rc}"),
            )
            registry.close_subscribers(run_id)
            if on_done is not None:
                with contextlib.suppress(Exception):
                    on_done(rc)

    threading.Thread(target=_reader, daemon=True, name=f"runs-{run_id}").start()
    registry.attach_process(run_id, proc)
    registry.publish_threadsafe(
        run_id, StatusEvent(run_id=run_id, status="running", message=f"pid={proc.pid}")
    )
    return proc


# ---- Energy sampler -----------------------------------------------------


async def energy_loop(run_id: str, registry: RunRegistry, interval_s: float = 1.0) -> None:
    """Sample GPU power every `interval_s` and publish EnergyEvents.

    Uses `mindxtrain.operator.telemetry.energy.sample_power_w()` if available,
    else degrades to 0.0. Stops when the run reaches a terminal status.
    """
    try:
        from mindxtrain.operator.telemetry.energy import sample_power_w
    except ImportError:
        def sample_power_w(_gpu: int = 0) -> float:  # type: ignore[no-redef]
            return 0.0

    terminal: set[RunStatus] = {"succeeded", "failed", "cancelled"}
    while True:
        run = registry.get(run_id)
        if run is None or run.status in terminal:
            return
        try:
            watts = float(sample_power_w(0))
        except Exception:
            watts = 0.0
        registry.publish(run_id, EnergyEvent(run_id=run_id, watts=watts, gpu_index=0))
        await asyncio.sleep(interval_s)


# ---- Single-process default registry ------------------------------------


_DEFAULT_REGISTRY: RunRegistry | None = None


def default_registry() -> RunRegistry:
    """Return the process-wide singleton registry (lazily constructed)."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = RunRegistry()
    return _DEFAULT_REGISTRY


def reset_default_registry() -> None:
    """Test helper: drop the singleton so each test starts clean."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None


# ---- Loopback guard for /ingest -----------------------------------------


def is_loopback(host: str | None) -> bool:
    """True iff `host` is a loopback address.

    Accepts the strings FastAPI's `request.client.host` returns ('127.0.0.1',
    '::1', 'localhost', 'testclient' for the in-process TestClient).
    """
    if host is None:
        return False
    if host in ("127.0.0.1", "::1", "localhost", "testclient"):
        return True
    return host.startswith("127.")


__all__ = [
    "EnergyEvent",
    "EvalEvent",
    "LogEvent",
    "Run",
    "RunRegistry",
    "RunStatus",
    "StatusEvent",
    "StepEvent",
    "TrainEvent",
    "default_registry",
    "energy_loop",
    "format_sse",
    "is_loopback",
    "parse_trainer_log_line",
    "reset_default_registry",
    "spawn_subprocess_streaming",
]
