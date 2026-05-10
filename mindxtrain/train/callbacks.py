"""Training callbacks — eval-during-training + checkpoint mgmt + UI stream.

Subclasses `transformers.TrainerCallback` (lazy import). Returned as
configuration objects whose `.callback()` method materializes the
TrainerCallback when the trainer actually constructs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict


def _ensure_transformers() -> Any:
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError as exc:
        msg = "transformers not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc


class EvalDuringTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")

    every_n_steps: int = 200
    suite: Literal["mmlu", "gsm8k", "bfcl"] = "mmlu"

    def callback(self, eval_fn: Callable[[int], dict[str, float]]) -> Any:
        TrainerCallback = _ensure_transformers()
        every = self.every_n_steps

        class _CB(TrainerCallback):  # type: ignore[misc, valid-type]
            def on_step_end(self, args: Any, state: Any, control: Any, **_kw: Any) -> None:
                if state.global_step % every == 0 and state.global_step > 0:
                    metrics = eval_fn(state.global_step)
                    for k, v in metrics.items():
                        state.log_history.append({"step": state.global_step, k: v})

        return _CB()


class BestCheckpointKeeper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: Path
    metric: str = "eval_loss"
    keep: int = 3
    minimize: bool = True

    def callback(self) -> Any:
        TrainerCallback = _ensure_transformers()
        out_dir = self.out_dir
        metric = self.metric
        keep = self.keep
        minimize = self.minimize

        class _CB(TrainerCallback):  # type: ignore[misc, valid-type]
            def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, float] | None = None, **_kw: Any) -> None:
                if not metrics or metric not in metrics:
                    return
                # Naive top-k: keep `keep` checkpoints with the best metric.
                ckpts: list[tuple[float, Path]] = []
                for p in sorted(out_dir.glob("checkpoint-*")):
                    log = p / "trainer_state.json"
                    if not log.exists():
                        continue
                    # Load the most recent metric value for this checkpoint.
                    try:
                        import json

                        st = json.loads(log.read_text())
                        last = next(
                            (h.get(metric) for h in reversed(st.get("log_history", [])) if metric in h),
                            None,
                        )
                        if last is None:
                            continue
                        ckpts.append((float(last), p))
                    except (OSError, json.JSONDecodeError, ValueError):
                        continue
                ckpts.sort(reverse=not minimize)
                for _, path in ckpts[keep:]:
                    import shutil

                    shutil.rmtree(path, ignore_errors=True)

        return _CB()


class StreamCallback(BaseModel):
    """Push step + eval events to the operator's loopback ingest endpoint.

    Lives next to the other two callbacks; like them, the actual
    `TrainerCallback` subclass is materialized lazily so this module
    imports without `--extra ml`. The ingest endpoint is bound to
    127.0.0.1 by `mindxtrain.operator.runs.is_loopback`, which is why
    the default `sink_url` host is loopback and not configurable beyond it.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    sink_url: str = "http://127.0.0.1:8080/coach/api/runs/{run_id}/ingest"
    timeout_s: float = 2.0
    suite: Literal["mmlu", "gsm8k", "bfcl"] = "mmlu"

    def _post(self, event: dict[str, Any]) -> None:
        url = self.sink_url.format(run_id=self.run_id)
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                client.post(url, json=event)
        except httpx.HTTPError:
            # Best-effort: a failed ingest never blocks training.
            pass

    def callback(self) -> Any:
        TrainerCallback = _ensure_transformers()
        run_id = self.run_id
        post = self._post
        suite = self.suite

        class _CB(TrainerCallback):  # type: ignore[misc, valid-type]
            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: dict[str, float] | None = None,
                **_kw: Any,
            ) -> None:
                if not logs or "loss" not in logs:
                    return
                post(
                    {
                        "kind": "step",
                        "run_id": run_id,
                        "step": int(state.global_step),
                        "loss": float(logs["loss"]),
                        "lr": float(logs["learning_rate"]) if "learning_rate" in logs else None,
                        "grad_norm": float(logs["grad_norm"]) if "grad_norm" in logs else None,
                        "tokens_per_s": None,
                    }
                )

            def on_evaluate(
                self,
                args: Any,
                state: Any,
                control: Any,
                metrics: dict[str, float] | None = None,
                **_kw: Any,
            ) -> None:
                if not metrics:
                    return
                clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                if not clean:
                    return
                post(
                    {
                        "kind": "eval",
                        "run_id": run_id,
                        "step": int(state.global_step),
                        "suite": suite,
                        "metrics": clean,
                    }
                )

            def on_train_end(self, args: Any, state: Any, control: Any, **_kw: Any) -> None:
                post(
                    {
                        "kind": "status",
                        "run_id": run_id,
                        "status": "succeeded",
                        "message": f"step={state.global_step}",
                    }
                )

        return _CB()


def eval_during_training(every_n_steps: int = 200, suite: Literal["mmlu", "gsm8k", "bfcl"] = "mmlu") -> EvalDuringTraining:
    return EvalDuringTraining(every_n_steps=every_n_steps, suite=suite)


def best_checkpoint_keeper(out_dir: Path, metric: str = "eval_loss", keep: int = 3) -> BestCheckpointKeeper:
    return BestCheckpointKeeper(out_dir=out_dir, metric=metric, keep=keep)


def stream_callback(run_id: str, sink_url: str | None = None) -> StreamCallback:
    if sink_url is None:
        return StreamCallback(run_id=run_id)
    return StreamCallback(run_id=run_id, sink_url=sink_url)


__all__ = [
    "BestCheckpointKeeper",
    "EvalDuringTraining",
    "StreamCallback",
    "best_checkpoint_keeper",
    "eval_during_training",
    "stream_callback",
]
