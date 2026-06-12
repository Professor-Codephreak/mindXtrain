"""Public /v1/training/jobs API — versioned surface for external callers.

mindX agents and any other client (CLI scripts, other services) dispatch
training through this API. It's a thin facade over the same `RunRegistry`
the Coach UI uses, so a job_id IS a run_id — both UIs see the same
in-memory state.

Two reasons for a separate router under `/v1/`:

1. **Stability contract.** Coach endpoints under `/coach/api/runs/*` are
   internal and may change between minor releases. The `/v1/training/jobs`
   surface is the one external callers should pin to.
2. **Auth.** Coach is intended for the operator's own host (often behind a
   reverse proxy); `/v1` accepts requests from arbitrary clients and gates
   them on a bearer token when `MINDXTRAIN_API_KEY` is set in env.

Body for POST /v1/training/jobs accepts one of (mutually exclusive):

- `recipe`: name of a built-in recipe (`mindxtrain init --list`).
- `config_yaml`: raw YAML of an `XTrainConfig`.
- `config`: parsed JSON of an `XTrainConfig`.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mindxtrain.autotune.benchmark import run_autotune
from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.loader import list_recipes, render_recipe
from mindxtrain.config.schema import XTrainConfig
from mindxtrain.operator import runs as _runs

router = APIRouter(prefix="/v1/training", tags=["training"])

_REGISTRY = _runs.default_registry()


# ---- auth dependency -------------------------------------------------------


def _bearer(authorization: str | None = Header(default=None)) -> None:
    """Enforce `Authorization: Bearer <MINDXTRAIN_API_KEY>` if the env var is set.

    Unset key = open in dev mode. Set key = strict comparison. Use 401 for
    missing/wrong tokens (not 403) so client SDKs can prompt for a key.
    """
    expected = os.environ.get("MINDXTRAIN_API_KEY", "").strip()
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    presented = authorization[len("Bearer "):].strip()
    if presented != expected:
        raise HTTPException(status_code=401, detail="invalid bearer token")


# ---- request/response models ----------------------------------------------


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe: str | None = Field(default=None, description="Built-in recipe name.")
    config_yaml: str | None = Field(default=None, description="Raw YAML body of an XTrainConfig.")
    config: dict[str, Any] | None = Field(default=None, description="Parsed XTrainConfig JSON.")
    out_dir: str | None = Field(default=None, description="Optional override for the run output directory.")
    settlement_tx: str | None = Field(
        default=None,
        description="Algorand USDC settlement tx id, required when x402 metering is enabled.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> CreateJobRequest:
        provided = [bool(self.recipe), bool(self.config_yaml), bool(self.config)]
        if sum(provided) != 1:
            msg = "exactly one of `recipe`, `config_yaml`, `config` is required"
            raise ValueError(msg)
        return self


class JobInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: _runs.RunStatus
    recipe: str
    out_dir: str
    created_at: str
    backend: str
    base_model: str
    manifest_path: str | None = None

    @classmethod
    def from_run(cls, run: _runs.Run, cfg: XTrainConfig) -> JobInfo:
        manifest = run.out_dir / "manifest.json"
        return cls(
            job_id=run.id,
            status=run.status,
            recipe=run.recipe,
            out_dir=str(run.out_dir),
            created_at=run.created_at.isoformat(),
            backend=cfg.train.backend,
            base_model=cfg.model.name,
            manifest_path=str(manifest) if manifest.exists() else None,
        )


# ---- helpers ---------------------------------------------------------------


def _resolve_config(req: CreateJobRequest) -> tuple[str, XTrainConfig]:
    """Turn a CreateJobRequest into (recipe_label, parsed XTrainConfig)."""
    if req.recipe is not None:
        if req.recipe not in list_recipes():
            raise HTTPException(status_code=404, detail=f"unknown recipe {req.recipe!r}")
        cfg = XTrainConfig.model_validate(yaml.safe_load(render_recipe(req.recipe)))
        return req.recipe, cfg
    if req.config_yaml is not None:
        try:
            cfg = XTrainConfig.model_validate(yaml.safe_load(req.config_yaml))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"config_yaml invalid: {exc}") from exc
        return f"adhoc:{cfg.meta.run_name}", cfg
    assert req.config is not None
    try:
        cfg = XTrainConfig.model_validate(req.config)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"config invalid: {exc}") from exc
    return f"adhoc:{cfg.meta.run_name}", cfg


def _spawn_for_backend(run: _runs.Run, cfg: XTrainConfig, plan: AutotunePlan) -> None:
    """Route the launch based on `cfg.train.backend`.

    - `trl_cpu` runs in-process on a daemon thread (no subprocess); status
      events are published to the registry from the thread.
    - Anything else falls through to the Axolotl-style prepare_run +
      subprocess streamer (the same code path Coach uses).
    """
    if cfg.train.backend in ("trl_cpu", "trl_local"):
        _spawn_inprocess_cpu(run, cfg, plan)
        return

    from mindxtrain.train.sft import prepare_run

    prepared = prepare_run(cfg, plan, run.out_dir)
    _runs.spawn_subprocess_streaming(
        cmd=prepared.cmd,
        env=prepared.env,
        log_path=prepared.log_path,
        run_id=run.id,
        registry=_REGISTRY,
    )


def _spawn_inprocess_cpu(run: _runs.Run, cfg: XTrainConfig, plan: AutotunePlan) -> None:
    """Daemon-thread launcher for the in-process TRL lanes (`trl_cpu`/`trl_local`).

    Both lanes run in-process and synchronously; we wrap them in a thread so the
    FastAPI handler returns immediately. `trl_local` auto-detects a local GPU
    (else CPU fallback); `trl_cpu` pins CPU. Log lines are forwarded as
    `LogEvent`s; final status is `succeeded`/`failed`.
    """
    from mindxtrain.train.backend_trl_cpu import run_trl_cpu, run_trl_local

    runner = run_trl_local if cfg.train.backend == "trl_local" else run_trl_cpu
    lane = cfg.train.backend

    def _on_line(line: str) -> None:
        _REGISTRY.publish_threadsafe(
            run.id, _runs.LogEvent(run_id=run.id, line=line, level="stdout"),
        )

    def _thread() -> None:
        _REGISTRY.publish_threadsafe(
            run.id, _runs.StatusEvent(run_id=run.id, status="running", message=f"{lane} lane"),
        )
        try:
            runner(cfg, plan, run.out_dir, on_line=_on_line)
        except Exception as exc:
            _REGISTRY.publish_threadsafe(
                run.id,
                _runs.StatusEvent(run_id=run.id, status="failed", message=str(exc)),
            )
            _REGISTRY.close_subscribers(run.id)
            return
        from mindxtrain.operator.receipt_emit import emit_run_receipt
        emit_run_receipt(_REGISTRY, run, cfg, plan)
        _REGISTRY.publish_threadsafe(
            run.id, _runs.StatusEvent(run_id=run.id, status="succeeded", message=f"{lane} lane done"),
        )
        _REGISTRY.close_subscribers(run.id)

    threading.Thread(target=_thread, daemon=True, name=f"{lane}-{run.id}").start()


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


# ---- endpoints -------------------------------------------------------------


def _x402_required() -> bool:
    """Whether to gate training jobs behind an x402 USDC settlement.

    Off by default. Set `MINDXTRAIN_X402_REQUIRED` to a truthy value to require
    payment. This is a thin stub: it issues an invoice and verifies an Algorand
    USDC settlement, but does NOT submit the on-chain `recordSettlement` proof to
    the x402_receiver contract — that facilitator half is post-hackathon work.
    """
    return os.environ.get("MINDXTRAIN_X402_REQUIRED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _x402_price_usdc() -> float:
    try:
        return float(os.environ.get("MINDXTRAIN_X402_PRICE_USDC", "1.0"))
    except ValueError:
        return 1.0


def _enforce_x402(req: CreateJobRequest, recipe_label: str) -> None:
    """Raise 402 with an invoice when payment is required but unsettled.

    When a settlement tx is supplied, verify it on Algorand and proceed only if
    confirmed. Verifying needs `--extra chain` (algosdk); the unpaid 402 path
    does not (the invoice is constructed locally).
    """
    if not _x402_required():
        return

    from mindxtrain.provenance.x402 import Invoice, validate_settlement

    price = _x402_price_usdc()
    receiver = os.environ.get("MINDXTRAIN_X402_RECEIVER", "")

    if not req.settlement_tx:
        invoice = Invoice(
            invoice_id=f"job-{recipe_label}",
            run_id=recipe_label,
            amount_usdc=price,
            receiver=receiver,
            pay_url=os.environ.get("MINDXTRAIN_FACILITATOR_URL", ""),
        )
        raise HTTPException(
            status_code=402,
            detail={"error": "payment required", "invoice": invoice.model_dump()},
        )

    settlement = validate_settlement(
        req.settlement_tx,
        expected_amount_usdc=price,
        expected_receiver=receiver or None,
    )
    if not settlement.confirmed:
        raise HTTPException(
            status_code=402,
            detail={"error": "settlement not confirmed", "tx_id": req.settlement_tx},
        )


@router.post("/jobs", response_model=JobInfo, dependencies=[Depends(_bearer)])
async def create_job(req: CreateJobRequest) -> JobInfo:
    recipe_label, cfg = _resolve_config(req)
    _enforce_x402(req, recipe_label)
    plan = run_autotune(dry_run=True)
    out_dir = Path(req.out_dir) if req.out_dir else Path("./out/runs") / cfg.meta.run_name

    run = _REGISTRY.create(recipe_label, out_dir)
    _REGISTRY.attach_loop(asyncio.get_running_loop())
    _REGISTRY.publish(run.id, _runs.StatusEvent(run_id=run.id, status="pending", message="launching"))

    try:
        _spawn_for_backend(run, cfg, plan)
    except RuntimeError as exc:
        _REGISTRY.publish(
            run.id,
            _runs.StatusEvent(run_id=run.id, status="failed", message=str(exc)),
        )
        _REGISTRY.close_subscribers(run.id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    snap = _REGISTRY.get(run.id)
    assert snap is not None
    return JobInfo.from_run(snap, cfg)


@router.get("/jobs", response_model=list[JobInfo], dependencies=[Depends(_bearer)])
async def list_jobs() -> list[JobInfo]:
    out: list[JobInfo] = []
    for run in _REGISTRY.list_runs():
        cfg = _try_load_cfg_for_recipe(run.recipe)
        if cfg is None:
            continue
        out.append(JobInfo.from_run(run, cfg))
    return out


@router.get("/jobs/{job_id}", response_model=JobInfo, dependencies=[Depends(_bearer)])
async def get_job(job_id: str) -> JobInfo:
    snap = _REGISTRY.get(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    cfg = _try_load_cfg_for_recipe(snap.recipe)
    if cfg is None:
        raise HTTPException(status_code=500, detail="job recipe no longer resolvable")
    return JobInfo.from_run(snap, cfg)


@router.get("/jobs/{job_id}/events", dependencies=[Depends(_bearer)])
async def stream_job_events(job_id: str) -> StreamingResponse:
    if _REGISTRY.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")

    async def _stream() -> AsyncIterator[str]:
        async for event in _REGISTRY.subscribe(job_id, kinds=None):
            yield _runs.format_sse(event)

    return StreamingResponse(_stream(), media_type="text/event-stream", headers=_sse_headers())


@router.post("/jobs/{job_id}/cancel", dependencies=[Depends(_bearer)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    if _REGISTRY.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    cancelled = await _REGISTRY.cancel(job_id, grace_s=2.0)
    return {"job_id": job_id, "cancelled": cancelled}


def _try_load_cfg_for_recipe(recipe: str) -> XTrainConfig | None:
    """Best-effort cfg resolver for read endpoints (handles adhoc + built-in)."""
    if recipe.startswith("adhoc:"):
        # Adhoc configs aren't persisted yet — return a stub-shaped placeholder.
        # The job_id + status are still meaningful; backend/base_model are unknown.
        return None
    if recipe not in list_recipes():
        return None
    try:
        return XTrainConfig.model_validate(yaml.safe_load(render_recipe(recipe)))
    except Exception:
        return None


__all__ = ["CreateJobRequest", "JobInfo", "router"]
