"""mindXtrain Coach — FastAPI router.

Surfaces the differentiating pieces of the pipeline (recipes, autotune,
Axolotl compilation, cost vs H100) behind tiny JSON endpoints the static
HTML/JS UI consumes.

The /api/runs/* routes own the live training-feedback loop. Events are
pushed to the browser via Server-Sent Events; see `mindxtrain.operator.runs`
for the registry + event schema.

The /api/{github,droplet}/* routes share the same SSE pipeline by creating
synthetic Runs with reserved recipe names (`_github_push`,
`_droplet_provision`, `_droplet_sync`) and chaining shell-out steps via
`mindxtrain.deploy._orchestrator`.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from mindxtrain.autotune.benchmark import run_autotune
from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.budget.pricing import MI300X_USDC_PER_HOUR, gpu_hour_price
from mindxtrain.config.loader import list_recipes, render_recipe
from mindxtrain.config.schema import XTrainConfig
from mindxtrain.deploy import (
    amd_dev_cloud as _adc,
)
from mindxtrain.deploy import (
    droplet as _droplet_mod,
)
from mindxtrain.deploy import (
    github_push as _gh,
)
from mindxtrain.deploy._orchestrator import (
    droplet_provision_pipeline,
    droplet_sync_pipeline,
    github_push_pipeline,
)
from mindxtrain.operator import runs as _runs
from mindxtrain.train import compile_axolotl_yaml

router = APIRouter(prefix="/coach", tags=["coach"])

_STATIC_DIR = Path(__file__).parent / "static"
_REGISTRY = _runs.default_registry()

# Reference H100 cost numbers used in the cost slide. Lifted from
# docs/benchmarks.md so the UI shows the same comparison the README does.
H100_USDC_PER_HOUR = 4.00
H200_USDC_PER_HOUR = 6.00


class RecipeSummary(BaseModel):
    name: str
    base_model: str
    method: str
    gpus: int
    description: str


class RecipeDetail(BaseModel):
    name: str
    yaml: str
    summary: RecipeSummary


class CompileRequest(BaseModel):
    recipe: str = Field(description="recipe name, e.g. qwen3_8b_sft_lora")
    plan: AutotunePlan | None = None


class CompileResponse(BaseModel):
    recipe: str
    config_summary: RecipeSummary
    plan: AutotunePlan
    axolotl_yaml: dict[str, Any]
    overrides: list[str]


class CostRequest(BaseModel):
    gpus: int = Field(default=1, ge=1, le=64)
    hours: float = Field(default=1.5, gt=0.0, le=720.0)
    safety_margin: float = Field(default=1.15, ge=1.0, le=2.0)


class CostBreakdown(BaseModel):
    name: str
    rate_usdc_per_hour: float
    gpus: int
    cost_usdc: float
    fits_qwen3_8b_bf16_bs8_seq4096: bool
    note: str


class CostResponse(BaseModel):
    hours: float
    safety_margin: float
    mi300x: CostBreakdown
    h100: CostBreakdown
    h200: CostBreakdown
    speedup_vs_h100_x: float


class CoachHealthResponse(BaseModel):
    coach_version: str = "0.1.0"
    chat_backend_ready: bool = False
    chat_backend_name: str = ""
    recipes_available: int


def _summarize(cfg: XTrainConfig, name: str) -> RecipeSummary:
    method = cfg.train.method.kind
    desc = cfg.meta.description or f"{method.upper()} of {cfg.model.name} on {cfg.data.hf_id}."
    return RecipeSummary(
        name=name,
        base_model=cfg.model.name,
        method=method,
        gpus=cfg.hardware.gpus,
        description=desc,
    )


# ---- routes ---------------------------------------------------------------

@router.get("/", response_class=FileResponse, include_in_schema=False)
async def coach_index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@router.get("/api/recipes", response_model=list[RecipeSummary])
async def api_recipes() -> list[RecipeSummary]:
    out: list[RecipeSummary] = []
    for name in list_recipes():
        cfg = XTrainConfig.model_validate(yaml.safe_load(render_recipe(name)))
        out.append(_summarize(cfg, name))
    return out


@router.get("/api/recipes/{name}", response_model=RecipeDetail)
async def api_recipe(name: str) -> RecipeDetail:
    if name not in list_recipes():
        raise HTTPException(status_code=404, detail=f"unknown recipe {name!r}")
    yaml_text = render_recipe(name)
    cfg = XTrainConfig.model_validate(yaml.safe_load(yaml_text))
    return RecipeDetail(name=name, yaml=yaml_text, summary=_summarize(cfg, name))


@router.post("/api/bench", response_model=AutotunePlan)
async def api_bench() -> AutotunePlan:
    """Day-1 dry-run; Day-2 swap to GPU-backed `run_autotune(dry_run=False)`."""
    return run_autotune(dry_run=True)


@router.post("/api/compile", response_model=CompileResponse)
async def api_compile(req: CompileRequest) -> CompileResponse:
    if req.recipe not in list_recipes():
        raise HTTPException(status_code=404, detail=f"unknown recipe {req.recipe!r}")
    cfg = XTrainConfig.model_validate(yaml.safe_load(render_recipe(req.recipe)))
    plan = req.plan or run_autotune(dry_run=True)
    axolotl_yaml = compile_axolotl_yaml(cfg, plan)
    from mindxtrain.train.axolotl_compile import autotune_overrides_summary

    return CompileResponse(
        recipe=req.recipe,
        config_summary=_summarize(cfg, req.recipe),
        plan=plan,
        axolotl_yaml=axolotl_yaml,
        overrides=autotune_overrides_summary(plan),
    )


@router.post("/api/cost", response_model=CostResponse)
async def api_cost(req: CostRequest) -> CostResponse:
    mi300x_cost = gpu_hour_price(gpus=req.gpus, hours=req.hours, safety_margin=req.safety_margin)
    # H100 OOMs at Qwen3-8B BF16 bs=8 seq=4096, must use 2 GPUs to fit.
    h100_gpus = max(2, req.gpus * 2)
    h100_cost = h100_gpus * req.hours * H100_USDC_PER_HOUR * req.safety_margin
    h200_cost = req.gpus * req.hours * H200_USDC_PER_HOUR * req.safety_margin

    return CostResponse(
        hours=req.hours,
        safety_margin=req.safety_margin,
        mi300x=CostBreakdown(
            name="MI300X (192 GB HBM3)",
            rate_usdc_per_hour=MI300X_USDC_PER_HOUR,
            gpus=req.gpus,
            cost_usdc=round(mi300x_cost, 2),
            fits_qwen3_8b_bf16_bs8_seq4096=True,
            note="Fits unquantized with massive headroom.",
        ),
        h100=CostBreakdown(
            name="H100 (80 GB HBM3)",
            rate_usdc_per_hour=H100_USDC_PER_HOUR,
            gpus=h100_gpus,
            cost_usdc=round(h100_cost, 2),
            fits_qwen3_8b_bf16_bs8_seq4096=False,
            note=f"OOMs at this bs/seq; needs {h100_gpus}x cards or FP8 fallback.",
        ),
        h200=CostBreakdown(
            name="H200 (141 GB HBM3e)",
            rate_usdc_per_hour=H200_USDC_PER_HOUR,
            gpus=req.gpus,
            cost_usdc=round(h200_cost, 2),
            fits_qwen3_8b_bf16_bs8_seq4096=True,
            note="Fits with less headroom than MI300X.",
        ),
        speedup_vs_h100_x=round(h100_cost / mi300x_cost, 2) if mi300x_cost > 0 else 0.0,
    )


@router.get("/api/health", response_model=CoachHealthResponse)
async def api_health() -> CoachHealthResponse:
    """Coach health — does NOT require the chat backend to be live."""
    import os

    backend = os.environ.get("AUTOMINDX_BACKEND", "")
    return CoachHealthResponse(
        chat_backend_ready=False,  # Day-5 wiring flips this when vLLM is reachable.
        chat_backend_name=backend,
        recipes_available=len(list_recipes()),
    )


# ---- preflight + dream-corpus (training-run launch gate) ----------------

# Env vars the Coach UI surfaces as a preflight gate before kicking off a
# production training run. Required = the run will fail without them.
# Optional = the run still works but post-train steps (publish to HF Hub,
# Lighthouse pin, mindX fallback swap) silently no-op.
_PREFLIGHT_REQUIRED = (
    "AMD_DEV_CLOUD_TOKEN",
    "AMD_DEV_CLOUD_SSH_KEY_ID",
    "HF_TOKEN",
    "HF_HUB_USERNAME",
)
_PREFLIGHT_OPTIONAL = (
    "MINDXTRAIN_API_KEY",
    "MINDXTRAIN_MINDX_HOME",
    "LIGHTHOUSE_API_KEY",
)


class PreflightResponse(BaseModel):
    """Per-env-var presence (no values exposed) + readiness summary."""

    vars: dict[str, bool] = Field(
        description="Which env vars are present (True) or unset (False).",
    )
    required: list[str] = Field(description="Subset of vars considered required.")
    optional: list[str] = Field(description="Subset of vars considered optional.")
    required_missing: list[str] = Field(
        description="Required vars currently unset — the run is gated until these are populated.",
    )
    ready: bool = Field(description="True iff required_missing is empty.")


class DreamCorpusResponse(BaseModel):
    """Sanity check that mindX's dream-cycle JSONL corpus is reachable."""

    root: str = Field(description="Filesystem root inspected.")
    exists: bool
    files: int = 0
    raw_lines: int = 0
    unique_rows: int = 0
    ready: bool = Field(description="True iff exists and unique_rows > 0.")
    note: str | None = Field(
        default=None,
        description="Friendly error message when the path is missing or empty.",
    )


@router.get("/api/preflight", response_model=PreflightResponse)
async def api_preflight() -> PreflightResponse:
    """Report which env vars the launch flow needs, without exposing values.

    Used by the Coach UI's first step card to gate the training-run launch.
    Returns `ready=False` when any required var is unset so the UI can halt
    the auto-advance flow and prompt the operator to populate `.env`.
    """
    all_vars = list(_PREFLIGHT_REQUIRED) + list(_PREFLIGHT_OPTIONAL)
    vars_present = {name: bool(os.environ.get(name, "").strip()) for name in all_vars}
    required_missing = [n for n in _PREFLIGHT_REQUIRED if not vars_present[n]]
    return PreflightResponse(
        vars=vars_present,
        required=list(_PREFLIGHT_REQUIRED),
        optional=list(_PREFLIGHT_OPTIONAL),
        required_missing=required_missing,
        ready=not required_missing,
    )


@router.get("/api/dream-corpus", response_model=DreamCorpusResponse)
async def api_dream_corpus(root: str | None = None) -> DreamCorpusResponse:
    """Stats for the mindX dream-cycle JSONL corpus the recipe will consume.

    Resolution order for the corpus root:
    1. Explicit `?root=` query arg.
    2. `$MINDXTRAIN_MINDX_HOME/data/memory` if the env var is set.
    3. `/home/hacker/mindX/data/memory` (the documented default).

    Returns `ready=False` with a `note` if the path doesn't exist or has no
    unique rows yet (e.g. a fresh mindX install before its first dream cycle).
    """
    from mindxtrain.data.sources.mindx_dreams import count_mindx_dreams

    if root is not None:
        corpus_root = Path(root).expanduser()
    else:
        home = os.environ.get("MINDXTRAIN_MINDX_HOME", "/home/hacker/mindX")
        corpus_root = Path(home).expanduser() / "data" / "memory"

    if not corpus_root.exists():
        return DreamCorpusResponse(
            root=str(corpus_root),
            exists=False,
            ready=False,
            note=(
                f"corpus root not found: {corpus_root}. "
                "Set MINDXTRAIN_MINDX_HOME or pass ?root=… to point at the "
                "mindX data/memory directory."
            ),
        )

    stats = count_mindx_dreams(corpus_root)
    unique = stats["unique_rows"]
    ready = unique > 0
    note = (
        None
        if ready
        else (
            "corpus root exists but contains no dream JSONL — run a dream "
            "cycle in mindX (agents/machine_dreaming.py) before training."
        )
    )
    return DreamCorpusResponse(
        root=str(corpus_root),
        exists=True,
        files=stats["files"],
        raw_lines=stats["raw_lines"],
        unique_rows=unique,
        ready=ready,
        note=note,
    )


# ---- live training runs (SSE) -------------------------------------------


class LaunchRequest(BaseModel):
    recipe: str = Field(description="recipe name, e.g. qwen3_8b_sft_lora")
    plan: AutotunePlan | None = None
    out_dir: str | None = Field(
        default=None,
        description="optional override for the run output directory",
    )


SpawnFn = Callable[["_runs.Run", XTrainConfig, AutotunePlan], None]


def _real_spawn(run: _runs.Run, cfg: XTrainConfig, plan: AutotunePlan) -> None:
    """Default spawn: compile Axolotl YAML and stream the subprocess.

    Tests monkey-patch the module-level `_SPAWN` to bypass the real
    subprocess and emit canned events instead.
    """
    from mindxtrain.train.sft import prepare_run

    prepared = prepare_run(cfg, plan, run.out_dir)
    _runs.spawn_subprocess_streaming(
        cmd=prepared.cmd,
        env=prepared.env,
        log_path=prepared.log_path,
        run_id=run.id,
        registry=_REGISTRY,
    )


_SPAWN: SpawnFn = _real_spawn


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }


@router.post("/api/runs/launch", response_model=_runs.Run)
async def api_runs_launch(req: LaunchRequest) -> _runs.Run:
    """Spawn a training run and return its `Run` snapshot immediately.

    Does not block on the subprocess — the spawn helper attaches a
    background line-reader thread that publishes events into the registry.
    """
    if req.recipe not in list_recipes():
        raise HTTPException(status_code=404, detail=f"unknown recipe {req.recipe!r}")
    cfg = XTrainConfig.model_validate(yaml.safe_load(render_recipe(req.recipe)))
    plan = req.plan or run_autotune(dry_run=True)

    out_dir = Path(req.out_dir) if req.out_dir else Path("./out/runs") / cfg.meta.run_name
    run = _REGISTRY.create(req.recipe, out_dir)
    _REGISTRY.attach_loop(asyncio.get_running_loop())
    _REGISTRY.publish(run.id, _runs.StatusEvent(run_id=run.id, status="pending", message="launching"))

    try:
        _SPAWN(run, cfg, plan)
    except RuntimeError as exc:
        # Most common cause: `accelerate` not on PATH (no --extra ml).
        # Surface as a 503 + emit a failure event so any subscriber sees it.
        _REGISTRY.publish(
            run.id,
            _runs.StatusEvent(run_id=run.id, status="failed", message=str(exc)),
        )
        _REGISTRY.close_subscribers(run.id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    snapshot = _REGISTRY.get(run.id)
    assert snapshot is not None
    return snapshot


@router.get("/api/runs", response_model=list[_runs.Run])
async def api_runs_list() -> list[_runs.Run]:
    return _REGISTRY.list_runs()


@router.get("/api/runs/{run_id}", response_model=_runs.Run)
async def api_run_get(run_id: str) -> _runs.Run:
    snap = _REGISTRY.get(run_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return snap


async def _stream(run_id: str, kinds: tuple[str, ...] | None) -> AsyncIterator[str]:
    if _REGISTRY.get(run_id) is None:
        # Yield a single error frame and close.
        yield "event: error\ndata: {\"detail\":\"unknown run\"}\n\n"
        return
    async for event in _REGISTRY.subscribe(run_id, kinds=kinds):
        yield _runs.format_sse(event)


@router.get("/api/runs/{run_id}/events")
async def api_run_events(run_id: str) -> StreamingResponse:
    return StreamingResponse(
        _stream(run_id, kinds=None),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@router.get("/api/runs/{run_id}/logs")
async def api_run_logs(run_id: str) -> StreamingResponse:
    return StreamingResponse(
        _stream(run_id, kinds=("log",)),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@router.post("/api/runs/{run_id}/cancel")
async def api_run_cancel(run_id: str) -> dict[str, Any]:
    if _REGISTRY.get(run_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    cancelled = await _REGISTRY.cancel(run_id, grace_s=2.0)
    return {"run_id": run_id, "cancelled": cancelled}


@router.post("/api/runs/{run_id}/ingest")
async def api_run_ingest(run_id: str, request: Request) -> dict[str, str]:
    """Loopback-only ingest used by the in-process StreamCallback."""
    host = request.client.host if request.client else None
    if not _runs.is_loopback(host):
        raise HTTPException(status_code=403, detail="loopback only")
    if _REGISTRY.get(run_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    body = await request.json()
    body["run_id"] = run_id  # trust the URL, never the body
    try:
        # Re-validate via the discriminated union so unknown kinds 422 cleanly.
        from pydantic import TypeAdapter

        ta = TypeAdapter(_runs.TrainEvent)
        event = ta.validate_python(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    _REGISTRY.publish(run_id, event)
    return {"status": "ok"}


# ---- deploy: github push + droplet sync/provision -----------------------

_GITHUB_PUSH_RECIPE = "_github_push"
_DROPLET_SYNC_RECIPE = "_droplet_sync"
_DROPLET_PROVISION_RECIPE = "_droplet_provision"
_DEPLOY_BUSY_RECIPES = frozenset({_DROPLET_SYNC_RECIPE, _DROPLET_PROVISION_RECIPE})


class DeployStatus(BaseModel):
    """Returned by /api/{github,droplet}/status to drive the UI's enabled state."""

    configured: bool
    missing: list[str]
    target: str


class GithubPushRequest(BaseModel):
    commit_message: str = "mindXtrain initial push"
    force: bool = False


class DropletSyncRequest(BaseModel):
    run_bench: bool = True
    fetch_plan: bool = True


class DropletProvisionRequest(BaseModel):
    name: str = "mindxtrain"
    repo: str | None = None
    branch: str | None = None
    container: str | None = None
    extras: str = "ml,eval,data,obs"
    wait_for_bootstrap: bool = True


# Spawn shims — same _SPAWN injection pattern used by training. Tests
# monkey-patch these to bypass real subprocess execution.
GithubSpawnFn = Callable[["_runs.Run", GithubPushRequest], None]
DropletSyncSpawnFn = Callable[["_runs.Run", DropletSyncRequest], None]
DropletProvisionSpawnFn = Callable[["_runs.Run", DropletProvisionRequest], None]


def _real_github_spawn(run: _runs.Run, req: GithubPushRequest) -> None:
    cfg = _gh.GithubConfig(
        token=os.environ["GITHUB_TOKEN"],
        repo=os.environ["GITHUB_REPO"],
        branch=os.environ.get("GITHUB_DEFAULT_BRANCH", "main"),
        author_name=os.environ.get("GITHUB_AUTHOR_NAME", "mindXtrain bot"),
        author_email=os.environ.get("GITHUB_AUTHOR_EMAIL", "noreply@pythai.net"),
    )
    github_push_pipeline(
        cfg,
        run_id=run.id,
        out_dir=run.out_dir,
        commit_message=req.commit_message,
        force=req.force,
        registry=_REGISTRY,
    )


def _real_droplet_sync_spawn(run: _runs.Run, req: DropletSyncRequest) -> None:
    cfg = _droplet_mod.from_env()
    droplet_sync_pipeline(
        cfg,
        repo_root=Path.cwd(),
        run_id=run.id,
        out_dir=run.out_dir,
        run_bench=req.run_bench,
        fetch_plan=req.fetch_plan,
        registry=_REGISTRY,
    )


def _real_droplet_provision_spawn(run: _runs.Run, req: DropletProvisionRequest) -> None:
    cloud_cfg = _adc.from_env()
    droplet_provision_pipeline(
        cloud_cfg,
        name=req.name,
        repo=req.repo or os.environ.get("GITHUB_REPO", "professor-codephreak/mindXtrain"),
        branch=req.branch or os.environ.get("GITHUB_DEFAULT_BRANCH", "main"),
        container=req.container or os.environ.get("DROPLET_CONTAINER", "rocm/primus:v26.2"),
        extras=req.extras,
        run_id=run.id,
        out_dir=run.out_dir,
        wait_for_bootstrap=req.wait_for_bootstrap,
        registry=_REGISTRY,
    )


_GITHUB_SPAWN: GithubSpawnFn = _real_github_spawn
_DROPLET_SYNC_SPAWN: DropletSyncSpawnFn = _real_droplet_sync_spawn
_DROPLET_PROVISION_SPAWN: DropletProvisionSpawnFn = _real_droplet_provision_spawn


def _bootstrap_run(recipe: str) -> _runs.Run:
    out_dir = Path("./out/deploy") / recipe.lstrip("_")
    run = _REGISTRY.create(recipe, out_dir / "pending")  # path is rewritten below
    final_out = Path("./out/deploy") / recipe.lstrip("_") / run.id
    final_out.mkdir(parents=True, exist_ok=True)
    _REGISTRY._update(run.id, out_dir=final_out)
    _REGISTRY.attach_loop(asyncio.get_running_loop())
    _REGISTRY.publish(run.id, _runs.StatusEvent(run_id=run.id, status="pending", message="launching"))
    snap = _REGISTRY.get(run.id)
    assert snap is not None
    return snap


def _busy_deploy_run() -> _runs.Run | None:
    """Return the first in-flight deploy run, or None."""
    busy: set[_runs.RunStatus] = {"pending", "running"}
    for run in _REGISTRY.list_runs():
        if run.recipe in _DEPLOY_BUSY_RECIPES and run.status in busy:
            return run
    return None


def _fail_run(run: _runs.Run, message: str) -> None:
    _REGISTRY.publish(run.id, _runs.StatusEvent(run_id=run.id, status="failed", message=message))
    _REGISTRY.close_subscribers(run.id)


# -- /api/github/status + /api/github/push --------------------------------


@router.get("/api/github/status", response_model=DeployStatus)
async def api_github_status() -> DeployStatus:
    missing = _gh.status_missing()
    return DeployStatus(
        configured=not missing,
        missing=missing,
        target=_gh.status_target(),
    )


@router.post("/api/github/push", response_model=_runs.Run)
async def api_github_push(req: GithubPushRequest) -> _runs.Run:
    missing = _gh.status_missing()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"error": "github push not configured", "missing": missing},
        )
    run = _bootstrap_run(_GITHUB_PUSH_RECIPE)
    try:
        _GITHUB_SPAWN(run, req)
    except Exception as exc:
        _fail_run(run, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    snap = _REGISTRY.get(run.id)
    assert snap is not None
    return snap


# -- /api/droplet/{status,sync,provision,list} ----------------------------


@router.get("/api/droplet/status", response_model=dict)
async def api_droplet_status() -> dict[str, Any]:
    """Both modes' configured-ness in one payload — UI uses each independently."""
    sync_missing = _droplet_mod.status_missing()
    provision_missing = _adc.missing_env()
    return {
        "sync": DeployStatus(
            configured=not sync_missing,
            missing=sync_missing,
            target=_droplet_mod.status_target(),
        ).model_dump(),
        "provision": DeployStatus(
            configured=not provision_missing,
            missing=provision_missing,
            target=_adc.status_target(),
        ).model_dump(),
    }


@router.post("/api/droplet/sync", response_model=_runs.Run)
async def api_droplet_sync(req: DropletSyncRequest) -> _runs.Run:
    busy = _busy_deploy_run()
    if busy is not None:
        raise HTTPException(status_code=409, detail={
            "error": "another deploy run is in progress",
            "active_run_id": busy.id,
            "active_recipe": busy.recipe,
        })
    missing = _droplet_mod.status_missing()
    if missing:
        raise HTTPException(status_code=503, detail={
            "error": "droplet sync not configured",
            "missing": missing,
        })
    run = _bootstrap_run(_DROPLET_SYNC_RECIPE)
    try:
        _DROPLET_SYNC_SPAWN(run, req)
    except Exception as exc:
        _fail_run(run, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    snap = _REGISTRY.get(run.id)
    assert snap is not None
    return snap


@router.post("/api/droplet/provision", response_model=_runs.Run)
async def api_droplet_provision(req: DropletProvisionRequest) -> _runs.Run:
    busy = _busy_deploy_run()
    if busy is not None:
        raise HTTPException(status_code=409, detail={
            "error": "another deploy run is in progress",
            "active_run_id": busy.id,
            "active_recipe": busy.recipe,
        })
    missing = _adc.missing_env()
    if missing:
        raise HTTPException(status_code=503, detail={
            "error": "AMD Dev Cloud provision not configured",
            "missing": missing,
        })
    run = _bootstrap_run(_DROPLET_PROVISION_RECIPE)
    try:
        _DROPLET_PROVISION_SPAWN(run, req)
    except Exception as exc:
        _fail_run(run, str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    snap = _REGISTRY.get(run.id)
    assert snap is not None
    return snap


@router.get("/api/droplet/list", response_model=list[dict])
async def api_droplet_list(name: str | None = None) -> list[dict[str, Any]]:
    """Proxy `GET /v2/droplets` (optionally filtered by name)."""
    missing = _adc.missing_env()
    if missing:
        raise HTTPException(status_code=503, detail={
            "error": "AMD Dev Cloud not configured",
            "missing": missing,
        })
    cfg = _adc.from_env()
    with _adc.AmdDevCloudClient(cfg) as client:
        return client.list(name=name)
