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
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindxtrain.autotune.benchmark import run_autotune
from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.budget.pricing import MI300X_USDC_PER_HOUR
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

# Strong refs to per-run watchdog tasks (otherwise the garbage collector
# can reap them mid-await and the sampler keeps running after a terminal
# status). Cleaned up by the watchdog itself once it returns.
_METRICS_WATCHDOGS: dict[str, asyncio.Task] = {}

_log = logging.getLogger("mindxtrain.operator.coach")

# Hands-free CPU training: the operator can auto-launch a run at boot so
# the Coach UI is live without anyone pressing "Run training". Safe ~90s
# smoke recipe by default; override with MINDXTRAIN_AUTOSTART_RECIPE.
_DEFAULT_AUTOSTART_RECIPE = "mindx_fallback_qwen3_1_5b_cpu_smoke"

# Reference datacenter-GPU $/hr + VRAM for the (background) cost calculator.
H100_USDC_PER_HOUR = 4.00
H200_USDC_PER_HOUR = 6.00
A100_USDC_PER_HOUR = 1.50

# GPU VRAM (GB) — used to compute whether a workload fits per card.
_GPU_VRAM_GB = {"mi300x": 192, "h200": 141, "h100": 80, "a100": 80}

# Approximate per-parameter training memory (bytes) by method:
# full = weights(2) + grads(2) + AdamW fp32 m/v + master (~12) ≈ 16; LoRA/QLoRA
# freeze the base so only a small adapter carries grads/opt state.
_BYTES_PER_PARAM = {"full": 16.0, "lora": 3.0, "qlora": 1.5}


def _workload_vram_gb(params_b: float, method: str, batch: int, seq_len: int) -> float:
    """Rough peak training VRAM (GB) for a workload — weights/opt + activations."""
    base = params_b * _BYTES_PER_PARAM.get(method, 16.0)
    # Activation memory grows with batch×seq and (weakly) model width.
    activations = batch * seq_len * 1.0e-4 * (params_b ** 0.5)
    return base + activations


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
    model_config = ConfigDict(extra="forbid")
    gpus: int = Field(default=1, ge=1, le=64)
    hours: float = Field(default=1.5, gt=0.0, le=720.0)
    safety_margin: float = Field(default=1.15, ge=1.0, le=2.0)
    # Generalize beyond the hardcoded Qwen3-8B: the calculator now sizes VRAM
    # from the actual workload (params, method, batch, seq).
    params_b: float = Field(default=8.0, gt=0.0, le=2000.0, description="model size, billions of params")
    method: Literal["full", "lora", "qlora"] = "full"
    batch: int = Field(default=8, ge=1, le=4096)
    seq_len: int = Field(default=4096, ge=64, le=1_048_576)


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
    needed_vram_gb: float
    mi300x: CostBreakdown
    h100: CostBreakdown
    h200: CostBreakdown
    a100: CostBreakdown
    comparisons: list[CostBreakdown] = Field(default_factory=list)
    cheapest_that_fits: str = ""
    speedup_vs_h100_x: float


class CoachHealthResponse(BaseModel):
    coach_version: str = "0.1.0"
    chat_backend_ready: bool = False
    chat_backend_name: str = ""
    chat_backend_model: str = Field(
        default="",
        description=(
            "When the detected backend is ollama, the first available model "
            "name (e.g. 'qwen3:0.6b'). Empty for vllm/openai_compat or when "
            "the probe fails."
        ),
    )
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


@router.get("/modelfile", response_class=FileResponse, include_in_schema=False)
async def coach_modelfile_page() -> FileResponse:
    """Standalone Ollama Modelfile builder (opened in a separate window)."""
    return FileResponse(_STATIC_DIR / "modelfile.html")


@router.get("/api/modelfile/params")
async def api_modelfile_params() -> dict[str, Any]:
    """The full PARAMETER catalogue so the builder can render toggles + inputs."""
    from mindxtrain.deploy.modelfile import MODELFILE_PARAMS

    return {"parameters": [p.model_dump() for p in MODELFILE_PARAMS]}


@router.post("/api/modelfile/build")
async def api_modelfile_build(spec: dict[str, Any]) -> dict[str, str]:
    """Render a Modelfile from a spec body → `{modelfile: <text>}`."""
    from pydantic import ValidationError

    from mindxtrain.deploy.modelfile import ModelfileSpec, render_modelfile

    try:
        parsed = ModelfileSpec.model_validate(spec)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"modelfile": render_modelfile(parsed)}


@router.post("/api/modelfile/create")
async def api_modelfile_create(body: dict[str, Any]) -> dict[str, str]:
    """Write the Modelfile and run `ollama create <tag>` (off the event loop)."""
    from pydantic import ValidationError

    from mindxtrain.deploy.modelfile import ModelfileSpec, create_model

    tag = str(body.get("tag", "")).strip()
    if not tag:
        raise HTTPException(status_code=422, detail="a `tag` is required to create the model")
    try:
        parsed = ModelfileSpec.model_validate(body.get("spec", {}))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await asyncio.to_thread(create_model, tag, parsed)


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
    """Cost + fit comparison vs datacenter GPUs (background; not shown in the UI).

    Sizes peak training VRAM from the workload (params/method/batch/seq), then for
    each GPU computes the card count needed to fit, the cost, and whether a single
    card fits. Generalized beyond the old hardcoded Qwen3-8B slide.
    """
    needed = _workload_vram_gb(req.params_b, req.method, req.batch, req.seq_len)
    rates = {
        "mi300x": MI300X_USDC_PER_HOUR, "h200": H200_USDC_PER_HOUR,
        "h100": H100_USDC_PER_HOUR, "a100": A100_USDC_PER_HOUR,
    }
    labels = {
        "mi300x": "MI300X (192 GB HBM3)", "h200": "H200 (141 GB HBM3e)",
        "h100": "H100 (80 GB HBM3)", "a100": "A100 (80 GB)",
    }

    def _breakdown(key: str) -> CostBreakdown:
        vram = _GPU_VRAM_GB[key]
        fits_one = vram >= needed
        # Cards needed to hold the workload (sharded), honoring the user's gpu count.
        cards = max(req.gpus, -(-int(needed) // vram))  # ceil-div
        cost = cards * req.hours * rates[key] * req.safety_margin
        note = (
            f"fits on one card ({vram} GB ≥ {needed:.0f} GB needed)."
            if fits_one
            else f"needs {cards}x to fit {needed:.0f} GB (or quantize)."
        )
        return CostBreakdown(
            name=labels[key], rate_usdc_per_hour=rates[key], gpus=cards,
            cost_usdc=round(cost, 2), fits_qwen3_8b_bf16_bs8_seq4096=fits_one, note=note,
        )

    mi300x, h200, h100, a100 = (_breakdown(k) for k in ("mi300x", "h200", "h100", "a100"))
    comparisons = [mi300x, h200, h100, a100]
    fitting = [c for c in comparisons if c.fits_qwen3_8b_bf16_bs8_seq4096] or comparisons
    cheapest = min(fitting, key=lambda c: c.cost_usdc)

    return CostResponse(
        hours=req.hours,
        safety_margin=req.safety_margin,
        needed_vram_gb=round(needed, 1),
        mi300x=mi300x, h100=h100, h200=h200, a100=a100,
        comparisons=comparisons,
        cheapest_that_fits=cheapest.name,
        speedup_vs_h100_x=round(h100.cost_usdc / mi300x.cost_usdc, 2) if mi300x.cost_usdc > 0 else 0.0,
    )


@router.get("/api/health", response_model=CoachHealthResponse)
async def api_health() -> CoachHealthResponse:
    """Coach health.

    Reports the auto-detected chat backend (ollama if reachable on the
    loopback, vllm otherwise) plus a `chat_backend_ready` boolean from a
    live reachability probe. For ollama, also includes the first model
    name so the UI can render "ollama (qwen3:0.6b) ready".
    """
    from mindxtrain.operator.app import (
        backend_first_model,
        backend_reachable,
        resolve_backend_name,
    )

    backend = resolve_backend_name()
    ready = backend_reachable(backend)
    model_name = (backend_first_model(backend) or "") if ready else ""
    return CoachHealthResponse(
        chat_backend_ready=ready,
        chat_backend_name=backend,
        chat_backend_model=model_name,
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


class CorpusBucketStats(BaseModel):
    """File / line / unique-row counts for one bucket of dream-cycle output."""

    files: int = 0
    raw_lines: int = 0
    unique_rows: int = 0


class DreamCorpusResponse(BaseModel):
    """Sanity check that mindX's dream-cycle JSONL corpus is reachable.

    The dream cycle writes two JSONL streams per cycle:
    - `*_training.jsonl` — STM-to-insight consolidation (phase 5b)
    - `*_evolutions.jsonl` — insight-to-evolution proposals (phase 5c)

    Both are reported here; a recipe with `data.include_evolutions: true`
    consumes the union.
    """

    root: str = Field(description="Filesystem root inspected.")
    exists: bool
    consolidation: CorpusBucketStats = Field(default_factory=CorpusBucketStats)
    evolutions: CorpusBucketStats = Field(default_factory=CorpusBucketStats)
    ready: bool = Field(
        description="True iff exists and at least one bucket has unique rows.",
    )
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
    from mindxtrain.data.sources.mindx_dreams import (
        count_mindx_dreams,
        count_mindx_evolutions,
    )

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

    consolidation = CorpusBucketStats(**count_mindx_dreams(corpus_root))
    evolutions = CorpusBucketStats(**count_mindx_evolutions(corpus_root))
    ready = (consolidation.unique_rows + evolutions.unique_rows) > 0
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
        consolidation=consolidation,
        evolutions=evolutions,
        ready=ready,
        note=note,
    )


# ---- create dataset (author a script for an actor) ----------------------
# A model is an actor; an actor has a persona (voice) and a script (the
# training examples). This lets the operator author a small script in the
# browser and save it as `source: local` JSONL the recipes can imprint from.

_SAFE_NAME = re.compile(r"[^a-z0-9_-]+")


def _datasets_root() -> Path:
    """Where authored scripts live. Override with MINDXTRAIN_DATASETS_DIR."""
    return Path(os.environ.get("MINDXTRAIN_DATASETS_DIR", "./out/datasets"))


def _safe_dataset_name(name: str) -> str:
    cleaned = _SAFE_NAME.sub("-", name.strip().lower()).strip("-")
    return cleaned or "script"


class ExchangeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str
    assistant: str


class CreateScriptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="dataset name; becomes out/datasets/<name>/script.jsonl")
    persona: str = Field(default="", description="built-in persona key (overrides persona_name/system_prompt)")
    persona_name: str = "actor"
    system_prompt: str = ""
    voice_examples: list[str] = Field(default_factory=list)
    exchanges: list[ExchangeIn] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list, description="skill bundles to mix in")
    seed_voice: bool = True


class ScriptInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str
    rows: int
    persona_name: str
    skills: list[str] = Field(default_factory=list)
    train_params: dict[str, int] = Field(default_factory=dict)


class ScriptPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str
    rows: int
    sample: list[dict[str, Any]]


@router.get("/api/persona", response_model=dict)
async def api_persona() -> dict[str, Any]:
    """The persona Coach pre-fills the create-script form with (clean-room).

    Loaded from `MINDXTRAIN_PERSONA_PATH` if set, else a minimal default. Never
    copies mindX bytes — reads recognised fields at runtime.
    """
    from mindxtrain.data.scripts import load_persona

    p = load_persona()
    return {
        "name": p.name,
        "system_prompt": p.system_prompt,
        "voice_examples": list(p.voice_examples),
    }


@router.get("/api/personas")
async def api_personas() -> dict[str, Any]:
    """Built-in personas + toggleable skills for the Create-script picker."""
    from mindxtrain.data import personas as _pz

    return {"personas": _pz.list_personas(), "skills": _pz.list_skills()}


@router.post("/api/datasets", response_model=ScriptInfo)
async def api_create_dataset(req: CreateScriptRequest) -> ScriptInfo:
    """Author a script from a persona + optional skills + exchanges → `source: local` JSONL.

    Skills (software_engineer / platform_architect / bash / solidity) mix their
    in-domain exchanges into the script. Returns the row count + training params
    auto-derived from the dataset size.
    """
    from mindxtrain.data import personas as _pz
    from mindxtrain.data.scripts import (
        Exchange,
        Persona,
        build_script_rows,
        derive_training_params,
        write_script_jsonl,
    )

    # Base persona: a built-in (with skills mixed in) or the explicit fields.
    if req.persona:
        persona, skill_exchanges = _pz.compose(req.persona, req.skills)
    else:
        base = Persona(
            name=req.persona_name or "actor",
            system_prompt=req.system_prompt,
            voice_examples=list(req.voice_examples),
        )
        persona, skill_exchanges = _pz.compose(base, req.skills)

    exchanges = [Exchange(user=e.user, assistant=e.assistant) for e in req.exchanges]
    exchanges.extend(skill_exchanges)

    if not exchanges and not (req.seed_voice and persona.voice_examples):
        raise HTTPException(
            status_code=422,
            detail="provide an exchange, a skill, or a voice example to seed.",
        )

    name = _safe_dataset_name(req.name)
    out_path = _datasets_root() / name / "script.jsonl"
    rows_list = build_script_rows(persona, exchanges, seed_voice=req.seed_voice)
    write_script_jsonl(rows_list, out_path)
    rows = len(rows_list)
    return ScriptInfo(
        name=name, path=str(out_path), rows=rows, persona_name=persona.name,
        skills=[s for s in req.skills if s in _pz.SKILLS],
        train_params=derive_training_params(rows),
    )


@router.get("/api/datasets", response_model=list[ScriptInfo])
async def api_list_datasets() -> list[ScriptInfo]:
    """List authored scripts under the datasets root (newest dirs first)."""
    if not _datasets_root().exists():
        return []
    out: list[ScriptInfo] = []
    for d in sorted(_datasets_root().iterdir(), reverse=True):
        script = d / "script.jsonl"
        if not script.is_file():
            continue
        rows = sum(1 for line in script.read_text().splitlines() if line.strip())
        out.append(ScriptInfo(name=d.name, path=str(script), rows=rows, persona_name=""))
    return out


@router.get("/api/datasets/{name}", response_model=ScriptPreview)
async def api_preview_dataset(name: str) -> ScriptPreview:
    """Preview the first few rows of an authored script."""
    script = _datasets_root() / _safe_dataset_name(name) / "script.jsonl"
    if not script.is_file():
        raise HTTPException(status_code=404, detail=f"no script for {name!r}")
    sample: list[dict[str, Any]] = []
    rows = 0
    for line in script.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows += 1
        if len(sample) < 5:
            try:
                sample.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ScriptPreview(name=_safe_dataset_name(name), path=str(script), rows=rows, sample=sample)


# ---- imprint measurement (recall before/after) --------------------------
# Score how much an actor's utterances moved toward the persona voice after
# training. Scoring is fast + dependency-light; the heavy generation that
# produces the before/after utterances runs in `mindxtrain imprint` (CLI) or
# the e2e test so the operator event loop never blocks on model inference.


class ImprintScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inquiries: list[str]
    before: list[str]
    after: list[str]
    baseline: list[str]


@router.post("/api/imprint/score")
async def api_imprint_score(req: ImprintScoreRequest) -> dict[str, Any]:
    """Score a persona imprint from supplied before/after utterances + baseline."""
    from mindxtrain.eval.imprint import score_imprint

    report = score_imprint(req.inquiries, req.before, req.after, req.baseline)
    return report.model_dump()


# ---- governance: boardroom (any-N) + dojo (prime-N) ---------------------
# A model is an actor; the classroom graduates it; the boardroom decides about
# the graduation; a disputed boardroom is settled by a prime-sized dojo. The
# boardroom/dojo can be backed by real models (use_models) or tallied from
# supplied votes. Model deliberation runs in a worker thread so the operator
# event loop never blocks on inference.


class MemberIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    role: str = "generalist"
    model: str = ""


class ConveneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motion: str
    members: list[MemberIn]
    quorum: float = Field(default=0.5, ge=0.0, le=1.0)
    votes: dict[str, str] | None = None
    use_models: bool = False
    base_url: str | None = None


class DojoSettleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    motion: str
    size: int = 3
    model: str = ""
    votes: dict[str, str] | None = None
    use_models: bool = False
    base_url: str | None = None


@router.get("/api/boardroom/presets")
async def api_boardroom_presets() -> dict[str, list[str]]:
    """Named preset boards → their advisor roles."""
    from mindxtrain.governance.boardroom import PRESET_BOARDS

    return {name: list(roles) for name, roles in PRESET_BOARDS.items()}


@router.get("/api/models")
async def api_models() -> dict[str, Any]:
    """Model ids the configured chat backend exposes, so the Boardroom card can
    pick a model that's actually installed (best-effort; `[]` if unreachable)."""
    import httpx

    from mindxtrain.governance.panel import resolve_chat_base_url

    base = resolve_chat_base_url()
    models: list[str] = []
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base}/models")
            resp.raise_for_status()
            data = resp.json()
        models = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    except (httpx.HTTPError, OSError, ValueError):
        models = []
    # Local models first so the chat/boardroom default isn't a :cloud model.
    models.sort(key=lambda m: (":cloud" in m, m))
    return {"base_url": base, "models": models}


# ---- chat: AI-SDK-style streaming + ollama controls ---------------------
# The chat streams token deltas as Server-Sent Events (the AI SDK "text stream"
# pattern) so responses render live, and exposes start/stop/status for the local
# ollama server + model interaction. See docs/coach.md.


def _resolve_chat_backend() -> Any:
    """Build the active chat backend (ollama / vllm / openai_compat)."""
    from mindxtrain.models.registry import build_backend
    from mindxtrain.operator.app import resolve_backend_name

    name = resolve_backend_name()
    kwargs: dict[str, Any] = {}
    if name == "vllm":
        kwargs["base_url"] = os.environ.get(
            "MINDXTRAIN_VLLM_BASE_URL",
            os.environ.get("AUTOMINDX_VLLM_BASE_URL", "http://localhost:8000/v1"),
        )
    elif name == "ollama":
        kwargs["base_url"] = os.environ.get("MINDXTRAIN_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    elif name == "openai_compat":
        kwargs["base_url"] = os.environ.get("MINDXTRAIN_OPENAI_BASE_URL", "")
        kwargs["api_key"] = os.environ.get("MINDXTRAIN_OPENAI_API_KEY", "")
    return build_backend(name, **kwargs)


@router.post("/api/chat/stream")
async def api_chat_stream(body: dict[str, Any]) -> StreamingResponse:
    """Stream a chat completion as an SSE text stream of token deltas.

    Body: `{model, messages:[{role,content}], max_tokens?, temperature?}`. Each SSE
    `data:` line is a JSON-encoded token; the stream ends with `data: [DONE]`. Mirrors
    the AI SDK text-stream protocol so the client renders tokens as they arrive.
    """
    from pydantic import ValidationError

    from mindxtrain.models.registry import ChatRequest

    payload = {
        "model": str(body.get("model") or "").strip() or "default",
        "messages": body.get("messages") or [],
        "max_tokens": int(body.get("max_tokens") or 512),
        "temperature": float(body.get("temperature", 0.7)),
        "stream": True,
    }
    try:
        req = ChatRequest.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    backend = _resolve_chat_backend()

    async def _gen() -> Any:
        try:
            stream = await backend.stream_chat(req)
            async for token in stream:
                yield f"data: {json.dumps(token)}\n\n"
        except Exception as exc:  # surface backend errors in-stream, never 500 mid-stream
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_sse_headers())


def _ollama_serve_pids() -> list[int]:
    """PIDs of running `ollama serve` processes (best-effort)."""
    import subprocess

    try:
        out = subprocess.run(
            ["pgrep", "-f", "ollama serve"], capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


@router.get("/api/ollama/status")
async def api_ollama_status() -> dict[str, Any]:
    """Whether the local ollama server is reachable + installed + running."""
    import shutil

    import httpx

    from mindxtrain.governance.panel import resolve_chat_base_url

    base = resolve_chat_base_url()
    reachable = False
    try:
        with httpx.Client(timeout=2.0) as client:
            reachable = client.get(f"{base}/models").status_code == 200
    except (httpx.HTTPError, OSError):
        reachable = False
    return {
        "reachable": reachable,
        "has_ollama_bin": shutil.which("ollama") is not None,
        "serve_pids": _ollama_serve_pids(),
        "base_url": base,
    }


@router.post("/api/ollama/start")
async def api_ollama_start() -> dict[str, Any]:
    """Start `ollama serve` (detached) if it isn't already running."""
    import shutil
    import subprocess

    if shutil.which("ollama") is None:
        raise HTTPException(status_code=422, detail="ollama binary not found on PATH")
    if _ollama_serve_pids():
        return {"started": False, "note": "ollama serve already running"}
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=500, detail=f"failed to start ollama: {exc}") from exc
    return {"started": True}


@router.post("/api/ollama/stop")
async def api_ollama_stop() -> dict[str, Any]:
    """Stop the local `ollama serve` process(es)."""
    import subprocess

    pids = _ollama_serve_pids()
    if not pids:
        return {"stopped": False, "note": "no `ollama serve` process found"}
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"stopped": True, "pids": pids}


@router.post("/api/boardroom/convene")
async def api_boardroom_convene(req: ConveneRequest) -> dict[str, Any]:
    """Convene a boardroom on a motion. Tally supplied `votes`, or `use_models`
    to have each member's model deliberate (run off the event loop)."""
    from pydantic import ValidationError

    from mindxtrain.governance import Boardroom, Member

    try:
        members = [Member(id=m.id, role=m.role, model=m.model) for m in req.members]  # type: ignore[arg-type]
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not members:
        raise HTTPException(status_code=422, detail="a boardroom needs at least one member")
    board = Boardroom(members=members, quorum=req.quorum)

    deliberations: list[dict[str, Any]] = []
    if req.use_models:
        from mindxtrain.governance import panel as _panel

        async def _one(m: Member) -> Any:
            return await asyncio.to_thread(_panel.deliberate, m, req.motion, base_url=req.base_url)

        delibs = await asyncio.gather(*[_one(m) for m in members])
        votes = {d.member_id: d.vote for d in delibs}
        deliberations = [d.model_dump() for d in delibs]
        decision = board.convene(req.motion, votes)
    elif req.votes is not None:
        try:
            decision = board.convene(req.motion, dict(req.votes))  # type: ignore[arg-type]
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=422, detail="provide `votes` or set `use_models: true`")

    return {"decision": decision.model_dump(), "deliberations": deliberations}


@router.post("/api/dojo/settle")
async def api_dojo_settle(req: DojoSettleRequest) -> dict[str, Any]:
    """Settle a dispute with a prime-sized dojo. Tally supplied `votes` (keyed
    `judge-0..`) or `use_models` to have the judges rule (off the event loop)."""
    from mindxtrain.governance import Dojo

    dojo = Dojo.sized(req.size)
    if req.use_models:
        from mindxtrain.governance import panel as _panel

        kw = {"base_url": req.base_url}
        if req.model:
            kw["default_model"] = req.model
        ballot = _panel.model_judge_ballot(**kw)
        verdict = await asyncio.to_thread(dojo.settle, req.motion, ballot)
    elif req.votes is not None:
        try:
            verdict = dojo.settle(req.motion, dict(req.votes))  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=422, detail="provide `votes` or set `use_models: true`")
    return verdict.model_dump()


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
    """Default spawn: route by backend.

    - `trl_cpu` runs in-process on a daemon thread (uses the same code
      path as `/v1/training/jobs` so the Coach UI sees the same events
      whether the run was kicked off via Coach or the public API).
    - Everything else compiles to Axolotl YAML + streams a subprocess.

    Tests monkey-patch the module-level `_SPAWN` to bypass the real
    subprocess and emit canned events instead.
    """
    if cfg.train.backend in ("trl_cpu", "trl_local"):
        import threading

        from mindxtrain.train.backend_trl_cpu import run_trl_cpu, run_trl_local

        # trl_local auto-detects a local GPU (else CPU fallback); trl_cpu pins CPU.
        _run_inprocess = run_trl_local if cfg.train.backend == "trl_local" else run_trl_cpu

        def _on_line(line: str) -> None:
            _REGISTRY.publish_threadsafe(
                run.id, _runs.LogEvent(run_id=run.id, line=line, level="stdout"),
            )

        def _on_event(ev: dict[str, Any]) -> None:
            """Translate trl_cpu structured logs into registry events.

            Drives Coach's Chart.js loss curve directly — same kind=step
            and kind=eval contract the axolotl subprocess streamer
            satisfies via stdout regex.
            """
            kind = ev.get("kind")
            if kind == "step":
                _REGISTRY.publish_threadsafe(run.id, _runs.StepEvent(
                    run_id=run.id,
                    step=int(ev["step"]),
                    loss=float(ev["loss"]),
                    lr=ev.get("lr"),
                    grad_norm=ev.get("grad_norm"),
                    tokens_per_s=ev.get("tokens_per_s"),
                    # Realtime-feedback fields — drive the Coach progress
                    # bar + "is it learning" accuracy chart.
                    total_steps=ev.get("total_steps"),
                    mean_token_accuracy=ev.get("mean_token_accuracy"),
                    entropy=ev.get("entropy"),
                ))
            elif kind == "eval":
                _REGISTRY.publish_threadsafe(run.id, _runs.EvalEvent(
                    run_id=run.id,
                    step=int(ev["step"]),
                    suite=str(ev.get("suite", "mid_train")),
                    metrics={k: float(v) for k, v in ev.get("metrics", {}).items()},
                ))

        def _thread() -> None:
            _REGISTRY.publish_threadsafe(
                run.id,
                _runs.StatusEvent(run_id=run.id, status="running", message="cpu lane"),
            )
            try:
                _run_inprocess(
                    cfg, plan, run.out_dir, on_line=_on_line, on_event=_on_event,
                )
            except Exception as exc:
                _REGISTRY.publish_threadsafe(
                    run.id,
                    _runs.StatusEvent(run_id=run.id, status="failed", message=str(exc)),
                )
                _REGISTRY.close_subscribers(run.id)
                return
            # Bind the AutotunePlan + checkpoint hashes into a verifiable
            # manifest before announcing success, so a UI subscriber can fetch
            # the receipt the moment it sees `succeeded`.
            from mindxtrain.operator.receipt_emit import emit_run_receipt
            emit_run_receipt(_REGISTRY, run, cfg, plan)
            _REGISTRY.publish_threadsafe(
                run.id,
                _runs.StatusEvent(
                    run_id=run.id, status="succeeded", message="cpu lane done",
                ),
            )
            _REGISTRY.close_subscribers(run.id)

        threading.Thread(target=_thread, daemon=True, name=f"trl-cpu-{run.id}").start()
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

    # Start the per-run system-metrics sampler. For trl_cpu the trainer
    # runs in-process so trainer-PID == operator-PID. Axolotl-subprocess
    # would need its own PID — handled in a follow-up.
    from mindxtrain.operator.coach.run_metrics import start_metrics_sampler
    start_metrics_sampler(run.id, os.getpid())
    # Watchdog stops the sampler on terminal status. The reference is
    # stored alongside the sampler tasks so the task survives until
    # cancellation; without this binding GC could reap it mid-watch.
    _METRICS_WATCHDOGS[run.id] = asyncio.create_task(
        _stop_metrics_on_terminal(run.id),
        name=f"metrics-watchdog-{run.id}",
    )

    snapshot = _REGISTRY.get(run.id)
    assert snapshot is not None
    return snapshot


async def _stop_metrics_on_terminal(run_id: str) -> None:
    """Watchdog — stops the metrics sampler when its run hits a terminal status.

    Subscribes to the run's event stream filtered to `status` kinds and
    bails on the first terminal value. If the registry is torn down or
    the run vanishes, the subscribe iterator ends and the task exits
    quietly.
    """
    from mindxtrain.operator.coach.run_metrics import stop_metrics_sampler

    terminal = {"succeeded", "failed", "cancelled"}
    try:
        async for ev in _REGISTRY.subscribe(run_id, kinds=("status",)):
            if isinstance(ev, _runs.StatusEvent) and ev.status in terminal:
                break
    except Exception:
        pass
    await stop_metrics_sampler(run_id)
    _METRICS_WATCHDOGS.pop(run_id, None)


def autostart_enabled() -> bool:
    """True when MINDXTRAIN_AUTOSTART opts the operator into hands-free training."""
    return os.environ.get("MINDXTRAIN_AUTOSTART", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def autostart_recipe() -> str:
    """Recipe the operator auto-launches at boot (MINDXTRAIN_AUTOSTART_RECIPE)."""
    return (
        os.environ.get("MINDXTRAIN_AUTOSTART_RECIPE", "").strip()
        or _DEFAULT_AUTOSTART_RECIPE
    )


def _mindx_root() -> Path:
    """Filesystem root of the mindX install (MINDXTRAIN_MINDX_ROOT, else ~/mindX)."""
    explicit = os.environ.get("MINDXTRAIN_MINDX_ROOT", "").strip()
    return Path(explicit) if explicit else Path.home() / "mindX"


def sea_decision_path() -> Path:
    """Path to the SEA agent's training-recommendation file.

    The mindX StrategicEvolutionAgent writes its go/no-go verdict here;
    the operator reads it to gate autonomous training. Override with
    `MINDXTRAIN_SEA_DECISION`, else default under the mindX data dir.
    """
    explicit = os.environ.get("MINDXTRAIN_SEA_DECISION", "").strip()
    if explicit:
        return Path(explicit)
    return _mindx_root() / "data" / "training_recommendation.json"


def read_sea_decision() -> dict[str, Any] | None:
    """Parse the SEA decision file. `None` when absent or unreadable/invalid."""
    try:
        raw = sea_decision_path().read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def sea_training_gate() -> dict[str, Any]:
    """Evaluate the SEA agent's training recommendation.

    Returns a status dict consumed by both the autostart path and the
    `/api/sea-decision` endpoint. `open` is True only when SEA explicitly
    recommends training *and* the record is still fresh (within `ttl_s`).
    A missing file means SEA has not spoken — the gate stays closed.
    """
    data = read_sea_decision()
    if data is None:
        return {
            "open": False, "available": False, "decision": None,
            "reason": "no SEA decision file — autonomous training stands down",
        }
    reason = str(data.get("reason", "")).strip() or "no reason given"
    if not bool(data.get("recommend", False)):
        return {
            "open": False, "available": True, "decision": data,
            "reason": f"SEA decided against training: {reason}",
        }
    ts = data.get("ts")
    ttl = data.get("ttl_s", 3600)
    if isinstance(ts, (int, float)) and isinstance(ttl, (int, float)):
        age = time.time() - float(ts)
        if age > float(ttl):
            return {
                "open": False, "available": True, "decision": data,
                "reason": (
                    f"SEA recommendation is stale "
                    f"({age:.0f}s old > ttl {float(ttl):.0f}s)"
                ),
            }
    return {
        "open": True, "available": True, "decision": data,
        "reason": f"SEA recommends training — {reason}",
    }


async def autostart_cpu_training() -> _runs.Run | None:
    """Launch a CPU training run at boot — autonomous, but only if SEA agrees.

    Two gates. `MINDXTRAIN_AUTOSTART` arms autonomous mode (off by
    default so a `TestClient` / CI lifespan never spawns a trainer). The
    mindX `StrategicEvolutionAgent`'s decision file is the actual
    decider — the run launches only when SEA recommends training and the
    record is fresh. SEA's chosen recipe (if any) wins; otherwise the
    `MINDXTRAIN_AUTOSTART_RECIPE` default applies. Idempotent — skips
    when a run is already pending/running. Returns the launched `Run`,
    or `None` when a gate is closed / the launch failed (logged, never
    raised).
    """
    if not autostart_enabled():
        return None
    gate = sea_training_gate()
    if not gate["open"]:
        _log.info("autostart: SEA gate closed — %s", gate["reason"])
        return None
    for existing in _REGISTRY.list_runs():
        if existing.status in ("pending", "running"):
            _log.info(
                "autostart: run %s already %s — skipping",
                existing.id, existing.status,
            )
            return None
    decision = gate["decision"] or {}
    recipe = str(decision.get("recipe") or "").strip() or autostart_recipe()
    _log.info("autostart: SEA gate OPEN — %s; launching %r", gate["reason"], recipe)
    try:
        run = await api_runs_launch(LaunchRequest(recipe=recipe))
    except HTTPException as exc:
        _log.warning(
            "autostart: launch failed (HTTP %s) — %s. The Coach UI stays "
            "available; press 'Run training' to start a session manually.",
            exc.status_code, exc.detail,
        )
        return None
    _log.info("autostart: run %s launched on SEA's recommendation", run.id)
    return run


@router.get("/api/sea-decision")
async def api_sea_decision() -> dict[str, Any]:
    """The SEA agent's current training recommendation + the gate verdict.

    The Coach UI polls this so the user can see whether autonomous
    training is armed and why SEA did (or did not) recommend a run.
    `open` True means a boot right now would auto-launch; the user can
    always start a session by hand with the "Run training" button.
    """
    gate = sea_training_gate()
    gate["autostart_enabled"] = autostart_enabled()
    gate["decision_path"] = str(sea_decision_path())
    return gate


@router.get("/api/runs", response_model=list[_runs.Run])
async def api_runs_list() -> list[_runs.Run]:
    return _REGISTRY.list_runs()


@router.get("/api/runs/{run_id}", response_model=_runs.Run)
async def api_run_get(run_id: str) -> _runs.Run:
    snap = _REGISTRY.get(run_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return snap


@router.get("/api/runs/{run_id:path}/metrics")
async def api_run_metrics(run_id: str, since: float = 0.0) -> dict[str, Any]:
    """Backfill the system-metrics sparklines on tab-switch.

    Returns samples newer than `since` (unix seconds, default 0 = all
    cached). 404 when the run id is unknown to the registry, [] when
    the run exists but the sampler hasn't produced any samples yet
    (e.g., between launch and the first 1 Hz tick).
    """
    from mindxtrain.operator.coach.run_metrics import get_buffer

    if _REGISTRY.get(run_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return {"samples": get_buffer(run_id, since=since)}


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


class PushToOllamaRequest(BaseModel):
    tag: str | None = Field(
        default=None,
        description="Ollama tag for the new model. Defaults to the run's recipe name.",
    )
    system_prompt: str | None = None
    base_model: str | None = Field(
        default=None,
        description=(
            "Override the base model name resolved from the recipe. Useful "
            "when the training adapter was produced against a snapshot that "
            "differs from the recipe's `model.name` field."
        ),
    )
    register_with_mindx: bool = Field(
        default=False,
        description=(
            "After ollama create succeeds, PATCH the new tag into mindX as "
            "the local fallback (best-effort; failure does NOT abort the push)."
        ),
    )
    mindx_base_url: str | None = Field(
        default=None,
        description="Override the mindX base URL for the fallback PATCH.",
    )


class PushToOllamaResponse(BaseModel):
    run_id: str
    tag: str
    merged_dir: str
    modelfile: str
    mindx_fallback_swapped: bool = False
    mindx_fallback_swap: dict[str, str] | None = None
    message: str = "pushed"


@router.post(
    "/api/runs/{run_id:path}/push-to-ollama",
    response_model=PushToOllamaResponse,
)
async def api_run_push_to_ollama(
    run_id: str, req: PushToOllamaRequest,
) -> PushToOllamaResponse:
    """Merge the run's LoRA adapter into the base weights, write a
    Modelfile, and call `ollama create`. Streams the push log into the
    run's SSE channel so the Coach UI can replay it in the train card.

    The adapter is expected at `<run.out_dir>/checkpoint/` — the same
    location both trl_cpu and the axolotl subprocess write to.
    """
    snap = _REGISTRY.get(run_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")

    adapter = snap.out_dir / "checkpoint"
    if not adapter.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"no checkpoint at {adapter}; let the training run finish "
                f"before pushing to ollama"
            ),
        )

    # Resolve the base model from the recipe unless the caller overrode it.
    if req.base_model:
        base_model = req.base_model
    else:
        try:
            cfg = XTrainConfig.model_validate(
                yaml.safe_load(render_recipe(snap.recipe)),
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=f"can't resolve base model from recipe {snap.recipe!r}: {exc}",
            ) from exc
        base_model = cfg.model.name

    tag = req.tag or snap.recipe

    # Bind the registry to the current loop so the threaded merge can publish
    # back into it via call_soon_threadsafe. Without this, repeat requests
    # under TestClient (which closes its loop after each call) hit
    # "Event loop is closed" the second time around.
    _REGISTRY.attach_loop(asyncio.get_running_loop())

    def _log(line: str) -> None:
        _REGISTRY.publish_threadsafe(
            run_id, _runs.LogEvent(run_id=run_id, line=line, level="stdout"),
        )

    _REGISTRY.publish(
        run_id,
        _runs.StatusEvent(
            run_id=run_id, status="running",
            message=f"push-to-ollama: {base_model} + {adapter} -> {tag}",
        ),
    )

    try:
        from mindxtrain.deploy.ollama_push import push_to_ollama
        result = await asyncio.to_thread(
            push_to_ollama,
            base_model=base_model,
            adapter_dir=adapter,
            tag=tag,
            system_prompt=req.system_prompt,
            sink=_log,
            register_with_mindx=req.register_with_mindx,
            mindx_base_url=req.mindx_base_url,
        )
    except (FileNotFoundError, ImportError) as exc:
        _REGISTRY.publish(
            run_id,
            _runs.StatusEvent(run_id=run_id, status="failed", message=str(exc)),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        msg = (exc.stderr or exc.stdout or "").strip() or f"ollama create exit={exc.returncode}"
        _REGISTRY.publish(
            run_id,
            _runs.StatusEvent(run_id=run_id, status="failed", message=msg),
        )
        raise HTTPException(status_code=502, detail=msg) from exc

    _REGISTRY.publish(
        run_id,
        _runs.StatusEvent(
            run_id=run_id, status="succeeded", message=f"pushed to ollama as {result.tag}",
        ),
    )
    swap = result.mindx_fallback_swap
    return PushToOllamaResponse(
        run_id=run_id,
        tag=result.tag,
        merged_dir=str(result.merged_dir),
        modelfile=str(result.modelfile),
        mindx_fallback_swapped=swap is not None,
        mindx_fallback_swap=swap,
    )


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
    recipe: str | None = Field(
        default=None,
        description=(
            "Built-in recipe to run on the droplet via `mindxtrain train` "
            "after cloud-init bench. When set, the operator SSH-tails the "
            "training log and bridges per-step events into this run's SSE "
            "stream so the Coach Train card populates live."
        ),
    )


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
        recipe=req.recipe,
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
    if req.recipe is not None and req.recipe not in list_recipes():
        raise HTTPException(
            status_code=404,
            detail=f"unknown recipe {req.recipe!r}",
        )
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


# ---- hardware diagnostics ------------------------------------------------


@router.get("/api/diagnostics/hardware")
async def api_diagnostics_hardware() -> dict[str, Any]:
    """Return a CPU/AMD/NVIDIA hardware profile + recommended training lane.

    The probes shell out to `rocm-smi` / `nvidia-smi` when available.
    Each probe has a short timeout so a hung driver tool can't stall the
    Coach UI. The composite profile is JSON-stable: the UI can poll this
    repeatedly to refresh hardware state (e.g., after `rocm` installs).
    """
    from mindxtrain.operator.coach.hw_diagnostics import probe_all

    profile = probe_all()
    return profile.model_dump()


@router.get("/api/diagnostics/live")
async def api_diagnostics_live() -> dict[str, Any]:
    """Cheap (~ms) live sample of host pressure + operator process state.

    Backbone of the Advanced Admin card. Returns load avgs, RAM%, disk%,
    and the operator's own RSS + thread count. The UI polls this at 1-2 Hz
    while the admin card is visible.
    """
    from mindxtrain.operator.coach.hw_diagnostics import probe_live_metrics

    return probe_live_metrics().model_dump()


@router.get("/api/diagnostics/chronos")
async def api_diagnostics_chronos() -> dict[str, Any]:
    """Aggregate chronos.agent state for the UI's promised-time card.

    Calls mindX's `/v1/oracle/{time,anchors,drift}` and merges the
    responses into one payload. Degrades to `consensus: unavailable`
    when mindX is unreachable so the UI never blanks out.
    """
    from mindxtrain.operator.coach import chronos_client

    promised = await chronos_client.now()
    anchors_resp = await chronos_client.anchors(limit=100)
    drift_resp = await chronos_client.drift(hours=24)
    return {
        "promised_time": promised,
        "anchors": anchors_resp.get("anchors", []),
        "anchor_count": anchors_resp.get("n", 0),
        "drift_history": drift_resp,
    }


@router.get("/api/diagnostics/measurement-confidence")
async def api_diagnostics_measurement_confidence() -> dict[str, Any]:
    """psutil vs `ps -A` cross-check — flags container/cgroup bias.

    `confidence_band` is the headline: `tight` < 5pp / 100 MB,
    `loose` < 15pp / 500 MB, `divergent` otherwise, `unknown` when
    either source isn't available.
    """
    from mindxtrain.operator.coach.cli_diagnostics import measurement_confidence

    return measurement_confidence()


@router.get("/api/diagnostics/cli-samplers")
async def api_diagnostics_cli_samplers() -> dict[str, Any]:
    """All six Linux terminal samplers in one shot."""
    from mindxtrain.operator.coach.cli_diagnostics import run_samplers

    return run_samplers()


@router.get("/api/diagnostics/runs", response_model=list[_runs.Run])
async def api_diagnostics_runs() -> list[_runs.Run]:
    """Snapshot of every run the registry currently knows about.

    Newest first. Powers the admin card's "Active runs" panel — the user
    can see at a glance what's training, what's deploying, and which
    runs have terminated. Same data as `/api/runs` (returned all-runs
    rather than filtered) but lives under /api/diagnostics/* for
    discoverability."""
    rows = _REGISTRY.list_runs()
    # Newest first by created_at.
    rows.sort(key=lambda r: r.created_at, reverse=True)
    return rows


# ---- MEI (mindX Efficiency Index) endpoints -----------------------------
# Surface the score layer for the Coach UI. The score itself is computed
# in `mindxtrain.eval.mei.score`; this layer reads the history ledger and
# exposes promotion gating to the operator.


class MEIHistoryRow(BaseModel):
    """Compact row for the Coach's history list. Maps a HistoryEntry to a
    flat shape the JS renderer can consume without nested unwrapping."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    run_id: str
    model_id: str
    promoted: bool
    composite: float
    quality: float
    decode_throughput: float
    prefill_throughput: float
    memory: float
    energy: float
    mab_provisional: bool


class MEIScoreView(BaseModel):
    """Full score plus promotion preview for one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    model_id: str
    composite: float
    quality: float
    decode_throughput: float
    prefill_throughput: float
    memory: float
    energy: float
    quality_bands: dict[str, float]
    mab_provisional: bool
    notes: list[str]
    promotable: bool
    promotion_reasons: list[str]


class MEIPromoteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    promoted: bool
    reasons: list[str] = Field(
        default_factory=list,
        description="When promoted=False, the failing-gate reasons.",
    )


def _history_row_from_entry(entry: Any) -> MEIHistoryRow:
    s = entry.score
    return MEIHistoryRow(
        timestamp=entry.timestamp,
        run_id=entry.run_id,
        model_id=entry.model_id,
        promoted=entry.promoted,
        composite=s.composite,
        quality=s.quality,
        decode_throughput=s.decode_throughput,
        prefill_throughput=s.prefill_throughput,
        memory=s.memory,
        energy=s.energy,
        mab_provisional=s.mab_provisional,
    )


@router.get("/api/mei/history", response_model=list[MEIHistoryRow])
async def api_mei_history(last: int = 20) -> list[MEIHistoryRow]:
    """Return the last N MEI history entries, newest-first.

    Empty list when no scores exist yet. The Coach UI renders this as the
    "Recent MEI scores" mini-list under the MEI card.
    """
    from mindxtrain.eval.mei import history as _mei_history

    rows = _mei_history.read_all()
    if last > 0:
        rows = rows[-last:]
    return [_history_row_from_entry(e) for e in reversed(rows)]


@router.get("/api/mei/score/{run_id:path}", response_model=MEIScoreView)
async def api_mei_score(run_id: str) -> MEIScoreView:
    """Return the most recent MEIScore for `run_id`, plus promotability.

    The run-id is the registry id used by the training pipeline. Returns
    404 when no score has been recorded for that run yet (the operator
    can rerun `mindxtrain mei score` against the run's record to populate).
    """
    from mindxtrain.eval.mei import history as _mei_history
    from mindxtrain.eval.mei.score import is_promotable

    entries = [e for e in _mei_history.read_all() if e.run_id == run_id]
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"no MEI score recorded for run_id={run_id!r}",
        )
    # Most-recent (file-order last) wins when a run was scored multiple times.
    entry = entries[-1]
    prior = _mei_history.currently_promoted()
    prior_score = prior.score if prior is not None and prior.run_id != run_id else None
    ok, reasons = is_promotable(entry.score, prior_promoted=prior_score)
    sc = entry.score
    return MEIScoreView(
        run_id=entry.run_id,
        model_id=entry.model_id,
        composite=sc.composite,
        quality=sc.quality,
        decode_throughput=sc.decode_throughput,
        prefill_throughput=sc.prefill_throughput,
        memory=sc.memory,
        energy=sc.energy,
        quality_bands=dict(sc.quality_bands),
        mab_provisional=sc.mab_provisional,
        notes=list(sc.notes),
        promotable=ok,
        promotion_reasons=reasons,
    )


@router.post("/api/mei/promote/{run_id:path}", response_model=MEIPromoteResponse)
async def api_mei_promote(run_id: str) -> MEIPromoteResponse:
    """Promote `run_id` to AgenticPlace if all §8 gates pass.

    Idempotent in the sense that repeated promotion of the same run only
    appends new history entries (each with promoted=True). The currently-
    promoted entry is whatever the last `promoted=True` row says — append-
    only ledger semantics.
    """
    from mindxtrain.eval.mei import history as _mei_history
    from mindxtrain.eval.mei.score import is_promotable

    entries = [e for e in _mei_history.read_all() if e.run_id == run_id]
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"no MEI score recorded for run_id={run_id!r}",
        )
    entry = entries[-1]
    prior = _mei_history.currently_promoted()
    prior_score = prior.score if prior is not None and prior.run_id != run_id else None
    ok, reasons = is_promotable(entry.score, prior_promoted=prior_score)
    if not ok:
        return MEIPromoteResponse(run_id=run_id, promoted=False, reasons=reasons)
    _mei_history.append(
        entry.score,
        run_id=entry.run_id,
        model_id=entry.model_id,
        model_sha256=entry.model_sha256,
        promoted=True,
    )
    return MEIPromoteResponse(run_id=run_id, promoted=True, reasons=[])


# ---- Verifiable training receipt ----------------------------------------
# The AOT-only discipline is a verification primitive: a frozen AutotunePlan
# hash bound to the checkpoint hash proves which compiled backend/heuristic/
# RCCL config produced these weights (cf. Verde/RepOps bitwise reproducibility).
# This layer re-verifies the manifest emitted at run completion and surfaces a
# "verified ✓" badge in the Coach UI.


class ReceiptHashesView(BaseModel):
    """Flat BLAKE3 hashes for the Coach receipt card. Empty string = artifact
    not produced by this run (e.g. a CPU run has no dataset/eval JSON)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_yaml: str
    checkpoint: str
    autotune_plan: str
    dataset: str
    eval_json: str


class ReceiptView(BaseModel):
    """Re-verified receipt for one run, consumed directly by coach.js."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    base_model: str
    git_sha: str
    created_at: str
    hashes: ReceiptHashesView
    verified: bool
    checks: dict[str, bool]


@router.get("/api/receipt/{run_id:path}", response_model=ReceiptView)
async def api_receipt(run_id: str) -> ReceiptView:
    """Re-verify the manifest emitted at run completion.

    404 when the run id is unknown; 409 when the run exists but hasn't produced
    a manifest yet (still training, or it failed before the receipt was sealed).
    """
    from mindxtrain.provenance.manifest import Manifest
    from mindxtrain.provenance.verify import verify_receipt

    snap = _REGISTRY.get(run_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")

    run_dir = Path(snap.out_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="no receipt yet for this run (run is still training or failed)",
        )

    manifest = Manifest.model_validate_json(manifest_path.read_text())
    plan_path = run_dir / "autotune_plan.json"
    plan_json = plan_path.read_bytes() if plan_path.is_file() else None
    config_snapshot = run_dir / "config.snapshot.yaml"

    try:
        checks = verify_receipt(
            manifest,
            config_yaml_path=config_snapshot,
            dataset_manifest_path=run_dir / "dataset_manifest.json",
            checkpoint_dir=run_dir / "checkpoint",
            eval_json_path=run_dir / "eval" / "lm_eval.json",
            plan_json=plan_json,
        )
    except (FileNotFoundError, NotADirectoryError):
        # A required artifact (config snapshot or checkpoint dir) vanished after
        # the receipt was written — report unverified rather than 500.
        checks = {
            "config_yaml": False,
            "checkpoint": False,
            "dataset": False,
            "eval_json": False,
            "autotune_plan": False,
        }

    return ReceiptView(
        run_id=manifest.run_id,
        base_model=manifest.base_model,
        git_sha=manifest.git_sha,
        created_at=manifest.created_at.isoformat(),
        hashes=ReceiptHashesView(
            config_yaml=manifest.blake3.config_yaml,
            checkpoint=manifest.blake3.checkpoint,
            autotune_plan=manifest.blake3.autotune_plan,
            dataset=manifest.blake3.dataset,
            eval_json=manifest.blake3.eval_json,
        ),
        verified=all(checks.values()),
        checks=checks,
    )
