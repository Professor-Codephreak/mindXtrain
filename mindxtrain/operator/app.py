"""automindXtrain FastAPI app.

Exposes:
    GET  /                        — coach UI (mindXtrain Coach)
    GET  /health                  — liveness check
    POST /v1/chat/completions     — OpenAI-compatible chat
    POST /v1/agentic              — mindX-native agentic dispatch (Day 5+)
    /v1/training/jobs/*           — public training-jobs API (mindX agents,
                                    external clients). Bearer auth via
                                    MINDXTRAIN_API_KEY when set.
    GET  /coach/*                 — Coach UI + API (recipes, autotune, cost)

The production deployment lives at https://mindx.pythai.net — the Coach UI
is at /coach/ and the public training-jobs API is at /v1/training/jobs.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mindxtrain import __version__
from mindxtrain.models.registry import ChatRequest, ChatResponse, build_backend
from mindxtrain.operator.coach import router as coach_router
from mindxtrain.operator.training_api import router as training_router

# ---- backend resolution --------------------------------------------------


def _ollama_reachable(timeout_s: float = 1.0) -> bool:
    """Probe ollama at MINDXTRAIN_OLLAMA_BASE_URL.

    Used by auto-detect to pick `ollama` as the default backend on hosts
    where ollama is the only thing running (e.g., the laptop dev
    environment). Strips `/v1` from the configured base URL because
    ollama's health-style endpoint is `/api/tags`, not OpenAI-shaped.
    """
    base = os.environ.get("MINDXTRAIN_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    probe_url = base.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            return client.get(probe_url).status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _vllm_reachable(timeout_s: float = 1.0) -> bool:
    """Probe vLLM at MINDXTRAIN_VLLM_BASE_URL.

    Hits `/v1/models` — the OpenAI-compatible models listing endpoint
    vLLM always exposes. This is what flips the production Coach chat
    card on the MI300X droplet from "(no backend configured)" to
    live, and what `/health` consults so a load balancer knows when
    inference is actually warm.
    """
    base = os.environ.get(
        "MINDXTRAIN_VLLM_BASE_URL",
        os.environ.get("AUTOMINDX_VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    probe_url = base.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            return client.get(probe_url).status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _vllm_first_model() -> str | None:
    """Return the first model id vLLM lists, or None on failure.

    Lets `/coach/api/health` render "vllm (Qwen/Qwen3-8B) ready" in
    prod the same way ollama does on the laptop. Best-effort: probe
    failure → None and the UI degrades to just the backend name.
    """
    base = os.environ.get(
        "MINDXTRAIN_VLLM_BASE_URL",
        os.environ.get("AUTOMINDX_VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    probe_url = base.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(probe_url)
            if resp.status_code != 200:
                return None
            body = resp.json()
        models = body.get("data", [])
        if not models:
            return None
        first = models[0]
        return first.get("id") if isinstance(first, dict) else None
    except (httpx.HTTPError, OSError, ValueError, IndexError):
        return None


def backend_reachable(name: str) -> bool:
    """Live probe for a backend by name. Used by both /health and /coach health."""
    if name == "ollama":
        return _ollama_reachable()
    if name == "vllm":
        return _vllm_reachable()
    # openai_compat and unknown backends: we don't have a generic probe,
    # so the chat-completions failure path remains the authoritative signal.
    return False


def backend_first_model(name: str) -> str | None:
    """Best-effort first-model lookup; None when the backend doesn't list one."""
    if name == "ollama":
        return ollama_first_model()
    if name == "vllm":
        return _vllm_first_model()
    return None


def resolve_backend_name() -> str:
    """Pick the active backend.

    Resolution order:
    1. Explicit `MINDXTRAIN_BACKEND` env var (canonical).
    2. Legacy `AUTOMINDX_BACKEND` (back-compat with the pre-rename code).
    3. Auto-detect: ollama if reachable on localhost:11434, else vllm.
    """
    explicit = (
        os.environ.get("MINDXTRAIN_BACKEND")
        or os.environ.get("AUTOMINDX_BACKEND")
    )
    if explicit:
        return explicit
    if _ollama_reachable():
        return "ollama"
    return "vllm"


def ollama_first_model() -> str | None:
    """Return the name of the first model ollama lists, or None on failure.

    Used by the Coach health endpoint to render
    `ollama (qwen3:0.6b) ready` instead of just `ollama ready`. Best-effort:
    a timeout / parse failure returns None, the UI still shows the backend
    name without a model qualifier.
    """
    base = os.environ.get("MINDXTRAIN_OLLAMA_BASE_URL", "http://localhost:11434/v1")
    probe_url = base.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(probe_url)
            if resp.status_code != 200:
                return None
            body = resp.json()
        models = body.get("models", [])
        # Prefer local (non-cloud) models first; the user's qwen3:0.6b
        # ranks ahead of glm-5.1:cloud, deepseek-v4-pro:cloud, etc.
        local = [m for m in models if ":cloud" not in (m.get("name") or "")]
        chosen = (local or models)[0] if (local or models) else None
        return chosen.get("name") if chosen else None
    except (httpx.HTTPError, OSError, ValueError, IndexError):
        return None

app = FastAPI(
    title="automindXtrain",
    version=__version__,
    description="Pluggable LLM cognitive runtime for the mindXtrain pipeline.",
)

# --- coach UI -------------------------------------------------------------

_COACH_STATIC = Path(__file__).parent / "coach" / "static"
app.mount("/coach/static", StaticFiles(directory=_COACH_STATIC), name="coach-static")
app.include_router(coach_router)
app.include_router(training_router)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Land on the Coach UI."""
    return RedirectResponse(url="/coach/")


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    backend_ready: bool
    backend_model: str = ""
    coach_url: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness — always 200.

    `status` is "ok" even when the backend is unreachable so simple
    load-balancer health checks don't take the operator out of
    rotation just because vLLM is still warming. The structured
    `backend_ready` field is what an inference-aware probe should
    consult; `/readyz` enforces it as the HTTP status.
    """
    backend = resolve_backend_name()
    ready = backend_reachable(backend)
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=backend,
        backend_ready=ready,
        backend_model=(backend_first_model(backend) or "") if ready else "",
        coach_url="/coach/",
    )


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict[str, object]:
    """Readiness gate — 503 when the resolved backend is unreachable.

    Use this when you want a probe that *fails* until inference is
    actually warm (e.g., k8s readiness probe, uptime monitor that
    pages on inference outage rather than process death).
    """
    backend = resolve_backend_name()
    if not backend_reachable(backend):
        raise HTTPException(
            status_code=503,
            detail={"backend": backend, "reachable": False},
        )
    return {"backend": backend, "reachable": True}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    backend_name = resolve_backend_name()
    backend_kwargs: dict[str, object] = {}
    if backend_name == "vllm":
        backend_kwargs["base_url"] = os.environ.get(
            "MINDXTRAIN_VLLM_BASE_URL",
            os.environ.get("AUTOMINDX_VLLM_BASE_URL", "http://localhost:8000/v1"),
        )
    elif backend_name == "ollama":
        backend_kwargs["base_url"] = os.environ.get(
            "MINDXTRAIN_OLLAMA_BASE_URL", "http://localhost:11434/v1",
        )
    elif backend_name == "openai_compat":
        backend_kwargs["base_url"] = os.environ["MINDXTRAIN_OPENAI_BASE_URL"]
        backend_kwargs["api_key"] = os.environ.get("MINDXTRAIN_OPENAI_API_KEY", "")

    try:
        backend = build_backend(backend_name, **backend_kwargs)
        return await backend.chat(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/agentic")
async def agentic() -> dict[str, str]:
    """mindX-native agentic endpoint (Day 5+)."""
    raise HTTPException(status_code=501, detail="TODO Day 5: wire mindX MASTERMIND dispatch")
