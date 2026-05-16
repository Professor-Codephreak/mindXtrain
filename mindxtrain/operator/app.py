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
    coach_url: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=resolve_backend_name(),
        coach_url="/coach/",
    )


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
