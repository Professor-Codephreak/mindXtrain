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

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mindxtrain import __version__
from mindxtrain.models.registry import ChatRequest, ChatResponse, build_backend
from mindxtrain.operator.coach import router as coach_router
from mindxtrain.operator.training_api import router as training_router

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
        backend=os.environ.get("AUTOMINDX_BACKEND", "vllm"),
        coach_url="/coach/",
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    backend_name = os.environ.get("AUTOMINDX_BACKEND", "vllm")
    backend_kwargs: dict[str, object] = {}
    if backend_name == "vllm":
        backend_kwargs["base_url"] = os.environ.get(
            "AUTOMINDX_VLLM_BASE_URL",
            "http://localhost:8000/v1",
        )
    elif backend_name == "openai_compat":
        backend_kwargs["base_url"] = os.environ["AUTOMINDX_OPENAI_BASE_URL"]
        backend_kwargs["api_key"] = os.environ.get("AUTOMINDX_OPENAI_API_KEY", "")

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
