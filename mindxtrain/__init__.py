"""mindxtrain — production training framework for aGLM-BANKON derivatives.

Layout per mindXtrain2.md §Part 4 "Repository layout":

    cli/         entry point (typer)
    config/      Pydantic schema + JSON / YAML loaders
    data/        curate -> dedupe -> filter -> tokenize -> pack -> synth -> verify
    models/      ModelRegistry + ChatTemplate + per-base backends
    train/       sft, dpo, grpo, rlhf, tool_use, distributed, callbacks
    eval/        lighteval, inspect_ai, bfcl, persona_regression, agenda_regression, tau_bench, card
    autotune/    60-second AOT MI300X probe (the hackathon differentiator)
    operator/    ml-intern-pattern derived: tool_router, agent_loop, context, approval, FastAPI api, coach UI
    storage/     StorageProvider interface + local_fs / hf_hub / lighthouse / ipfs
    provenance/  TrainingRun manifest, BLAKE3 hashing, ERC-8004, Algorand, x402
    deploy/      content-addressed registry, hot_swap with canary, ab_test, api_client
    budget/      psutil-derived ResourceBudget (carry-over from aGLM)
"""

__version__ = "0.1.0"
