# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mindxtrain` is a single-package Python training framework for fine-tuning open-weight LLMs on AMD MI300X and serving them through an OpenAI-compatible API. The architectural differentiator is a **60-second AOT autotune probe** (`mindxtrain bench`) that fixes attention backend (CK vs Triton), GEMM heuristic, and RCCL config at training start — **JIT autotune is forbidden in the production training loop**.

Built for the AMD × lablab.ai hackathon (May 4–10 2026). The base install is CPU-only and runs the CLI, Coach UI, `bench --dry-run`, manifest verify, and the operator FastAPI; heavyweight paths gate on opt-in dep groups.

## Commands

The repo uses `uv` with Python 3.12 (pinned `>=3.12,<3.13`). All commands run from the repo root.

```bash
uv sync                         # base install (CPU-only; 112 tests pass)
uv sync --extra ml --extra eval --extra data    # opt into heavyweight groups
uv sync --all-extras            # everything except amd-quark (ships in container)

# Standard local cycle — CI runs the same:
uv run ruff check .                                       # lint
uv run mypy mindxtrain/config mindxtrain/provenance       # mypy --strict (only these two)
uv run pytest -q                                          # → 112 passed
uv run pytest tests/test_config_schema.py -q              # single test file
uv run pytest tests/test_config_schema.py::test_xgmi_2gpu_rejected   # single test

# CLI entry point (typer; 9 verbs):
uv run mindxtrain --help
uv run mindxtrain init --list                             # list 12 built-in YAML recipes
uv run mindxtrain init --template qwen3_8b_sft_lora --out run.yaml
uv run mindxtrain bench --dry-run --out plan.json         # CPU-safe (real probe needs MI300X)
uv run mindxtrain receipt ./out/runs/<name>/manifest.json --config run.yaml

# Operator FastAPI + Coach UI (no GPU required):
uv run uvicorn mindxtrain.operator.app:app --host 0.0.0.0 --port 8080
# → http://localhost:8080/coach/
```

GPU verbs (`bench` without `--dry-run`, `train`, `quantize`, `serve`) require an AMD MI300X with ROCm 7.2.1 inside `rocm/primus:v26.2`. The full operator path is in `HANDOFF.md`.

Solidity contracts live in `contracts/` (Foundry, solc 0.8.26): `forge install && forge test` from inside `contracts/`.

## Architecture (concentric layers)

The codebase is organized so each inner layer is consumed by the next, never the reverse:

1. **CLI** (`mindxtrain/cli/main.py`, typer) — `init | bench | train | dataset prep | eval | quantize | serve | publish | receipt`. Never reaches into the training backend; consumes a Pydantic-validated config + `AutotunePlan` and dispatches downward.
2. **Autotune** (`mindxtrain/autotune/`) — the differentiator. Emits `AutotunePlan` JSON, AOT-only.
3. **Dataset** (`mindxtrain/data/`) — curate → dedupe (MinHash + SemDeDup) → filter → tokenize → pack → synth → verify.
4. **Training** (`mindxtrain/train/`) — backend dispatch (`dispatch.py` → axolotl / unsloth / torchtune / primus / TRL in-process). Methods: SFT, DPO, ORPO, GRPO, GSPO, RLHF, tool-use, CPT.
5. **Artifact + Integration** (`mindxtrain/{eval,deploy,storage,provenance,operator}`) — Quark FP8/MXFP4 → lm-eval-harness → HF Hub push → Lighthouse pin → mindX register → AgenticPlace → BANKON ENS → x402 metering → ERC-8004 attestation.

Key end-to-end flow: `XTrainConfig` (Pydantic) + `AutotunePlan` → `dispatch_training()` → `checkpoint_dir/` → `eval.json` → `quantized/` → `manifest.json` (BLAKE3 of YAML+dataset+ckpt+eval, plus HF/Lighthouse/INFT/ASA pointers) → operator serves on `/v1/chat/completions`. `mindxtrain receipt` re-hashes and verifies the manifest round-trip.

Recipes live as YAML at `mindxtrain/train/recipes/<name>.yaml` and are auto-picked up by `mindxtrain init --list` and validated by `tests/test_config_schema.py::test_all_recipes_validate`.

## Non-negotiable invariants

These are encoded in the schema/recipes; violating them is a deployment bug, not a style issue.

1. **AOT-only.** No `torch.compile(mode="max-autotune")` in production paths. No JIT autotune in vLLM (`VLLM_USE_TRITON_FLASH_ATTN=0` if needed). `autotune.policy: aot_only` is the YAML contract.
2. **`hardware.gpus: Literal[1, 8]` only.** 2/4-GPU MI300X FSDP groups hit asymmetric xGMI bandwidth — schema rejects them at parse time. (Tested in `test_config_schema.py::test_xgmi_2gpu_rejected`, `test_distributed.py`.)
3. **Seven MI300X env vars** are defaults in every recipe's `train.env` (autotune plan may override values, never remove keys): `PYTORCH_ROCM_ARCH=gfx942`, `HSA_NO_SCRATCH_RECLAIM=1`, `HIP_FORCE_DEV_KERNARG=1`, `GPU_MAX_HW_QUEUES=1`, `NVTE_CK_USES_BWD_V3=1`, `NVTE_CK_IS_V3_ATOMIC_FP32=1`, `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1`, `NCCL_MIN_NCHANNELS=112`.
4. **`extra: forbid` + `frozen: true`** on every Pydantic model — unknown YAML keys raise `ValidationError`; loaded configs are immutable.
5. **Solidity contracts are write-once.** No proxies, no `Ownable`, no admin keys, no setters in `contracts/src/{mindxtrain_registry,x402_receiver}.sol`. Rotating any parameter requires a fresh deploy.
6. **Numpy pinned `<2.0`** against `torch==2.9.1+rocm7.2.1.lw`.
7. **Container is `rocm/primus:v26.2`**; SHA256 digest snapshot in `ops/containerfiles/digest.lock`.

## Lazy-import pattern (mandatory for optional deps)

Optional dep groups: `ml` (trl, transformers, peft, accelerate, datasets), `eval` (lm-eval, lighteval, inspect-ai, jinja2), `data` (datasketch, sentence-transformers, faiss-cpu, pyarrow), `serve` (vllm), `chain` (web3, py-algorand-sdk, huggingface-hub), `obs` (opentelemetry-sdk, prometheus-client, psutil).

Every module that wants an optional dep guards the import inside the function that needs it. `import mindxtrain.eval.harness` must always succeed even without `--extra eval`. Error messages must include the exact `uv sync --extra <group>` to run. New modules taking optional deps must follow this pattern.

## Reuse boundaries

- **From `/home/hacker/mindX/`** (production codebase): Codephreak persona JSON loaded at runtime via `MINDXTRAIN_PERSONA_PATH`. Do not copy file bytes — load via env var.
- **Not** from `/home/hacker/aglm/` — broken per its own README.

## Adding things

- **New recipe** → drop YAML at `mindxtrain/train/recipes/<name>.yaml`; `test_all_recipes_validate` picks it up.
- **New training backend** → add `mindxtrain/train/backend_<name>.py` exposing `run_<name>(cfg, plan, out_dir) -> Path`; wire into `train/dispatch.py`; add to `TrainingBackend` literal in `config/schema.py`.
- **New operator backend** → subclass `Backend` in `mindxtrain/operator/backends/<name>.py` decorated `@register_backend("<name>")`; side-effect import from `models/registry.py`; add runtime branch in `operator/app.py::chat_completions`.
- **New training method** → add `_MethodBase` subclass in `config/schema.py` with `kind: Literal["<name>"]`; extend `TrainMethod` discriminated union; add `train/<name>.py` runner; update dispatch; add a recipe.

## Documentation hub

| Doc | What it covers |
|-----|----------------|
| `HANDOFF.md` | 11-step operator checklist (local → MI300X droplet → submission). |
| `docs/architecture.md` | 5-layer architecture + MI300X invariants + data flow. |
| `docs/development.md` | Toolchain, optional-deps, lazy-import pattern, debugging table. |
| `docs/actualization_status.md` | Per-module map of what's real vs. requires extras. |
| `docs/autotune.md` | The 60-second AOT probe — the differentiator. |
| `docs/cli.md` | Every verb with synopsis, options, exit codes. |
| `docs/yaml_schema.md` | Every field of the 10-section `XTrainConfig`. |
| `docs/coach.md` | Interactive `/coach/` web UI bundled in the operator. |
| `docs/blueprints/` | Frozen source design briefs (the spec the project was built against). |
