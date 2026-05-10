# AGENTS.md

Canonical entry point for any agent (Claude Code, Codex, Cursor, Copilot, etc.) working in this repository. Read this first.

## TL;DR

`mindxtrain` is a single-package Python framework that fine-tunes open-weight LLMs on AMD MI300X and serves them OpenAI-compatible. The differentiator is a **60-second AOT autotune probe** (`mindxtrain bench`) — the plan is fixed at training start; **JIT autotune is forbidden in the production loop**.

The base install is CPU-only and runs the CLI, Coach UI, `bench --dry-run`, manifest verify, and the operator FastAPI. Heavyweight paths gate on opt-in `--extra` groups.

## Where to look

| You want to… | Read |
|---|---|
| Understand the architecture and invariants | [`CLAUDE.md`](CLAUDE.md), [`docs/architecture.md`](docs/architecture.md) |
| Take the repo from "code done" to "demo live" | [`HANDOFF.md`](HANDOFF.md) (11 ordered steps) |
| Know what's real Python vs. what needs `--extra <group>` | [`docs/actualization_status.md`](docs/actualization_status.md) |
| Add a recipe / training backend / operator backend / training method | [`docs/development.md`](docs/development.md) §"Adding…" |
| Look up every CLI verb's options + exit codes | [`docs/cli.md`](docs/cli.md) |
| Look up every YAML field | [`docs/yaml_schema.md`](docs/yaml_schema.md) |
| Understand the autotune probe | [`docs/autotune.md`](docs/autotune.md) |
| Understand the Coach UI | [`docs/coach.md`](docs/coach.md) |
| See the frozen design briefs | [`docs/blueprints/`](docs/blueprints/) |

## Verification gates (must all pass before pushing)

```bash
uv run ruff check .
uv run mypy mindxtrain/config mindxtrain/provenance
uv run pytest -q       # → 112 passed
```

CI runs the same three commands on Ubuntu 24.04 / Python 3.12 (CPU-only).

## Non-negotiable invariants

These are encoded in the schema/recipes; violating them is a deployment bug. Full reasoning in [`docs/development.md`](docs/development.md).

1. **AOT-only** — no `torch.compile(mode="max-autotune")`, no JIT autotune in vLLM. The YAML field `autotune.policy: aot_only` is the contract.
2. **`hardware.gpus: Literal[1, 8]`** — 2/4-GPU MI300X FSDP groups hit asymmetric xGMI; schema rejects them at parse time.
3. **Seven MI300X env vars** are defaults in every recipe's `train.env` (`PYTORCH_ROCM_ARCH=gfx942`, `HSA_NO_SCRATCH_RECLAIM=1`, `HIP_FORCE_DEV_KERNARG=1`, `GPU_MAX_HW_QUEUES=1`, `NVTE_CK_USES_BWD_V3=1`, `NVTE_CK_IS_V3_ATOMIC_FP32=1`, `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1`, `NCCL_MIN_NCHANNELS=112`).
4. **`extra: forbid` + `frozen: true`** on every Pydantic model.
5. **Solidity contracts are write-once** — no proxies, no `Ownable`, no admin keys, no setters.
6. **Numpy pinned `<2.0`** against `torch==2.9.1+rocm7.2.1.lw`.
7. **Lazy imports for optional deps** — `import mindxtrain.eval.harness` must succeed even without `--extra eval`. Error messages must include the exact `uv sync --extra <group>` to run.
8. **Reuse boundaries** — Codephreak persona JSON loads at runtime via `MINDXTRAIN_PERSONA_PATH` from `/home/hacker/mindX/`, not by copying bytes. Never reuse `/home/hacker/aglm/` (broken per its own README).

## Quick commands

```bash
uv sync                                                       # base, CPU-only
uv sync --all-extras                                          # everything except amd-quark
uv run mindxtrain --help                                      # 9 verbs
uv run mindxtrain init --list                                 # 12 built-in YAML recipes
uv run mindxtrain bench --dry-run --out plan.json             # synthetic plan, no GPU
uv run mindxtrain receipt manifest.json --config run.yaml     # BLAKE3 round-trip
uv run uvicorn mindxtrain.operator.app:app --port 8080        # → /coach/ UI
```

GPU verbs (`bench` without `--dry-run`, `train`, `quantize`, `serve`) require an AMD MI300X with ROCm 7.2.1 inside `rocm/primus:v26.2`. See [`HANDOFF.md`](HANDOFF.md).

## Available skills

For Algorand-related work (the provenance/x402/ASA paths) the following skills are available and should be preferred over ad-hoc patterns:

- `algorand-typescript`, `build-smart-contracts`, `test-smart-contracts`, `call-smart-contracts` — contract authoring/testing/deploy.
- `use-algokit-utils`, `use-algokit-cli`, `troubleshoot-errors`, `implement-arc-standards` — client-side AlgoKit work.
- `search-algorand-examples` — patterns from official Algorand repos.
- `algorand-ts-migration` — TEALScript / beta → Algorand TypeScript 1.0.
- `mindx` — for cross-cutting work that touches the broader mindX system.

For the Solidity side (`contracts/`):

- `foundry-framework`, `solidity-dev`, `solidity-style-guide`, `slither-analysis`, `echidna-fuzzer`, `gas-optimization`, `hardhat-framework`.

## When in doubt

- The schema is the source of truth for "what's a valid YAML." Run `uv run python -c "from mindxtrain.config.loader import load_config; load_config('path/to.yaml')"`.
- The blueprints under `docs/blueprints/` are frozen — do not edit them; they are the historical specification.
- Per-module status: `docs/actualization_status.md`. If a module says "stub" there, do not silently turn it on; it's stubbed deliberately.
