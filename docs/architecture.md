# Architecture

`mindxtrain` is a single-package training framework producing checkpoints with
verifiable provenance, served through an OpenAI-compatible API. The repository
is organized per `docs/blueprints/mindxtrain2.md` §Part 4.

```
mindxtrain/
├── cli/         entry point (typer): init|bench|train|eval|quantize|serve|publish|receipt
├── config/      Pydantic schema + JSON / YAML loaders
├── data/        curate -> dedupe -> filter -> tokenize -> pack -> synth -> verify
├── models/      ModelRegistry + ChatTemplate + per-base presets
├── train/       sft, dpo, grpo, rlhf, tool_use, distributed, callbacks, recipes/*.yaml
├── eval/        lighteval, inspect_ai, bfcl, persona/agenda regression, tau_bench, card
├── autotune/    60-second AOT MI300X probe (the differentiator)
├── operator/    FastAPI app, Coach UI, ml-intern patterns (tool_router, agent_loop, …)
├── storage/     StorageProvider interface + local_fs / hf_hub / lighthouse / ipfs
├── provenance/  TrainingRun manifest, BLAKE3, ERC-8004, Algorand, x402
├── deploy/      content-addressed registry, hot_swap, ab_test, vllm/sglang launchers, quark
└── budget/      psutil-derived ResourceBudget + per-provider pricing
```

## The five conceptual layers

The codebase is concentric — each inner layer is consumed by the next, never
the reverse.

```
┌──────────────────────────────────────────────────────────┐
│ 1. CLI layer (typer)                                     │
│    init | bench | train | dataset prep | eval | quantize │
│    serve | publish | receipt                             │
├──────────────────────────────────────────────────────────┤
│ 2. Autotune layer (60s AOT probe — DIFFERENTIATOR)       │
│    attention_probe (CK vs Triton) │ gemm_probe │ rccl    │
│      ↓                                                   │
│    AutotunePlan (JSON, AOT — JIT autotune is forbidden)  │
├──────────────────────────────────────────────────────────┤
│ 3. Dataset layer                                         │
│    HF datasets streaming → MinHash + SemDeDup → packing  │
│    → FSDP sharding → Lighthouse-pinned CIDs              │
├──────────────────────────────────────────────────────────┤
│ 4. Training layer (backend dispatch)                     │
│    axolotl │ unsloth │ torchtune │ primus                │
│    LoRA │ QLoRA │ full SFT │ DPO │ ORPO │ GRPO │ GSPO    │
├──────────────────────────────────────────────────────────┤
│ 5. Artifact + Integration layer                          │
│    Quark FP8 / MXFP4 → lm-eval-harness                   │
│    → HF Hub push → Lighthouse pin                        │
│    → mindX register → AgenticPlace listing               │
│    → BANKON ENS subname → x402 Algorand metering         │
└──────────────────────────────────────────────────────────┘
```

The CLI never reaches into the training backend; it consumes the autotune plan
and a Pydantic-validated config and dispatches downward through
`mindxtrain/train/dispatch.py`. The training backend never reaches up to the
CLI; it returns a checkpoint directory that the artifact layer consumes.

## Autotune is the spine

The single architectural choice that distinguishes mindxtrain from Axolotl,
LLaMA-Factory, Unsloth, torchtune and Primus is the autotune layer. It runs a
**60-second MI300X micro-benchmark** (CK-vs-Triton SDPA, hipBLASLt heuristic
check, RCCL bus-bandwidth probe) and emits a static `AutotunePlan` JSON
consumed at training start.

**AOT-only — JIT autotune is forbidden in production.** The plan is fixed at
training start; no Triton / Inductor / MIOpen JIT autotune runs in the live
training loop. This is reproducible, latency-stable, and the point of the
entire framework.

See [autotune.md](autotune.md) for the full probe taxonomy and the
`AutotunePlan` schema.

## MI300X-specific invariants (non-negotiable)

These are encoded in the schema and the recipe library; violating them is a
deployment bug.

1. **FSDP topology must be 1- or 8-GPU** (`hardware.gpus: Literal[1, 8]`). The
   2- and 4-GPU groups have asymmetric xGMI bandwidth on MI300X — kills
   throughput silently. Enforced by the schema; tested in
   `tests/test_config_schema.py`.
2. **`PYTORCH_ROCM_ARCH=gfx942`** must be set; AOTriton compiles for the GPU
   arch and `gfx942` is MI300X. Default in every recipe's `train.env`.
3. **`HSA_NO_SCRATCH_RECLAIM=1`** + **`HIP_FORCE_DEV_KERNARG=1`** +
   **`GPU_MAX_HW_QUEUES=1`** — the three runtime knobs that make Primus-Turbo
   MI300X paths stable. Default in every recipe's `train.env`.
4. **Numpy must be pinned `<2.0`** against `torch==2.9.1+rocm7.2.1.lw`. Pinned
   in the project `pyproject.toml`.
5. **Container is `rocm/primus:v26.2`**; SHA256 digest snapshot lives in
   `ops/containerfiles/digest.lock`.

## End-to-end data flow

```
examples/demo_qwen3_8b_sft.yaml
        │
        ├─[parse, validate]─► XTrainConfig (Pydantic v2)
        │
        ├─[mindxtrain bench]─► AutotunePlan {ck/triton, gemm, rccl, …}
        │                           │
        │                           ▼
        ├─[mindxtrain train]──► dispatch_training(cfg, plan, out_dir)
        │                           │
        │                           ▼ (Axolotl YAML, env vars set)
        │                       checkpoint_dir/  + train.log
        │                           │
        ├─[mindxtrain eval]─────►   eval.json (lm-eval-harness)
        │                           │
        ├─[mindxtrain quantize]─►   checkpoint_dir/quantized/  (Quark FP8 PTPC)
        │                           │
        └─[mindxtrain publish]──►   Manifest with BLAKE3 of YAML+dataset+
                                    checkpoint+eval, plus HF/Lighthouse/
                                    INFT/ASA pointers
                                                 │
                                                 ▼
                                    mindxtrain.operator.app serves the FP8
                                    weights on /v1/chat/completions
```

`mindxtrain receipt` re-hashes the artifacts and verifies the BLAKE3 fields
against the manifest. That round-trip is the cypherpunk2048 reproducibility
guarantee.

## Model strategy (per mindxtrain2.md §Part 6)

mindxtrain targets a **family**, not a single flagship: edge → mid →
flagship → specialist. Per the rigorous comparison in
[`blueprints/mindXtrain2.md`](blueprints/mindXtrain2.md) §Part 6:

- **Primary base = Qwen3.5** (Apache-2.0, contiguous family from 0.6 B →
  235 B → Qwen3.5-122B-A10B; mature PEFT/Axolotl/Unsloth recipes; BFCL
  leadership in the Qwen lineage).
- **Specialist track = GLM-5.1** (MIT-licensed weights; SOTA SWE-Bench Pro
  58.4; long-horizon agentic reasoning with 200 K-context DSA). Used where
  8-hour autonomous SWE sessions matter; otherwise overkill.
- DeepSeek V3.2 / Mistral Large 3 / Phi-4-mini / Gemma 4 are watchlist or
  jurisdictional secondary tracks.

The five `mindxtrain.models.{glm51,qwen35,deepseek_v32,mistral3,phi4_mini}.py`
preset modules auto-register on import so `mindxtrain init` and the
`ModelRegistry` know all five from day one.

## Actualization status

The framework ships with **38 modules actualized** as real Python on a
CPU-only laptop, **6 optional-dep groups** (`ml`, `eval`, `data`, `serve`,
`chain`, `obs`) for the heavyweight paths, and **5 cloud-provider stubs**
preserved in `budget/providers/*` for post-hackathon work. Per-module map at
[actualization_status.md](actualization_status.md).

Verification gate that should always pass on the base install:

```bash
uv run pytest -q          # → 112 passed
uv run ruff check .       # clean
```

## What lives outside the Python tree

- `contracts/` — Foundry workspace for `mindxtrain_registry.sol` (write-once
  anchor) and `x402_receiver.sol` (immutable facilitator). No proxies, no
  admin keys, no setters.
- `examples/` — `demo_qwen3_8b_sft.yaml` (the hero config).
- `Containerfile`, `compose.yaml` — top-level Podman / podman-compose entries.
- `ops/` — per-role container files, compose stacks, k8s manifests, vmm and
  Gensyn definitions.
- `docs/blueprints/` — the source design briefs the project was built against.
