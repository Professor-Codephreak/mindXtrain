# Changelog

All notable changes to **mindxtrain** are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Streaming chat + ollama controls** (Coach "Try the model" card). Responses now
  **stream token-by-token** (the AI-SDK text-stream pattern over SSE) via
  `POST /coach/api/chat/stream`, with a **model picker** (local models first, no more
  silently-broken `:cloud` default) and **start/stop/status** for the local ollama
  server (`/coach/api/ollama/{status,start,stop}`). Fixes the chat that returned no
  visible response. Reference: `docs/Vercel AI SDK 6_ … .md`.

- **Ollama Modelfile builder** (`mindxtrain.deploy.modelfile`) — render a valid
  `Modelfile` from a typed `ModelfileSpec` (every instruction: FROM/SYSTEM/TEMPLATE/
  ADAPTER/LICENSE/MESSAGE/REQUIRES + the full PARAMETER catalogue). A standalone Coach
  window at `/coach/modelfile` exposes a toggle + input for every instruction and
  parameter; `POST /coach/api/modelfile/{build,create}` render it and run `ollama create`.
- **Default personas + skills** (`mindxtrain.data.personas`) — built-in personas
  (codephreak / assistant / mentor) and toggleable **skills** (software engineer, platform
  architect, bash, solidity) that mix in-domain exchanges into a script. The Create-script
  card gains a persona picker + skill toggles; `derive_training_params` auto-tunes CPU
  imprint epochs/grad_accum from the dataset size.

- **Governance layer** (`mindxtrain.governance`) — classroom / boardroom / dojo,
  a clean-room reimplementation of openmindx/openmind's Boardroom-consensus +
  Dojo-evaluation behaviour. The **classroom** graduates an actor on its imprint;
  the **boardroom** (any N role-based members) convenes on the promotion motion →
  approved / rejected / disputed; the **dojo** settles a dispute with a panel of a
  **prime number** of judges (odd prime ≥ 3 → no tie, always resolves). See
  [`docs/governance.md`](governance.md).
- **Model-backed boardroom/dojo** (`governance.panel`) — members + judges
  deliberate with real models over any OpenAI-compatible backend
  (`deliberate`, `model_ballot`, `model_judge_ballot`). Coach **Boardroom** card +
  `GET /coach/api/boardroom/presets`, `POST /coach/api/boardroom/convene`,
  `POST /coach/api/dojo/settle` (model deliberation runs off the event loop).

## [1.0.0] — 2026-06-11

First production release. **CPU training is active end-to-end**; the GPU
(MI300X / consumer) path is code-complete and validated on CPU dry-run + unit
tests, pending real ROCm hardware to execute. Honest per-module status is in
[`docs/actualization_status.md`](actualization_status.md).

### Added (1.0.0)

- **Device-aware local-GPU lane** (`trl_local`,
  `mindxtrain.train.backend_trl_cpu.run_trl_local`). Auto-detects a consumer GPU
  (CUDA or ROCm Radeon, bf16/fp16) and falls back to CPU; `trl_cpu` is the
  force-CPU wrapper. Recipe `mindx_fallback_qwen3_1_5b_local`. Honest limit:
  integrated Vega/`gfx90c` APUs are unsupported and fall back to CPU.
- **Verifiable training receipt.** Every operator/CPU run emits `manifest.json`
  binding the frozen `AutotunePlan` hash to the checkpoint + config hashes
  (`provenance.manifest.emit_receipt_for_run`); re-verified via `mindxtrain
  receipt`, the `GET /coach/api/receipt/{run_id}` endpoint, and a Coach
  "Verifiable receipt" card. Optional x402 metering gate on `/v1/training/jobs`.
- **Actor / persona / script + imprint.** Author a training *script* for an
  *actor* (`mindxtrain.data.scripts`), ingest as `source: local`, and measure the
  persona **imprint** by recall before/after training
  (`mindxtrain.eval.imprint`, `mindxtrain imprint`,
  `POST /coach/api/imprint/score`). Coach **Create script** card +
  `POST/GET /coach/api/datasets`. Recipe `mindx_persona_imprint_local`.
- **mindX dream-cycle trigger** (`deploy.api_client.trigger_dream_ingestion`) —
  hands an imprinted actor to mindX's `machine.dream` 8hr cycle via HTTP or an
  inbox drop (clean-room: a pointer, never mindX code). `mindxtrain imprint
  --trigger-dream`.
- **Coach live-training diagnostics** — accurate depiction with accordion
  compression: rolling loss-chart window, per-step metrics + log accordions with
  honest "showing last N" counts, system-metric sparklines, chat re-probe.
- **Autotune on real hardware** — runtime GPU-count autodetect (`rccl_probe`),
  GEMM microbenchmark with timings (`gemm_probe`), `serve --to sglang`.
- **Clean-room policy** codified in `CLAUDE.md` + `AGENTS.md`: reimplement/adapt
  locally, never copy mindX/external source bytes.

### Added (pre-1.0 foundation)

- **mindX self-training loop.** `mindx_dreams` dataset source adapter walks
  `<mindx>/data/memory/ltm/*/*_training.jsonl`, deduplicates by content hash,
  and yields OpenAI-chat rows. Pure stdlib; no GPU/heavy deps to load
  (`mindxtrain.data.sources.mindx_dreams`).
- **CPU training lane** (`mindxtrain.train.backend_trl_cpu.run_trl_cpu`).
  Real TRL SFT/LoRA on CPU, produces a real HF-format checkpoint compatible
  with `quantize`/`receipt`/`publish`. Wired via `TrainingBackend = "trl_cpu"`.
- **Public training-jobs API** at `/v1/training/jobs` with bearer auth
  (`MINDXTRAIN_API_KEY`). Versioned facade over the same `RunRegistry` Coach
  uses, so mindX agents and external clients become peer dispatchers
  (`mindxtrain.operator.training_api`).
- **mindX fallback-swap caller** (`mindxtrain.deploy.api_client.swap_mindx_fallback_model`).
  PATCHes mindX's `/v1/config/fallback-model` so a freshly published HF Hub
  checkpoint becomes mindX's active default without a source edit. Pairs with
  the mindX-side endpoint shipped in AgenticPlace/mindX@ad8193ea3.
- Two new YAML recipes: `mindx_fallback_qwen3_1_5b_sft_lora` (Qwen3-1.5B
  LoRA on MI300X, the production target) and `mindx_fallback_qwen3_1_5b_cpu_smoke`
  (SmolLM2-135M CPU smoke).
- `.env.example`: `MINDXTRAIN_API_KEY` and `MINDXTRAIN_MINDX_HOME`.

### Changed

- Schema: `DataSource` extends to include `"mindx_dreams"`; `DataCfg.hf_id`
  is now defaultable with a `model_validator` requiring it only for
  `source: "hf"`; `DataCfg.path` added (used by `local` + `mindx_dreams`).
- `HardwareCfg.gpus: Literal[0, 1, 8]` — `0` = CPU lane.
- Production URL flipped from `mindx.pythai.net/hackathon` to
  `mindx.pythai.net/coach`. Hackathon-era references preserved in
  `HACKATHON.md` and the build-in-public posts for the historical record.
- CI: ruff scoped to `mindxtrain/ tests/`; mypy points at the canonical
  `mindxtrain/config mindxtrain/provenance` (fixing stale paths). Container
  build added on push-to-main; GHCR publish on tag.

### Notes

- Adapter smoke against the real corpus saw 1051 unique rows in
  `/home/hacker/mindX/data/memory` (corpus snapshot 2026-05-14).

## [0.1.0] — 2026-05-06

Initial public release. Submitted to the AMD × lablab.ai Developer Hackathon
(build window May 4–10 2026, on-site finale May 9–10 in San Francisco).

### Added

- Single-package canonical layout per `docs/blueprints/mindxtrain2.md` §Part 4:
  `mindxtrain/{cli,config,data,models,train,eval,autotune,operator,storage,provenance,deploy,budget}`.
- CLI with 8 verbs: `init`, `bench`, `train`, `eval`, `quantize`, `serve`,
  `publish`, `receipt` (`mindxtrain.cli.main`).
- 60-second AOT autotune probe for AMD MI300X — the hackathon differentiator.
  CK vs Triton attention selection + hipBLASLt GEMM heuristic + RCCL config
  (`mindxtrain.autotune.{benchmark,attention_probe,gemm_probe,rccl_probe,plan}`).
- Pydantic v2 `XTrainConfig` with discriminated union over 9 training methods
  (full, lora, qlora, dpo, orpo, grpo, gspo, kto, cpt) (`mindxtrain.config.schema`).
- 12 YAML training recipes covering Qwen3.5/Qwen3.6/Instella across
  SFT-LoRA, full-FSDP, DPO, ORPO, GRPO, CPT, and VL.
- Axolotl YAML compiler; alt backends (Unsloth, torchtune, Primus) wired as
  dispatch stubs (`mindxtrain.train`).
- Provenance manifest with BLAKE3 content addressing, ROCm/git/gfx capture,
  and on-chain pointers (ERC-7857 INFT, Algorand ASA, ERC-8004 attestation)
  (`mindxtrain.provenance`).
- Operator FastAPI app exposing `/v1/chat/completions`, `/v1/agentic`, and
  the interactive `/coach/` UI (`mindxtrain.operator`).
- Pluggable inference backends: vLLM, OpenAI-compatible
  (`mindxtrain.operator.backends`).
- Pluggable storage providers: local fs, HF Hub, Lighthouse, IPFS
  (`mindxtrain.storage`).
- Foundry contracts for ERC-8004 attestation registry (`contracts/`).
- Containerfiles + compose + k8s manifests for MI300X (`ops/`).

### Notes

- This release replaces the earlier 3-package layout
  (`mindXtrain/`, `automindXtrain/`, `custmodel/`) with a single ordered
  package `mindxtrain/`. Old import paths (`xtrain.*`, `automindx.*`,
  `custmodel.*`) are not preserved — adopters must rewrite to `mindxtrain.*`.
- Most operator and trainer surfaces ship as honest minimal stubs that raise
  `NotImplementedError` for paths not yet implemented in the hackathon scope.
  The autotune probe, config schema, manifest, recipes, and Axolotl compiler
  are the production-ready paths.

[0.1.0]: https://example.invalid/mindxtrain/releases/tag/v0.1.0
