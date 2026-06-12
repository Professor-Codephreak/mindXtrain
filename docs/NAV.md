# mindXtrain Documentation Index

Every doc lives in `docs/`. The only Markdown at the repo root is `README.md` (entry
point) plus `CLAUDE.md` / `AGENTS.md` (agent-tooling entrypoints, required at root).
Start at [Quickstart](quickstart.md); operators running the demo read [HANDOFF.md](HANDOFF.md).

## Getting started

- [Quickstart](quickstart.md) — install (with optional-dep groups), init, bench, train.
- [HANDOFF.md](HANDOFF.md) — **operator checklist**: local setup → MI300X provision → train/eval/quantize → publish → contracts → deploy.

## Architecture & invariants

- [Architecture](architecture.md) — the 5-layer single-package layout + MI300X invariants + data flow.
- [Development workflow](development.md) — toolchain, optional-deps, lazy-import pattern, invariants, **training lanes** (CPU / local-GPU / MI300X), how to add recipes/backends/methods.
- [Actualization status](actualization_status.md) — per-module map of what's real vs. needs `--extra` vs. v1.0.0 CPU-active / GPU-pending / stub.
- [Autotune deep-dive](autotune.md) — the 60-second AOT probe (the differentiator).

## Coach UI & training workflow

- [Coach UI](coach.md) — the interactive `/coach/` operator UI: create-script (personas + skills), live-training diagnostics, verifiable receipt, streaming chat + ollama controls, Modelfile builder.
- [dcoach](dcoach.md) — `/coach/dcoach`: prove a CPU-trained model recalls its training (imprint → classroom → boardroom → autotune feedback), clean-room llama-style eval tools, prompt-tools page, and how mindXtrain fits decentralized training.
- [Governance](governance.md) — classroom (graduation) / boardroom (any-N consensus) / dojo (prime-N dispute settlement), model-backed deliberation.

## Decentralized training landscape (2026)

- [Decentralized training deep-dive](decentralized-training-deep-dive-2026.md) — Prime Intellect / Nous Psyche / Gensyn / Templar / Pluralis, the DiLoCo/SparseLoCo algorithms, verification (TOPLOC / Verde / Gauntlet), and where mindXtrain fits (AOT-only = verifiable).
- [LLM training-stack landscape](mindxtrain-llm-training-landscape-2026.md) — the open-source training/eval/quantize stack survey anchored on mindXtrain.
- [Vercel AI SDK 6 deep-dive](<Vercel AI SDK 6_ A Framework-Agnostic Deep Dive (June 2026).md>) — the streaming/agent toolkit the Coach chat patterns after (clean-room, vanilla JS).

## Reference

- [CLI reference](cli.md) — every `mindxtrain` verb with synopsis, options, exit codes.
- [YAML schema](yaml_schema.md) — every field of the 10-section `XTrainConfig`.
- [Benchmarks](benchmarks.md) — target metrics + framework comparison.
- [CHANGELOG](CHANGELOG.md) — version history (current: v1.0.0).
- [LICENSE-NOTICE](LICENSE-NOTICE.md) — Apache-2.0 + MIT-compatibility statement.

## Source briefs (`blueprints/`)

The frozen design briefs the project was built against — historical specification; for
current state read the docs above. Do not edit.

- [`blueprints/mindXtrain.md`](blueprints/mindXtrain.md) — operating brief; three-track pitch, day-by-day execution.
- [`blueprints/mindXtrain2.md`](blueprints/mindXtrain2.md) — technical reference; canonical Part 4 layout.
- [`blueprints/mindXtrain_ Production Blueprint for the AMD and lablab.ai Hackathon.md`](<blueprints/mindXtrain_ Production Blueprint for the AMD and lablab.ai Hackathon.md>) — repo skeleton, hero recipe, immutable registry stub.

## On-chain

- [`contracts/README.md`](../contracts/README.md) — Foundry workspace for the immutable run-receipt registry + x402 receiver.
