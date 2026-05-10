# mindxtrain Documentation Hub

Index of every doc in this directory. Start at [Quickstart](quickstart.md);
operators executing the demo pipeline should read [HANDOFF.md](../HANDOFF.md).

## Getting started

- [Quickstart](quickstart.md) — install (with optional-dep groups), init, bench, train.
- [HANDOFF.md](../HANDOFF.md) — **operator checklist** for taking the project from "code is done" to "demo is live": local setup → MI300X provision → train/eval/quantize → publish → contracts → submission.

## Architecture

- [Architecture](architecture.md) — canonical single-package layout per mindxtrain2.md §Part 4.
- [Actualization status](actualization_status.md) — per-module map of what's real vs. requires `--extra` install vs. requires runtime.
- [Autotune deep-dive](autotune.md) — the 60-second AOT probe (the differentiator).

## Reference

- [CLI reference](cli.md) — every `mindxtrain` verb with synopsis, options, exit codes.
- [YAML schema](yaml_schema.md) — every field of the 10-section `XTrainConfig`.
- [Coach UI](coach.md) — the interactive `/coach/` web UI.
- [Benchmarks](benchmarks.md) — target metrics + framework comparison table.

## Hackathon

- [Hackathon submission tracker](hackathon_submission.md) — judging-criteria mapping, video script, deck outline, lablab form fields.
- [`HACKATHON.md`](../HACKATHON.md) (repo root) — daily verification gates.

## Development

- [Development workflow](development.md) — toolchain, optional-deps, lazy-import pattern, invariants, how to add recipes/backends/methods.

## Source briefs (`blueprints/`)

The canonical design briefs the project was built against. Treat as historical specification; for current state read the docs above.

- [`blueprints/mindXtrain.md`](blueprints/mindXtrain.md) — operating brief; three-track integrated pitch, day-by-day execution.
- [`blueprints/mindXtrain2.md`](blueprints/mindXtrain2.md) — technical reference; canonical Part 4 layout, Qwen3.5-primary / GLM-5.1-specialist strategy, ml-intern operator patterns.
- [`blueprints/mindXtrain_ Production Blueprint for the AMD and lablab.ai Hackathon.md`](blueprints/mindXtrain_%20Production%20Blueprint%20for%20the%20AMD%20and%20lablab.ai%20Hackathon.md) — repo skeleton, hero `demo_qwen3_8b_sft.yaml`, immutable Solidity registry stub.

PDF copies of the same briefs live alongside the markdown.

## On-chain

- [`contracts/README.md`](../contracts/README.md) — Foundry workspace for the immutable run-receipt registry and x402 receiver.
