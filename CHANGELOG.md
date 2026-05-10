# Changelog

All notable changes to **mindxtrain** are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
