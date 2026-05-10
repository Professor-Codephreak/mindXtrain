# Benchmarks

The numbers we are chasing on the hero workload, and the framework comparison that goes in the README.

## Hero workload

**Qwen3-8B SFT, 1× MI300X, bs=8, seq=4096, BF16, AdamW, 1B tokens.**

| Metric                   | Target          | Why                                                            |
|--------------------------|-----------------|----------------------------------------------------------------|
| Throughput               | **>15 000 tok/s** | Comparable to AMD's published Llama-3.1-8B numbers.           |
| MFU                      | **>40 %**         | Floor for "MI300X is being exercised, not idled."              |
| Time to eval-loss = 1.5  | **<90 minutes**   | Lets the demo video show the loop converging in real time.     |
| Total cost               | **<$3**           | $1.99 / hr × 1 GPU × ~1.5 hr × safety margin.                  |
| Peak HBM                 | ~80 GB          | Headroom on MI300X's 192 GB; impossible on H100 80 GB.         |

Hit those four and the cost slide writes itself: **MI300X $1.99/hr × 1 GPU × 1.5 hr ≈ $3** versus **H100 $4/hr × 2 GPUs × 4 hr ≈ $32** — 4× cheaper for the same workload, and the H100 baseline can't even fit BF16 at this batch/seq combo without quantization.

## H100 cost baseline

The argument the judges remember is "MI300X is 4× cheaper for this exact workload." The cost numbers come from public list prices and need to hold up under questioning.

| GPU       | $/hr  | Memory  | Qwen3-8B BF16 bs=8 seq=4096 | Cost for 1B tokens |
|-----------|-------|---------|-----------------------------|--------------------|
| H100 80 GB | $4.00 | 80 GB  | OOM unless bs/seq cut       | ~$32 (2× GPUs, 4 hr, FP8 fallback) |
| H200 141 GB | $6.00 | 141 GB | Fits, ~12k tok/s             | ~$24 (1× GPU, 4 hr) |
| MI300X 192 GB | $1.99 | 192 GB | Fits with headroom, >15k tok/s | **<$3** (1× GPU, 1.5 hr) |

## Framework comparison (the README differentiator)

This is the table the README prints. mindxtrain is the only row with all seven cells filled — that is the elevator pitch.

| Framework      | One-cmd ROCm 7.2.1 install | MI300X auto-tune | Qwen3.6 day-zero | FP8 via Quark | x402 micropayments | Decentralized fallback | Training-receipt manifest |
|----------------|----------------------------|------------------|------------------|---------------|--------------------|------------------------|---------------------------|
| Axolotl        | ⚠ (community fork)        | ✗                | ✓                | △ (torchao)  | ✗                  | ✗                      | ✗                         |
| LLaMA-Factory  | ✓ (AMD tutorial)           | ✗                | ✓                | △            | ✗                  | ✗                      | ✗                         |
| Unsloth        | ✓ (OneClickAMD)            | ✗                | △ (single-GPU)   | ✗            | ✗                  | ✗                      | ✗                         |
| torchtune      | ✓ (AMD CI)                 | ✗                | ✗ (no recipe)    | △            | ✗                  | ✗                      | ✗                         |
| Primus         | ✓ (`rocm/primus:v26.2`)    | ✗                | ✗ (pretrain only) | ✓           | ✗                  | ✗                      | ✗                         |
| **mindxtrain** | **✓**                      | **✓ (60s AOT)**  | **✓**            | **✓**         | **✓ (Algorand)**   | **✓ (Bacalhau/Akash)** | **✓ (BLAKE3 + INFT)**     |

Legend: ✓ = supported · ⚠ = supported via community fork · △ = partial / opt-in · ✗ = not supported.

## Capturing the numbers

The `mindxtrain` CLI emits structured logs that map onto the metrics above. The output tree:

```
runs/<run_id>/
├── config.yaml              # input, BLAKE3-hashed in the manifest
├── autotune_plan.json       # the AOT plan (the differentiator)
├── train.log                # accelerate stdout/stderr
├── metrics.jsonl            # one record per logging step: tok_per_s, mfu, hbm_gb, watts
├── checkpoint/              # HF safetensors + tokenizer, BLAKE3-hashed
├── quantized/               # Quark FP8 PTPC, vLLM-loadable
├── eval.json                # lm-evaluation-harness output
└── manifest.json            # mindxtrain.provenance.Manifest with BLAKE3 hashes
```

`metrics.jsonl` is the source of truth for the benchmark numbers. The cost slide is a one-liner over that file: average `tok_per_s` × seconds × $1.99 / 3600.

## Regression detection

`eval.regression.threshold_pct: -1.0` in every recipe means **fail the run if any benchmark task drops more than 1 percentage point** versus the base model baseline. That keeps a fine-tune that improves the target distribution but breaks general capability from being silently published. The baseline JSON is computed once per base model and cached alongside the run; comparison happens via `mindxtrain.eval.persona_regression.regression_score` and `mindxtrain.eval.agenda_regression.regression_score`.

## What's not measured (yet)

- Energy (kWh per training run) — `mindxtrain.operator.telemetry.energy.sample_power_w` wraps `rocm-smi --showpower` (returns 0.0 W gracefully on a CPU dev box). MI300X power baseline is ~750 W under load; a 90-minute run is ~1.1 kWh. Telemetry collection into `metrics.jsonl` is wired but the dashboard integration is post-hackathon work.
- Multi-node throughput — out of hackathon scope; the `mindxtrain receipt` manifest accommodates it (`hardware.gpus` field), and the autotune `rccl_probe` is the entry point for the multi-node version.
