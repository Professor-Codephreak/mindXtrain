# Autotune — the 60-second AOT probe

The single layer that distinguishes mindxtrain from Axolotl, LLaMA-Factory, Unsloth, torchtune, and Primus. Quoted from `docs/blueprints/mindxtrain_ Production Blueprint*.md` §"How the hackathon actually scores":

> The single most differentiating angle is the auto-selection layer. No competitor framework — not Axolotl, LLaMA-Factory, Unsloth, torchtune, or Optimum-AMD itself — runs a per-job MI300X micro-benchmark before training to pick CK vs Triton attention backends, hipBLASLt heuristic vs rocBLAS path, AITER vs reference MoE kernels, NCCL_MIN_NCHANNELS, gradient-checkpointing strategy, FSDP shard width, and LoRA rank against the actual (model, dataset shape, sequence length, GPU count) tuple. mindxtrain owns that AOT-only autotune layer.

## The AOT-only discipline

JIT autotune (Triton autotune in vLLM cold-start, `torch.compile(mode='max-autotune')` Inductor, MIOpen find-mode) is **forbidden in production training**. Reasons:

1. **Reproducibility.** A run with JIT autotune produces different kernels on different invocations of the same workload, breaking deterministic benchmarks.
2. **First-batch latency.** Triton autotune on cold start can stall a training step for 5-30 seconds, invisible in the loss curve and very visible in `tok/s`.
3. **Cypherpunk2048 standard.** Production paths must be statically declared at deployment. JIT compilation is an in-band runtime decision, which is exactly what the standard prohibits.

The `autotune.policy: aot_only` field in the YAML is the contract. The training layer reads the `AutotunePlan` JSON at start, sets env vars + flags, and never re-tunes during the loop. AOTriton (the AOT version of Triton math) is loaded as a precompiled `.so`; Composable Kernel kernels are pulled from the offline-tuned hipBLASLt cache.

## The probe taxonomy

`mindxtrain bench` runs three probes in sequence inside its 60-second budget. The whole flow is at [`mindxtrain/autotune/benchmark.py`](../mindxtrain/autotune/benchmark.py).

### 1. attention_probe — CK vs Triton SDPA

[`mindxtrain/autotune/attention_probe.py`](../mindxtrain/autotune/attention_probe.py).

Times `torch.nn.functional.scaled_dot_product_attention` across four representative shapes (queries × keys × heads × head-dim per the recipe's `model.name` + `data.seq_len`) on both backends:

| Backend  | How                                                                |
|----------|--------------------------------------------------------------------|
| `ck`     | Composable Kernel (default) — hand-tuned ASM/CK kernels via AITER.  |
| `triton` | AOTriton 0.11.2b0 with `TORCH_BLAS_PREFER_HIPBLASLT=0` and `PYTORCH_TUNABLEOP_ENABLED=0` toggles. |

The probe is **real** — when `torch` (`--extra ml`) and a ROCm-visible GPU
are both present, it times the four representative shapes on each backend
via `torch.nn.attention.sdpa_kernel`. Without torch (typical CPU dev box),
the probe gracefully returns the canonical `("ck", [])` default so
`bench --dry-run` parity holds and the AutotunePlan downstream consumers
keep working unchanged.

```
budget: ~30 s
shapes: 4 representative (qlen, klen, num_heads, head_dim)
output: AttentionBackend ∈ {ck, triton}, list[ProbeTiming]
```

`ProbeTiming` is `{ label, backend, median_ms, iterations }` — captured per (shape × backend) so the demo can render a side-by-side timing table in the video.

### 2. gemm_probe — hipBLASLt heuristic

[`mindxtrain/autotune/gemm_probe.py`](../mindxtrain/autotune/gemm_probe.py).

Per the user-confirmed Day-1 plan ("1 real probe + 2 hardcoded heuristics"), this returns `hipblaslt_default` for gfx942 based on AMD's documented MI300X tuning guidance. Reference: AMD ROCm 7.2.1 release notes, hipBLASLt 0.10 default heuristics are within 5 % of hand-tuned for the BF16/FP16 GEMMs mindxtrain hits (LoRA rank 16-64, hidden 2048-8192).

**Why we don't enumerate.** A real hipBLASLt heuristic enumeration is ~1.5 minutes and risks burning the entire 60-second budget. If MMLU eval shows GEMM-bound throughput regression on a specific recipe, revisit post-hackathon.

Output: `hipblaslt_default | hipblaslt_tuned | rocblas_fallback`.

### 3. rccl_probe — collective bandwidth

[`mindxtrain/autotune/rccl_probe.py`](../mindxtrain/autotune/rccl_probe.py).

For 1-GPU runs this is a no-op. For 8-GPU runs it returns `8gpu_xgmi` with `NCCL_MIN_NCHANNELS=112` set in the plan notes. **2-GPU and 4-GPU groupings raise `RuntimeError`** — MI300X xGMI bandwidth between subsets of 2/4 GPUs is asymmetric, and FSDP shards on those topologies will silently bottleneck.

```python
def probe_rccl(gpu_index: int = 0, gpu_count: int = 1) -> RcclConfig:
    if gpu_count == 1:
        return "1gpu_noop"
    if gpu_count == 8:
        return "8gpu_xgmi"
    raise RuntimeError(f"FSDP on {gpu_count} GPUs is unsafe...")
```

This is enforced in two places: the `rccl_probe` raises at probe time, and the `XTrainConfig.hardware.gpus` field is `Literal[1, 8]` so the schema rejects bad values at parse time.

## The `AutotunePlan` schema

[`mindxtrain/autotune/plan.py`](../mindxtrain/autotune/plan.py).

```python
class AutotunePlan(BaseModel):
    schema_version: Literal["1"] = "1"
    gpu_arch: str = "gfx942"
    rocm_version: str = "7.2.1"

    attention_backend: Literal["ck", "triton"] = "ck"
    gemm_heuristic: Literal["hipblaslt_default", "hipblaslt_tuned", "rocblas_fallback"] = "hipblaslt_default"
    rccl_config: Literal["1gpu_noop", "8gpu_xgmi", "unsupported_2_4_gpu"] = "1gpu_noop"

    fsdp_shard_width: Literal[1, 8] = 1
    suggested_lora_rank: int = 16
    suggested_micro_batch_size: int = 4

    probe_timings: list[ProbeTiming] = []
    notes: list[str] = []
```

Pure data, content-addressed via BLAKE3 in the mindxtrain provenance manifest, fully reproducible across MI300X nodes.

## How the training layer consumes the plan

`mindxtrain/train/dispatch.py` reads the plan and applies it before invoking
the backend (real subprocess wrap of `accelerate launch -m axolotl.cli.train`
in `mindxtrain/train/sft.py`):

```python
def dispatch_training(cfg: XTrainConfig, plan: AutotunePlan, out_dir: Path) -> Path:
    # 1. set env vars: cfg.train.env + plan-driven additions
    #    e.g. plan.rccl_config == "8gpu_xgmi" → set NCCL_MIN_NCHANNELS=112
    #         plan.attention_backend == "ck" → NVTE_CK_USES_BWD_V3=1, etc.
    # 2. compile cfg → Axolotl YAML, override:
    #    train.flash_attention.backend ← plan.attention_backend
    #    train.method.r ← plan.suggested_lora_rank if cfg.train.method.kind == "lora"
    #    train.batch.per_device ← min(cfg, plan.suggested_micro_batch_size)
    # 3. subprocess: accelerate launch -m axolotl.cli.train <yaml>
    # 4. capture stdout/stderr to out_dir/train.log
    # 5. return checkpoint dir
```

## Dry-run / CI path

Every CI pipeline runs `mindxtrain bench --dry-run`, which skips the GPU probes entirely and emits a hardcoded reference plan. The reference plan has `attention_backend: ck`, `gemm_heuristic: hipblaslt_default`, `rccl_config: 1gpu_noop`, `fsdp_shard_width: 1` — sane MI300X 1-GPU defaults that exercise the same code path the real probe writes.

```bash
$ uv run mindxtrain bench --dry-run --out plan.json
wrote plan.json (dry_run=True, attention=ck, gemm=hipblaslt_default)
```

The dry-run path is what makes the GitHub Actions CI matrix CPU-only.

## Day 2 implementation budget (target ~30 minutes per probe)

| Probe          | Target time | Risk                                              |
|----------------|-------------|---------------------------------------------------|
| attention_probe | 30 min     | First-time AOTriton compilation may be slow; warm cache mounted from persistent volume. |
| gemm_probe     | 0 (hardcoded) | None — heuristic is documented.                |
| rccl_probe     | 5 min       | 1-GPU is no-op; 8-GPU only relevant if we rent the 8× SKU. |

Total Day 2 budget: ~35 minutes of MI300X time + writing time. The remaining hours go to verifying the plan flows into Axolotl correctly via Day 3's dispatch wiring.

## Where the demo wow-moment lives

Capture `autotune_plan.json` and the streaming probe output for the 5-minute video. The 60-second autotune dashboard is the single most quotable visual asset in the submission — a measurable kernel selection that no competitor framework ships.

```
$ mindxtrain bench --gpu 0 --out plan.json
[autotune] CK FA forward, shape=(8, 4096, 32, 128): 12.4 ms (median over 50)
[autotune] Triton FA forward, shape=(8, 4096, 32, 128): 14.7 ms (median over 50)
[autotune] CK FA forward, shape=(8, 4096, 16, 128): 11.0 ms
[autotune] Triton FA forward, shape=(8, 4096, 16, 128): 13.2 ms
[autotune] picked ck (avg 1.2× faster)
[autotune] gemm: hipblaslt_default (gfx942 documented heuristic)
[autotune] rccl: 1gpu_noop
[autotune] wrote plan.json (1.2 KB) in 47 s
```
