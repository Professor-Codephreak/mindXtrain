# YAML schema reference

mindxtrain takes one YAML per training run, validated against
`mindxtrain.config.schema.XTrainConfig` (Pydantic v2). The canonical hero
config lives at [`examples/demo_qwen3_8b_sft.yaml`](../examples/demo_qwen3_8b_sft.yaml).
Every recipe under `mindxtrain/train/recipes/` round-trips through this schema
(proven by `tests/test_config_schema.py::test_all_recipes_validate`).

Source of truth: [`mindxtrain/config/schema.py`](../mindxtrain/config/schema.py).
When the schema changes, update this doc.

> **YAML recipes vs JSON defaults.** The 12 YAML recipes in
> `mindxtrain/train/recipes/` are full `XTrainConfig` instances for a specific
> training run (they're what `mindxtrain init --template <name>` writes).
> Separately, `mindxtrain/config/{train_default,eval_default,deploy_default}.json`
> provide ml-intern-style runtime defaults with `${ENV}` interpolation — these
> are runtime defaults for `train`/`eval`/`serve` orchestration, not training-job
> recipes.

Top-level shape:

```yaml
meta:       { project, run_name, seed, license, description }
hardware:   { name, gfx_arch, gpus, expected_hbm_gb }
autotune:   { enabled, plan_path, budget_seconds, policy }
model:      { name, revision, attn_implementation, torch_dtype, trust_remote_code }
data:       { source, hf_id, split, streaming, max_samples, seq_len, packing, dedupe, shard }
train:      { backend, method, optimizer, schedule, batch, precision,
              gradient_checkpointing, flash_attention, fsdp, env }
eval:       { harness, regression }
quantize:   { enabled, scheme, ptpc }
serve:      { backend, reasoning_parser, tool_call_parser, tensor_parallel,
              max_model_len, port }
publish:    { enabled, hf, lighthouse, mindx, agenticplace, bankon, billing }
receipt:    { output, include }
```

`extra: forbid` is set on every model — unknown fields raise `ValidationError`. `frozen: true` is set on every model — configs are immutable once loaded.

## `meta`

| Field         | Type | Default      | Notes                                        |
|---------------|------|--------------|----------------------------------------------|
| `project`     | str  | _required_   | Logical group, e.g. `mindxtrain_demo`.        |
| `run_name`    | str  | _required_   | Slug for the run, e.g. `qwen3_8b_sft_lora`.   |
| `seed`        | int  | `2048`       | RNG seed; cypherpunk2048 reference.           |
| `license`     | str  | `apache-2.0` | SPDX-style license string.                    |
| `description` | str  | `""`         | Free-form.                                    |

## `hardware`

| Field             | Type                         | Default   | Notes                                                                                  |
|-------------------|------------------------------|-----------|----------------------------------------------------------------------------------------|
| `name`            | `mi300x \| mi325x \| mi350x \| mi355x` | `mi300x`  | Cloud SKU.                                                                             |
| `gfx_arch`        | `gfx942 \| gfx950`           | `gfx942`  | Must match `name`. AOTriton compiles per arch.                                         |
| `gpus`            | `Literal[1, 8]`              | `1`       | **Hard constraint** — 2/4-GPU FSDP groups hit MI300X xGMI bandwidth asymmetry. |
| `expected_hbm_gb` | int                          | `192`     | Used by autotune to size FSDP shards.                                                  |

## `autotune`

| Field             | Type                | Default                               | Notes                                       |
|-------------------|---------------------|---------------------------------------|---------------------------------------------|
| `enabled`         | bool                | `true`                                | Skip with `--dry-run` on CPU.               |
| `plan_path`       | Path                | `./out/mindxtrain.tuned.yaml`         | AOT plan output location.                   |
| `budget_seconds`  | int (10-600)        | `60`                                  | MoE recipes use 90-120 s.                   |
| `policy`          | `Literal[aot_only]` | `aot_only`                            | **JIT autotune is forbidden in production.** |

## `model`

| Field                  | Type                                  | Default              | Notes                                              |
|------------------------|---------------------------------------|----------------------|----------------------------------------------------|
| `name`                 | str                                   | _required_           | HF Hub model ID.                                   |
| `revision`             | str \| null                           | `null`               | git revision pin; `null` means default branch.     |
| `attn_implementation`  | `flash_attention_2 \| sdpa \| eager`  | `flash_attention_2`  | autotune may override.                             |
| `torch_dtype`          | `bfloat16 \| float16 \| float32 \| fp8_e4m3 \| mxfp4` | `bfloat16` | BF16 is the safe default on MI300X.                |
| `trust_remote_code`    | bool                                  | `false`              | Reject untrusted custom code paths.                |

## `data`

| Field         | Type                          | Default          | Notes                                                |
|---------------|-------------------------------|------------------|------------------------------------------------------|
| `source`      | `hf \| local \| lighthouse`   | `hf`             |                                                      |
| `hf_id`       | str                           | _required_       | Dataset ID, e.g. `HuggingFaceH4/ultrachat_200k`.     |
| `split`       | str                           | `train`          |                                                      |
| `streaming`   | bool                          | `true`           | Avoid storing 100 GB+ corpora locally.               |
| `max_samples` | int \| null                   | `null`           | Truncate for fast demos.                             |
| `seq_len`     | int (64 .. 1 048 576)         | `4096`           |                                                      |
| `packing`     | bool                          | `true`           | Pack-to-cutoff Qwen3-style.                          |
| `dedupe`      | `DedupeCfg`                   | `{}`             | Optional `minhash` and `semdedup` sub-configs.       |
| `shard`       | `ShardCfg`                    | `{ num_shards: 1 }` |                                                   |

`DedupeCfg.minhash`: `{ threshold: 0.0..1.0 }`.
`DedupeCfg.semdedup`: `{ threshold, model: <ST model id> }`.

## `train`

| Field                    | Type                                    | Default                | Notes                                                              |
|--------------------------|-----------------------------------------|------------------------|--------------------------------------------------------------------|
| `backend`                | `axolotl \| unsloth \| torchtune \| primus` | `axolotl`           | Only `axolotl` is real in week 1.                                  |
| `method`                 | discriminated union (see below)         | `lora` defaults        | Tag with `kind:`.                                                  |
| `optimizer`              | `OptimizerCfg`                          | adamw_torch_fused 1e-4 | `name`, `lr`, `betas`, `weight_decay`, `grad_clip`.                |
| `schedule`               | `ScheduleCfg`                           | cosine, warmup 0.03    | `type`, `warmup_ratio`, `epochs`.                                  |
| `batch`                  | `BatchCfg`                              | per_device 8, ga 4     | `per_device`, `grad_accum`.                                        |
| `precision`              | `DType`                                 | `bfloat16`             | training-time precision; quantize step changes serving precision.   |
| `gradient_checkpointing` | bool                                    | `true`                 |                                                                    |
| `flash_attention`        | `FlashAttentionCfg`                     | `{ backend: ck }`      | autotune may flip to `triton`.                                     |
| `fsdp`                   | `FsdpCfg`                               | `{ enabled: false }`   |                                                                    |
| `env`                    | `dict[str, str]`                        | seven MI300X knobs     | Set in subprocess before training-backend launch.                  |

The default `train.env` carries the non-negotiable MI300X knobs:

```yaml
env:
  HSA_NO_SCRATCH_RECLAIM: "1"
  NVTE_CK_USES_BWD_V3: "1"
  NVTE_CK_IS_V3_ATOMIC_FP32: "1"
  PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32: "1"
  NCCL_MIN_NCHANNELS: "112"
  HIP_FORCE_DEV_KERNARG: "1"
  PYTORCH_ROCM_ARCH: "gfx942"
```

### `train.method` (discriminated union)

| `kind`   | Required fields                                   | Notes                                       |
|----------|---------------------------------------------------|---------------------------------------------|
| `full`   | _none_                                            | Full-parameter SFT.                         |
| `lora`   | `r`, `alpha`, `dropout`, `target_modules`         | Default LoRA recipe.                        |
| `qlora`  | `r`, `alpha`, `dropout`, `quant_bits` (4 or 8), `target_modules` | bitsandbytes opt-in only.    |
| `dpo`    | `beta`                                            | Direct Preference Optimization.             |
| `orpo`   | `beta`                                            | Odds Ratio Preference Optimization.         |
| `grpo`   | `num_generations`, `kl_coef`                      | Group Relative Policy Optimization.         |
| `gspo`   | `num_generations`                                 | Qwen team's preferred RL on hybrid + MoE.   |
| `kto`    | `beta`                                            | Kahneman-Tversky Optimization.              |
| `cpt`    | _none_                                            | Continued Pretraining.                      |

Unknown `kind` raises `ValidationError` (tested in `tests/test_config_schema.py::test_method_discriminator_rejects_unknown_kind`).

## `eval`

| Field        | Type                | Default                                            | Notes                                          |
|--------------|---------------------|----------------------------------------------------|------------------------------------------------|
| `harness`    | `EvalHarnessCfg`    | `{ tasks: [mmlu, gsm8k, ifeval, humaneval], fewshot: 5 }` | Wraps `lm-evaluation-harness`.        |
| `regression` | `EvalRegressionCfg` | `{ baseline: "", threshold_pct: -1.0 }`            | Fail run if any task drops > 1 pct vs baseline. |

## `quantize`

| Field      | Type                                       | Default       | Notes                                                  |
|------------|--------------------------------------------|---------------|--------------------------------------------------------|
| `enabled`  | bool                                       | `true`        |                                                        |
| `scheme`   | `quark_fp8 \| quark_mxfp4 \| gptq_rocm \| none` | `quark_fp8` | AMD Quark FP8 (E4M3) is the default.                   |
| `ptpc`     | bool                                       | `true`        | Per-tensor-per-channel — 15-30 % faster than BlockScale on MI300X. |

## `serve`

| Field                  | Type                          | Default       | Notes                                            |
|------------------------|-------------------------------|---------------|--------------------------------------------------|
| `backend`              | `vllm-rocm \| sglang`         | `vllm-rocm`   |                                                  |
| `reasoning_parser`     | `deepseek_r1 \| qwen3 \| none`| `qwen3`       | Use `qwen3` for Qwen3 / 3.5 / 3.6.               |
| `tool_call_parser`     | `hermes \| qwen3_coder \| none` | `hermes`    | Use `qwen3_coder` for Qwen3-Coder family.        |
| `tensor_parallel`      | int (≥1)                      | `1`           | tp size for multi-GPU serving.                   |
| `max_model_len`        | int (≥512)                    | `8192`        | KV cache cap.                                    |
| `port`                 | int (1024..65535)             | `8000`        |                                                  |

## `publish`

| Field         | Type                       | Notes                                                            |
|---------------|----------------------------|------------------------------------------------------------------|
| `enabled`     | bool, default `true`       |                                                                  |
| `hf`          | `HfPublishCfg \| null`     | `{ repo, private }`. `null` skips HF push.                       |
| `lighthouse`  | `LighthousePublishCfg`     | `{ api_key_env }`. Defaults to env var `LIGHTHOUSE_API_KEY`.     |
| `mindx`       | `MindxPublishCfg`          | `{ api_url, register_as_capability }`.                           |
| `agenticplace`| `AgenticPlacePublishCfg`   | `{ api_url, chain_map_url }`.                                    |
| `bankon`      | `BankonPublishCfg`         | `{ ens_parent, subname }`.                                       |
| `billing`     | `BillingPublishCfg`        | `{ x402: { network, asset, receiver_via, price_per_1k_tokens } }`.|

`publish.billing.x402.network` is `algorand | base | base-sepolia`. The hackathon defaults are `algorand` + `USDC` ASA `203977300`.

## `receipt`

| Field    | Type                  | Default                             | Notes                                                |
|----------|-----------------------|-------------------------------------|------------------------------------------------------|
| `output` | Path                  | `./out/receipt.json`                | Manifest output.                                     |
| `include`| `list[ReceiptIncludeKey]` | all 8 keys (see schema.py)        | Which provenance fields to capture.                  |

`ReceiptIncludeKey` is one of: `rocm_version`, `gfx_arch`, `container_digest`, `all_git_shas`, `yaml_hash`, `dataset_cids`, `eval_report`, `energy_kwh`.

## How the schema is enforced

```bash
uv run pytest tests/test_config_schema.py -v
```

12 tests cover the full surface: every recipe round-trips, the demo example validates, the `hardware.gpus: 1|8` constraint rejects 2 and 4, the discriminator rejects unknown method kinds, and `extra: forbid` rejects unknown keys at every level.
