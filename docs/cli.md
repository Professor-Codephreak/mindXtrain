# CLI reference

The `mindxtrain` Typer app: 9 verbs (8 top-level + a `dataset` subgroup).
Every verb that consumes a YAML config validates it against the
[10-section schema](yaml_schema.md) before doing anything else.

```
mindxtrain [--version] <verb> [options]
```

All verbs dispatch into real Python in the canonical `mindxtrain.*` modules.
Verbs that require optional dependencies surface a clean
`run `uv sync --extra <group>`` hint and exit `3`.

## Global options

| Flag         | Purpose                              |
|--------------|--------------------------------------|
| `--version`  | Print the version and exit.          |
| `--help`     | Show help for the top-level command. |

## `init` — scaffold a YAML

Render a built-in recipe to disk.

```
mindxtrain init [--template <name>] [--out <path>] [--list]
```

| Option                | Default              | Description                                      |
|-----------------------|----------------------|--------------------------------------------------|
| `--template`, `-t`    | `qwen3_8b_sft_lora`  | recipe name; see `--list`                        |
| `--out`, `-o`         | `run.yaml`           | output path                                      |
| `--list`              | _flag_               | print every built-in recipe and exit             |

Available recipes (12 total):

```
instella_3b_lora        qwen3_30b_a3b_lora      qwen3_32b_dpo
qwen3_32b_full_fsdp     qwen3_32b_grpo          qwen3_32b_orpo
qwen3_6_27b_lora        qwen3_6_35b_a3b_lora    qwen3_8b_cpt
qwen3_8b_sft_full       qwen3_8b_sft_lora       qwen3_vl_8b_sft
```

```bash
$ uv run mindxtrain init --template qwen3_8b_sft_lora --out run.yaml
wrote run.yaml (2785 bytes, recipe='qwen3_8b_sft_lora')
```

## `bench` — run the 60-second AOT autotune probe

The differentiator. See [autotune.md](autotune.md) for probe taxonomy.

```
mindxtrain bench [--gpu N] [--out <path>] [--dry-run]
```

| Option       | Default                | Description                                                         |
|--------------|------------------------|---------------------------------------------------------------------|
| `--gpu`      | `0`                    | HIP/ROCm device index                                               |
| `--out`, `-o`| `autotune_plan.json`   | output path                                                         |
| `--dry-run`  | _flag_                 | skip GPU probes; emit a synthetic reference plan (CPU-safe)         |

`--dry-run` is the CPU-only path used in tests and CI. A real
`mindxtrain bench --gpu 0` requires `torch` (`--extra ml`) and an MI300X
with ROCm 7.2.1; if torch is unavailable, the attention probe gracefully
falls back to the canonical `ck` default.

## `train` — dispatch a training run

```
mindxtrain train <config.yaml> [--plan <plan.json>] [--out <run-dir>]
```

| Option       | Default                | Description                                                  |
|--------------|------------------------|--------------------------------------------------------------|
| `--plan`     | (uses dry-run plan)    | autotune plan JSON from `mindxtrain bench`                   |
| `--out`, `-o`| `./out/runs`           | output root for `<run_id>/` directory                        |

Loads the YAML, dispatches to `train.backend` (`axolotl`, `unsloth`,
`torchtune`, `primus`). The Axolotl path subprocess-wraps
`accelerate launch -m axolotl.cli.train`. Plan-derived env vars
(`PYTORCH_ROCM_ARCH=gfx942`, `HSA_NO_SCRATCH_RECLAIM=1`, etc.) are injected
before launch.

Requires `--extra ml` plus the chosen backend on `PATH`.

Exits `3` with a clean install hint if `accelerate` (or the backend) is missing.

## `dataset prep` — run the dataset pipeline

```
mindxtrain dataset prep <config.yaml> [--out <dir>]
```

Streams the HF dataset (`datasets`), runs heuristic + optional MinHash/SemDeDup
filters, tokenizes (`AutoTokenizer`), packs to `data.seq_len`, emits sharded
`.tar` files. Pin the resulting tars via
`mindxtrain.storage.lighthouse` or `mindxtrain.storage.ipfs`.

Requires `--extra ml` (datasets, transformers).

## `eval` — run lm-evaluation-harness

```
mindxtrain eval <config.yaml> [--checkpoint <path>]
```

| Option       | Default                                           | Description                |
|--------------|---------------------------------------------------|----------------------------|
| `--checkpoint`, `-c` | `./out/runs/<run_name>/checkpoint`        | path to the checkpoint dir |

Subprocess-wraps `lm_eval --model hf --tasks <comma-sep>`. Tasks come from
`cfg.eval.harness.tasks`. Output JSON written under
`<checkpoint>/eval/lm_eval.json`. Summary printed via
`mindxtrain.eval.harness.parse_summary`.

Requires `--extra eval`.

## `quantize` — Quark FP8 / MXFP4

```
mindxtrain quantize <config.yaml> [--checkpoint <path>]
```

Wraps `python -m amd_quark.quantize` with `--scheme fp8_e4m3` (default) or
`--scheme mxfp4` (CDNA 4 / MI350X+). Output is a `quantized/` directory next
to the checkpoint, vLLM-loadable.

Requires the `amd-quark` package — typically only available inside the
`rocm/primus:v26.2` container or per
[Quark docs](https://quark.docs.amd.com/).

## `serve` — print the vLLM-ROCm launch command

```
mindxtrain serve <config.yaml> [--checkpoint <path>]
```

Builds the `vllm serve` argv from `cfg.serve` and prints it. We deliberately
don't `exec` — the user pipes it into their own orchestrator (or
`ops/compose/compose_dev.yaml`).

The chat-template parsers map per `serve.reasoning_parser` (`qwen3` for Qwen3,
`deepseek_r1` for DeepSeek-style) and `serve.tool_call_parser` (`hermes`,
`qwen3_coder`).

## `publish` — push to HF + Lighthouse + register

```
mindxtrain publish <config.yaml> --manifest <manifest.json> [--skip-hf] [--skip-pin]
```

1. `mindxtrain.storage.hf_hub.publish_to_hf` — uploads the checkpoint dir to
   HuggingFace Hub (uses `HF_TOKEN`). `--skip-hf` to bypass.
2. `mindxtrain.storage.lighthouse.publish_to_lighthouse` — pins to
   Lighthouse via direct httpx POST (uses `LIGHTHOUSE_API_KEY`). Falls back
   to a stub `cid://stub-...` derived from the checkpoint's BLAKE3 if the
   key is unset. `--skip-pin` to bypass entirely.
3. `mindxtrain.deploy.api_client.register_with_mindx` — POSTs the run-id /
   HF URL / CID to `MINDXTRAIN_API_BASE_URL/v1/agents`. Skipped gracefully
   if the endpoint isn't reachable.
4. The manifest JSON file is updated in-place with the resulting `hf_repo_id`
   and `lighthouse_cid` fields.

## `receipt` — verify a provenance manifest

```
mindxtrain receipt <manifest.json> [--config <run.yaml>]
```

Loads the manifest and prints the run-id + BLAKE3 fields. With `--config`,
also re-hashes the on-disk artifacts (`config_yaml`, `dataset`, `checkpoint`,
`eval_json`) and emits a per-field pass/fail dict — exits `0` if every hash
verifies, `2` if any drift is detected.

```bash
$ uv run mindxtrain receipt out/runs/<run_id>/manifest.json --config run.yaml
{
  "config_yaml": true,
  "dataset": true,
  "checkpoint": true,
  "eval_json": true
}
```

## Exit-code summary

| Code | Meaning                                                         |
|------|-----------------------------------------------------------------|
| 0    | Success.                                                        |
| 1    | Bad input — missing file, hash mismatch, schema error.          |
| 2    | Verify failed — at least one BLAKE3 field doesn't match disk.   |
| 3    | Optional dep missing — install with `uv sync --extra <group>`.  |

## Where the verbs live

| Verb              | Module                                                                                |
|-------------------|---------------------------------------------------------------------------------------|
| `init`            | `mindxtrain.cli.main.init` + `mindxtrain.config.loader.render_recipe`                 |
| `bench`           | `mindxtrain.cli.main.bench` + `mindxtrain.autotune.benchmark.run_autotune`            |
| `train`           | `mindxtrain.cli.main.train` + `mindxtrain.train.dispatch.dispatch_training`           |
| `dataset prep`    | `mindxtrain.cli.main.dataset_prep` + `mindxtrain.data.{curate,filter,tokenize,pack}`  |
| `eval`            | `mindxtrain.cli.main.eval_` + `mindxtrain.eval.harness.run_lm_eval`                   |
| `quantize`        | `mindxtrain.cli.main.quantize` + `mindxtrain.deploy.quark.quark_fp8`                  |
| `serve`           | `mindxtrain.cli.main.serve` + `mindxtrain.deploy.vllm_launcher.build_vllm_command`    |
| `publish`         | `mindxtrain.cli.main.publish` + `mindxtrain.storage.{hf_hub,lighthouse}` + `mindxtrain.deploy.api_client` |
| `receipt`         | `mindxtrain.cli.main.receipt` + `mindxtrain.provenance.verify.verify_receipt`         |
