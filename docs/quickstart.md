# Quickstart

Install in a minute, get to a validated YAML and rendered autotune plan in
five commands. CPU-only laptop is fine for everything in this doc; GPU paths
are flagged.

## Prerequisites

- Python 3.12 (managed by `uv`).
- `uv` 0.8+ (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- ~500 MB disk for the base install; up to ~6 GB with `--all-extras`.
- Optional: an MI300X with ROCm 7.2.1 for the real `bench`, `train`,
  `quantize`, `serve` paths.

## Install — base

```bash
git clone <repo-url> mindxtrain
cd mindxtrain
uv sync                       # one project, ~40 deps, no GPU stack
uv run pytest -q              # → 112 passed
```

The base install gives you the CLI, the Coach UI, the autotune dry-run, the
provenance manifest + verify path, the operator FastAPI app, and all
in-process Python utilities (registry, hot-swap, agent loop, ContextManager,
data filter, sequence packing, ...).

## Install — with extras

Install only the deps you need; multiple `--extra` flags allowed.

```bash
uv sync --extra ml                # +trl, transformers, peft, accelerate, datasets
uv sync --extra ml --extra eval   # + lm-eval, lighteval, inspect-ai, jinja2
uv sync --extra data              # + datasketch, sentence-transformers, faiss-cpu
uv sync --extra serve             # + vllm
uv sync --extra chain             # + web3, py-algorand-sdk, huggingface-hub
uv sync --extra obs               # + opentelemetry-sdk, prometheus-client, psutil
uv sync --all-extras              # everything except amd-quark
```

Full unlocks-table at [actualization_status.md](actualization_status.md).

## 30-second tour (CPU-only)

```bash
# 1. list every built-in recipe
uv run mindxtrain init --list

# 2. scaffold the hero config
uv run mindxtrain init --template qwen3_8b_sft_lora --out run.yaml

# 3. dry-run autotune (no GPU needed)
uv run mindxtrain bench --dry-run --out plan.json

# 4. inspect the YAML
head -30 run.yaml

# 5. inspect the plan
cat plan.json
```

Expected step-5 output:

```json
{
  "schema_version": "1",
  "gpu_arch": "gfx942",
  "rocm_version": "7.2.1",
  "attention_backend": "ck",
  "gemm_heuristic": "hipblaslt_default",
  "rccl_config": "1gpu_noop",
  "fsdp_shard_width": 1,
  "suggested_lora_rank": 16,
  "suggested_micro_batch_size": 4,
  "probe_timings": [{ "label": "dry-run-reference", "backend": "ck", "median_ms": 0.0, "iterations": 1 }],
  "notes": ["dry-run reference plan; replace with real probe output on MI300X via Day-2 bench"]
}
```

That's the contract the training layer consumes.

## What you have on a CPU (base install)

| Verb                  | CPU?       | Notes                                          |
|-----------------------|------------|------------------------------------------------|
| `mindxtrain --version`| ✓          |                                                |
| `mindxtrain init`     | ✓          | Renders any of the 12 built-in recipes.        |
| `mindxtrain bench --dry-run` | ✓   | Synthetic reference plan.                      |
| `mindxtrain bench` (real) | ✗ (needs MI300X + torch) | Real CK vs Triton SDPA timing. |
| `mindxtrain train`    | ✗ (needs `--extra ml` + axolotl + GPU) | Subprocess wraps `accelerate launch -m axolotl.cli.train`. |
| `mindxtrain dataset prep` | ✗ (needs `--extra ml` + reachable dataset) | Streams HF dataset → filter → tokenize → pack → tar shards. |
| `mindxtrain eval`     | ✗ (needs `--extra eval`) | Subprocess wraps `lm_eval`. |
| `mindxtrain quantize` | ✗ (needs `amd-quark` + GPU) | Wraps `python -m amd_quark.quantize`. |
| `mindxtrain serve`    | ✗ (needs `--extra serve` + GPU) | Builds the `vllm serve` command. |
| `mindxtrain publish`  | ◑ (works without HF/Lighthouse keys; skips gracefully) | HF Hub + Lighthouse + mindX register. |
| `mindxtrain receipt`  | ✓          | BLAKE3 reverify against the manifest.          |
| `mindxtrain.operator.app` (uvicorn) | ✓ | FastAPI + `/coach/` UI; no chat backend until vLLM is reachable. |

## On the MI300X (operator path)

The full ordered checklist is in [HANDOFF.md](../HANDOFF.md). Quick version:

```bash
# pull the canonical container
podman pull docker.io/rocm/primus:v26.2

# launch with device passthrough + cache volumes
podman run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 8G --ipc=host \
  -v $(pwd):/workspace/mindxtrain \
  -v ~/.cache/miopen:/root/.cache/miopen \
  -v ~/.cache/aiter:/root/.cache/aiter \
  -v ~/.cache/torch_extensions:/root/.cache/torch_extensions \
  -e PYTORCH_ROCM_ARCH=gfx942 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e GPU_MAX_HW_QUEUES=1 \
  rocm/primus:v26.2

# inside the container
cd /workspace/mindxtrain
pip install -e ".[ml,eval,data,obs]"
mindxtrain bench --gpu 0 --out plan.json     # the real 60s probe
mindxtrain dataset prep run.yaml --out ./out/dataset
mindxtrain train run.yaml --plan plan.json
mindxtrain eval run.yaml
mindxtrain quantize run.yaml
mindxtrain serve run.yaml                     # → vLLM-ROCm command
```

The volume mounts pre-warm the MIOpen kernel cache, AITER JIT cache, and Torch
extensions — without them, first-iteration latency on every container
restart will burn your demo window.

## Useful one-liners

```bash
# print the schema as JSON
uv run python -c "from mindxtrain.config.schema import XTrainConfig; import json; print(json.dumps(XTrainConfig.model_json_schema(), indent=2))" | head

# validate any YAML against the schema
uv run python -c "from mindxtrain.config.loader import load_config; print(load_config('examples/demo_qwen3_8b_sft.yaml').meta.run_name)"

# verify a provenance manifest
uv run mindxtrain receipt out/runs/<run_id>/manifest.json --config run.yaml

# launch the operator inference server
uv run uvicorn mindxtrain.operator.app:app --host 0.0.0.0 --port 8080
```

## Next steps

- [HANDOFF.md](../HANDOFF.md) — operator checklist.
- [architecture.md](architecture.md) — canonical layout + 5-layer architecture.
- [actualization_status.md](actualization_status.md) — what's real, what gates on extras.
- [autotune.md](autotune.md) — the 60-second probe.
- [yaml_schema.md](yaml_schema.md) — every field of the canonical YAML.
