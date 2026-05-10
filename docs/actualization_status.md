# Actualization status

A per-module map of what's real Python vs. what gracefully degrades to an
install hint or runtime requirement. Reflects the state after the
"actualize stubs" pass; counts and labels track the canonical layout from
[`blueprints/mindxtrain2.md`](blueprints/mindxtrain2.md) §Part 4.

## Headline numbers

- **99** Python modules under `mindxtrain/`.
- **38 actualized** (real implementations using stdlib / already-installed deps + lazy imports).
- **5 cloud-provider stubs** preserved in `budget/providers/*` (post-hackathon).
- **2 deliberate-redirects** that raise with a pointer to a sibling module
  (`storage/lighthouse.py:get_dir` → use `storage.ipfs`).
- **112 tests** pass on a CPU-only laptop (`uv run pytest -q`).
- **0** OLD-namespace imports anywhere (`from xtrain.`, `from automindx.`,
  `from custmodel` are all gone).

## What `uv sync` (no extras) gives you

Every module is *importable*. Anything that doesn't need a heavyweight
runtime works directly:

| Surface | Status |
|---|---|
| `mindxtrain --help` / `--version` / `init` / `init --list` | works |
| `mindxtrain bench --dry-run` | works (synthetic plan) |
| `mindxtrain receipt <manifest.json>` | works (BLAKE3 verify) |
| `mindxtrain.operator.app` (FastAPI, no chat backend) | boots; `/coach/` UI live |
| `mindxtrain.deploy.{registry,hot_swap,ab_test}` | atomic JSON-backed registry |
| `mindxtrain.operator.{tool_router,agent_loop,context,trajectory,approval}` | bounded ReAct, ContextManager, etc. |
| `mindxtrain.provenance.{manifest,hashing,verify}` | BLAKE3 manifest round-trip |
| `mindxtrain.storage.local_fs` | working |
| `mindxtrain.train.distributed` (FSDP/DeepSpeed config builders) | works |
| `mindxtrain.budget.{pricing,resource}` | works (psutil if installed) |

## What the optional-dep groups unlock

Install with `uv sync --extra <group>` (multiple `--extra` flags allowed,
or `--all-extras`):

| Group | Adds | Unlocks |
|---|---|---|
| `ml` | `trl`, `transformers`, `peft`, `accelerate`, `datasets` | `mindxtrain train`, `mindxtrain dataset prep`, `mindxtrain.train.{sft,dpo,grpo,rlhf,tool_use}`, `mindxtrain.data.{curate,tokenize}`, `mindxtrain.train.callbacks` |
| `eval` | `lm-eval`, `lighteval`, `inspect-ai`, `jinja2` | `mindxtrain eval`, `mindxtrain.eval.{harness,lighteval_adapter,inspect_ai_adapter,bfcl,tau_bench,card}` |
| `data` | `datasketch`, `sentence-transformers`, `faiss-cpu`, `pyarrow` | `mindxtrain.data.{dedupe,filter}` semantic paths, `mindxtrain.eval.persona_regression` |
| `serve` | `vllm` | in-process vLLM (the operator FastAPI app proxies via httpx by default) |
| `chain` | `web3`, `py-algorand-sdk`, `huggingface-hub` | `mindxtrain.provenance.{erc8004.broadcast_attestation,x402.validate_settlement,algorand}`, `mindxtrain.storage.hf_hub` |
| `obs` | `opentelemetry-sdk`, `prometheus-client`, `psutil` | `mindxtrain.operator.telemetry.*`, `mindxtrain.budget.resource.detect` |

The `all` extra installs everything except `amd-quark` (which ships with the
rocm/primus container — see [HANDOFF.md](../HANDOFF.md) §3).

## Per-subpackage status

### `mindxtrain.cli`

`main.py` — **real**. All 9 verbs (`init`, `bench`, `train`, `eval`,
`quantize`, `serve`, `publish`, `receipt`, `dataset prep`) dispatch into
canonical modules. Exit codes: `0` = ok, `1` = bad input / missing file,
`3` = optional dep missing.

### `mindxtrain.config`

`schema.py` (Pydantic 10-section `XTrainConfig`) and `loader.py`
(YAML render + load) — **real, frozen**. Three runtime-defaults JSON files
(`train_default.json`, `eval_default.json`, `deploy_default.json`) ship as
`${ENV}`-interpolated templates per mindxtrain2.md ml-intern style.

### `mindxtrain.data`

| Module | Status | Dep group |
|---|---|---|
| `curate.py` | real (HF datasets streaming) | `--extra ml` |
| `dedupe.py` | real MinHash + SemDeDup | `--extra data` |
| `filter.py` | real (length/repeat/alpha heuristics + optional KenLM) | none (stdlib) |
| `pack.py` | real (greedy first-fit + tar shards) | none (stdlib) |
| `synth.py` | real (httpx → vLLM teacher endpoint) | needs reachable `MINDXTRAIN_TEACHER_BASE_URL` |
| `tokenize.py` | real (AutoTokenizer wrap) | `--extra ml` |
| `verify.py` | real (BLAKE3 walk vs manifest) | none |

### `mindxtrain.models`

| Module | Status |
|---|---|
| `registry.py` | real (Backend ABC + ModelRegistry + preset registry) |
| `chat_template.py` | real (Hermes/Qwen3-Coder/Qwen3-Reasoning parsers) |
| `glm51.py`, `qwen35.py`, `deepseek_v32.py`, `mistral3.py`, `phi4_mini.py` | real Pydantic presets, auto-register on import |

### `mindxtrain.train`

| Module | Status | Dep group |
|---|---|---|
| `dispatch.py` | real 4-way switch | none |
| `axolotl_compile.py` | real (XTrainConfig → Axolotl YAML) | none |
| `sft.py` | real subprocess wrap of `accelerate launch -m axolotl.cli.train` | `--extra ml` + axolotl on PATH |
| `dpo.py`, `grpo.py`, `rlhf.py`, `tool_use.py` | real TRL trainer wraps | `--extra ml` |
| `distributed.py` | real (FSDP / DeepSpeed dict builders, 1- or 8-GPU only) | none |
| `callbacks.py` | real `EvalDuringTraining` + `BestCheckpointKeeper` | `--extra ml` (lazy) |
| `backend_unsloth.py`, `backend_torchtune.py`, `backend_primus.py` | real subprocess wraps | each backend's own install |

### `mindxtrain.eval`

| Module | Status | Dep group |
|---|---|---|
| `harness.py` | real `lm_eval` subprocess + JSON parser | `--extra eval` |
| `lighteval_adapter.py` | real `lighteval accelerate` wrap | `--extra eval` |
| `inspect_ai_adapter.py` | real `inspect eval` wrap | `--extra eval` |
| `bfcl.py` | real `bfcl evaluate` wrap | external (BFCL harness) |
| `tau_bench.py` | real subprocess wrap | external |
| `persona_regression.py` | real (sentence-transformer cosine vs baseline) | `--extra data` |
| `agenda_regression.py` | real (keyword overlap + optional LLM judge) | none + optional `MINDXTRAIN_TEACHER_BASE_URL` |
| `card.py` | real (Jinja2 with stdlib `string.Template` fallback) | optional `--extra eval` |

### `mindxtrain.autotune`

| Module | Status |
|---|---|
| `benchmark.py`, `plan.py`, `gemm_probe.py`, `rccl_probe.py` | real |
| `attention_probe.py` | real (CK vs Triton SDPA timing if torch+ROCm available; CPU fallback returns canonical default) |

### `mindxtrain.operator`

| Module | Status |
|---|---|
| `app.py` (FastAPI), `coach/api.py`, `coach/static/*` | real |
| `tool_router.py` | real (typed `ToolSpec` + dispatch) |
| `agent_loop.py` | real (bounded ReAct + doom-loop detector) |
| `context.py` | real (170k-token compaction + summarize fallback) |
| `trajectory.py` | real (JSONL append-only writer) |
| `approval.py` | real (CLI / Web / Slack transports) |
| `backends/{vllm,openai_compat}.py` | real (httpx clients to OpenAI-compat endpoints) |
| `telemetry/{energy,otel_hooks,prometheus_exporter}.py` | real, gracefully no-op if optional deps missing |
| `prompts/{system_v1,codephreak}.yaml` | real prompt-as-data |

### `mindxtrain.storage`

| Module | Status | Dep group |
|---|---|---|
| `provider.py` | real ABC | none |
| `local_fs.py` | real | none |
| `hf_hub.py` | real (huggingface_hub upload_folder) | `--extra chain` |
| `lighthouse.py` | real httpx POST to Lighthouse REST API; falls back to stub-CID without `LIGHTHOUSE_API_KEY` | none |
| `ipfs.py` | real httpx to kubo `/api/v0/add` | needs running kubo |

### `mindxtrain.provenance`

| Module | Status | Dep group |
|---|---|---|
| `manifest.py` | real (`Manifest` + `emit_receipt`) | none |
| `hashing.py` | real (BLAKE3 file/dir) | none |
| `verify.py` | real (re-hash on-disk artifacts) | none |
| `x402.py` | real httpx invoice + Algorand verify | `--extra chain` |
| `erc8004.py` | real ABI encode + web3 broadcast | `--extra chain` |
| `algorand.py` | real BANKON ENS allocator + ASA info | `--extra chain` |

### `mindxtrain.deploy`

| Module | Status | Dep group |
|---|---|---|
| `registry.py` | real atomic JSON-backed registry | none |
| `hot_swap.py` | real canary-promote + rollback | none |
| `ab_test.py` | real deterministic Splitter | none |
| `api_client.py` | real httpx → mindx.pythai.net + agenticplace.pythai.net | needs deployed services |
| `vllm_launcher.py`, `sglang_rocm.py` | real argv builders | none |
| `quark.py` | real subprocess wrap of `python -m amd_quark.quantize` | rocm/primus container |
| `gptq_rocm.py` | real subprocess wrap | `auto-gptq` ROCm wheel |

### `mindxtrain.budget`

| Module | Status | Dep group |
|---|---|---|
| `pricing.py` | real | none |
| `resource.py` | real (psutil + rocm-smi probes; falls back to defaults) | optional `--extra obs` |
| `providers/{akash,amd_dev_cloud,bacalhau,ionet,tensorwave}.py` | **stubs** (post-hackathon) | each provider's SDK |

## What stays as `NotImplementedError`

7 residual `NotImplementedError` raises across the package:

- `budget/providers/akash.py`, `amd_dev_cloud.py`, `bacalhau.py`, `ionet.py`,
  `tensorwave.py` — cloud-burst provisioning. Out of hackathon scope.
- `storage/lighthouse.py:LighthouseProvider.get_dir` — deliberately
  redirects to `mindxtrain.storage.ipfs.IpfsProvider.get_dir`.
- `train/dispatch.py` — string match in a docstring, not an actual raise.

Run `grep -r "raise NotImplementedError" mindxtrain` to confirm.

## Test coverage

```
tests/
├── test_ab_test.py                   # canary splitter distribution
├── test_agent_loop.py                # bounded ReAct + doom-loop
├── test_autotune_plan.py             # AutotunePlan invariants
├── test_axolotl_compile.py           # XTrainConfig → Axolotl YAML
├── test_cli_smoke.py                 # all 9 verbs reachable
├── test_coach_api.py                 # /coach/api/* endpoints
├── test_config_schema.py             # 10-section schema, recipe round-trip
├── test_context_manager.py           # ContextManager compaction
├── test_data_pipeline.py             # filter / synth / verify
├── test_deploy_registry.py           # registry + hot-swap atomicity
├── test_distributed.py               # FSDP/DeepSpeed builders, xGMI invariant
├── test_manifest.py                  # Manifest + BLAKE3 round-trip
├── test_models_registry.py           # preset + chat-template lookup
├── test_pack.py                      # greedy first-fit packer + tar shards
├── test_parsers.py                   # chat templates
├── test_pricing.py                   # MI300X $/hr math
├── test_provenance_verify.py         # tamper detection
├── test_tool_router.py               # ToolSpec dispatch
└── test_vllm_launcher.py             # vLLM cmd builder
```

`uv run pytest -q` → **112 passed**.

## See also

- [HANDOFF.md](../HANDOFF.md) — ordered checklist for taking the project from "code is done" to "demo is live."
- [development.md](development.md) — toolchain, lazy-import pattern, how to add features.
- [architecture.md](architecture.md) — canonical layout + 5-layer architecture.
