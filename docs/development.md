# Development workflow

Conventions and invariants for working in this repo. Read once before opening a PR.

## Toolchain

- **Python 3.12** (`>=3.12,<3.13`) — pinned; matches `rocm/primus:v26.2`.
- **uv** — single project (no workspace). `uv sync` installs the base deps;
  `uv sync --extra <group>` adds optional groups.
- **ruff** — replaces black/isort/flake8/pyupgrade. Config in
  [`pyproject.toml`](../pyproject.toml).
- **mypy --strict** — only on `mindxtrain/config` and `mindxtrain/provenance`
  (the schemas + manifest paths). Training / eval code is exempt.
- **pytest** + `pytest-asyncio` — fast unit tests; GPU tests are manual on
  the MI300X.
- **Foundry** — Solidity contracts in `contracts/`. Installed on the MI300X
  droplet for the on-chain anchoring path.

## Optional dependency groups

`pyproject.toml` defines six `[project.optional-dependencies]` groups:

| Group   | Adds                                                        |
|---------|-------------------------------------------------------------|
| `ml`    | trl, transformers, peft, accelerate, datasets               |
| `eval`  | lm-eval, lighteval, inspect-ai, jinja2                      |
| `data`  | datasketch, sentence-transformers, faiss-cpu, pyarrow       |
| `serve` | vllm                                                        |
| `chain` | web3, py-algorand-sdk, huggingface-hub                      |
| `obs`   | opentelemetry-sdk, prometheus-client, psutil                |

Plus `all` which pulls everything except `amd-quark` (which ships in the
rocm/primus container, see [HANDOFF.md](../HANDOFF.md) §3).

The base install (no extras) is enough for: the CLI, the Coach UI, the
autotune dry-run, manifest verify, the operator FastAPI app, and every
in-process Python utility (registry, hot-swap, agent loop, ContextManager,
data filter, sequence packing). See
[actualization_status.md](actualization_status.md) for the per-module map.

## Lazy-import pattern

Every module that wants an optional dep guards the import inside the
function that needs it:

```python
def run_lm_eval(model_dir: Path, tasks: list[str]) -> Path:
    if not _lm_eval_available():
        msg = "lm-eval not installed; run `uv sync --extra eval`."
        raise RuntimeError(msg)
    ...  # subprocess wrap that uses the dep
```

Two implications:

1. `import mindxtrain.eval.harness` always succeeds even without `--extra eval`.
2. The error message includes the exact `uv sync --extra <group>` to run.

This is the canonical pattern; new modules that take optional deps must
follow it.

## The standard local cycle

```bash
uv sync                                                    # base install
uv run ruff check --fix .                                  # lint + auto-fix
uv run mypy mindxtrain/config mindxtrain/provenance        # types where strict
uv run pytest -q                                           # → 112 passed in ~5s
```

CI runs the same four commands on Ubuntu 24.04 / Python 3.12 (CPU-only).
See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Repository layout

```
.
├── pyproject.toml                # single project; optional-dep groups
├── HANDOFF.md                    # operator checklist
├── HACKATHON.md                  # daily verification gates
├── README.md, NOTICE, LICENSE-*  # legal + entry doc
├── CHANGELOG.md
├── Containerfile, compose.yaml   # podman entry points
├── mindxtrain/                   # the package — 12 subpackages, ~99 modules
│   ├── cli/                      # typer CLI (9 verbs)
│   ├── config/                   # 10-section Pydantic schema + JSON defaults
│   ├── data/                     # curate → dedupe → filter → tokenize → pack → synth → verify
│   ├── models/                   # registry + chat templates + 5 base presets
│   ├── train/                    # sft, dpo, grpo, rlhf, tool_use, distributed, callbacks, recipes/
│   ├── eval/                     # lighteval / inspect_ai / bfcl / persona / agenda / card
│   ├── autotune/                 # 60s AOT probe — the differentiator
│   ├── operator/                 # FastAPI app, Coach UI, ml-intern patterns
│   ├── storage/                  # local_fs / hf_hub / lighthouse / ipfs
│   ├── provenance/               # manifest, hashing, verify, erc8004, algorand, x402
│   ├── deploy/                   # registry, hot_swap, ab_test, vllm_launcher, quark
│   └── budget/                   # ResourceBudget + cloud-provider stubs
├── contracts/                    # Foundry workspace (ERC-8004 attestation)
├── ops/                          # containerfiles, compose, k8s, vmm, gensyn
├── examples/                     # demo YAMLs
├── tests/                        # pytest — 112 tests (CPU-only smoke)
└── docs/
    ├── *.md                      # current state (this directory)
    └── blueprints/               # source design briefs (frozen)
```

## Reuse boundaries

- **From `/home/hacker/mindX/`** (production codebase): Codephreak persona
  JSON loaded at runtime via `MINDXTRAIN_PERSONA_PATH`. Do not copy file
  bytes — load via env var.
- **Not** from `/home/hacker/aglm/` — broken per its own README. Use only
  for reference to legacy class names mindxtrain2.md flagged as needing
  refactor.

## Invariants

These are non-negotiable; violating them is a deployment bug, not a style
preference.

1. **AOT-only.** No `torch.compile(mode="max-autotune")` in production paths.
   No JIT autotune in vLLM serving (set `VLLM_USE_TRITON_FLASH_ATTN=0` if
   needed). The `autotune.policy: aot_only` field in the YAML is the
   contract; tested at
   `tests/test_config_schema.py::test_qwen3_8b_sft_lora_validates`.
2. **`hardware.gpus: 1 | 8` only.** 2/4-GPU MI300X FSDP groups hit
   asymmetric xGMI; the schema rejects them at parse time. Tested at
   `tests/test_config_schema.py::test_xgmi_2gpu_rejected` and
   `tests/test_distributed.py`.
3. **Seven MI300X env vars in `train.env`** (defaults, can be overridden by
   the autotune plan but never removed): `HSA_NO_SCRATCH_RECLAIM=1`,
   `NVTE_CK_USES_BWD_V3=1`, `NVTE_CK_IS_V3_ATOMIC_FP32=1`,
   `PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32=1`, `NCCL_MIN_NCHANNELS=112`,
   `HIP_FORCE_DEV_KERNARG=1`, `PYTORCH_ROCM_ARCH=gfx942`.
4. **`extra: forbid` on every Pydantic model.** Unknown YAML keys raise
   `ValidationError`. Tested at
   `tests/test_config_schema.py::test_extra_field_forbidden`.
5. **Configs are immutable once loaded** (`frozen: true`).
6. **Solidity contracts: no proxies, no `Ownable`, no admin keys, no setters.**
   `mindxtrain_registry.sol` is write-once. Rotating any parameter requires a
   fresh deploy.
7. **Lazy imports for optional deps** — see the pattern above.

## Adding a new recipe

1. Drop a YAML at `mindxtrain/train/recipes/<name>.yaml`. Validate locally:
   ```bash
   uv run python -c "from mindxtrain.config.loader import load_config; load_config('mindxtrain/train/recipes/<name>.yaml')"
   ```
2. The `tests/test_config_schema.py::test_all_recipes_validate` test will
   pick it up automatically — re-run pytest.
3. Add a row to [docs/yaml_schema.md](yaml_schema.md) only if the recipe
   exercises a previously-unused field.

## Adding a new training backend

1. Add `mindxtrain/train/backend_<name>.py` exposing a
   `run_<name>(cfg, plan, out_dir) -> Path` function (or for in-process TRL
   trainers, a `run_<name>(cfg, out_dir) -> Path` function).
2. Wire it into `mindxtrain/train/dispatch.py`'s `if backend == ...` ladder.
3. Add `<name>` to the `TrainingBackend` literal in
   `mindxtrain/config/schema.py`.
4. Update [docs/cli.md](cli.md) "Where the verbs live" table.

## Adding a new model backend (operator)

1. Add `mindxtrain/operator/backends/<name>.py` with a `Backend` subclass
   decorated `@register_backend("<name>")`.
2. Side-effect import it from `mindxtrain/models/registry.py` so registration
   runs on package import.
3. Add a runtime branch in `mindxtrain/operator/app.py::chat_completions` for
   the env-var-driven kwargs.

## Adding a new training method

1. Define a `_MethodBase` subclass in `mindxtrain/config/schema.py` with
   `kind: Literal["<name>"] = "<name>"` and the method-specific fields.
2. Add it to the `TrainMethod` discriminated union.
3. Add a `mindxtrain/train/<name>.py` runner (TRL wrap or subprocess).
4. Update the dispatch path so a YAML with `train.method.kind == "<name>"`
   reaches the runner.
5. Add a recipe under `mindxtrain/train/recipes/` exercising it.
6. Update `docs/yaml_schema.md` "train.method" table.

## Adding a new optional-dep group

1. Add the entry to `[project.optional-dependencies]` in `pyproject.toml`.
2. Add a row to the table in [actualization_status.md](actualization_status.md).
3. Update [development.md](development.md) and [quickstart.md](quickstart.md).

## Adding a new doc

1. Write `docs/<name>.md`.
2. Add a one-line entry to [`docs/NAV.md`](NAV.md) under the appropriate section.

## Live training UI

The Coach UI's "Train" step (`#step-train` in
[`coach/static/index.html`](../mindxtrain/operator/coach/static/index.html))
launches a training run and streams loss / lr / log lines back into the
browser over Server-Sent Events. Architecture:

- **Registry**: `mindxtrain.operator.runs.RunRegistry` is an in-memory
  singleton (one per uvicorn process) keyed by `run_id`. Snapshots are
  immutable `Run` records (frozen Pydantic); state changes produce new
  snapshots via `model_copy`.
- **Event schema**: `TrainEvent` is a tagged union over `StatusEvent`,
  `StepEvent`, `EvalEvent`, `LogEvent`, `EnergyEvent` — all with
  `extra="forbid", frozen=True`. Wire format: `event: <kind>\ndata:
  <event.model_dump_json()>\n\n`.
- **Two ingestion paths**, deduped by `(run_id, step)` in
  `RunRegistry.publish`:
  1. Subprocess stdout regex (`parse_trainer_log_line`) — works on the
     base install, parses HF Trainer's `'loss': … 'learning_rate': …`
     log lines.
  2. In-process `mindxtrain.train.callbacks.StreamCallback` — POSTs to
     `/coach/api/runs/{id}/ingest` (loopback only). Requires `--extra ml`.
- **Subprocess orchestration**: `spawn_subprocess_streaming` uses
  `subprocess.Popen(stdout=PIPE, bufsize=1, text=True)` and tees lines
  to both `train.log` (the durable on-disk artifact) and
  `RunRegistry.publish_threadsafe` from a daemon thread. We use
  `Popen` (not `asyncio.create_subprocess_exec`, not `BackgroundTasks`)
  so the child outlives the launch HTTP request and `SIGINT`-then-`SIGTERM`
  cancellation matches the CLI Ctrl-C path.

### Routes

All under `/coach/api/runs`:

| Verb | Path | Purpose |
|---|---|---|
| POST | `/launch` | Spawn a run; returns `Run` immediately. 503 if `accelerate` is missing. |
| GET  | `/` | List active + last 20 runs. |
| GET  | `/{id}` | `Run` snapshot. |
| GET  | `/{id}/events` | SSE — all event kinds. Replays last 200 buffered on connect. |
| GET  | `/{id}/logs` | SSE — `kind="log"` only. |
| POST | `/{id}/cancel` | `SIGINT` then `SIGTERM` after grace. |
| POST | `/{id}/ingest` | Loopback-only — used by `StreamCallback`. |

SSE responses set `Cache-Control: no-cache`, `X-Accel-Buffering: no`,
`Connection: keep-alive` so reverse proxies don't buffer the stream.

### Invariants

- `import mindxtrain.operator.runs` succeeds **without** `--extra ml`. The
  in-process `StreamCallback` requires `transformers`; the subprocess-stdout
  path does not. UI degrades gracefully.
- `Run` and every `*Event` are `frozen=True, extra="forbid"`.
- The subprocess line reader runs in a daemon thread; events reach the
  asyncio loop via `loop.call_soon_threadsafe(registry.publish, …)`.

### Frontend

Vanilla JS, no build step. Live view uses the browser-native `EventSource`:

```js
const es = new EventSource(`/coach/api/runs/${id}/events`);
es.addEventListener("step",   e => pushPoint(JSON.parse(e.data)));
es.addEventListener("log",    e => appendLog(JSON.parse(e.data)));
es.addEventListener("status", e => updateBadge(JSON.parse(e.data)));
```

**Chart.js is vendored locally** at `coach/static/vendor/chart.umd.min.js`
(pinned to v4.4.0; SHA256 in `coach/static/vendor/VERSIONS.md`). No CDN
dependency at demo time. If the vendored bundle is missing, the page
degrades to a metrics table — `coach.js` checks `typeof Chart === "undefined"`
and shows the table-only fallback.

### Why not Selenium / WebSocket / Streamlit

- **Selenium** is a browser-test framework, not a UI library — it
  can't push live data into a browser. (It might appear later as CI
  smoke for the dashboard; that's E2E testing, not UI.)
- **WebSocket** is bidirectional; we don't need browser→server streaming.
  Held in reserve for v2 "edit hyperparam mid-run."
- **Streamlit / Gradio** each spin up their own ASGI server on a separate
  port, which breaks the single-URL operator demo and the lazy-import
  invariant. SSE on the existing `:8080` is the right shape.

## Common debugging

| Symptom                                        | Cause                                                                                              |
|------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: No module named 'mindxtrain'` | Forgot `uv sync`. Fixed by `uv sync`.                                                          |
| `RuntimeError: ... not installed; run uv sync --extra <group>` | Optional dep gating — install the named group.                                       |
| `pydantic.ValidationError: extra keys not permitted` | YAML has a typo or stale field name. Compare to [yaml_schema.md](yaml_schema.md).            |
| `ValueError: MI300X xGMI permits only 1 or 8 GPUs` | `hardware.gpus` is 2 or 4. Use 1 or 8.                                                          |
| `Failed to download due to network timeout` (uv) | `UV_HTTP_TIMEOUT=120 uv sync`.                                                                  |
| First-iteration training is 30s slow on MI300X | Cold AITER / MIOpen / Triton caches. Volume-mount `~/.cache/miopen`, `AITER_JIT_DIR`, `TORCH_EXTENSIONS_DIR`. |
| `vllm serve` stalls on first batch             | Triton autotune cold-start. Set `VLLM_USE_TRITON_FLASH_ATTN=0` or warm-up batch in `mindxtrain serve`. |

## What not to commit

- `*.safetensors`, `*.bin`, `*.pt`, `*.onnx` (large model weights).
- `out/`, `runs/`, `checkpoints/` (run outputs).
- `.env` (use `.env.example`).
- `contracts/lib/` (Foundry submodules — pulled with `forge install`).
- `.venv/`, `.uv-cache/`, `.cache/`.

All of the above are in [`.gitignore`](../.gitignore).
