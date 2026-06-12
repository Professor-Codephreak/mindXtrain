# mindxtrain Coach (UI)

A single-page web UI that walks judges and new contributors through the mindxtrain pipeline without needing a GPU. Bundled inside the mindxtrain.operator FastAPI app at `/coach/`.

## Why it exists

Hackathon judges have ~3 minutes per submission. The Coach lets them poke at the differentiator (the 60-second AOT autotune) and the cost story (4× cheaper than H100) interactively, in a browser, without setting up ROCm.

## Boot

```bash
uv run uvicorn mindxtrain.operator.app:app --host 0.0.0.0 --port 8080
```

Open http://localhost:8080 — the root path redirects to `/coach/`.

The Coach works **without** a backend GPU:
- The autotune endpoint runs `run_autotune(dry_run=True)` and emits the reference plan.
- The compile endpoint produces a real Axolotl YAML against the dry-run plan.
- The cost calculator is pure arithmetic.

The chat panel stays disabled until `MINDXTRAIN_BACKEND=vllm` is set and a vLLM-ROCm server is reachable.

## Layout

```
mindxtrain/operator/coach/
├── __init__.py            # exports the FastAPI router
├── api.py                 # routes (recipes / bench / compile / cost / health /
│                          #   runs / metrics / receipt / sea-decision / mei / diagnostics)
├── run_metrics.py         # 1 Hz system-metrics sampler (psutil + /proc)
├── chronos_client.py      # mindX promised-time client
└── static/
    ├── index.html         # multi-card SPA shell (preflight → … → train → receipt → chat)
    ├── style.css          # minimal dark-friendly CSS, AMD orange accent
    └── coach.js           # vanilla JS state machine, no framework
```

The Coach mounts under `/coach/`; static assets are at `/coach/static/*`. The UI
has grown well past the original five-step demo: it now covers preflight, hardware
detection, dream-corpus stats, recipe pick, autotune, compile, **live training with
diagnostic feedback**, the **verifiable receipt**, MEI scoring, cost, deploy, and chat.

## Routes

| Method | Path                                | Body / Query              | Returns                                    |
|--------|-------------------------------------|---------------------------|--------------------------------------------|
| GET    | `/`                                 | —                         | 307 redirect to `/coach/`                  |
| GET    | `/coach/`                           | —                         | `index.html`                               |
| GET    | `/coach/static/{path}`              | —                         | static files                               |
| GET    | `/coach/api/recipes`                | —                         | `list[RecipeSummary]` (12 items)           |
| GET    | `/coach/api/recipes/{name}`         | —                         | `{ name, yaml, summary }`                  |
| POST   | `/coach/api/bench`                  | (none)                    | `AutotunePlan` (dry-run reference)         |
| POST   | `/coach/api/compile`                | `{recipe, plan?}`         | `{recipe, config_summary, plan, axolotl_yaml, overrides}` |
| POST   | `/coach/api/cost`                   | `{gpus, hours, safety_margin}` | `{mi300x, h100, h200, speedup_vs_h100_x}`  |
| GET    | `/coach/api/health`                 | —                         | `{coach_version, chat_backend_ready, recipes_available}` |
| POST   | `/coach/api/runs/launch`            | `{recipe, plan?, out_dir?}` | `Run` snapshot (spawns training)         |
| GET    | `/coach/api/runs/{id}/events`       | —                         | SSE stream (`status`/`step`/`eval`/`log`/`metrics`/`energy`) |
| GET    | `/coach/api/runs/{id}/metrics`      | `?since=`                 | system-metrics backfill for the sparklines |
| GET    | `/coach/api/receipt/{run_id}`       | —                         | `ReceiptView` — re-verified BLAKE3 hashes + `verified` |
| GET    | `/coach/api/sea-decision`           | —                         | mindX SEA autonomous-training gate state   |
| GET    | `/coach/api/mei/score/{run_id}`     | —                         | `MEIScoreView` (mindX Efficiency Index)    |
| GET    | `/coach/api/diagnostics/live`       | —                         | host load / RAM% / disk% / operator RSS    |

The full schema is rendered at `/docs` (Swagger).

## Live training diagnostics

The **Train (live)** card is the accurate, real-time depiction of a run. Events
arrive over Server-Sent Events (`/coach/api/runs/{id}/events`) — `step`, `eval`,
`log`, and 1 Hz `metrics` — and drive these surfaces:

- **Session headline** — status badge, wall-clock + CPU-time elapsed, throttle%,
  last loss, freshest eval. The at-a-glance "is it healthy" line.
- **Phase + progress** — friendly phase narration ("Loading base model…",
  "Training…", "Saving checkpoint…") plus a progress bar with `step N / total · ETA`,
  driven by `StepEvent.total_steps`.
- **Loss curve** (Chart.js) — dual-axis loss (orange) + `mean_token_accuracy`
  (green, NaN-gapped where a backend omits it). The primary "is it learning" signal.

Because a real MI300X run logs **thousands of steps**, the heavy detail is kept
accurate but compressed behind accordions, with the truncation always shown — never
silent:

- **Loss curve** keeps a rolling window of the last `MAX_CHART_POINTS` (1500) points;
  once it rolls, a `showing last 1500 of N steps` note appears under the chart.
- **Per-step metrics** (step, loss, acc, entropy, lr, grad_norm) live in a collapsed
  `<details>` accordion; the DOM table caps at 50 rows but the summary reports the
  true total — `per-step metrics (N steps · last 50 shown)`.
- **train.log (live tail)** is a `<details>` accordion that auto-folds older lines
  and shows a running `(N lines)` count, capping the DOM at `MAX_LOG_LINES` (2000)
  and labelling `· oldest dropped` once it does.
- **System metrics** — five d3 sparklines (host cpu%/ram%/load, trainer rss MB,
  trainer cpu-s/s) sampled at 1 Hz, in their own `<details>` (open by default).

This keeps the page legible on a laptop while the underlying data stays faithful.

## Verifiable receipt card

When a run finishes, the operator emits `manifest.json` (BLAKE3 of the config
snapshot, checkpoint, and the frozen `AutotunePlan`) into the run directory. The
**Verifiable receipt** card fetches `/coach/api/receipt/{run_id}`, which re-hashes
the on-disk artifacts and returns a `verified` flag plus the per-field checks. A
`verified ✓` badge and the truncated hashes render in the card; the same check runs
from a shell via `mindxtrain receipt out/runs/<run>/manifest.json --config <recipe>.yaml`.
Binding the AutotunePlan hash to the checkpoint is the AOT-as-verification primitive —
it proves which compiled backend/heuristic/RCCL config produced the weights.

## Create script + imprint (actor / persona / script)

mindXtrain (and Coach) **train models**. The model is an **actor**; an actor has a
**persona** (identity / voice) and a **script** (the training examples — the
"impression"). The **Create script** card authors a small script in the browser and
saves it as `source: local` JSONL the recipes ingest.

- **`POST /coach/api/datasets`** — `{name, persona_name, system_prompt, voice_examples,
  exchanges:[{user,assistant}], seed_voice}` → writes
  `out/datasets/<name>/script.jsonl` (override the root with `MINDXTRAIN_DATASETS_DIR`).
  `GET /coach/api/datasets` lists them; `GET /coach/api/datasets/{name}` previews.
- **`GET /coach/api/persona`** — pre-fills the form from `MINDXTRAIN_PERSONA_PATH`
  (clean-room: recognised fields only, never copies mindX bytes).
- Point the **`mindx_persona_imprint_local`** recipe's `data.path` at the saved script
  and train the tiny actor (`trl_local`, CPU or local GPU).

**Imprint = recall, before vs after.** Pose the script's own user-turns back to the
actor and compare the base model (before) with the trained adapter (after) against the
script's assistant voice:

```bash
mindxtrain imprint mindxtrain/train/recipes/mindx_persona_imprint_local.yaml
```

prints an `ImprintReport` (`before_voice`, `after_voice`, `imprint_delta`, `shift`,
`imprinted`); exit 4 if no imprint took. `POST /coach/api/imprint/score` scores supplied
utterances without blocking the event loop on inference. `mindxtrain imprint
--trigger-dream` hands the imprinted actor to mindX's `machine.dream` 8-hour cycle (via
`MINDXTRAIN_API_BASE_URL` `/v1/dream/ingest`, else a `data/incoming/` inbox drop under
`MINDXTRAIN_MINDX_HOME`) — clean-room, an artifact pointer, never mindX code.

## Create script — personas + skills

The **Create script** card authors a `source: local` JSONL from a persona and toggleable
skills:

- **Built-in personas** (`GET /coach/api/personas`) — `codephreak`, `assistant`, `mentor`
  (`mindxtrain.data.personas.BUILTIN_PERSONAS`). Pick one, or use the custom fields.
- **Skills** — toggle **Software Engineer / Platform Architect / Bash / Solidity** to mix
  each skill's in-domain exchanges into the script (`mindxtrain.data.personas.SKILLS`,
  `compose(persona, skills)`). A skill is a system-prompt addendum + representative turns.
- `POST /coach/api/datasets` composes persona + skills + your exchanges and returns the row
  count plus **training params auto-derived from the dataset size**
  (`derive_training_params` — small scripts overfit to imprint: more epochs, grad_accum 1).

## Build an Ollama Modelfile (separate window)

The **Build Modelfile…** button (in the train card's push-to-ollama row) opens a standalone
builder at `/coach/modelfile` (a separate browser window), pre-filled for the current run:

- Every instruction is a toggle: `FROM` (required), `SYSTEM`, `TEMPLATE`, `ADAPTER`,
  `LICENSE`, `REQUIRES`, plus `MESSAGE` examples and `stop` sequences.
- Every `PARAMETER` (`num_ctx`, `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`,
  `mirostat`, `seed`, … — the full catalogue from `GET /coach/api/modelfile/params`) is a
  toggle + input, rendered dynamically with defaults and ranges.
- `POST /coach/api/modelfile/build` renders the `Modelfile` text;
  `POST /coach/api/modelfile/create` runs `ollama create <tag>`. Core logic:
  `mindxtrain.deploy.modelfile` (`ModelfileSpec`, `render_modelfile`, `create_model`).

## The core storyboard

The original CPU-only demo path, top-to-bottom (the cards above and below it —
preflight, hardware, dream-corpus, live training, receipt, MEI, deploy — flank it):

1. **Pick a recipe** — clickable grid of all built-in recipes; the selected one's YAML expands inline.
2. **Run the autotune probe** — single button; shows the `AutotunePlan` JSON plus a six-chip summary (`attention=ck`, `gemm=hipblaslt_default`, `rccl=1gpu_noop`, …).
3. **Compile to Axolotl YAML** — translates `(recipe, plan)` into the trainer-side YAML, surfaces the plan-driven overrides as chips above the YAML.
4. **Train (live)** — spawns the run and streams the diagnostic feedback described in [Live training diagnostics](#live-training-diagnostics); on a CPU box the `trl_cpu` lane trains a small model in-process so the whole loop is demoable without a GPU. The `trl_local` lane is the device-aware variant — it uses a local consumer GPU (CUDA or ROCm Radeon) when present and falls back to CPU otherwise, so the same recipe runs on a laptop or a gaming GPU. `recommend_lane` sends an Instinct/MI300X card to `axolotl_amd` and any other local GPU to `trl_local`.
5. **Verifiable receipt** — the `verified ✓` badge + bound hashes appear the moment the run completes.
6. **Cost vs H100** — sliders for GPUs and hours; emits a three-row comparison table (MI300X / H100 / H200) with a headline like "MI300X is 5.4× cheaper than the H100 baseline".
7. **Try the model** — chat panel that proxies to `/v1/chat/completions`. Stays disabled and explains why until the backend reports ready; a **Check now** button re-probes on demand.

## Demo storyboard

```
0:00–0:30  open localhost:8080, point at the three-stage diagram in the header
0:30–1:00  click qwen3_8b_sft_lora; show the YAML preview
1:00–2:00  click "Run autotune (dry-run)"; show the plan JSON streaming in
           and the six-chip summary populating
2:00–3:00  click "Compile"; show the Axolotl YAML diff (the autotune
           plan's attention_backend appears as flash_attn_backend=ck)
3:00–4:00  drag the cost slider to 1 GPU × 1.5 hours; show the
           "5× cheaper than H100" headline
4:00–5:00  the chat panel; show that it's gracefully disabled because
           the backend isn't booted, then close
```

Every Coach interaction is screen-recordable on a CPU-only laptop. The MI300X work happens behind the scenes for the actual training run; the Coach surfaces the *outcome* judges care about.

## Dependencies

- FastAPI — already a dep of mindxtrain.operator.
- `mindxtrain` — workspace dep added to `pyproject.toml` so the Coach can call `mindxtrain.config.loader.list_recipes()`, `mindxtrain.autotune.benchmark.run_autotune()`, and `mindxtrain.train.compile_axolotl_yaml()`.
- `pyyaml` — added for the recipe→summary path.

No JavaScript framework, no build step, no node_modules.

## Tests

`tests/test_coach_api.py` covers every endpoint via FastAPI's `TestClient`:

- root redirects to `/coach/`
- index serves HTML with the right `<title>`
- static files serve (CSS + JS)
- recipes list returns 12 items
- recipe detail returns YAML + summary
- 404 on unknown recipe
- bench returns a valid `AutotunePlan`
- compile returns Axolotl YAML + overrides; 404 on unknown recipe
- cost returns three breakdowns; 422 on invalid input
- health endpoint reports `recipes_available=12`
- `/health` mentions `coach_url=/coach/`
- the train card exposes the diagnostic accordions (`metrics-table-wrap`,
  `metrics-table-count`, `train-log-count`, `chart-window-note`) and coach.js wires
  the rolling-window cap + counters (`MAX_CHART_POINTS`, `_updateMetricsTableCount`,
  `_updateLogCount`)
- the receipt card + loader are present (`step-receipt`, `loadReceiptForRun`)

The live-training + receipt round-trip is covered in `tests/test_coach_receipt_api.py`
(canned spawn → `/coach/api/receipt/{id}` returns `verified=True`).

Run with `uv run pytest tests/test_coach_api.py -v`.

## Customizing for the demo

Tweak the cost-comparison constants in `mindxtrain/operator/coach/api.py`:

```python
H100_USDC_PER_HOUR = 4.00
H200_USDC_PER_HOUR = 6.00
```

The MI300X rate is sourced from `mindxtrain.budget.pricing.MI300X_USDC_PER_HOUR` ($1.99/hr, AMD Developer Cloud list price).

## Streaming chat + ollama controls (Try the model)

The **Try the model** card chats with a local model and **streams the response
token-by-token** — the [AI SDK](<Vercel AI SDK 6_ A Framework-Agnostic Deep Dive (June 2026).md>)
text-stream pattern, implemented in vanilla JS (no build step): `coach.js` consumes a
`text/event-stream` whose `data:` lines are JSON token deltas, ending with `data: [DONE]`.

- **`POST /coach/api/chat/stream`** — `{model, messages, max_tokens?}` → SSE token stream.
  Relays `backend.stream_chat()` (the OpenAI-compatible streaming the ollama/vLLM backends
  already speak). Backend errors are surfaced in-stream (`event: error`), never as a mid-stream 500.
- **Model picker** — populated from `GET /coach/api/models` (local models sorted ahead of
  `:cloud`), so the chat no longer defaults to a cloud model that silently returns nothing.
- **ollama controls** — `GET /coach/api/ollama/status` + `POST /coach/api/ollama/{start,stop}`
  start/stop the local `ollama serve` and report its state; `↻ models` re-lists.

For a remote vLLM-ROCm endpoint instead, set `MINDXTRAIN_BACKEND=vllm` +
`MINDXTRAIN_VLLM_BASE_URL`; the same streaming chat works against it
(see [HANDOFF.md](HANDOFF.md) §§ 5–6).
