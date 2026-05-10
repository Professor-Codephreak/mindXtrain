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
├── api.py                 # routes (recipes / bench / compile / cost / health)
└── static/
    ├── index.html         # five-step SPA shell
    ├── style.css          # minimal dark-friendly CSS, AMD orange accent
    └── coach.js           # vanilla JS state machine, no framework
```

The Coach mounts under `/coach/`; static assets are at `/coach/static/*`.

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

The full schema is rendered at `/docs` (Swagger).

## The five-step storyboard

The UI walks left-to-right through:

1. **Pick a recipe** — clickable grid of all 12 built-in recipes; the selected one's YAML expands inline.
2. **Run the autotune probe** — single button; shows the `AutotunePlan` JSON plus a six-chip summary (`attention=ck`, `gemm=hipblaslt_default`, `rccl=1gpu_noop`, …).
3. **Compile to Axolotl YAML** — translates `(recipe, plan)` into the trainer-side YAML, surfaces the plan-driven overrides as chips above the YAML.
4. **Cost vs H100** — sliders for GPUs and hours; emits a three-row comparison table (MI300X / H100 / H200) with a headline like "MI300X is 5.4× cheaper than the H100 baseline".
5. **Try the model** — chat panel that proxies to `/v1/chat/completions`. Stays disabled and explains why until the backend reports ready.

## Demo storyboard for the 5-min hackathon video

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

Run with `uv run pytest tests/test_coach_api.py -v`.

## Customizing for the demo

Tweak the cost-comparison constants in `mindxtrain/operator/coach/api.py`:

```python
H100_USDC_PER_HOUR = 4.00
H200_USDC_PER_HOUR = 6.00
```

The MI300X rate is sourced from `mindxtrain.budget.pricing.MI300X_USDC_PER_HOUR` ($1.99/hr, AMD Developer Cloud list price).

## Wiring the chat panel to a live vLLM endpoint

To enable the chat panel:

```bash
export MINDXTRAIN_BACKEND=vllm
export MINDXTRAIN_VLLM_BASE_URL=http://localhost:8000/v1
podman-compose -f ops/compose/compose_dev.yaml up -d   # boots vLLM-ROCm + the mindxtrain operator
```

Then `coach.js` `probeChat()` flips the panel from disabled to enabled, and the textarea proxies to `/v1/chat/completions` against the FP8 model produced by `mindxtrain quantize`. The full pipeline lives in [HANDOFF.md](../HANDOFF.md) §§ 5–6.
