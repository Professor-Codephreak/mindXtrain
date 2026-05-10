# mindxtrain

Production training framework for fine-tuning open-weight LLMs on AMD MI300X
and serving them through an OpenAI-compatible API. Single ordered package,
canonical layout per [`docs/blueprints/mindXtrain2.md`](docs/blueprints/mindXtrain2.md)
§Part 4.

The single architectural feature that distinguishes mindxtrain from Axolotl,
LLaMA-Factory, Unsloth, torchtune and Primus is its **60-second AOT autotune
probe**: CK-vs-Triton attention, hipBLASLt heuristic, RCCL config — the plan
is fixed at training start, JIT autotune is forbidden in the production loop.

**Status**: scaffolded, single canonical package, 112/112 tests pass, ruff
clean. ~38 modules ship as real Python on a CPU-only laptop; heavyweight
training, eval, and quantization paths gate on opt-in extra dep groups.
See [`docs/actualization_status.md`](docs/actualization_status.md) for the
per-module map and [`HANDOFF.md`](HANDOFF.md) for the operator checklist.

## Hackathon

AMD × lablab.ai Developer Hackathon, build window May 4–10 2026, on-site
finale May 9–10 in San Francisco.

- **Demo URL:** _TBD — `mindx.pythai.net/hackathon`_
- **Video (5 min, ≤300 MB):** _TBD_
- **Tracks:** AI Agents & Agentic Workflows, Fine-Tuning on AMD GPUs, Vision & Multimodal AI

See [`HACKATHON.md`](HACKATHON.md) for daily verification gates and
[`docs/NAV.md`](docs/NAV.md) for the full documentation hub.

## Quickstart

```bash
uv sync                                                    # base install
uv run pytest -q                                           # → 112 passed
uv run mindxtrain --help                                   # 9 verbs
uv run mindxtrain init --template qwen3_8b_sft_lora --out run.yaml
uv run mindxtrain bench --dry-run --out plan.json
uv run uvicorn mindxtrain.operator.app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080/coach/  for the interactive UI
```

To unlock training / eval / quantize / publish, install the matching dep group:

```bash
uv sync --extra ml --extra eval --extra data         # train + eval + curate
# or
uv sync --all-extras                                  # everything except amd-quark
```

GPU steps (`bench` without `--dry-run`, `train`, `quantize`, `serve`) require
an AMD MI300X with ROCm 7.2.1; run inside `rocm/primus:v26.2`. The full
operator checklist lives in [`HANDOFF.md`](HANDOFF.md).

## Layout

```
mindxtrain/{cli,config,data,models,train,eval,autotune,
            operator,storage,provenance,deploy,budget}/   # 99 modules
contracts/        Foundry workspace for ERC-8004 attestation registry
ops/              containerfiles, compose, k8s, vmm, gensyn
tests/            pytest suite — 112 tests, CPU-only smoke
examples/         demo YAML configs
docs/             user-facing documentation + frozen blueprints
scripts/          dev helpers
```

## Documentation

| Doc | What it covers |
|-----|----------------|
| [`HANDOFF.md`](HANDOFF.md) | **Operator checklist** — 11 ordered steps from local setup to submission. |
| [`docs/quickstart.md`](docs/quickstart.md) | Install + base-vs-extras command tour. |
| [`docs/architecture.md`](docs/architecture.md) | Canonical layout + 5-layer architecture + MI300X invariants. |
| [`docs/actualization_status.md`](docs/actualization_status.md) | Per-module map of what's real vs. requires extras. |
| [`docs/autotune.md`](docs/autotune.md) | The 60-second AOT probe — the hackathon differentiator. |
| [`docs/coach.md`](docs/coach.md) | Interactive `/coach/` web UI bundled in the operator. |
| [`docs/cli.md`](docs/cli.md) | Every `mindxtrain` verb with synopsis, options, exit codes. |
| [`docs/yaml_schema.md`](docs/yaml_schema.md) | Every field of the 10-section `XTrainConfig`. |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Target metrics + the 7-cell framework comparison. |
| [`docs/hackathon_submission.md`](docs/hackathon_submission.md) | Title/desc/tracks/video/deck + risks & mitigations. |
| [`docs/development.md`](docs/development.md) | Toolchain, optional-deps, lazy-import pattern, invariants. |
| [`docs/blueprints/`](docs/blueprints/) | Source design briefs (frozen specification). |

## License

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the upstream-license
notices in [`LICENSE-MIT-upstream-glm51`](LICENSE-MIT-upstream-glm51) and
[`LICENSE-NOTICE.md`](LICENSE-NOTICE.md). Version history in
[`CHANGELOG.md`](CHANGELOG.md).
