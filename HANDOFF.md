# HANDOFF — what you need to do next

This is the ordered checklist for taking the mindxtrain repo from "code is
done" to "demo is live." Each step is concrete; check it off when finished.

The repo state at handoff:

- Single canonical package at `mindxtrain/` (12 subpackages, ~100 modules).
- All stub `NotImplementedError` paths replaced with real Python (lazy imports
  for heavyweight deps).
- 112/112 tests pass on a CPU-only laptop (`uv sync` + `uv run pytest -q`).
- Optional dep groups in `pyproject.toml`: `ml`, `eval`, `data`, `serve`,
  `chain`, `obs`. Install only what you need.
- 12 YAML training recipes wired through the CLI.
- Coach UI (`/coach/`) serves all 12 recipes without GPU.

---

## 1. Local setup (no GPU; 10 minutes)

```bash
cd /home/hacker/Desktop/mindXtrain
cp .env.example .env       # then edit .env to fill in HF_TOKEN, etc.
uv sync                    # base install
uv run pytest -q           # → 112 passed
uv run mindxtrain --help   # all 9 verbs listed
```

**What goes in `.env`** (rest of the file is sane defaults):

| Var | Where to get it |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens (write scope) |
| `HF_HUB_USERNAME` | your HF handle |
| `LIGHTHOUSE_API_KEY` | https://files.lighthouse.storage/dashboard/apikey |
| `MINDXTRAIN_OPENAI_API_KEY` | optional; only if you want to use openai_compat backend |

> **Defer for post-hackathon:** `MINDXTRAIN_REGISTRY_ADDR` (ERC-8004 contract),
> `MINDXTRAIN_FACILITATOR_URL` (x402 facilitator). The publish path skips
> these gracefully if unset.

## 2. Provision the MI300X droplet (sign-up + 30 min)

> **Fast path (Coach UI):** if you've populated `GITHUB_TOKEN`,
> `AMD_DEV_CLOUD_TOKEN`, and `AMD_DEV_CLOUD_SSH_KEY_ID` in `.env`, you can skip
> the manual SSH dance entirely:
>
> 1. `uv run uvicorn mindxtrain.operator.app:app --port 8080`
> 2. Open <http://localhost:8080/coach/>, scroll to step 6 ("Deploy").
> 3. Click ① **Push to GitHub** → ② **Provision MI300X droplet**. The droplet
>    boots, cloud-init clones the repo from the SHA you just pushed, pulls the
>    container, and runs `mindxtrain bench` automatically. All output streams
>    live in the browser via SSE.
>
> Equivalent CLI: `mindxtrain github push && mindxtrain droplet provision`.
>
> The manual sequence below is preserved for scripted / CI use and as a
> fallback when the Coach UI isn't available.

```bash
# Sign up at https://devcloud.amd.com — request a single MI300X.
# Wait for the droplet (typically same-day).
# SSH in:
ssh ubuntu@<droplet-ip>

# Install podman if missing:
sudo apt-get update && sudo apt-get install -y podman podman-compose

# Pull the canonical training container:
podman pull docker.io/rocm/primus:v26.2

# Snapshot the digest into the repo so others can reproduce:
podman inspect --format '{{index .RepoDigests 0}}' rocm/primus:v26.2 \
  | tee -a ops/containerfiles/digest.lock

# Verify the GPU is visible:
podman run --rm --device=/dev/kfd --device=/dev/dri rocm/primus:v26.2 \
  rocminfo | head -50
# → should show gfx942, 192 GB HBM3
```

> **Cost watch:** $1.99/hr × planned hours. Budget ~$30 for the full demo
> pipeline (~15 GPU-hours). Leave the droplet **stopped** when not actively
> training.

## 3. Install heavyweight deps inside the container

```bash
# On the MI300X:
git clone <your-repo-url> /workspace/mindxtrain
cd /workspace/mindxtrain
podman run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v /workspace/mindxtrain:/workspace/mindxtrain \
  -w /workspace/mindxtrain \
  rocm/primus:v26.2 bash

# Inside the container:
pip install -e ".[ml,eval,data,obs]"
# (skip `serve` and `chain` until you need them — they pull large wheels)
```

## 4. Run the autotune probe (real, ~60 s)

```bash
mindxtrain bench --gpu 0 --out plan.json
cat plan.json | jq '.attention_backend, .gemm_heuristic, .rccl_config'
# → "ck", "hipblaslt_default", "1gpu_noop"
```

Snapshot `plan.json` into the repo so the run is reproducible:

```bash
cp plan.json ops/k8s/plan-mi300x.json
git add ops/k8s/plan-mi300x.json
git commit -m "snapshot autotune plan from mi300x"
```

## 5. Train + eval + quantize (~ 2 hours total for the demo recipe)

```bash
# Pick a recipe: instella_3b_lora is the AMD-on-AMD demo path (~30 min).
# Or qwen3_8b_sft_lora for the Qwen side prize (~75 min).
mindxtrain init --template instella_3b_lora --out run.yaml

# Optional: edit run.yaml for your project name, dataset, output path.
$EDITOR run.yaml

# Dataset prep (pulls + dedupes + tokenizes + packs):
mindxtrain dataset prep run.yaml --out ./out/dataset

# Training:
mindxtrain train run.yaml --plan plan.json
# → ./out/runs/<run_name>/checkpoint/

# Evaluation (MMLU subset):
mindxtrain eval run.yaml
# → ./out/runs/<run_name>/eval/lm_eval.json

# Quantize to FP8:
mindxtrain quantize run.yaml
# → ./out/runs/<run_name>/quantized/
```

If `mindxtrain train` fails with `accelerate not found`: you forgot
`pip install -e ".[ml]"` inside the container (step 3).

## 6. Build the manifest + verify

```bash
# Generate the provenance manifest by hashing every artifact:
uv run python -c "
from pathlib import Path
from mindxtrain.config.loader import load_config
from mindxtrain.provenance.manifest import emit_receipt, ProvenanceHashes
cfg = load_config('run.yaml')
run = Path('./out/runs') / cfg.meta.run_name
m = emit_receipt(
    cfg,
    cfg.meta.run_name,
    config_yaml_path=Path('run.yaml'),
    dataset_manifest_path=run / 'dataset_manifest.json',
    checkpoint_dir=run / 'checkpoint',
    eval_json_path=run / 'eval/lm_eval.json',
)
out = run / 'manifest.json'
out.write_text(m.model_dump_json(indent=2))
print(out)
"

# Verify it round-trips:
mindxtrain receipt ./out/runs/<run_name>/manifest.json --config run.yaml
# → all four BLAKE3 fields = true
```

## 7. Publish (HF Hub + Lighthouse + mindX register)

```bash
# Push to HF (uses HF_TOKEN; private=False for the demo):
mindxtrain publish run.yaml --manifest ./out/runs/<run_name>/manifest.json
# → updates manifest.json in-place with hf_repo_id + lighthouse_cid
```

If `LIGHTHOUSE_API_KEY` is unset, the pin step skips gracefully and the
manifest gets a `cid://stub-…` placeholder.

## 8. Deploy contracts (optional, post-hackathon)

The demo can ship without on-chain anchors. Do these once, post-hackathon:

```bash
cd contracts
forge install
forge test                       # local Foundry tests pass
forge script script/Deploy.s.sol \
  --rpc-url $MINDXTRAIN_BASE_RPC_URL \
  --private-key $DEPLOYER_KEY \
  --broadcast
# → records contract address; paste into .env as MINDXTRAIN_REGISTRY_ADDR
```

Once `MINDXTRAIN_REGISTRY_ADDR` is set, `mindxtrain.provenance.erc8004.broadcast_attestation`
can anchor the manifest BLAKE3 on-chain.

## 9. Serve the model + wire the demo URL

```bash
# Inside the rocm/vllm-dev container:
podman-compose -f ops/compose/compose_dev.yaml up -d
# → vLLM-ROCm at :8000, mindxtrain operator FastAPI at :8080

# Verify locally:
curl http://localhost:8080/coach/api/health
# → {"coach_version":"0.1.0", "recipes_available":12, ...}

# Reverse-proxy `mindx.pythai.net/hackathon` → MI300X:8080
# (set up via Caddy/Cloudflare; specifics depend on your DNS provider)
```

Once the proxy is live, `curl mindx.pythai.net/hackathon/coach/api/health`
returns 200 from the public internet.

## 10. Hackathon submission

```bash
# Push code:
git push origin main

# Record demo video (5 min, ≤ 300 MB):
#   1. mindxtrain init  → show CLI verbs
#   2. mindxtrain bench → 60-second autotune (the differentiator)
#   3. mindxtrain train → timelapse of training
#   4. mindxtrain quantize → FP8 weights
#   5. curl /v1/chat/completions → live inference
#   6. mindxtrain receipt → BLAKE3 reverify
#   7. Open /coach/ → click through the UI

# Submit at lablab.ai:
#   - Title: ≤ 50 chars
#   - Description: ≤ 255 chars
#   - Tracks: AI Agents & Agentic Workflows, Fine-Tuning on AMD GPUs
#   - GitHub: link to your public fork
#   - Video: HF Spaces or YouTube unlisted
#   - Demo URL: mindx.pythai.net/hackathon (or static fallback if credit ran out)

# Travel: SF on May 9. On-site finale May 10.
```

## 11. Quality gates (run before every push)

```bash
uv run ruff check .
uv run mypy mindxtrain/config mindxtrain/provenance
uv run pytest -q       # → 112 passed
```

All three must pass before pushing to `main`. CI runs the same gates on the
`main` branch.

---

## What's still TODO (post-hackathon)

These paths are wired but require runtime/contracts/services to actually
flow end-to-end:

- **x402 metering** (`mindxtrain.provenance.x402`) — wired to httpx, needs
  a deployed facilitator URL.
- **ERC-8004 broadcast** (`mindxtrain.provenance.erc8004.broadcast_attestation`)
  — needs deployed attestation registry + signer key.
- **BANKON ENS** allocation (`mindxtrain.provenance.algorand.allocate_ens_subname`)
  — needs the BANKON allocation service deployed.
- **AgenticPlace listing** (`mindxtrain.deploy.api_client.list_on_agenticplace`)
  — needs `agenticplace.pythai.net` live.
- **mindX agent register** (`mindxtrain.deploy.api_client.register_with_mindx`)
  — needs `mindx.pythai.net/v1/agents` live.

The framework itself ships as production-ready Apache-2.0; the integrations
above are paid/external services you stand up at your own pace.

---

## Quick reference

| What | Where |
|---|---|
| All CLI verbs | `mindxtrain --help` |
| All recipes | `mindxtrain init --list` |
| Coach UI | http://localhost:8080/coach/ |
| Per-module status | `docs/actualization_status.md` |
| Architecture | `docs/architecture.md` |
| Autotune detail | `docs/autotune.md` |
| Coach detail | `docs/coach.md` |
| CLI reference | `docs/cli.md` |
| YAML schema | `docs/yaml_schema.md` |
| Submission notes | `docs/hackathon_submission.md` |
| Frozen blueprints | `docs/blueprints/{mindXtrain,mindXtrain2}.md` |

Good luck at the finale.
