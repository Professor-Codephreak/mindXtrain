# Hackathon Submission Plan

AMD × lablab.ai Developer Hackathon — build window May 4–10 2026, on-site finale May 9–10 SF.

## Submission deliverables (lablab form)

| Field                   | Length / format            | Status         |
|-------------------------|----------------------------|----------------|
| Submission Title        | ≤50 chars                  | drafted below  |
| Short Description       | ≤255 chars                 | drafted below  |
| Long Description        | ≥100 words                 | drafted below  |
| Main Tracks             | multi-select               | three primary  |
| Cover Image             | 16:9                       | TODO Day 6     |
| Video Presentation URL  | YouTube unlisted, ≤300 MB  | TODO Day 6     |
| Demo Application URL    | live during 72-hr judging  | mindx.pythai.net/hackathon |
| GitHub URL              | Apache-2.0 + MIT-compat    | TODO Day 6     |
| Build-in-Public box     | three #AMDDevHackathon posts | tracked below |

### Title (≤50 chars)

```
mindxtrain — one-command Qwen3 on MI300X
```

(40 chars.)

### Short description (≤255 chars)

```
60-second AOT autotune + one-command LoRA / SFT / DPO / GRPO of Qwen3 family on AMD MI300X.
ROCm 7.2.1 + Primus-Turbo + AITER + Composable Kernel + AMD Quark FP8 + vLLM-ROCm in one CLI.
4× cheaper than H100 for the same workload, with x402 Algorand metering and BLAKE3 provenance.
```

(254 chars.)

### Long description (≥100 words)

```
mindxtrain is the first one-command Qwen3 fine-tuner natively optimized for AMD MI300X.
Before each training run, a 60-second on-device micro-benchmark probes attention kernels
(Composable Kernel vs Triton), GEMM heuristics (hipBLASLt), and collective bandwidth (RCCL),
then emits a static AOT plan that the training loop consumes — no JIT autotune in production,
in line with cypherpunk2048 reproducibility. The training layer dispatches to Axolotl, Unsloth,
torchtune, or Primus-Turbo on a per-job basis driven by the plan. Quantization is via AMD Quark
FP8 PTPC (15-30% faster than BlockScale on MI300X). Serving is vLLM-ROCm or SGLang with the
correct Hermes / Qwen3-Coder / Qwen3 reasoning parsers. Every run produces a BLAKE3-hashed
provenance manifest pinned to Lighthouse Storage, anchored to an immutable on-chain registry,
and listed on AgenticPlace with x402-Algorand metering. The cost slide writes itself: $1.99/hr
on a single MI300X versus $4/hr × 2 H100s for the same Qwen3-8B BF16 workload — 4× cheaper,
192 GB makes OOM impossible.
```

(184 words.)

## Tracks targeted

| Track                                    | Primary deliverable                                       | Status         |
|------------------------------------------|-----------------------------------------------------------|----------------|
| Fine-Tuning on AMD GPUs                  | LoRA SFT of `amd/Instella-3B` and `Qwen/Qwen3-8B` on MI300X | Day 3-4       |
| AI Agents & Agentic Workflows            | mindxtrain.operator `/v1/chat/completions` serving the trained model | Day 5      |
| Vision & Multimodal AI                   | Stretch only — `qwen3_vl_8b_sft.yaml` recipe is shipped   | Day 4 if green |
| Build-in-Public (meta)                   | 3+ technical posts tagged `#AMDDevHackathon`              | tracked below  |
| Best Use of Qwen (cross-cutting)         | Qwen3-8B secondary run + Qwen3.6 recipes                  | Day 3 if green |
| x402 Payments / Launch & Fund            | Schema + contracts shipped; full integration deferred     | post-hackathon |

## Judging-criteria mapping

The four lablab judging axes are equally weighted; here is what we lead with for each.

### 1. Application of Technology

The 60-second AOT autotune layer is the spine of this axis. Show the probe running on the MI300X, picking CK over Triton based on measured timing, then the training loop launching with the `AutotunePlan` env vars set. No competitor framework ships this. Pair with the seven-cell comparison table from [benchmarks.md](benchmarks.md).

Backup evidence: full AMD stack exercised end-to-end — ROCm 7.2.1, AOTriton, AITER, Composable Kernel, hipBLASLt, RCCL, Optimum-AMD, Quark, Primus-Turbo, vLLM-ROCm, SGLang.

### 2. Presentation

5-minute video script (see "Video script" below). 10-slide deck in Sequoia format. Live demo URL the judges hit during the 72-hour judging window. README opens with BLUF — demo URL + video URL + three-track pitch in five sentences.

### 3. Business Value

The cost slide. **$3 versus $32** for the same Qwen3-8B BF16 LoRA workload. Plus x402-Algorand metering: every training job and every inference call settles in seconds, no off-chain trust required. Real revenue model wired into the artifact pipeline, not a post-hoc afterthought.

### 4. Originality

The autotune layer is the originality story. AOT-only discipline (no JIT autotune in production) is unique among open training frameworks. The integration into mindX / AgenticPlace / BANKON / x402-Algorand makes the trained model a directly-rentable agent, not just a checkpoint on HF Hub.

## Video script (5 minutes, ≤300 MB)

```
0:00 - 0:30  framing: mindX + mindxtrain; hackathon thesis
0:30 - 1:00  cost slide: $3 vs $32 for the same Qwen3-8B BF16 workload
1:00 - 1:30  60-second autotune live: CK vs Triton timing, plan emission
1:30 - 3:00  training run: loss curve + tok/s + MFU + power streaming
3:00 - 3:30  Quark FP8 quantize + vLLM-ROCm serve coming up
3:30 - 4:00  mindxtrain.operator /v1/chat/completions in browser, persona-conditioned
4:00 - 4:30  provenance manifest + BLAKE3 reverify + on-chain anchor tx hash
4:30 - 5:00  close: integrated three-track architecture diagram
```

Record at 1080p 16:9, hosted on YouTube unlisted. Upload to a separate Google Drive as the ≤300 MB direct file for the lablab form fallback.

## Deck outline (10 slides, Sequoia format)

1. **Title** — mindxtrain on AMD MI300X. Demo URL + video URL.
2. **Problem** — Fine-tuning Qwen3-class models is multi-day effort across mismatched tools.
3. **Approach** — One CLI, one YAML, one container, end-to-end on AMD.
4. **The differentiator** — 60-second AOT autotune. No competitor framework ships this.
5. **Architecture diagram** — single `mindxtrain` package; 5-layer architecture; autotune as the spine.
6. **Live demo** — screenshot grid of `mindxtrain bench`, train, quantize, serve.
7. **Cost** — $3 vs $32. The H100 baseline can't even fit unquantized.
8. **Why MI300X specifically** — 192 GB HBM3, xGMI gotchas, AITER kernels, the autotune layer's measured CK win.
9. **What's next** — full ERC-7857 INFT + AgenticPlace listing + x402 loop for revenue model.
10. **Team + ask** — Best Overall + the three primary tracks. Demo URL again.

## Build-in-Public posts (three required)

Tag every post `#AMDDevHackathon @AIatAMD @lablabai @huggingface @Alibaba_Qwen`.

| #  | Day        | Topic                                                              | Status |
|----|------------|--------------------------------------------------------------------|--------|
| 1  | Day 1 May 4  | Why MI300X for sovereign cognition; `mindxtrain init` screenshot  | TODO   |
| 2  | Day 2 May 5  | The 60-second AOT autotune wow moment; CK-vs-Triton table         | TODO   |
| 3  | Day 5 May 8  | Demo URL live; cost diff vs H100; full stack call-out             | TODO   |

Optional fourth post: Day 6 recap with the video and the deck.

## Verification gates per day

| Day | Date  | Gate                                                                    |
|-----|-------|-------------------------------------------------------------------------|
| 1   | May 4 | `uv run pytest -q` → 112 passed; `mindxtrain init` writes valid YAML; Coach UI serves 12 recipes. |
| 2   | May 5 | `mindxtrain bench` on MI300X writes `autotune_plan.json` in <60 s with measured CK-vs-Triton timings. |
| 3   | May 6 | `mindxtrain dataset prep` + `mindxtrain train` on `amd/Instella-3B-Instruct` produce `checkpoint/`. If green: `Qwen/Qwen3.5-8B` second run kicks off. |
| 4   | May 7 | `mindxtrain eval` lm-eval JSON; `mindxtrain quantize` writes `quantized/`; `mindxtrain receipt --config run.yaml` round-trips. |
| 5   | May 8 | `curl mindx.pythai.net/hackathon/coach/api/health` → 200; `/v1/chat/completions` returns a completion through the operator → vLLM. |
| 6   | May 9 | Video rendered, deck exported, lablab form submitted, GitHub repo public. |

The `HANDOFF.md` at the repo root is the single ordered checklist that
takes Day-1 through Day-6; treat this table as the per-day green/red.

## License compatibility

Project is **Apache-2.0** ([LICENSE](../LICENSE)) with an explicit **MIT-compatibility statement** in [LICENSE-NOTICE.md](../LICENSE-NOTICE.md). The Apache 2.0 license is fully MIT-compatible per §4(d) of the Apache License — downstream consumers can relicense MIT while preserving the original NOTICE. Verify with the lablab Discord `#ineedhelp` channel if challenged at submission time.

## Wallet pre-funding (Day 5)

Demo wallets need to hold a small balance so judges can run the full paid loop end-to-end without leaving the demo URL.

| Network            | Balance | Purpose                                          |
|--------------------|---------|--------------------------------------------------|
| Algorand mainnet   | $5 USDC | x402 invoice settlement on the demo training job |
| Base mainnet       | $5 USDC | ERC-7857 INFT mint + ERC-8004 attestation        |

Stretch only if the spine is green by EOD Day 5.

## Risks & mitigations

| Risk                                                              | Mitigation                                                                                |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| MI300X droplet credit exhausts mid-judging                        | Static fallback page on `mindx.pythai.net/hackathon` with a "GPU paused, restart at $X" link. |
| `bitsandbytes` 4-bit unstable on ROCm 7.x                         | QLoRA path is opt-in; default is Quark FP8 + LoRA.                                        |
| Qwen3.6 hybrid GDN layers need `flash-linear-attention` wheels    | Build wheels in CI, ship in the Podman image; reference-impl fallback.                    |
| 60-second autotune insufficient for MoE expert imbalance          | Recipes for `qwen3_30b_a3b_lora` and `qwen3_6_35b_a3b_lora` set `budget_seconds: 90/120`. |
| `vLLM-ROCm` Triton autotune stalls first batch on cold start      | Warm-up batch in `mindxtrain serve` before exposing the endpoint; or `VLLM_USE_TRITON_FLASH_ATTN=0`. |
| lablab MIT-only requirement vs Apache-2.0 preference              | LICENSE-NOTICE.md MIT-compat statement; verify with Discord pre-submission.                |
