# Hackathon Submission Tracker

AMD × lablab.ai Developer Hackathon. Build window **May 4 – May 10, 2026**. On-site finale May 9–10 at MindsDB SF AI Collective, 3154 17th St (invitation-only).

## Tracks targeted

- [ ] **AI Agents & Agentic Workflows** — the mindxtrain operator serves a fine-tuned model behind an OpenAI-compatible API; mindX agents consume it.
- [ ] **Fine-Tuning on AMD GPUs** — mindxtrain runs a real LoRA fine-tune of `amd/Instella-3B-Instruct` on a single MI300X with the 60-second AOT autotune layer.
- [ ] **Vision & Multimodal AI** — _stretch only._

## Side challenges

- [x] **Build-in-Public** — three technical posts tagged `#AMDDevHackathon` published on `rage.pythai.net` (links below).
- [ ] **Best Use of Qwen** — secondary config template for Qwen3-8B if Days 3–4 stay green.
- [ ] **x402 Payments / Launch & Fund** — _deferred to post-hackathon TODO._
- [ ] **HuggingFace Spaces "Most Likes"** — _stretch._

## Judging axes (equally weighted)

1. **Application of Technology** — meaningful MI300X / ROCm / Dev Cloud integration. The 60-second autotune probe is the spine of this axis.
2. **Presentation** — 5-minute video (<300 MB), 10-slide deck, live demo URL the judges can hit.
3. **Business Value** — 4× cheaper training than H100, single-GPU 70B-class LoRA in BF16 unquantized.
4. **Originality** — autotune layer is unique among Axolotl, LLaMA-Factory, Unsloth, torchtune, Primus.

## Submission deliverables checklist

- [ ] Working live demo URL (judges will hit it in real time).
- [ ] 5-minute video presentation link (≤300 MB).
- [ ] Public GitHub repo (Apache-2.0 with MIT-compat notice for lablab).
- [ ] Hugging Face submission form filled:
  - Title (≤50 chars)
  - Short description (≤255 chars)
  - Long description (≥100 words)
  - 16:9 cover image
  - Main Tracks (multiple)
  - Technologies, Video URL, Demo URL.

## Daily verification gates

- **Day 1 (May 4):** `uv run pytest -q` → 4 passed; `mindxtrain bench --dry-run` writes a plan JSON.
- **Day 2 (May 5):** `mindxtrain bench` on MI300X writes `autotune_plan.json` in <60 s with measured CK-vs-Triton timings.
- **Day 3 (May 6):** Axolotl LoRA run on `amd/Instella-3B-Instruct` produces a checkpoint directory.
- **Day 4 (May 7):** `lm-eval-harness` MMLU subset JSON; Quark FP8 quantize succeeds; `out/runs/<run_id>/manifest.json` validates.
- **Day 5 (May 8):** `curl mindx.pythai.net/hackathon/v1/chat/completions` returns a chat completion routed through the mindxtrain operator → vLLM.
- **Day 6 (May 9):** video rendered, deck exported, HF Spaces submission form submitted, GitHub repo public.

## Build-in-Public links (published)

Cite these in the lablab submission form's "Additional Information" / Build-in-Public box. All four published 2026-05-09 from the local `wordpress_agent` toolchain via the `JWT Authentication for WP REST API` plugin against rage.pythai.net.

| # | Post | URL | Post ID |
|---|------|-----|---------|
| 0 | Anchor — mindXtrain (project overview) | https://rage.pythai.net/mindxtrain/ | 650 |
| 1 | Day 1 — Why MI300X for Sovereign Cognition | https://rage.pythai.net/mindxtrain-day-1-mi300x/ | 651 |
| 2 | Day 2 — The 60-Second AOT Autotune Probe | https://rage.pythai.net/mindxtrain-day-2-autotune/ | 652 |
| 3 | Day 5 — Demo Live, Qwen3-8B for <$3 | https://rage.pythai.net/mindxtrain-day-5-demo/ | 653 |
