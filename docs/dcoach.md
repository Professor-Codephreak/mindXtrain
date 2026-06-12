# dcoach — prove a CPU-trained model recalls its training

`dcoach` is the decentralized-aware extension of the [Coach](coach.md). It closes
mindXtrain's founding loop: **author a dataset → imprint a persona on a tiny model
(CPU) → prove the model recalls the training → let governance rule on it → feed the
verdict back into autotune.** It is also the on-ramp to the 2026 decentralized-training
landscape (see [the deep dive](decentralized-training-deep-dive-2026.md)).

Open it at **`/coach/dcoach`** (linked from the Coach header).

## The proof loop

```
persona + skills ─► script.jsonl ─► imprint-train (trl_local, CPU)
                                          │
            ┌─────────────────────────────┘
            ▼
   probe recall  ──► classroom (before vs after)  ──► boardroom (rule)  ──► feedback
   (base vs adapter)   recall ↑? persona kept?        approve / reject       tune next run
```

1. **Author** — a persona (e.g. `codephreak`) plus optional skills (software engineer,
   platform architect, bash, solidity) is composed into chat rows
   (`data/scripts.py::build_script_rows`). Each row carries the persona **system prompt**
   + a user→assistant turn.
2. **Imprint-train** — a tiny actor (default `HuggingFaceTB/SmolLM2-135M`) is LoRA-trained
   on the script on the CPU lane (`train/backend_trl_cpu.py::run_trl_local`). The autotune
   plan is frozen AOT — no JIT autotune in the loop.
3. **Probe recall** — `eval/imprint.py::probe_recall` generates the actor's answer to each
   inquiry **before** (base model) and **after** (base + adapter). The probe prepends the
   *same persona system prompt the adapter trained under*, so the comparison measures what
   the imprint actually learned rather than penalising a missing conditioning turn.
4. **Classroom** — `governance/classroom.py::evaluate_classroom` scores before vs after
   against the persona baseline (clean-room [llama-style evaluators](#clean-room-eval-tools)):
   recall up? persona maintained? `passed = persona_maintained and pairwise ≥ 0.5`.
5. **Boardroom** — the classroom graduation becomes a motion; a board (any-N, preset or
   model-backed) rules **approve / reject**. A disputed board is settled by a prime-sized
   **dojo**.
6. **Feedback** — `autotune/feedback.py` records `(run_id, params, classroom_score,
   outcome)` to an append-only ledger and `suggest_next_params` nudges the next run: a weak
   or rejected imprint trains harder (more epochs, `grad_accum=1`); a clean pass holds.
   `suggest_from_history` feeds the nudge back into `derive_training_params`.

The whole chain is `governance/proof_loop.py::run_proof_loop`, streamed phase-by-phase to
the UI via **`POST /coach/api/dcoach/run`** (SSE). It is heavy (real CPU training +
generation) — expect a few minutes per run.

## Clean-room eval tools

`eval/llama_evals.py` reimplements the *behaviour* of LlamaIndex's evaluators (MIT) from
their public contract — never copied. Each returns an `EvalScore{score∈[0,1], passing,
reasoning, method}`:

| Evaluator | What it measures | Backed by |
|-----------|------------------|-----------|
| `SemanticSimilarityEvaluator` | embedding/lexical closeness of two texts | `eval/imprint.py::_voice_similarity` |
| `CorrectnessEvaluator` | response vs reference (LLM judge, 1–5 → [0,1]) | `governance/panel.chat_once` |
| `PairwiseEvaluator` | after-utterance better than before toward the persona | judge (A/B/TIE) |
| `GuidelineEvaluator` | rubric/agenda compliance | LLM judge |

Endpoints: `POST /coach/api/classroom/evaluate`, `POST /coach/api/eval/prompt`,
`POST /coach/api/autotune/feedback`.

## Prompt tools — test cheap, promote if it wins

**`/coach/prompts`** treats prompting as the cheapest pseudo-training: craft a system
prompt + few-shot demonstrations, run them against a base model (streaming, **no
training**), evaluate the outcome with the eval tools, and only if it's advantageous
**make it permanent** by baking the prompt + demonstrations into an Ollama Modelfile
(`POST /coach/api/modelfile/create`). Non-permanent experiment → promote on results.

## How mindXtrain fits decentralized training

The dcoach page renders a read-only panel (`GET /coach/api/decentralized`) mapping each
2026 network to where mindXtrain plugs in. mindXtrain **does not mine** on any of them —
every one is CUDA-first / hardware-gated. Instead it exposes a *verifiable, payable*
training surface compatible with their verification primitives:

| mindXtrain primitive | Maps to |
|----------------------|---------|
| AOT-only autotune plan (bit-reproducible run) | Gensyn **Verde + RepOps** training verification |
| BLAKE3 verifiable receipt (`mindxtrain receipt`) | TOPLOC / checkpoint-hash verification; Templar **Gauntlet** auditing |
| x402-metered training job | Per-job crypto metering — unbuilt territory across all networks |
| AgenticPlace / ERC-8004 registration | Pluralis unextractable-model ownership / on-chain attribution |

Networks covered: **Prime Intellect** (open stack, RL post-training), **Templar · Bittensor
SN3** (Covenant-72B, the only live incentivized training market), **Nous · Psyche**
(DisTrO on Solana), **Gensyn** (verification-first, Verde — the closest match), **Pluralis ·
Node0** (model-parallel over WAN, unextractable models). Full analysis in
[decentralized-training-deep-dive-2026.md](decentralized-training-deep-dive-2026.md) and
[mindxtrain-llm-training-landscape-2026.md](mindxtrain-llm-training-landscape-2026.md).

## Why this matters

This is mindXtrain's **first-run proof**: that a model trained on the CPU lane actually
*recalls* what it was trained on — measured, ruled on, and fed back, not asserted. It is
also the bridge to the [mindX self-training loop](../README.md): the same loop that imprints
`codephreak` here consumes the `machine.dream` corpus to produce the small model mindX falls
back to.
