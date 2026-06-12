# Governance — classroom / boardroom / dojo

A clean-room reimplementation (from the behaviour of
[`github.com/openmindx/openmind`](https://github.com/openmindx/openmind) — Boardroom
multi-model consensus + Dojo head-to-head evaluation) of the decision layer that governs
training. Lives in `mindxtrain/governance/`; pure stdlib + pydantic, base-install importable.

## The model

- **Classroom** (`governance/classroom.py`) — where an actor (model) trains. An actor
  **graduates** when its persona imprint took: `graduate(imprint_report, min_delta=…)`
  returns a `Graduation` (the motion the boardroom convenes on). Ties the governance layer
  to `mindxtrain.eval.imprint`.
- **Boardroom** (`governance/boardroom.py`) — a panel of **any number** of role-based
  members (advocate, critic, analyst, devil's advocate, expert, generalist). `convene(motion,
  ballot)` tallies votes → `approved` / `rejected` / `disputed`. The boardroom **governs the
  classroom**: it decides about training given a graduation. Preset boards: `classic_triad`,
  `devils_court`, `full_board`, `peer_review`.
- **Dojo** (`governance/dojo.py`) — the boardroom's **dispute-settlement** extension. When a
  boardroom is `disputed` (a tie or no quorum), a dojo settles it. **A dojo panel is always an
  odd prime (≥ 3)** — an odd number of decisive judges cannot tie, so the dispute always
  resolves. `Dojo.sized(n)` rounds a requested size to the nearest valid prime; `settle(motion,
  ballot)` returns a final `DojoVerdict`. 2 is prime but even (can tie), so it is excluded.

## Flow

```
classroom: train actor → measure imprint → graduate(report) ─► Graduation.motion
                                                                      │
boardroom: convene(motion, ballot) ─► approved / rejected / disputed │
                                                  │ disputed         │
dojo (prime panel): settle_dispute(decision, dojo, ballot) ─► DojoVerdict (no tie)
```

Members vote via an explicit `{id: vote}` map or a callable `(member, motion) -> vote`, so
the whole layer is testable with no LLM and can later be backed by real models
(boardroom-of-LLMs, dojo head-to-head) — clean-room, never vendoring openmind's TypeScript.

## Why prime

A boardroom can be any size because deliberation tolerates abstention and "no decision"
(escalate). A dojo must *settle* — so its panel is an odd prime: `approvals + rejections`
is odd, the majority is strict, and the verdict is final. See `governance/primes.py`
(`is_prime`, `next_prime`, `nearest_prime`) and `dojo.prime_dojo_size`.

## Model-backed deliberation

`governance/panel.py` backs members + judges with **real models** (any OpenAI-compatible
backend — the same ollama / vLLM the operator serves). `deliberate(member, motion)` prompts a
member from its role stance and parses a `VERDICT: APPROVE|REJECT|ABSTAIN`; `model_ballot()` /
`model_judge_ballot()` return ballots you pass straight to `Boardroom.convene` /
`Dojo.settle`. Lazy + best-effort: a model that errors or returns no parseable verdict abstains
(boardroom) or is recorded as reject (dojo). The base URL resolves from
`MINDXTRAIN_OPENAI_BASE_URL` / `MINDXTRAIN_VLLM_BASE_URL` / `MINDXTRAIN_OLLAMA_BASE_URL`.

## Coach surface

The **Boardroom** card (after the receipt card) convenes a board on a promotion motion and,
if disputed, settles it in a prime dojo:

- `GET /coach/api/boardroom/presets` — named boards → roles.
- `POST /coach/api/boardroom/convene` — `{motion, members:[{id,role,model}], quorum, votes?,
  use_models?, base_url?}`. Tally supplied `votes`, or `use_models: true` to have each member's
  model deliberate. Model calls run in a worker thread (`asyncio.to_thread`) so the operator
  event loop never blocks on inference. Returns the decision + per-member deliberations.
- `POST /coach/api/dojo/settle` — `{motion, size, model?, votes?, use_models?, base_url?}`.
  Sizes the panel to the nearest odd prime and settles.

## Tests

- `tests/test_governance.py` — primes, any-N boardroom (majority / tie / no-quorum), prime-only
  dojo (rejects non-prime panels, settles without tie), end-to-end classroom → disputed → dojo.
- `tests/test_governance_panel.py` — verdict parsing, role stances, model-backed ballots driving
  a boardroom + dojo over a mocked chat backend, graceful backend-error handling.
- `tests/test_coach_governance_api.py` — convene (votes + model mode), dojo settle (prime sizing),
  422 paths, and the Coach card/JS presence.
