"""dcoach proof loop — prove a CPU-trained model recalls its training.

Chains the whole thing end-to-end: compose a persona/skills script → derive training
params (nudged by past feedback) → imprint-train a tiny actor on CPU → probe before
(base) vs after (adapter) recall → the **classroom** tests recall/persona → the
**boardroom** decides success/failure → record **autotune feedback** that improves the
next run. This is mindXtrain's first-run proof + the autotune feedback loop.

Heavy (real training + generation) — runs on the `trl_local` CPU lane. `on_event(phase,
msg)` reports progress so a UI can stream it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from mindxtrain.governance.classroom import ClassroomReport

_DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
_IMPRINT_RECIPE = "mindx_persona_imprint_local"


class ProofResult(BaseModel):
    """The outcome of one proof-loop run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    dataset_path: str
    rows: int
    train_params: dict[str, int]
    classroom: ClassroomReport
    boardroom_outcome: str
    boardroom_rationale: str
    passed: bool
    next_params: dict[str, int]


def _build_cfg(base_model: str, script_path: Path, params: dict[str, int], run_id: str) -> Any:
    """Render the imprint recipe, override base/data/params, and load an XTrainConfig."""
    from mindxtrain.config.loader import load_config, render_recipe

    raw = yaml.safe_load(render_recipe(_IMPRINT_RECIPE))
    raw["meta"]["run_name"] = run_id
    raw["model"]["name"] = base_model
    raw["data"]["path"] = str(script_path)
    raw["data"]["max_samples"] = 64
    raw["data"]["seq_len"] = 128
    raw["train"]["schedule"]["epochs"] = int(params.get("epochs", 12))
    raw["train"]["batch"]["grad_accum"] = int(params.get("grad_accum", 1))
    raw["train"]["batch"]["per_device"] = int(params.get("per_device", 1))
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml.safe_dump(raw))
        tmp = fh.name
    return load_config(tmp)


def run_proof_loop(
    *,
    run_id: str,
    persona: str = "codephreak",
    skills: list[str] | None = None,
    exchanges: list[Any] | None = None,
    base_model: str = _DEFAULT_BASE_MODEL,
    out_dir: str | Path,
    inquiries: list[str] | None = None,
    board_preset: str = "classic_triad",
    board_model: str | None = None,
    force_cpu: bool = True,
    max_new_tokens: int = 32,
    on_event: Callable[[str, str], None] | None = None,
    feedback_path: Path | None = None,
) -> ProofResult:
    """Run the full proof loop and return a structured `ProofResult`."""
    emit = on_event or (lambda _phase, _msg: None)
    out = Path(out_dir)

    # 1) Compose persona + skills → script.
    from mindxtrain.data import personas as _pz
    from mindxtrain.data.scripts import (
        build_script_rows,
        derive_training_params,
        write_script_jsonl,
    )

    base_persona, skill_exchanges = _pz.compose(persona, skills or [])
    all_exchanges = list(exchanges or []) + skill_exchanges
    rows_list = build_script_rows(base_persona, all_exchanges, seed_voice=True)
    script_path = write_script_jsonl(rows_list, out / "script.jsonl")
    rows = len(rows_list)
    emit("dataset", f"authored {rows} rows for persona '{base_persona.name}'")

    # 2) Derive params, nudged by past feedback.
    from mindxtrain.autotune import feedback as _fb

    params = _fb.suggest_from_history(derive_training_params(rows), path=feedback_path)
    emit("params", f"epochs={params['epochs']} grad_accum={params['grad_accum']}")

    # 3) Imprint-train the tiny actor.
    from mindxtrain.autotune.benchmark import run_autotune
    from mindxtrain.train.backend_trl_cpu import run_trl_local

    cfg = _build_cfg(base_model, script_path, params, run_id)
    run_dir = out / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    emit("train", "imprinting the persona…")
    run_trl_local(cfg, run_autotune(dry_run=True), run_dir, force_cpu=force_cpu)
    adapter = run_dir / "checkpoint"

    # 4) Probe before (base) vs after (adapter) recall.
    from mindxtrain.eval.imprint import default_inquiries, probe_recall

    inq = inquiries or [e.user for e in all_exchanges][:4] or default_inquiries(base_persona.name)
    baseline = [e.assistant for e in all_exchanges] or list(base_persona.voice_examples)
    emit("probe", "recall before training…")
    before = probe_recall(base_model, inq, force_cpu=force_cpu, max_new_tokens=max_new_tokens)
    emit("probe", "recall after training…")
    after = probe_recall(base_model, inq, adapter_dir=adapter, force_cpu=force_cpu, max_new_tokens=max_new_tokens)

    # 5) Classroom test.
    from mindxtrain.governance.classroom import evaluate_classroom, graduate

    classroom = evaluate_classroom(inq, before, after, baseline)
    emit("classroom", f"passed={classroom.passed} recall {classroom.before_recall}→{classroom.recall}")

    # 6) Boardroom decision.
    from mindxtrain.eval.imprint import score_imprint
    from mindxtrain.governance import Boardroom
    from mindxtrain.governance.boardroom import board_from_preset

    grad = graduate(score_imprint(inq, before, after, baseline), run_id=run_id)
    board = Boardroom(members=board_from_preset(board_preset, model=board_model or ""))
    if board_model:
        from mindxtrain.governance import panel as _panel

        ballot: Any = _panel.model_ballot(default_model=board_model)
    else:
        vote = "approve" if classroom.passed else "reject"
        ballot = {m.id: vote for m in board.members}
    decision = board.convene(grad.motion, ballot)
    emit("boardroom", f"{decision.outcome}: {decision.rationale}")

    # 7) Record feedback + suggest the next run's params.
    outcome = decision.outcome
    _fb.record(
        run_id=run_id, params=params, classroom_score=classroom.imprint_delta,
        passed=classroom.passed, boardroom_outcome=outcome, path=feedback_path,  # type: ignore[arg-type]
    )
    next_params = _fb.suggest_next_params(
        params, passed=classroom.passed, classroom_score=classroom.imprint_delta,
    )
    emit("feedback", f"next: epochs={next_params['epochs']} grad_accum={next_params['grad_accum']}")

    return ProofResult(
        run_id=run_id, dataset_path=str(script_path), rows=rows, train_params=params,
        classroom=classroom, boardroom_outcome=outcome, boardroom_rationale=decision.rationale,
        passed=classroom.passed and decision.outcome == "approved", next_params=next_params,
    )


__all__ = ["ProofResult", "run_proof_loop"]
