"""`mindxtrain publish` consults the MEI promotion gate.

The verb refuses to push to HF Hub when an MEI score on file for the run
fails the §8 gates, unless --force is given (in which case the manifest
records the bypass).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mindxtrain.cli.main import app
from mindxtrain.eval.mei import history as _hist
from mindxtrain.eval.mei.score import MEIScore

runner = CliRunner()


def _score(composite: float, *, q: float = 0.5, dt: float = 0.5,
           pp: float = 0.5, m: float = 0.5, e: float = 0.5) -> MEIScore:
    return MEIScore(
        composite=composite, quality=q, decode_throughput=dt,
        prefill_throughput=pp, memory=m, energy=e,
        quality_bands={"instruction": q, "reasoning": q, "knowledge_code": q},
        mab_provisional=True, notes=[],
    )


@pytest.fixture
def setup_publish_env(tmp_path: Path, monkeypatch):
    """Wire a tmp ledger + a minimal config and manifest the publish verb
    can consume. Returns (config_path, manifest_path, run_id)."""
    ledger = tmp_path / "history.jsonl"
    monkeypatch.setattr(_hist, "DEFAULT_HISTORY_PATH", ledger)

    config_yaml = tmp_path / "run.yaml"
    config_yaml.write_text(
        "meta:\n"
        "  project: test\n"
        "  run_name: smoke_test\n"
        "model:\n"
        "  name: HuggingFaceTB/SmolLM2-135M\n"
        "data:\n"
        "  source: hf\n"
        "  hf_id: tatsu-lab/alpaca\n"
    )

    run_id = "abc12345"
    manifest = {
        "schema_version": "1",
        "run_id": run_id,
        "owner": "mindx",
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "blake3": {
            "config_yaml": "0" * 64,
            "dataset": "0" * 64,
            "checkpoint": "0" * 64,
            "eval_json": "0" * 64,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Force the heavyweight subroutines into clean no-ops so the verb's
    # gate logic is what we're actually testing.
    import mindxtrain.cli.main as cli_main
    from mindxtrain.storage import hf_hub, lighthouse

    monkeypatch.setattr(hf_hub, "publish_to_hf", lambda *a, **kw: "")
    monkeypatch.setattr(lighthouse, "publish_to_lighthouse", lambda *a, **kw: "")
    # register_with_mindx is imported inside the verb body; patch on the
    # source module so the inner import resolves to our stub.
    import mindxtrain.deploy.api_client as api_client
    monkeypatch.setattr(api_client, "register_with_mindx",
                        lambda **kw: {"agent_id": "stub"})
    _ = cli_main  # keep import resolved
    return config_yaml, manifest_path, run_id


def test_publish_proceeds_when_no_mei_score_on_file(setup_publish_env):
    """Backward compat: pre-MEI workflows continue to publish."""
    cfg, manifest, _run_id = setup_publish_env
    result = runner.invoke(app, ["publish", str(cfg), "--manifest", str(manifest)])
    assert result.exit_code == 0, result.stdout


def test_publish_proceeds_when_mei_score_passes_gates(setup_publish_env):
    cfg, manifest, run_id = setup_publish_env
    sc = _score(0.62, q=0.6, dt=0.6, pp=0.55, m=0.65, e=0.60)
    _hist.append(sc, run_id=run_id, model_id="m", model_sha256="x")
    result = runner.invoke(app, ["publish", str(cfg), "--manifest", str(manifest)])
    assert result.exit_code == 0, result.stdout
    assert "MEI gate" in result.stdout


def test_publish_refused_when_mei_score_fails_gates(setup_publish_env):
    cfg, manifest, run_id = setup_publish_env
    _hist.append(_score(0.40), run_id=run_id, model_id="m", model_sha256="x")
    result = runner.invoke(app, ["publish", str(cfg), "--manifest", str(manifest)])
    assert result.exit_code == 4
    assert "MEI gate refused" in result.stdout
    assert "composite" in result.stdout


def test_publish_force_bypasses_gate_and_marks_manifest(setup_publish_env):
    cfg, manifest, run_id = setup_publish_env
    _hist.append(_score(0.40), run_id=run_id, model_id="m", model_sha256="x")
    result = runner.invoke(app, ["publish", str(cfg), "--manifest", str(manifest), "--force"])
    assert result.exit_code == 0, result.stdout
    assert "--force" in result.stdout or "force" in result.stdout
    m = json.loads(manifest.read_text())
    assert m.get("promotion_bypassed") is True
    assert m.get("promotion_bypass_reasons"), "bypass reasons should be recorded"
