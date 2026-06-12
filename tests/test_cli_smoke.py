"""mindxtrain CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from mindxtrain.cli.main import app

runner = CliRunner()


def test_help_lists_all_eight_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("init", "bench", "train", "eval", "quantize", "serve", "publish",
                 "receipt", "dataset", "imprint"):
        assert verb in result.stdout


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "mindxtrain" in result.stdout


def test_init_writes_yaml(tmp_path):
    out = tmp_path / "run.yaml"
    result = runner.invoke(app, ["init", "--template", "qwen3_8b_sft_lora", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert out.stat().st_size > 0
    assert "Qwen/Qwen3-8B" in out.read_text()


def test_init_lists_recipes():
    result = runner.invoke(app, ["init", "--list"])
    assert result.exit_code == 0
    for name in ("qwen3_8b_sft_lora", "qwen3_32b_grpo", "instella_3b_lora"):
        assert name in result.stdout


def test_bench_dry_run_emits_plan(tmp_path):
    out = tmp_path / "plan.json"
    result = runner.invoke(app, ["bench", "--dry-run", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    plan = json.loads(out.read_text())
    assert plan["schema_version"] == "1"
    assert plan["attention_backend"] in ("ck", "triton")
    assert plan["gpu_arch"] == "gfx942"


def test_train_reports_missing_accelerate(tmp_path):
    """Without `--extra ml`, the trainer dispatch surfaces a clean install hint."""
    out = tmp_path / "run.yaml"
    runner.invoke(app, ["init", "--template", "qwen3_8b_sft_lora", "--out", str(out)])
    result = runner.invoke(app, ["train", str(out)])
    # Exit 3 = optional dep missing; 1 = bad config; 0 only if accelerate is on PATH.
    assert result.exit_code in (0, 1, 3)
    if result.exit_code == 3:
        assert "training failed" in result.stdout.lower() or "accelerate" in result.stdout.lower()


def _seed_run_for_receipt(tmp_path, monkeypatch):
    """Create out/runs/<run_name>/ with a full run-emitted manifest under tmp cwd."""
    from mindxtrain.autotune.plan import AutotunePlan
    from mindxtrain.config.loader import load_config, render_recipe
    from mindxtrain.provenance import manifest as _m

    monkeypatch.setattr(_m, "_fetch_time_attestation", lambda: _m.TimeAttestation())
    monkeypatch.chdir(tmp_path)

    recipe = tmp_path / "run.yaml"
    recipe.write_text(render_recipe("qwen3_8b_sft_lora"))
    cfg = load_config(recipe)

    run_dir = tmp_path / "out" / "runs" / cfg.meta.run_name
    ckpt = run_dir / "checkpoint"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"\x00" * 64)

    m = _m.emit_receipt_for_run(cfg, cfg.meta.run_name, run_dir=run_dir, plan=AutotunePlan())
    manifest_path = _m.write_run_manifest(m, run_dir)
    return recipe, run_dir, manifest_path


def test_receipt_verifies_run_emitted_manifest(tmp_path, monkeypatch):
    recipe, _run_dir, manifest_path = _seed_run_for_receipt(tmp_path, monkeypatch)
    result = runner.invoke(app, ["receipt", str(manifest_path), "--config", str(recipe)])
    assert result.exit_code == 0, result.stdout
    assert "autotune_plan" in result.stdout


def test_receipt_detects_checkpoint_tamper(tmp_path, monkeypatch):
    recipe, run_dir, manifest_path = _seed_run_for_receipt(tmp_path, monkeypatch)
    # Tamper a checkpoint file after the manifest is sealed.
    (run_dir / "checkpoint" / "adapter_model.safetensors").write_bytes(b"\xff" * 64)
    result = runner.invoke(app, ["receipt", str(manifest_path), "--config", str(recipe)])
    assert result.exit_code == 2, result.stdout


def test_serve_to_sglang_prints_command(tmp_path, monkeypatch):
    from mindxtrain.config.loader import load_config, render_recipe

    monkeypatch.chdir(tmp_path)
    recipe = tmp_path / "run.yaml"
    recipe.write_text(render_recipe("qwen3_8b_sft_lora"))
    cfg = load_config(recipe)
    quant = tmp_path / "out" / "runs" / cfg.meta.run_name / "quantized"
    quant.mkdir(parents=True)

    result = runner.invoke(app, ["serve", str(recipe), "--to", "sglang"])
    assert result.exit_code == 0, result.stdout
    assert "sglang cmd" in result.stdout
    assert "sglang.launch_server" in result.stdout


def test_dataset_prep_reports_missing_datasets(tmp_path):
    """Without `--extra ml`, dataset prep surfaces a clean install hint."""
    out = tmp_path / "run.yaml"
    runner.invoke(app, ["init", "--template", "qwen3_8b_sft_lora", "--out", str(out)])
    result = runner.invoke(app, ["dataset", "prep", str(out)])
    # Exit 3 = optional dep missing; 0 only if `datasets` is installed.
    assert result.exit_code in (0, 3)
    if result.exit_code == 3:
        assert "dataset prep failed" in result.stdout.lower()
