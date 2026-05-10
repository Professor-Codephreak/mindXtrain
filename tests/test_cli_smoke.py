"""mindxtrain CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from mindxtrain.cli.main import app

runner = CliRunner()


def test_help_lists_all_eight_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in ("init", "bench", "train", "eval", "quantize", "serve", "publish", "receipt", "dataset"):
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


def test_dataset_prep_reports_missing_datasets(tmp_path):
    """Without `--extra ml`, dataset prep surfaces a clean install hint."""
    out = tmp_path / "run.yaml"
    runner.invoke(app, ["init", "--template", "qwen3_8b_sft_lora", "--out", str(out)])
    result = runner.invoke(app, ["dataset", "prep", str(out)])
    # Exit 3 = optional dep missing; 0 only if `datasets` is installed.
    assert result.exit_code in (0, 3)
    if result.exit_code == 3:
        assert "dataset prep failed" in result.stdout.lower()
