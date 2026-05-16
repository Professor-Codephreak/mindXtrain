"""Push-to-Ollama: merge LoRA → Modelfile → ollama create.

Exercises the Modelfile writer and the ollama_create subprocess wrapper
in isolation. The peft merge path is integration-tested elsewhere (it
requires `--extra ml` + a real base checkpoint to be meaningful).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mindxtrain.deploy.ollama_push import (
    ollama_create,
    push_to_ollama,
    write_modelfile,
)

# ---- write_modelfile -----------------------------------------------------


def test_modelfile_minimal(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    out = write_modelfile(merged, tmp_path / "Modelfile")
    text = out.read_text(encoding="utf-8")
    assert text.startswith(f"FROM {merged}")
    # No optional stanzas → only the FROM line.
    assert "SYSTEM" not in text
    assert "PARAMETER" not in text


def test_modelfile_with_system_and_params(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    out = write_modelfile(
        merged, tmp_path / "Modelfile",
        system_prompt="You are mindX.",
        parameters={"temperature": 0.7, "num_ctx": 4096, "stop": "</s>"},
    )
    text = out.read_text(encoding="utf-8")
    assert f"FROM {merged}" in text
    assert 'SYSTEM """You are mindX."""' in text
    assert "PARAMETER temperature 0.7" in text
    assert "PARAMETER num_ctx 4096" in text
    # Strings get quoted; numbers don't.
    assert 'PARAMETER stop "</s>"' in text


def test_modelfile_creates_parent_dir(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    target = tmp_path / "nested" / "deeper" / "Modelfile"
    out = write_modelfile(merged, target)
    assert out.exists()
    assert out.parent == tmp_path / "nested" / "deeper"


# ---- ollama_create subprocess wrapper -----------------------------------


def test_ollama_create_invokes_cli(tmp_path: Path, monkeypatch) -> None:
    """The wrapper builds `ollama create <tag> -f <modelfile>` and surfaces stdout."""
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["check"] = kw.get("check")
        captured["timeout"] = kw.get("timeout")
        return subprocess.CompletedProcess(
            cmd, returncode=0, stdout="created: mindx-fallback\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM /tmp/merged\n")

    out = ollama_create(
        "mindx-fallback", modelfile, ollama_bin="/usr/local/bin/ollama",
    )
    assert "created: mindx-fallback" in out
    assert captured["cmd"] == [
        "/usr/local/bin/ollama", "create", "mindx-fallback", "-f", str(modelfile),
    ]
    assert captured["check"] is True
    # Long-running merges shouldn't be hit by a stingy default.
    assert (captured["timeout"] or 0) >= 60


def test_ollama_create_missing_binary_raises(tmp_path: Path, monkeypatch) -> None:
    """No ollama on PATH → FileNotFoundError with an install hint."""
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: None,
    )
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM /tmp/merged\n")

    with pytest.raises(FileNotFoundError, match="ollama"):
        ollama_create("anytag", modelfile)


def test_ollama_create_propagates_subprocess_error(tmp_path: Path, monkeypatch) -> None:
    """Non-zero exit from ollama bubbles as CalledProcessError (caller surfaces stderr)."""

    def _fake_run(cmd, **_kw):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output="", stderr="model arch not supported",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM /tmp/merged\n")

    with pytest.raises(subprocess.CalledProcessError):
        ollama_create("anytag", modelfile, ollama_bin="/usr/local/bin/ollama")


def test_ollama_create_streams_through_sink(tmp_path: Path, monkeypatch) -> None:
    """stdout and stderr both fan into the sink callback for live UI logging."""
    lines: list[str] = []

    def _fake_run(cmd, **_kw):
        return subprocess.CompletedProcess(
            cmd, returncode=0,
            stdout="creating layer 1 of 4\n",
            stderr="pulling weights\n",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM /tmp/merged\n")

    ollama_create(
        "tag", modelfile, ollama_bin="/u/bin/ollama", sink=lines.append,
    )
    joined = "\n".join(lines)
    assert "creating layer 1 of 4" in joined
    assert "pulling weights" in joined
    # The first line should be the command echo.
    assert lines[0].startswith("[push-ollama] $")


# ---- push_to_ollama end-to-end orchestration ----------------------------


def test_push_to_ollama_writes_modelfile_and_calls_create(tmp_path, monkeypatch) -> None:
    """End-to-end happy path with merge_lora_adapter monkey-patched out.

    Verifies push_to_ollama wires merge → Modelfile → ollama create in
    order and returns the structured result.
    """
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00")
    work = tmp_path / "ollama_push"

    # Skip the real peft merge — that's covered by the ml-extras integration test.
    def _fake_merge(base_model, adapter_dir, out_dir, sink=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_bytes(b"\x00")
        if sink:
            sink("[fake-merge] done")
        return out_dir

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter", _fake_merge,
    )

    def _fake_run(cmd, **_kw):
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which",
        lambda _b: "/usr/local/bin/ollama",
    )

    result = push_to_ollama(
        base_model="Qwen/Qwen3-0.6B",
        adapter_dir=adapter,
        tag="mindx-fallback-test",
        work_dir=work,
    )
    assert result.tag == "mindx-fallback-test"
    assert result.merged_dir == work / "merged"
    assert result.modelfile == work / "Modelfile"
    assert result.modelfile.exists()
    text = result.modelfile.read_text(encoding="utf-8")
    assert f"FROM {work / 'merged'}" in text


def test_push_to_ollama_default_workdir_under_adapter(tmp_path, monkeypatch) -> None:
    """When work_dir is None, artefacts live next to the adapter (parent/ollama_push)."""
    adapter = tmp_path / "run" / "checkpoint"
    adapter.mkdir(parents=True)

    def _fake_merge(_b, _a, out_dir, sink=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _fake_run(cmd, **_kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter", _fake_merge,
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which",
        lambda _b: "/u/bin/ollama",
    )

    result = push_to_ollama(
        base_model="base", adapter_dir=adapter, tag="default-workdir",
    )
    # Default: sibling to adapter_dir.
    assert result.merged_dir == adapter.parent / "ollama_push" / "merged"
