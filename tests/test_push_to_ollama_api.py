"""Coach API endpoint: POST /api/runs/{run_id}/push-to-ollama.

Verifies the operator wires push-to-ollama into the run registry —
404 for unknown runs, 409 when the checkpoint is missing, 503 when
ollama isn't installed, 200 + structured response on the happy path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app
from mindxtrain.operator.coach.api import _REGISTRY

client = TestClient(app)


def _create_run(tmp_path: Path, recipe: str = "mindx_fallback_qwen3_1_5b_cpu_smoke"):
    """Register a Run pointing at an on-disk out_dir with an adapter."""
    out_dir = tmp_path / "run"
    (out_dir / "checkpoint").mkdir(parents=True)
    (out_dir / "checkpoint" / "adapter_model.safetensors").write_bytes(b"\x00")
    return _REGISTRY.create(recipe, out_dir)


def test_push_to_ollama_unknown_run_returns_404() -> None:
    r = client.post("/coach/api/runs/does-not-exist/push-to-ollama", json={})
    assert r.status_code == 404


def test_push_to_ollama_missing_checkpoint_returns_409(tmp_path) -> None:
    out_dir = tmp_path / "no-ckpt"
    out_dir.mkdir()
    run = _REGISTRY.create("mindx_fallback_qwen3_1_5b_cpu_smoke", out_dir)
    r = client.post(f"/coach/api/runs/{run.id}/push-to-ollama", json={})
    assert r.status_code == 409
    assert "checkpoint" in r.json()["detail"].lower()


def test_push_to_ollama_happy_path(tmp_path, monkeypatch) -> None:
    """End-to-end with merge_lora_adapter + subprocess.run mocked out.

    The endpoint should resolve the base model from the recipe, run the
    pipeline in a thread, and return the structured response shape.
    """
    run = _create_run(tmp_path)

    def _fake_merge(base_model, adapter_dir, out_dir, sink=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        if sink:
            sink(f"[fake-merge] {base_model} + {adapter_dir} -> {out_dir}")
        return out_dir

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter", _fake_merge,
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which",
        lambda _b: "/usr/local/bin/ollama",
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "created\n", ""),
    )

    r = client.post(
        f"/coach/api/runs/{run.id}/push-to-ollama",
        json={"tag": "test-tag"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == run.id
    assert body["tag"] == "test-tag"
    assert body["merged_dir"].endswith("merged")
    assert body["modelfile"].endswith("Modelfile")
    assert body["message"] == "pushed"


def test_push_to_ollama_defaults_tag_to_recipe_name(tmp_path, monkeypatch) -> None:
    run = _create_run(tmp_path)

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter",
        lambda b, a, o, sink=None: (o.mkdir(parents=True, exist_ok=True) or o),
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: "/u/bin/ollama",
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    r = client.post(f"/coach/api/runs/{run.id}/push-to-ollama", json={})
    assert r.status_code == 200, r.text
    # Default tag is the recipe name.
    assert r.json()["tag"] == "mindx_fallback_qwen3_1_5b_cpu_smoke"


def test_push_to_ollama_returns_503_when_ollama_missing(tmp_path, monkeypatch) -> None:
    run = _create_run(tmp_path)

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter",
        lambda b, a, o, sink=None: (o.mkdir(parents=True, exist_ok=True) or o),
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: None,
    )

    r = client.post(f"/coach/api/runs/{run.id}/push-to-ollama", json={})
    assert r.status_code == 503
    assert "ollama" in r.json()["detail"].lower()


def test_push_to_ollama_registers_with_mindx_when_flag_set(tmp_path, monkeypatch) -> None:
    """register_with_mindx=True → calls swap_mindx_fallback_model + surfaces result."""
    run = _create_run(tmp_path)
    swap_calls: list[dict] = []

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter",
        lambda b, a, o, sink=None: (o.mkdir(parents=True, exist_ok=True) or o),
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: "/u/bin/ollama",
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    def _fake_swap(*, provider, model, api_url=None, timeout_s=30.0):
        swap_calls.append({"provider": provider, "model": model, "api_url": api_url})
        return {"previous": "qwen3:1.7b", "current": model, "config_file": "models/ollama.yaml"}

    monkeypatch.setattr(
        "mindxtrain.deploy.api_client.swap_mindx_fallback_model", _fake_swap,
    )

    r = client.post(
        f"/coach/api/runs/{run.id}/push-to-ollama",
        json={"tag": "trained-tag", "register_with_mindx": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mindx_fallback_swapped"] is True
    assert body["mindx_fallback_swap"]["current"] == "trained-tag"
    assert body["mindx_fallback_swap"]["previous"] == "qwen3:1.7b"
    # Provider + model were carried through correctly.
    assert swap_calls == [{"provider": "ollama", "model": "trained-tag", "api_url": None}]


def test_push_to_ollama_skips_mindx_registration_by_default(tmp_path, monkeypatch) -> None:
    """register_with_mindx defaults to False — no PATCH happens."""
    run = _create_run(tmp_path)

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter",
        lambda b, a, o, sink=None: (o.mkdir(parents=True, exist_ok=True) or o),
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: "/u/bin/ollama",
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    def _boom(**_kw):
        raise AssertionError("swap_mindx_fallback_model must NOT be called when flag is False")

    monkeypatch.setattr(
        "mindxtrain.deploy.api_client.swap_mindx_fallback_model", _boom,
    )

    r = client.post(f"/coach/api/runs/{run.id}/push-to-ollama", json={"tag": "x"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mindx_fallback_swapped"] is False
    assert body["mindx_fallback_swap"] is None


def test_push_to_ollama_mindx_failure_does_not_fail_push(tmp_path, monkeypatch) -> None:
    """mindX down → push still succeeds, swap reported as not happened."""
    import httpx

    run = _create_run(tmp_path)

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter",
        lambda b, a, o, sink=None: (o.mkdir(parents=True, exist_ok=True) or o),
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: "/u/bin/ollama",
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    def _raise(**_kw):
        raise httpx.ConnectError("mindX unreachable")

    monkeypatch.setattr(
        "mindxtrain.deploy.api_client.swap_mindx_fallback_model", _raise,
    )

    r = client.post(
        f"/coach/api/runs/{run.id}/push-to-ollama",
        json={"tag": "x", "register_with_mindx": True},
    )
    # The push still landed — the merged dir + tag are on disk.
    assert r.status_code == 200, r.text
    body = r.json()
    # But mindX swap is reported as not having succeeded.
    assert body["mindx_fallback_swapped"] is False
    assert body["mindx_fallback_swap"] is None


def test_push_to_ollama_returns_502_on_ollama_failure(tmp_path, monkeypatch) -> None:
    run = _create_run(tmp_path)

    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.merge_lora_adapter",
        lambda b, a, o, sink=None: (o.mkdir(parents=True, exist_ok=True) or o),
    )
    monkeypatch.setattr(
        "mindxtrain.deploy.ollama_push.shutil.which", lambda _b: "/u/bin/ollama",
    )

    def _boom(cmd, **_kw):
        raise subprocess.CalledProcessError(
            returncode=2, cmd=cmd, output="", stderr="arch not supported",
        )

    monkeypatch.setattr(subprocess, "run", _boom)

    r = client.post(f"/coach/api/runs/{run.id}/push-to-ollama", json={})
    assert r.status_code == 502
    assert "arch" in r.json()["detail"]
