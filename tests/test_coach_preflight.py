"""Preflight + dream-corpus endpoints — the Coach UI's first two gates
before kicking off a production training run.

Both are pure read endpoints; tests verify the response shape and that the
required-missing logic is correctly partitioned from optional vars.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


# ---- preflight -----------------------------------------------------------


_REQUIRED = ("AMD_DEV_CLOUD_TOKEN", "AMD_DEV_CLOUD_SSH_KEY_ID", "HF_TOKEN", "HF_HUB_USERNAME")
_OPTIONAL = ("MINDXTRAIN_API_KEY", "MINDXTRAIN_MINDX_HOME", "LIGHTHOUSE_API_KEY")


def _strip_preflight_env(monkeypatch):
    """Clear every env var the preflight endpoint inspects."""
    for name in _REQUIRED + _OPTIONAL:
        monkeypatch.delenv(name, raising=False)


def test_preflight_reports_required_missing(monkeypatch):
    _strip_preflight_env(monkeypatch)
    r = client.get("/coach/api/preflight")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is False
    assert set(data["required_missing"]) == set(_REQUIRED)
    # All vars present as keys, all False
    assert all(data["vars"][n] is False for n in _REQUIRED + _OPTIONAL)


def test_preflight_ready_when_required_set(monkeypatch):
    _strip_preflight_env(monkeypatch)
    for name in _REQUIRED:
        monkeypatch.setenv(name, "fake-value")
    r = client.get("/coach/api/preflight")
    assert r.status_code == 200
    data = r.json()
    assert data["ready"] is True
    assert data["required_missing"] == []
    assert all(data["vars"][n] is True for n in _REQUIRED)
    # Optional still unset — should not gate readiness
    assert all(data["vars"][n] is False for n in _OPTIONAL)


def test_preflight_empty_string_treated_as_unset(monkeypatch):
    """Whitespace-only / empty values must not count as 'set'."""
    _strip_preflight_env(monkeypatch)
    for name in _REQUIRED:
        monkeypatch.setenv(name, "   ")  # whitespace only
    r = client.get("/coach/api/preflight")
    assert r.status_code == 200
    assert r.json()["ready"] is False


def test_preflight_does_not_echo_values(monkeypatch):
    """Endpoint must never leak the actual secret values."""
    _strip_preflight_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_xyz")
    body = client.get("/coach/api/preflight").text
    assert "hf_super_secret_xyz" not in body


# ---- dream-corpus ----------------------------------------------------------


def _seed_corpus(tmp_path: Path, n_examples: int) -> Path:
    """Build a minimal LTM tree shaped like mindX/data/memory."""
    root = tmp_path / "data" / "memory"
    ltm = root / "ltm" / "agent_a"
    ltm.mkdir(parents=True)
    jsonl = ltm / "20260514_010000_training.jsonl"
    with jsonl.open("w") as fh:
        for i in range(n_examples):
            fh.write(
                json.dumps({
                    "messages": [
                        {"role": "system", "content": "test"},
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                })
                + "\n"
            )
    return root


def test_dream_corpus_uses_env_root(monkeypatch, tmp_path):
    root = _seed_corpus(tmp_path, n_examples=5)
    monkeypatch.setenv("MINDXTRAIN_MINDX_HOME", str(tmp_path))
    r = client.get("/coach/api/dream-corpus")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is True
    assert data["unique_rows"] == 5
    assert data["ready"] is True
    assert data["note"] is None
    assert data["root"] == str(root)


def test_dream_corpus_explicit_root_arg(monkeypatch, tmp_path):
    root = _seed_corpus(tmp_path, n_examples=3)
    monkeypatch.delenv("MINDXTRAIN_MINDX_HOME", raising=False)
    r = client.get(f"/coach/api/dream-corpus?root={root}")
    assert r.status_code == 200
    data = r.json()
    assert data["root"] == str(root)
    assert data["unique_rows"] == 3
    assert data["ready"] is True


def test_dream_corpus_missing_root_returns_friendly_note(monkeypatch, tmp_path):
    monkeypatch.setenv("MINDXTRAIN_MINDX_HOME", str(tmp_path / "nowhere"))
    r = client.get("/coach/api/dream-corpus")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is False
    assert data["ready"] is False
    assert data["note"] is not None
    assert "MINDXTRAIN_MINDX_HOME" in data["note"]


def test_dream_corpus_empty_tree_reports_not_ready(monkeypatch, tmp_path):
    (tmp_path / "data" / "memory" / "ltm").mkdir(parents=True)
    monkeypatch.setenv("MINDXTRAIN_MINDX_HOME", str(tmp_path))
    r = client.get("/coach/api/dream-corpus")
    assert r.status_code == 200
    data = r.json()
    assert data["exists"] is True
    assert data["unique_rows"] == 0
    assert data["ready"] is False
    assert data["note"] is not None
    assert "dream cycle" in data["note"].lower()
