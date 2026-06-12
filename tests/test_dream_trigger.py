"""mindX machine.dream ingestion trigger (clean-room handoff)."""

from __future__ import annotations

import json

from mindxtrain.deploy.api_client import trigger_dream_ingestion


def test_trigger_writes_inbox_pointer(tmp_path, monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_API_BASE_URL", raising=False)
    monkeypatch.setenv("MINDXTRAIN_MINDX_HOME", str(tmp_path))

    res = trigger_dream_ingestion(
        run_id="imprint-1",
        adapter_dir="/runs/imprint-1/checkpoint",
        base_model="HuggingFaceTB/SmolLM2-135M",
        persona_name="Codephreak",
        imprint_delta=0.42,
    )
    assert res["mode"] == "inbox"
    ptr = tmp_path / "data" / "incoming" / "imprint-1.dream.json"
    assert ptr.is_file()
    payload = json.loads(ptr.read_text())
    assert payload["run_id"] == "imprint-1"
    assert payload["persona"] == "Codephreak"
    assert payload["imprint_delta"] == "0.4200"
    assert payload["source"] == "mindxtrain.imprint"


def test_trigger_skips_cleanly_when_unconfigured(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_API_BASE_URL", raising=False)
    monkeypatch.delenv("MINDXTRAIN_MINDX_HOME", raising=False)
    res = trigger_dream_ingestion(
        run_id="x", adapter_dir="/a", base_model="m",
    )
    assert res["mode"] == "skipped"
