"""Script authoring (actor / persona / script) + local-JSONL ingestion."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mindxtrain.config.loader import list_recipes, load_config, render_recipe
from mindxtrain.data import scripts as S


def test_persona_from_dict_maps_recognised_keys():
    p = S.persona_from_dict(
        {"name": "Codephreak", "description": "You are Codephreak.",
         "examples": ["yo.", "let's build."], "irrelevant": 123},
    )
    assert p.name == "Codephreak"
    assert p.system_prompt == "You are Codephreak."
    assert p.voice_examples == ["yo.", "let's build."]


def test_load_persona_from_env(monkeypatch, tmp_path):
    pj = tmp_path / "persona.json"
    pj.write_text(json.dumps({"persona": "Mentor", "system": "Teach plainly."}))
    monkeypatch.setenv("MINDXTRAIN_PERSONA_PATH", str(pj))
    p = S.load_persona()
    assert p.name == "Mentor"
    assert p.system_prompt == "Teach plainly."


def test_load_persona_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_PERSONA_PATH", raising=False)
    p = S.load_persona()
    assert p.name == "actor"
    assert p.system_prompt


def test_build_script_rows_shape():
    persona = S.Persona(name="Codephreak", system_prompt="You are Codephreak.",
                        voice_examples=["augmentic intelligence."])
    rows = S.build_script_rows(
        persona,
        [S.Exchange(user="who are you?", assistant="i am codephreak.")],
        seed_voice=True,
    )
    # 1 exchange + 1 voice-seed row.
    assert len(rows) == 2
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "You are Codephreak."
    assert msgs[2]["content"] == "i am codephreak."
    # Voice-seed row carries the example as the assistant turn.
    assert rows[1]["messages"][2]["content"] == "augmentic intelligence."


def test_author_script_writes_ingestible_jsonl(tmp_path, monkeypatch):
    monkeypatch.delenv("MINDXTRAIN_PERSONA_PATH", raising=False)
    out = tmp_path / "ds" / "script.jsonl"
    path, n = S.author_script(
        out_path=out,
        exchanges=[
            S.Exchange(user="hi", assistant="hello, friend."),
            S.Exchange(user="what do you do?", assistant="i orchestrate agents."),
        ],
        persona=S.Persona(name="Codephreak", system_prompt="You are Codephreak."),
        seed_voice=False,
    )
    assert path == out and n == 2
    lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert all("messages" in r for r in lines)

    # The authored JSONL is ingestible by the local data source.
    from mindxtrain.config.schema import DataCfg
    from mindxtrain.data.curate import load_streaming_dataset

    cfg = DataCfg(source="local", path=out, max_samples=10, packing=False, seq_len=128)
    ingested = list(load_streaming_dataset(cfg))
    assert len(ingested) == 2
    assert ingested[0]["messages"][0]["role"] == "system"


def test_persona_imprint_recipe_validates():
    assert "mindx_persona_imprint_local" in list_recipes()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.yaml"
        p.write_text(render_recipe("mindx_persona_imprint_local"))
        cfg = load_config(p)
    assert cfg.train.backend == "trl_local"
    assert cfg.data.source == "local"
    assert cfg.data.path is not None
