"""Production test 1 (fast path): author a script → ingest local → score imprint.

The full loop (train the tiny actor, then generate real before/after utterances via
`mindxtrain imprint`) runs under `--extra ml`; this CPU/base-install test proves the
create-dataset → ingestion → measurement wiring without a model download.
"""

from __future__ import annotations

from mindxtrain.config.schema import DataCfg
from mindxtrain.data import scripts as S
from mindxtrain.data.curate import load_streaming_dataset
from mindxtrain.eval import imprint as I


def test_production_test_1_create_ingest_measure(tmp_path):
    # 1) Author a persona script (the "impression").
    persona = S.Persona(
        name="Codephreak",
        system_prompt="You are Codephreak, augmentic intelligence orchestrator.",
        voice_examples=["augmentic intelligence orchestration.", "let's build, together."],
    )
    exchanges = [
        S.Exchange(user="who are you?", assistant="i am codephreak, augmentic intelligence."),
        S.Exchange(user="what do you do?", assistant="i orchestrate autonomous agents."),
    ]
    out = tmp_path / "codephreak" / "script.jsonl"
    path, rows = S.author_script(out_path=out, exchanges=exchanges, persona=persona)
    assert rows == 4  # 2 exchanges + 2 voice seeds

    # 2) Ingest it via the local data source — the same path training uses.
    cfg = DataCfg(source="local", path=path, max_samples=50, packing=False, seq_len=128)
    ingested = list(load_streaming_dataset(cfg))
    assert len(ingested) == 4
    inquiries = [r["messages"][1]["content"] for r in ingested]
    baseline = [r["messages"][2]["content"] for r in ingested]

    # 3) Measure imprint: a "before" actor sounds generic; an "after" actor that
    #    learned the script recalls the voice. score_imprint must reflect the gain.
    before = ["I am a language model.", "I assist with various tasks.",
              "Hello there.", "I can help."]
    after = baseline  # perfect recall after imprint
    report = I.score_imprint(inquiries, before, after, baseline)

    assert report.after_voice > report.before_voice
    assert report.imprint_delta > 0.0
    assert report.imprinted is True
