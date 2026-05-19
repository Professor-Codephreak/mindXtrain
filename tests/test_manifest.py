"""custmodel Manifest + BLAKE3 hashing determinism."""

from __future__ import annotations

import json

from mindxtrain.provenance.hashing import blake3_dir, blake3_file
from mindxtrain.provenance.manifest import Manifest, ProvenanceHashes


def test_blake3_file_is_deterministic(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello mindXtrain")
    assert blake3_file(p) == blake3_file(p)
    assert len(blake3_file(p)) == 64  # hex digest of 32-byte blake3 output


def test_blake3_dir_walks_sorted(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "z.txt").write_text("z")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "m.txt").write_text("m")
    h1 = blake3_dir(tmp_path)
    h2 = blake3_dir(tmp_path)
    assert h1 == h2
    assert len(h1) == 64


def test_blake3_dir_changes_with_content(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    h1 = blake3_dir(tmp_path)
    (tmp_path / "a.txt").write_text("b")
    h2 = blake3_dir(tmp_path)
    assert h1 != h2


def test_manifest_round_trip():
    hashes = ProvenanceHashes(
        config_yaml="0" * 64,
        dataset="1" * 64,
        checkpoint="2" * 64,
        eval_json="3" * 64,
    )
    m = Manifest(
        run_id="instella-3b-alpaca-lora-demo",
        base_model="amd/Instella-3B-Instruct",
        blake3=hashes,
    )
    blob = m.model_dump_json()
    restored = Manifest.model_validate(json.loads(blob))
    assert restored.run_id == m.run_id
    assert restored.blake3.config_yaml == "0" * 64
    assert restored.on_chain.inft.chain == "base_sepolia"


def test_manifest_schema_dump():
    schema = Manifest.model_json_schema()
    assert schema["title"] == "Manifest"
    assert "blake3" in schema["properties"]


# ---- TimeAttestation + chronos integration ------------------------------


def test_manifest_carries_default_time_attestation():
    from mindxtrain.provenance.manifest import Manifest, ProvenanceHashes, TimeAttestation
    m = Manifest(
        run_id="t",
        base_model="b",
        blake3=ProvenanceHashes(
            config_yaml="0" * 64, dataset="0" * 64,
            checkpoint="0" * 64, eval_json="0" * 64,
        ),
    )
    assert isinstance(m.time_attestation, TimeAttestation)
    assert m.time_attestation.attested is False
    assert m.time_attestation.consensus == "offline"


def test_fetch_time_attestation_falls_back_when_mindx_down(monkeypatch):
    """No httpx connection → attested=False (graceful)."""
    import httpx

    from mindxtrain.provenance import manifest as _m

    def _raise(*_a, **_kw):
        raise httpx.ConnectError("mindX down")
    transport = httpx.MockTransport(_raise)
    orig = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    att = _m._fetch_time_attestation()
    assert att.attested is False
    assert att.consensus == "offline"


def test_fetch_time_attestation_populates_from_live_response(monkeypatch):
    """consensus=correlated → attested True, fields propagated."""
    import httpx

    from mindxtrain.provenance import manifest as _m

    body = {
        "unix_18dp": "1779169999.1",
        "utc": "2026-05-19T05:53:19+00:00",
        "consensus": "correlated",
        "confidence_ms": 12.5,
        "anchor_count_24h": 7,
        "promised_by": "chronos.agent",
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
    orig = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    att = _m._fetch_time_attestation()
    assert att.attested is True
    assert att.consensus == "correlated"
    assert att.confidence_ms == 12.5
    assert att.anchor_count_24h == 7
    assert att.promised_by == "chronos.agent"


def test_fetch_time_attestation_does_not_attest_when_unavailable(monkeypatch):
    """consensus='unavailable' must NOT flip attested=True."""
    import httpx

    from mindxtrain.provenance import manifest as _m

    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"consensus": "unavailable"}),
    )
    orig = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **kw: orig(*a, **{**kw, "transport": transport}),
    )
    att = _m._fetch_time_attestation()
    assert att.attested is False
    assert att.consensus == "unavailable"
