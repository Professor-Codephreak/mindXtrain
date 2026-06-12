"""Built-in personas + skills, and dataset-derived training params."""

from __future__ import annotations

import pytest

from mindxtrain.data import personas as PZ
from mindxtrain.data.scripts import build_script_rows, derive_training_params


def test_builtin_personas_present():
    names = {p["name"] for p in PZ.list_personas()}
    assert {"codephreak", "assistant", "mentor"} <= names
    assert PZ.get_persona("codephreak").name == "Codephreak"
    assert PZ.get_persona("unknown").name == "Assistant"  # fallback


def test_skills_cover_requested():
    names = {s["name"] for s in PZ.list_skills()}
    assert {"software_engineer", "platform_architect", "bash", "solidity"} <= names


def test_compose_merges_addenda_and_exchanges():
    persona, exchanges = PZ.compose("codephreak", ["software_engineer", "solidity"])
    # System prompt extended with both skill addenda.
    assert "Codephreak" in persona.system_prompt
    assert "tested" in persona.system_prompt or "code" in persona.system_prompt
    assert "Solidity" in persona.system_prompt or "on-chain" in persona.system_prompt
    # Exchanges from both skills (3 each).
    assert len(exchanges) == 6


def test_compose_ignores_unknown_skill():
    _, exchanges = PZ.compose("assistant", ["bash", "wizardry"])
    assert len(exchanges) == 3  # only bash


def test_compose_no_skills():
    persona, exchanges = PZ.compose("mentor", [])
    assert exchanges == []
    assert persona.name == "Mentor"


def test_composed_persona_builds_a_script():
    persona, exchanges = PZ.compose("codephreak", ["bash"])
    rows = build_script_rows(persona, exchanges, seed_voice=False)
    assert len(rows) == 3
    assert rows[0]["messages"][0]["role"] == "system"
    assert "Codephreak" in rows[0]["messages"][0]["content"]


@pytest.mark.parametrize(
    ("rows", "epochs", "grad_accum"),
    [(4, 24, 1), (8, 24, 1), (20, 16, 1), (64, 8, 1), (300, 4, 2), (1000, 2, 4)],
)
def test_derive_training_params(rows, epochs, grad_accum):
    p = derive_training_params(rows)
    assert p["epochs"] == epochs
    assert p["grad_accum"] == grad_accum
    assert p["per_device"] == 1
