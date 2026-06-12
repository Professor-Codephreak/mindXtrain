"""Author a training *script* for an *actor*.

The mindXtrain mental model: a **model is an actor**; an actor has a **persona**
(identity / voice) and a **script** (the training examples — the "impression" left
on the actor when it trains). This module is the clean-room primitive for building
a script from a persona + a handful of exchanges, written as the OpenAI-chat JSONL
that `data.source: local` ingests (`{"messages": [{role, content}, ...]}`).

Clean-room: the Codephreak persona is *loaded* at runtime from
`MINDXTRAIN_PERSONA_PATH` (or a caller-supplied path); we never copy mindX bytes —
we read recognised fields and ignore the rest.

Pure stdlib + pydantic; importable on a base install (no `--extra ml`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# Recognised keys for a persona's identity/voice in an mindX-style persona JSON.
# We map these defensively — any other keys are ignored (clean-room read).
_NAME_KEYS = ("name", "persona", "id", "title")
_SYSTEM_KEYS = ("system_prompt", "system", "description", "bio", "summary", "prompt")
_VOICE_KEYS = ("voice_examples", "examples", "utterances", "samples", "voice")


class Persona(BaseModel):
    """The identity to imprint onto an actor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "actor"
    system_prompt: str = ""
    voice_examples: list[str] = Field(
        default_factory=list,
        description="Example in-voice utterances; seed rows + the imprint baseline.",
    )


class Exchange(BaseModel):
    """One user→assistant turn in a script."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user: str
    assistant: str


def load_persona(path: str | Path | None = None) -> Persona:
    """Load a persona, clean-room, from JSON.

    Resolution: explicit `path` → `MINDXTRAIN_PERSONA_PATH` → a built-in minimal
    default. Reads only recognised fields; unknown keys are ignored so an
    arbitrary mindX persona file maps cleanly without copying its schema.
    """
    resolved = path or os.environ.get("MINDXTRAIN_PERSONA_PATH")
    if not resolved:
        return _default_persona()
    p = Path(resolved).expanduser()
    if not p.is_file():
        return _default_persona()
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return _default_persona()
    if not isinstance(raw, dict):
        return _default_persona()
    return persona_from_dict(raw)


def persona_from_dict(raw: dict) -> Persona:
    """Build a Persona from a loosely-shaped dict (recognised keys only)."""
    name = next((str(raw[k]) for k in _NAME_KEYS if raw.get(k)), "actor")
    system = next((str(raw[k]) for k in _SYSTEM_KEYS if raw.get(k)), "")
    voice: list[str] = []
    for k in _VOICE_KEYS:
        v = raw.get(k)
        if isinstance(v, list):
            voice.extend(str(x) for x in v if isinstance(x, (str, int, float)))
        elif isinstance(v, str):
            voice.append(v)
    return Persona(name=name, system_prompt=system, voice_examples=voice)


def _default_persona() -> Persona:
    return Persona(
        name="actor",
        system_prompt="You are a helpful, concise assistant.",
        voice_examples=[],
    )


def persona_system_prompt(persona: Persona) -> str:
    """The system message that fronts every row of the script.

    Uses the persona's own system prompt when present, otherwise synthesises a
    minimal one from the name so the actor still has an identity to imprint.
    """
    if persona.system_prompt.strip():
        return persona.system_prompt.strip()
    return f"You are {persona.name}. Stay in character and answer in your own voice."


def build_script_rows(
    persona: Persona,
    exchanges: list[Exchange],
    *,
    seed_voice: bool = True,
) -> list[dict]:
    """Turn a persona + exchanges into OpenAI-chat rows for `source: local`.

    Each row carries the persona system prompt + one user→assistant turn. When
    `seed_voice` is set, the persona's voice examples are added as extra
    assistant-only demonstrations so a tiny model has voice to imprint even from
    very few exchanges.
    """
    system = persona_system_prompt(persona)
    rows: list[dict] = []
    for ex in exchanges:
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": ex.user},
                    {"role": "assistant", "content": ex.assistant},
                ],
            },
        )
    if seed_voice:
        for sample in persona.voice_examples:
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Say something as {persona.name}."},
                        {"role": "assistant", "content": sample},
                    ],
                },
            )
    return rows


def write_script_jsonl(rows: list[dict], out_path: str | Path) -> Path:
    """Write script rows as JSONL; returns the path. Parent dirs are created."""
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def derive_training_params(num_rows: int) -> dict[str, int]:
    """Derive CPU-imprint training params from the dataset size.

    A small persona/skill script must *overfit* to imprint (many epochs, grad_accum 1
    so a few-row script still does many optimizer steps); larger datasets taper toward
    ordinary SFT. Returns `{epochs, grad_accum, per_device}` the imprint lane can apply.
    """
    n = max(1, num_rows)
    if n <= 8:
        epochs, grad_accum = 24, 1
    elif n <= 32:
        epochs, grad_accum = 16, 1
    elif n <= 128:
        epochs, grad_accum = 8, 1
    elif n <= 512:
        epochs, grad_accum = 4, 2
    else:
        epochs, grad_accum = 2, 4
    return {"epochs": epochs, "grad_accum": grad_accum, "per_device": 1}


def author_script(
    *,
    out_path: str | Path,
    exchanges: list[Exchange],
    persona: Persona | None = None,
    persona_path: str | Path | None = None,
    seed_voice: bool = True,
) -> tuple[Path, int]:
    """One-call script authoring: persona + exchanges → JSONL on disk.

    Returns (path, row_count). The persona is taken as-given, else loaded
    clean-room from `persona_path` / `MINDXTRAIN_PERSONA_PATH` / the default.
    """
    actor_persona = persona or load_persona(persona_path)
    rows = build_script_rows(actor_persona, exchanges, seed_voice=seed_voice)
    path = write_script_jsonl(rows, out_path)
    return path, len(rows)


__all__ = [
    "Exchange",
    "Persona",
    "author_script",
    "build_script_rows",
    "derive_training_params",
    "load_persona",
    "persona_from_dict",
    "persona_system_prompt",
    "write_script_jsonl",
]
