"""Ollama Modelfile builder.

Render a valid Ollama `Modelfile` from a typed spec, and expose the full parameter
catalogue so a UI can build toggles + input fields dynamically. Covers every
instruction (https://docs.ollama.com/modelfile): FROM, PARAMETER, TEMPLATE, SYSTEM,
ADAPTER, LICENSE, MESSAGE, REQUIRES.

Pure stdlib + pydantic; base-install importable. `create_model` (optional) shells out
to `ollama create` and is the only part needing the ollama binary.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ParamType = Literal["int", "float", "string", "bool"]


class ParamSpec(BaseModel):
    """Metadata for one PARAMETER — drives the UI toggle + input field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type: ParamType
    default: float | int | str | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""


# The full PARAMETER catalogue. Defaults follow Ollama's documented values.
MODELFILE_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(name="num_ctx", type="int", default=2048, minimum=64, description="context window size (tokens)"),
    ParamSpec(name="num_predict", type="int", default=-1, description="max tokens to predict (-1 = infinite)"),
    ParamSpec(name="num_keep", type="int", default=0, description="tokens kept from the initial prompt"),
    ParamSpec(name="seed", type="int", default=0, description="RNG seed for reproducible output"),
    ParamSpec(name="temperature", type="float", default=0.8, minimum=0.0, maximum=2.0, description="creativity / randomness"),
    ParamSpec(name="top_k", type="int", default=40, minimum=0, description="sample from the top-k tokens"),
    ParamSpec(name="top_p", type="float", default=0.9, minimum=0.0, maximum=1.0, description="nucleus sampling cumulative prob"),
    ParamSpec(name="min_p", type="float", default=0.0, minimum=0.0, maximum=1.0, description="min relative token probability"),
    ParamSpec(name="typical_p", type="float", default=1.0, minimum=0.0, maximum=1.0, description="locally-typical sampling"),
    ParamSpec(name="repeat_last_n", type="int", default=64, description="lookback for repeat penalty (-1 = num_ctx)"),
    ParamSpec(name="repeat_penalty", type="float", default=1.1, minimum=0.0, description="penalty strength for repetition"),
    ParamSpec(name="presence_penalty", type="float", default=0.0, description="penalize tokens already present"),
    ParamSpec(name="frequency_penalty", type="float", default=0.0, description="penalize by token frequency"),
    ParamSpec(name="mirostat", type="int", default=0, minimum=0, maximum=2, description="Mirostat sampling (0 off, 1 v1, 2 v2)"),
    ParamSpec(name="mirostat_tau", type="float", default=5.0, minimum=0.0, description="Mirostat target entropy"),
    ParamSpec(name="mirostat_eta", type="float", default=0.1, minimum=0.0, description="Mirostat learning rate"),
    ParamSpec(name="num_gpu", type="int", default=-1, description="layers to offload to GPU (-1 = auto)"),
    ParamSpec(name="num_thread", type="int", default=0, description="CPU threads (0 = auto)"),
    ParamSpec(name="num_batch", type="int", default=512, description="prompt-processing batch size"),
    ParamSpec(name="draft_num_predict", type="int", default=4, description="speculative draft tokens"),
)

_PARAM_TYPES: dict[str, ParamType] = {p.name: p.type for p in MODELFILE_PARAMS}
_VALID_ROLES = {"system", "user", "assistant"}


class ModelfileMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str


class ModelfileSpec(BaseModel):
    """A typed Modelfile spec. Only `from_model` is required."""

    model_config = ConfigDict(extra="forbid")

    from_model: str = Field(description="base model or path (the FROM instruction)")
    system: str = ""
    template: str = ""
    adapter: str = ""
    license: str = ""
    requires: str = Field(default="", description="minimum Ollama version (REQUIRES)")
    parameters: dict[str, float | int | str] = Field(default_factory=dict)
    stop: list[str] = Field(default_factory=list, description="stop sequences (PARAMETER stop)")
    messages: list[ModelfileMessage] = Field(default_factory=list)


def _fmt_value(name: str, value: float | int | str) -> str:
    """Format a PARAMETER value; quote strings that contain whitespace."""
    declared = _PARAM_TYPES.get(name)
    if declared in ("int",) and isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        return f'"{value}"' if (not value or any(c.isspace() for c in value)) else value
    return str(value)


def _block(value: str) -> str:
    """Render a multi-line value as a triple-quoted block, else inline."""
    if "\n" in value or '"' in value:
        return f'"""{value}"""'
    return f'"""{value}"""' if value else '""'


def render_modelfile(spec: ModelfileSpec) -> str:
    """Render a valid Modelfile text from the spec (deterministic field order)."""
    lines: list[str] = [f"FROM {spec.from_model}"]

    if spec.requires:
        lines.append(f"REQUIRES {spec.requires}")

    # PARAMETERs in catalogue order, then any extras, then stop sequences.
    ordered = [p.name for p in MODELFILE_PARAMS if p.name in spec.parameters]
    extras = [k for k in spec.parameters if k not in _PARAM_TYPES]
    for name in (*ordered, *sorted(extras)):
        lines.append(f"PARAMETER {name} {_fmt_value(name, spec.parameters[name])}")
    for stop in spec.stop:
        lines.append(f'PARAMETER stop "{stop}"')

    if spec.system:
        lines.append(f"SYSTEM {_block(spec.system)}")
    if spec.template:
        lines.append(f"TEMPLATE {_block(spec.template)}")
    if spec.adapter:
        lines.append(f"ADAPTER {spec.adapter}")
    if spec.license:
        lines.append(f"LICENSE {_block(spec.license)}")
    for m in spec.messages:
        # MESSAGE content is single-line in the instruction; collapse newlines.
        content = m.content.replace("\n", " ").strip()
        lines.append(f"MESSAGE {m.role} {content}")

    return "\n".join(lines) + "\n"


def write_modelfile(spec: ModelfileSpec, out_path: str | Path) -> Path:
    """Render + write a Modelfile to disk; returns the path."""
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_modelfile(spec))
    return out


def create_model(
    tag: str,
    spec: ModelfileSpec,
    *,
    out_dir: str | Path = "./out/modelfiles",
    ollama_bin: str = "ollama",
) -> dict[str, str]:
    """Write the Modelfile and run `ollama create <tag> -f <Modelfile>`.

    Returns `{tag, modelfile, status, output}`. Never raises on a failed
    `ollama create` — the failure is reported in the return dict.
    """
    path = write_modelfile(spec, Path(out_dir) / tag / "Modelfile")
    try:
        proc = subprocess.run(
            [ollama_bin, "create", tag, "-f", str(path)],
            capture_output=True, text=True, timeout=600, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"tag": tag, "modelfile": str(path), "status": "error", "output": str(exc)}
    status = "created" if proc.returncode == 0 else "failed"
    return {
        "tag": tag, "modelfile": str(path), "status": status,
        "output": (proc.stdout + proc.stderr).strip()[-2000:],
    }


__all__ = [
    "MODELFILE_PARAMS",
    "ModelfileMessage",
    "ModelfileSpec",
    "ParamSpec",
    "create_model",
    "render_modelfile",
    "write_modelfile",
]
