"""Push a trained LoRA adapter to a local ollama daemon.

Closes the local-learning loop: a CPU/MI300X training run produces a
PEFT adapter under `<run_dir>/checkpoint/`; this module merges it into
the base weights, writes a Modelfile, and calls `ollama create` so the
tag is immediately servable on the same host the operator runs on.

Ollama 0.13+ accepts `FROM <safetensors-directory>` natively for Llama,
Mistral, Gemma, Qwen, and Phi architectures — so we don't need
llama.cpp for the supported families that mindXtrain trains. The merged
HF directory is the canonical artefact; ollama internalizes it on
`create`.

Lazy-imports `peft` / `transformers` so `import mindxtrain.deploy` stays
cheap on the CPU-only base install. Callers must opt into `--extra ml`.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OllamaPushResult:
    tag: str
    merged_dir: Path
    modelfile: Path
    ollama_stdout: str
    # When push_to_ollama was called with register_with_mindx=True and
    # the PATCH succeeded, this holds the mindX response
    # ({previous, current, ...}). None when registration was skipped or
    # failed (failure logs through `sink` but does not abort the push —
    # the merged + tagged artefact still lands locally).
    mindx_fallback_swap: dict[str, str] | None = None


def merge_lora_adapter(
    base_model: str,
    adapter_dir: Path,
    out_dir: Path,
    *,
    sink: Callable[[str], None] | None = None,
) -> Path:
    """Merge a PEFT LoRA adapter into the base weights.

    Writes the resulting full-precision HF directory to `out_dir`. The
    directory is suitable as the `FROM` target of an ollama Modelfile.

    Raises ImportError if the `ml` extras aren't installed; the message
    points the user at the exact `uv sync` command.
    """
    _emit = sink or (lambda _line: None)

    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        msg = (
            "push-to-ollama needs `peft` + `transformers`. install with: "
            "uv sync --extra ml"
        )
        raise ImportError(msg) from exc

    _emit(f"[push-ollama] loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model)
    _emit(f"[push-ollama] applying adapter from {adapter_dir}")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    _emit(f"[push-ollama] saving merged weights to {out_dir}")
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    # Carry the tokenizer too — ollama needs it for chat templating.
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(out_dir))
    _emit(f"[push-ollama] merged dir ready: {out_dir}")
    return out_dir


def write_modelfile(
    merged_dir: Path,
    out_path: Path,
    *,
    system_prompt: str | None = None,
    template: str | None = None,
    parameters: dict[str, float | int | str] | None = None,
) -> Path:
    """Write an ollama Modelfile pointing at `merged_dir`.

    Minimal contract: `FROM <merged_dir>` plus optional SYSTEM, TEMPLATE,
    and PARAMETER stanzas. Ollama's docs are at
    https://github.com/ollama/ollama/blob/main/docs/modelfile.md — the
    safetensors-directory path is what ollama 0.13+ resolves natively.
    """
    lines: list[str] = [f"FROM {merged_dir}"]
    if system_prompt:
        # SYSTEM blocks support triple-quoted multiline payloads.
        lines.append(f'SYSTEM """{system_prompt}"""')
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    for key, value in (parameters or {}).items():
        # Strings need quoting; numbers don't.
        rendered = f'"{value}"' if isinstance(value, str) else value
        lines.append(f"PARAMETER {key} {rendered}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def ollama_create(
    tag: str,
    modelfile: Path,
    *,
    sink: Callable[[str], None] | None = None,
    ollama_bin: str | None = None,
    timeout_s: float = 1800.0,
) -> str:
    """Run `ollama create <tag> -f <modelfile>` and return its stdout.

    Raises FileNotFoundError if the ollama CLI isn't on PATH and no
    explicit `ollama_bin` is provided. Raises CalledProcessError if
    ollama returns non-zero (caller surfaces stderr to the user).
    """
    _emit = sink or (lambda _line: None)

    binary = ollama_bin or shutil.which("ollama")
    if not binary:
        msg = (
            "`ollama` CLI not found on PATH. install from https://ollama.com "
            "or pass --ollama-bin to point at a custom build"
        )
        raise FileNotFoundError(msg)

    cmd = [binary, "create", tag, "-f", str(modelfile)]
    _emit(f"[push-ollama] $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.stdout:
        _emit(result.stdout.rstrip())
    if result.stderr:
        # Ollama prints progress to stderr — surface it for visibility.
        _emit(result.stderr.rstrip())
    return result.stdout


def push_to_ollama(
    base_model: str,
    adapter_dir: Path,
    tag: str,
    *,
    work_dir: Path | None = None,
    system_prompt: str | None = None,
    template: str | None = None,
    parameters: dict[str, float | int | str] | None = None,
    sink: Callable[[str], None] | None = None,
    ollama_bin: str | None = None,
    register_with_mindx: bool = False,
    mindx_base_url: str | None = None,
) -> OllamaPushResult:
    """End-to-end: merge LoRA → Modelfile → ollama create.

    `work_dir` defaults to `<adapter_dir>/../ollama_push/`. The merged HF
    directory lives at `work_dir/merged/`; the Modelfile at
    `work_dir/Modelfile`. Both are kept around so the operator can re-run
    `ollama create` without redoing the merge.

    When `register_with_mindx=True`, after a successful `ollama create`
    we PATCH `<mindx_base_url>/v1/config/fallback-model` with
    `{provider: "ollama", model: <tag>}` so the freshly pushed tag
    becomes mindX's local fallback model. The PATCH is best-effort —
    a failure logs through `sink` and lands in `OllamaPushResult` as
    `mindx_fallback_swap=None`, but does NOT raise. The point of this
    flag is "close the dream → train → fallback loop without manual
    intervention"; a stopped mindX daemon shouldn't kill the push.
    """
    work = work_dir or adapter_dir.parent / "ollama_push"
    merged_dir = merge_lora_adapter(
        base_model, adapter_dir, work / "merged", sink=sink,
    )
    modelfile = write_modelfile(
        merged_dir, work / "Modelfile",
        system_prompt=system_prompt, template=template, parameters=parameters,
    )
    stdout = ollama_create(tag, modelfile, sink=sink, ollama_bin=ollama_bin)

    swap_result: dict[str, str] | None = None
    if register_with_mindx:
        _emit = sink or (lambda _line: None)
        try:
            # Lazy-imported so the deploy module stays importable on
            # hosts without the publish-side dep tree warmed up.
            from mindxtrain.deploy.api_client import swap_mindx_fallback_model

            _emit(f"[push-ollama] registering {tag} with mindX as fallback")
            swap_result = swap_mindx_fallback_model(
                provider="ollama", model=tag, api_url=mindx_base_url,
            )
            _emit(
                f"[push-ollama] mindX swap: "
                f"{swap_result.get('previous', '?')} -> "
                f"{swap_result.get('current', '?')}",
            )
        except Exception as exc:
            # Swallow — see docstring rationale.
            _emit(f"[push-ollama] mindX registration failed (push still ok): {exc}")
            swap_result = None

    return OllamaPushResult(
        tag=tag, merged_dir=merged_dir, modelfile=modelfile,
        ollama_stdout=stdout, mindx_fallback_swap=swap_result,
    )


__all__ = [
    "OllamaPushResult",
    "merge_lora_adapter",
    "ollama_create",
    "push_to_ollama",
    "write_modelfile",
]
