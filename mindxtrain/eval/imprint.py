"""Imprint measurement — did the persona take?

The mindXtrain model: training **imprints** a persona onto an actor. We measure the
imprint by **recall from utterance inquiry**: pose the same probe prompts (inquiries)
to the actor **before** and **after** training, then score how much the after-utterances
moved toward the persona's voice relative to before — a same-state before/after delta.

`score_imprint` is pure scoring over supplied utterances (base install, no GPU). It uses
sentence-transformer similarity when `--extra data` is present, else a stdlib lexical
fallback, so the measurement always runs. `probe_recall` (lazy `--extra ml`) generates the
utterances from a checkpoint; it's used by the end-to-end production test.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_TOKEN = re.compile(r"[a-z0-9']+")


def default_inquiries(name: str = "the actor") -> list[str]:
    """A small, persona-agnostic battery of recall probes."""
    return [
        "Who are you?",
        "What do you do?",
        f"Describe {name} in one sentence.",
        "What matters most to you?",
        "Say hello.",
    ]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _lexical_similarity(a: str, b: str) -> float:
    """Token Jaccard in [0, 1] — the dependency-free voice metric."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _voice_similarity(utterances: list[str], baseline: list[str]) -> tuple[float, str]:
    """Mean per-utterance max similarity to any baseline voice example.

    Returns (score in [0,1], method). Prefers sentence-transformer cosine when
    available; otherwise lexical Jaccard. Empty inputs score 0.
    """
    if not utterances or not baseline:
        return 0.0, "none"
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        u = enc.encode(utterances, normalize_embeddings=True)
        b = enc.encode(baseline, normalize_embeddings=True)
        sims = u @ b.T
        return float(np.mean(sims.max(axis=1))), "sentence-transformers"
    except (ImportError, OSError, RuntimeError):
        per = [max(_lexical_similarity(x, ref) for ref in baseline) for x in utterances]
        return (sum(per) / len(per)), "lexical"


class ImprintReport(BaseModel):
    """Before/after recall measurement of a persona imprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inquiries: list[str]
    before: list[str]
    after: list[str]
    before_voice: float = Field(description="mean similarity of before-utterances to persona voice")
    after_voice: float = Field(description="mean similarity of after-utterances to persona voice")
    imprint_delta: float = Field(description="after_voice - before_voice; >0 = imprinted toward persona")
    shift: float = Field(description="mean (1 - similarity(before_i, after_i)); how much utterances changed")
    method: str
    imprinted: bool = Field(description="imprint_delta > 0 and utterances actually shifted")


def score_imprint(
    inquiries: list[str],
    before: list[str],
    after: list[str],
    baseline: list[str],
) -> ImprintReport:
    """Score an imprint from same-state before/after utterances + a voice baseline.

    `before` / `after` are the actor's utterances for each inquiry (same order),
    captured from the same model state before vs after training. `baseline` is the
    persona's in-voice reference (e.g. `Persona.voice_examples`).
    """
    before_voice, m1 = _voice_similarity(before, baseline)
    after_voice, m2 = _voice_similarity(after, baseline)
    method = m1 if m1 != "none" else m2

    pairs = list(zip(before, after, strict=False))
    shift = (
        sum(1.0 - _lexical_similarity(b, a) for b, a in pairs) / len(pairs)
        if pairs
        else 0.0
    )
    delta = after_voice - before_voice
    return ImprintReport(
        inquiries=inquiries,
        before=before,
        after=after,
        before_voice=round(before_voice, 4),
        after_voice=round(after_voice, 4),
        imprint_delta=round(delta, 4),
        shift=round(shift, 4),
        method=method,
        imprinted=delta > 0.0 and shift > 0.0,
    )


def probe_recall(
    base_model: str,
    inquiries: list[str],
    *,
    adapter_dir: str | Path | None = None,
    max_new_tokens: int = 48,
    force_cpu: bool = False,
) -> list[str]:
    """Generate the actor's utterance for each inquiry (lazy `--extra ml`).

    With `adapter_dir` the persona-imprinted adapter is merged in (the "after"
    state); without it you get the base model (the "before" state). Same prompts,
    same decoding → a fair before/after recall comparison.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        msg = "transformers + torch not installed; run `uv sync --extra ml`."
        raise RuntimeError(msg) from exc

    device = "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(tok, "chat_template", None) is None:
        tok.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map={"": device}, attn_implementation="eager",
    )
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    out: list[str] = []
    for inquiry in inquiries:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": inquiry}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.3, no_repeat_ngram_size=3,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append(text.strip())
    return out


__all__ = ["ImprintReport", "default_inquiries", "probe_recall", "score_imprint"]
