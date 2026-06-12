"""CPU training backend — real SFT/LoRA via TRL, no GPU required.

The closed-loop case: mindX produces a small JSONL dataset (dream cycle),
mindXtrain needs to fine-tune a tiny base model on it locally without
provisioning a MI300X droplet. This backend exists so that mindX agents can
trigger self-training on commodity hardware; it is also the smoke lane for
any new recipe before burning AMD credits.

Slow but produces a *real* checkpoint and is compatible with the rest of the
pipeline (`mindxtrain quantize`, `mindxtrain receipt`, `publish` — all
expect a HF-format checkpoint directory).

This module follows the project lazy-import contract: importing the module
must succeed on a base install, but calling `run_trl_cpu` requires
`uv sync --extra ml`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.schema import (
    LoraMethod,
    QLoraMethod,
    XTrainConfig,
    resolve_thread_count,
)


def _apply_cpu_throttle(cfg: XTrainConfig, sink: Callable[[str], None]) -> int:
    """Apply the recipe's `cfg.train.cpu_throttle` to the current process.

    Resolves percent → thread count against the host's actual core count,
    sets every thread-pool env var the downstream stack respects (torch,
    OpenMP, MKL, OpenBLAS), optionally pins OpenMP threads to cores via
    OMP_PROC_BIND=close + OMP_PLACES=cores (Ryzen-friendly: keeps threads
    on the same CCX chiplet, reduces cross-CCX cache traffic for the
    small matmuls CPU training does), and shifts the process's POSIX nice
    level so the rest of the laptop stays usable.

    Must be called BEFORE torch is imported / first used, otherwise the
    thread-pool size is locked at whatever torch saw on first init.

    Returns the resolved thread count for logging.
    """
    throttle = cfg.train.cpu_throttle
    total_cores = os.cpu_count() or 1
    threads = resolve_thread_count(throttle.percent, total_cores)

    # Set every thread-pool env var the downstream stack reads. PyTorch's
    # ATen reads OMP_NUM_THREADS at first init; MKL and OpenBLAS each
    # have their own knob. All must agree to actually cap the workload.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    # tokenizers (HF Rust lib) has its own parallelism that BLAS doesn't
    # cap. Disable it during throttled runs — for a 135M smoke on 1-2
    # threads, the tokenizer-side parallelism only thrashes the cache.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false" if threads <= 2 else "true")

    if throttle.omp_proc_bind:
        # CCX-aware pinning. `close` = threads adjacent to the master;
        # `cores` = one thread per physical core. Both safe on non-AMD
        # CPUs; ignored by OpenMP runtimes that don't honor them.
        os.environ["OMP_PROC_BIND"] = "close"
        os.environ["OMP_PLACES"] = "cores"

    if throttle.nice_level != 0:
        try:
            os.nice(throttle.nice_level)
            sink(f"[trl_cpu] nice level set to {throttle.nice_level}")
        except (PermissionError, OSError) as exc:
            # Negative nice needs CAP_SYS_NICE. Surface, don't fail.
            sink(
                f"[trl_cpu] nice({throttle.nice_level}) refused "
                f"({type(exc).__name__}: {exc}) — continuing without it",
            )

    sink(
        f"[trl_cpu] throttle: {throttle.percent}% of {total_cores} cores "
        f"→ {threads} thread(s); OMP_PROC_BIND={'close' if throttle.omp_proc_bind else 'off'}",
    )
    return threads


def _require_ml_deps() -> dict[str, Any]:
    """Import TRL + transformers + peft + datasets eagerly; surface a single message."""
    missing: list[str] = []
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        missing.append("datasets")
        Dataset = None  # type: ignore
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError:
        missing.append("transformers")
        AutoModelForCausalLM = AutoTokenizer = None  # type: ignore
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore
    except ImportError:
        missing.append("trl")
        SFTConfig = SFTTrainer = None  # type: ignore
    try:
        from peft import LoraConfig  # type: ignore
    except ImportError:
        # peft only required for LoRA/QLoRA — kept optional here
        LoraConfig = None  # type: ignore

    if missing:
        msg = (
            f"CPU training backend requires {', '.join(missing)} — "
            "run `uv sync --extra ml`."
        )
        raise RuntimeError(msg)

    return {
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "LoraConfig": LoraConfig,
    }


def _stream_dataset_rows(cfg: XTrainConfig) -> Iterator[dict[str, Any]]:
    """Yield raw dataset rows for the configured DataCfg.source."""
    from mindxtrain.data.curate import load_streaming_dataset

    yield from load_streaming_dataset(cfg.data)


def _materialize_dataset(cfg: XTrainConfig, Dataset: Any) -> Any:
    """Pull the stream into an in-memory `datasets.Dataset` for TRL.

    CPU corpora are small by construction; we don't try to stream into TRL
    here since `SFTTrainer` wants a `Dataset` with `__len__`. Cap at
    `cfg.data.max_samples or 50000` to keep memory bounded.
    """
    cap = cfg.data.max_samples if cfg.data.max_samples is not None else 50_000
    rows: list[dict[str, Any]] = []
    for row in _stream_dataset_rows(cfg):
        rows.append(row)
        if len(rows) >= cap:
            break
    if not rows:
        msg = (
            f"data.source={cfg.data.source!r} yielded zero examples — "
            "check the `path` / `hf_id` and that the source has data."
        )
        raise RuntimeError(msg)
    return Dataset.from_list(rows)


def _build_lora_config(cfg: XTrainConfig, LoraConfig: Any) -> Any | None:
    method = cfg.train.method
    if isinstance(method, LoraMethod) or isinstance(method, QLoraMethod):
        if LoraConfig is None:
            msg = "peft not installed; `uv sync --extra ml` or drop method.kind to 'full'."
            raise RuntimeError(msg)
        return LoraConfig(
            r=method.r,
            lora_alpha=method.alpha,
            lora_dropout=method.dropout,
            target_modules=list(method.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
    return None


def _build_event_callback(
    on_event: Callable[[dict[str, Any]], None],
    sink: Callable[[str], None],
) -> Any:
    """Return a TrainerCallback that fires `on_event(dict)` per HF Trainer log.

    Bridges in-process Trainer logs into the operator's SSE event stream
    without going through stdout regex parsing. Each `on_log` call carries
    a `{loss, learning_rate, grad_norm, …}` dict — we lift that into the
    same shape `StepEvent` expects (step / loss / lr / grad_norm), and
    emit `eval` + `status` events on the other Trainer hooks.

    Lazy-imported `transformers.TrainerCallback` so this helper is only
    realised when the backend actually runs (consistent with the rest of
    this module).
    """
    from transformers import TrainerCallback  # type: ignore

    class _CB(TrainerCallback):  # type: ignore[misc, valid-type]
        def on_log(
            self, args: Any, state: Any, control: Any,
            logs: dict[str, float] | None = None, **_kw: Any,
        ) -> None:
            if not logs:
                return
            # HF Trainer emits multiple kinds of log: train step (has 'loss'),
            # final summary (has 'train_loss'), and eval (has 'eval_loss').
            # Map step-level logs to StepEvent.
            if "loss" in logs:
                on_event({
                    "kind": "step",
                    "step": int(state.global_step),
                    "loss": float(logs["loss"]),
                    "lr": float(logs["learning_rate"]) if "learning_rate" in logs else None,
                    "grad_norm": float(logs["grad_norm"]) if "grad_norm" in logs else None,
                    "tokens_per_s": None,
                    # Realtime-feedback fields for the Coach progress bar +
                    # "is it learning" chart. state.max_steps is the
                    # authoritative total HF resolved after packing.
                    "total_steps": (
                        int(state.max_steps)
                        if getattr(state, "max_steps", 0)
                        else None
                    ),
                    "mean_token_accuracy": (
                        float(logs["mean_token_accuracy"])
                        if "mean_token_accuracy" in logs
                        else None
                    ),
                    "entropy": (
                        float(logs["entropy"]) if "entropy" in logs else None
                    ),
                })
                sink(
                    f"[trl_cpu] step={state.global_step} loss={logs['loss']:.4f}"
                    + (f" lr={logs['learning_rate']:.2e}" if "learning_rate" in logs else "")
                    + (f" grad_norm={logs['grad_norm']:.3f}" if "grad_norm" in logs else ""),
                )

        def on_evaluate(
            self, args: Any, state: Any, control: Any,
            metrics: dict[str, float] | None = None, **_kw: Any,
        ) -> None:
            if not metrics:
                return
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            if clean:
                on_event({
                    "kind": "eval",
                    "step": int(state.global_step),
                    "suite": "mid_train",
                    "metrics": clean,
                })

    return _CB()


def _env_force_cpu() -> bool:
    """`MINDXTRAIN_FORCE_CPU` forces the CPU fallback even when a GPU exists.

    The escape hatch for parity testing and for pinning the deterministic CPU
    path on a box that happens to have an accelerator.
    """
    return os.environ.get("MINDXTRAIN_FORCE_CPU", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


@dataclass(frozen=True)
class _DevicePlan:
    """Resolved device/dtype decisions for one in-process training run."""

    label: str                  # human tag e.g. "cuda (bfloat16)" / "cpu (float32)"
    device_map: dict[str, str]  # passed to from_pretrained
    torch_dtype: Any            # a torch.dtype
    attn_impl: str              # "sdpa" on GPU, "eager" on CPU
    bf16: bool
    fp16: bool
    apply_cpu_throttle: bool    # only the CPU path throttles threads
    batch_cap: int | None       # cap per-device batch (CPU=2); None = uncapped
    grad_checkpointing: bool


def _resolve_device(cfg: XTrainConfig, torch_mod: Any, *, force_cpu: bool) -> _DevicePlan:
    """Pick the device/dtype for a run.

    GPU when one is visible (ROCm exposes itself through `torch.cuda`) and not
    forced off; otherwise the exact CPU defaults the legacy `trl_cpu` lane used.
    """
    if not force_cpu and torch_mod.cuda.is_available():
        bf16 = bool(torch_mod.cuda.is_bf16_supported())
        return _DevicePlan(
            label=f"cuda ({'bfloat16' if bf16 else 'float16'})",
            device_map={"": "cuda"},
            torch_dtype=torch_mod.bfloat16 if bf16 else torch_mod.float16,
            attn_impl="sdpa",
            bf16=bf16,
            fp16=not bf16,
            apply_cpu_throttle=False,
            batch_cap=None,
            grad_checkpointing=cfg.train.gradient_checkpointing,
        )
    return _DevicePlan(
        label="cpu (float32)",
        device_map={"": "cpu"},
        torch_dtype=torch_mod.float32,
        attn_impl="eager",
        bf16=False,
        fp16=False,
        apply_cpu_throttle=True,
        batch_cap=2,
        grad_checkpointing=False,  # CPU + checkpointing is pathologically slow
    )


def _capped_batch(per_device: int, cap: int | None) -> int:
    """Clamp the per-device batch to `cap` (CPU=2); `None` leaves it as-is."""
    return max(1, per_device if cap is None else min(per_device, cap))


def run_trl_cpu(
    cfg: XTrainConfig,
    plan: AutotunePlan,
    out_dir: Path,
    *,
    on_line: Callable[[str], None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Force-CPU wrapper around `run_trl_local` (the mindX self-training lane).

    Preserves the deterministic CPU behaviour the dream-cycle loop depends on:
    float32, eager attention, CPU throttle, batch cap 2, no gradient checkpointing.
    """
    return run_trl_local(
        cfg, plan, out_dir, force_cpu=True, on_line=on_line, on_event=on_event,
    )


def run_trl_local(
    cfg: XTrainConfig,
    plan: AutotunePlan,
    out_dir: Path,
    *,
    force_cpu: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Run a TRL SFT job on the local device; return the checkpoint directory.

    Auto-detects an accelerator: uses the GPU (CUDA or ROCm, bf16/fp16) when one
    is visible, otherwise falls back to CPU (float32) with a logged warning. Set
    `force_cpu=True` (or `MINDXTRAIN_FORCE_CPU=1`) to pin the CPU path.

    Same `on_line` / `on_event` streaming contract as the other lanes so the
    Coach UI's live log + loss curve work identically on GPU and CPU.

    `on_line` mirrors the axolotl backend signature so the Coach UI can
    stream log lines uniformly across lanes. TRL doesn't emit one-line-per-
    step by default, but we forward `transformers` log records via a tiny
    handler so the streaming surface stays consistent.

    `on_event` is the *structured* counterpart: each HF Trainer log fires
    a dict with `{kind: "step", step, loss, lr, grad_norm, ...}` so the
    Coach UI can populate its Chart.js loss curve directly, without
    stdout-parsing the way the axolotl subprocess path does. When
    provided, the loss chart fills in in real time during training.

    On the CPU path, applies `cfg.train.cpu_throttle` before torch initializes
    so the thread-pool size actually caps the workload (BLAS layers snapshot
    their thread count on first use). The GPU path skips throttling entirely.
    """
    sink = on_line if on_line is not None else (lambda _line: None)
    forced_cpu = force_cpu or _env_force_cpu()
    tag = "trl_cpu" if forced_cpu else "trl_local"

    # When the CPU path is known up front (force_cpu), throttle BEFORE importing
    # the ML stack — exactly the legacy ordering the dream-loop relies on.
    threads: int | None = None
    if forced_cpu:
        threads = _apply_cpu_throttle(cfg, sink)

    deps = _require_ml_deps()
    Dataset = deps["Dataset"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    SFTConfig = deps["SFTConfig"]
    SFTTrainer = deps["SFTTrainer"]
    LoraConfig = deps["LoraConfig"]

    import torch  # type: ignore

    device = _resolve_device(cfg, torch, force_cpu=forced_cpu)
    if device.apply_cpu_throttle:
        if threads is None:
            # Auto-detected CPU fallback (no accelerator visible). Throttle now;
            # `set_num_threads` still caps at runtime even post-import.
            if not forced_cpu:
                sink(f"[{tag}] no accelerator detected → CPU fallback (float32)")
            threads = _apply_cpu_throttle(cfg, sink)
        # ATen reads OMP_NUM_THREADS at init, but `set_num_threads` is the
        # canonical knob — call both belt-and-suspenders.
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            # Must be called before any aten op runs; OMP env var is the fallback.
            pass
    else:
        sink(f"[{tag}] accelerator detected → {device.label}")

    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sink(f"[{tag}] base={cfg.model.name} method={cfg.train.method.kind}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Some base tokenizers (e.g., SmolLM2-135M) don't ship a chat
    # template. The dream corpus is ChatML-shaped already, so we set
    # ChatML explicitly when missing — matches what mindX's machine_
    # dreaming phase 5b emits. Qwen / Llama base models keep their own
    # template untouched.
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        sink(f"[{tag}] tokenizer had no chat_template; set ChatML default.")

    sink(f"[{tag}] materializing dataset (in-memory)")
    full_dataset = _materialize_dataset(cfg, Dataset)
    sink(f"[{tag}] dataset size={len(full_dataset)}")

    # Optional train/eval split — deterministic via meta.seed so the same
    # recipe always carves the same held-out rows. When eval_split is
    # None we pass the full dataset to SFTTrainer (legacy behaviour);
    # otherwise we split, pass train_dataset + eval_dataset, and turn on
    # step-based evaluation so eval_loss appears in log_history.
    train_dataset = full_dataset
    eval_dataset = None
    if cfg.data.eval_split is not None:
        n = len(full_dataset)
        if n < 4:
            sink(
                f"[{tag}] dataset too small ({n} rows) for eval_split="
                f"{cfg.data.eval_split}; skipping held-out split",
            )
        else:
            split = full_dataset.train_test_split(
                test_size=cfg.data.eval_split, seed=cfg.meta.seed,
            )
            train_dataset, eval_dataset = split["train"], split["test"]
            sink(
                f"[{tag}] split: train={len(train_dataset)} "
                f"eval={len(eval_dataset)} (seed={cfg.meta.seed})",
            )

    sink(f"[{tag}] loading base model on {device.label}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=device.torch_dtype,
        device_map=device.device_map,
        attn_implementation=device.attn_impl,
    )

    peft_config = _build_lora_config(cfg, LoraConfig)

    # Estimate max_steps so we can floor logging_steps + eval_steps for
    # short runs. HF Trainer computes max_steps as
    # num_epochs * (len(train_dataset) // (batch * grad_accum)). When
    # packing is on, sample count drops post-tokenization — this estimate
    # is a lower bound but good enough for the cadence floor.
    per_device_batch = _capped_batch(cfg.train.batch.per_device, device.batch_cap)
    eff_batch = per_device_batch * cfg.train.batch.grad_accum
    est_max_steps = max(1, (len(train_dataset) // eff_batch) * cfg.train.schedule.epochs)
    effective_logging_steps = max(1, min(cfg.train.logging_steps, max(1, est_max_steps // 4)))
    effective_eval_steps = max(1, est_max_steps // 4)
    sink(
        f"[{tag}] est_max_steps={est_max_steps} "
        f"logging_steps={effective_logging_steps} "
        f"eval_steps={effective_eval_steps if eval_dataset is not None else 'off'}",
    )

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.train.schedule.epochs,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=cfg.train.batch.grad_accum,
        learning_rate=cfg.train.optimizer.lr,
        warmup_ratio=cfg.train.schedule.warmup_ratio,
        max_length=cfg.data.seq_len,
        packing=cfg.data.packing,
        logging_steps=effective_logging_steps,
        save_strategy="epoch",
        report_to="none",
        bf16=device.bf16,
        fp16=device.fp16,
        gradient_checkpointing=device.grad_checkpointing,
        seed=cfg.meta.seed,
    )
    if eval_dataset is not None:
        sft_kwargs["eval_strategy"] = "steps"
        sft_kwargs["eval_steps"] = effective_eval_steps
        sft_kwargs["per_device_eval_batch_size"] = per_device_batch
    sft_args = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = SFTTrainer(**trainer_kwargs)
    # Wire the structured event callback if the caller wants per-step
    # telemetry (Coach UI's loss chart). When `on_event` is None this is
    # a no-op — the CLI path doesn't need it.
    if on_event is not None:
        trainer.add_callback(_build_event_callback(on_event, sink))

    sink(f"[{tag}] starting trainer.train()")
    trainer.train()
    sink(f"[{tag}] training complete, saving checkpoint")
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    sink(f"[{tag}] checkpoint at {checkpoint_dir}")
    return checkpoint_dir


__all__ = ["run_trl_cpu", "run_trl_local"]
