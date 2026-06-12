"""mindxtrain CLI — Typer entry point for all 8 verbs."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from mindxtrain import __version__
from mindxtrain.autotune.benchmark import run_autotune
from mindxtrain.autotune.plan import AutotunePlan
from mindxtrain.config.loader import list_recipes, load_config, render_recipe

app = typer.Typer(
    name="mindxtrain",
    help="mindxtrain: 60s AOT autotune + multi-backend training + Quark FP8 quantize for MI300X.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(name="dataset", help="Dataset preparation subcommands.", no_args_is_help=True)
github_app = typer.Typer(name="github", help="GitHub source-tree publishing.", no_args_is_help=True)
droplet_app = typer.Typer(name="droplet", help="AMD Dev Cloud MI300X provision + sync.", no_args_is_help=True)
mei_app = typer.Typer(
    name="mei",
    help="mindX Efficiency Index — score, history, promotion checks.",
    no_args_is_help=True,
)
app.add_typer(dataset_app)
app.add_typer(github_app)
app.add_typer(droplet_app)
app.add_typer(mei_app)
console = Console()


def _version_cb(value: bool) -> None:
    if value:
        console.print(f"mindxtrain {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_cb,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """mindxtrain entry point."""


# ---- init / bench ---------------------------------------------------------


@app.command()
def init(
    template: str = typer.Option(
        "qwen3_8b_sft_lora",
        "--template",
        "-t",
        help="recipe name (run `mindxtrain init --list` to see all)",
    ),
    out: Path = typer.Option(
        Path("run.yaml"),
        "--out",
        "-o",
        help="output YAML path",
    ),
    list_only: bool = typer.Option(
        False,
        "--list",
        help="list all built-in recipe names and exit",
    ),
) -> None:
    """Write a starter YAML config from a built-in recipe."""
    if list_only:
        for name in list_recipes():
            console.print(f"  {name}")
        raise typer.Exit
    yaml_text = render_recipe(template)
    out.write_text(yaml_text)
    console.print(f"[green]wrote[/green] {out} ({len(yaml_text)} bytes, recipe={template!r})")


@app.command()
def bench(
    out: Path = typer.Option(Path("autotune_plan.json"), "--out", "-o"),
    gpu: int = typer.Option(0, "--gpu", help="HIP/ROCm device index"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Skip GPU probes; emit a hardcoded reference plan.",
    ),
) -> None:
    """Run the 60-second AOT autotune probe and write autotune_plan.json."""
    plan: AutotunePlan = run_autotune(gpu_index=gpu, dry_run=dry_run)
    out.write_text(plan.model_dump_json(indent=2))
    console.print(
        f"[green]wrote[/green] {out} (dry_run={dry_run}, "
        f"attention={plan.attention_backend}, gemm={plan.gemm_heuristic})",
    )


# ---- train / eval / quantize / serve --------------------------------------


def _load_plan(plan_path: Path | None) -> AutotunePlan:
    if plan_path and plan_path.exists():
        return AutotunePlan.model_validate_json(plan_path.read_text())
    return run_autotune(gpu_index=0, dry_run=True)


@app.command()
def train(
    config: Path = typer.Argument(..., help="path to XTrainConfig YAML"),
    plan_path: Path = typer.Option(None, "--plan", help="autotune plan JSON; falls back to dry-run."),
    out: Path = typer.Option(Path("./out/runs"), "--out", "-o", help="run output root"),
    cpu_percent: int | None = typer.Option(
        None, "--cpu-percent",
        help=(
            "Override `train.cpu_throttle.percent` at runtime. Applies "
            "only to the trl_cpu backend. 1-100; below 1 or above 100 errors."
        ),
    ),
    cpu_nice: int | None = typer.Option(
        None, "--cpu-nice",
        help="Override `train.cpu_throttle.nice_level`. -20..19.",
    ),
) -> None:
    """Dispatch a training run via the configured backend.

    With --cpu-percent N, the trl_cpu backend caps every thread pool
    (torch, OpenMP, MKL, OpenBLAS) at N% of the host's cores. Useful for
    leaving cycles free for the rest of the laptop while training runs in
    the background.
    """
    from mindxtrain.config.schema import CPUThrottleCfg
    from mindxtrain.train import dispatch_training

    cfg = load_config(config)
    # CLI override: rebuild train.cpu_throttle if either knob was passed.
    if cpu_percent is not None or cpu_nice is not None:
        throttle = cfg.train.cpu_throttle
        new_throttle = CPUThrottleCfg(
            percent=cpu_percent if cpu_percent is not None else throttle.percent,
            nice_level=cpu_nice if cpu_nice is not None else throttle.nice_level,
            omp_proc_bind=throttle.omp_proc_bind,
        )
        # Pydantic frozen=True forbids in-place mutation; rebuild via model_copy.
        new_train = cfg.train.model_copy(update={"cpu_throttle": new_throttle})
        cfg = cfg.model_copy(update={"train": new_train})
        console.print(
            f"[dim]cpu_throttle overridden: percent={new_throttle.percent} "
            f"nice={new_throttle.nice_level}[/dim]",
        )

    plan = _load_plan(plan_path)
    run_dir = out / cfg.meta.run_name
    try:
        ckpt = dispatch_training(cfg, plan, run_dir)
    except RuntimeError as exc:
        console.print(f"[red]training failed:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    console.print(f"[green]checkpoint:[/green] {ckpt}")


@app.command(name="eval")
def eval_(
    config: Path = typer.Argument(...),
    checkpoint: Path = typer.Option(None, "--checkpoint", "-c", help="checkpoint dir; default = ./out/runs/<run_name>/checkpoint"),
) -> None:
    """Run lm-eval-harness against a checkpoint."""
    from mindxtrain.eval.harness import parse_summary, run_lm_eval

    cfg = load_config(config)
    ckpt = checkpoint or Path("./out/runs") / cfg.meta.run_name / "checkpoint"
    if not ckpt.exists():
        console.print(f"[red]checkpoint not found:[/red] {ckpt}")
        raise typer.Exit(code=1)
    tasks = list(cfg.eval.harness.tasks) if cfg.eval and cfg.eval.harness else ["mmlu"]
    try:
        results = run_lm_eval(ckpt, tasks)
    except RuntimeError as exc:
        console.print(f"[red]eval failed:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    console.print(f"[green]results:[/green] {results}")
    console.print_json(data=parse_summary(results))


@app.command(name="eval-checkpoint")
def eval_checkpoint(
    config: Path = typer.Argument(...),
    checkpoint: Path = typer.Option(
        None, "--checkpoint", "-c",
        help="LoRA adapter dir; default = ./out/runs/<run_name>/checkpoint",
    ),
    jsonl: Path = typer.Option(
        None, "--jsonl",
        help=(
            "Path to a *_training.jsonl held-out file. Defaults to picking "
            "the newest one under `data.path/ltm/**/*_training.jsonl` "
            "(works for source='mindx_dreams')."
        ),
    ),
    max_samples: int = typer.Option(
        32, "--max-samples", "-n",
        help="Cap on how many rows to evaluate. Held-out CE is averaged.",
    ),
) -> None:
    """Compare base-model vs base+adapter cross-entropy on held-out chat rows.

    The training trainer_state.json gives a train-loss curve but doesn't
    answer whether the adapter generalises — this verb does. Prints
    `{base_loss, adapter_loss, delta}` where `delta < 0` means the
    adapter is actually moving the model toward the held-out dreams.

    The default jsonl picker walks the recipe's `data.path` for the
    newest `*_training.jsonl` — for mindX dreams this is the corpus the
    recipe trained on. For a rigorous held-out check, point `--jsonl` at
    a file the training run never saw.
    """
    from mindxtrain.eval.held_out_loss import score_checkpoint

    cfg = load_config(config)
    ckpt = checkpoint or Path("./out/runs") / cfg.meta.run_name / "checkpoint"
    if not ckpt.exists():
        console.print(f"[red]adapter dir not found:[/red] {ckpt}")
        raise typer.Exit(code=1)

    jsonl_path = jsonl
    if jsonl_path is None:
        if cfg.data.source != "mindx_dreams" or cfg.data.path is None:
            console.print(
                "[red]--jsonl required when data.source != mindx_dreams "
                "(no default picker for non-dream sources).",
            )
            raise typer.Exit(code=2)
        candidates = sorted(
            Path(cfg.data.path).glob("ltm/**/*_training.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            console.print(f"[red]no *_training.jsonl under {cfg.data.path}/ltm")
            raise typer.Exit(code=2)
        jsonl_path = candidates[0]
        console.print(f"[dim]using newest dream file: {jsonl_path}[/dim]")

    try:
        score = score_checkpoint(
            adapter_dir=ckpt,
            base_model=cfg.model.name,
            jsonl_path=jsonl_path,
            max_samples=max_samples,
            sink=lambda line: console.print(line),
        )
    except (ImportError, ValueError) as exc:
        console.print(f"[red]eval-checkpoint failed:[/red] {exc}")
        raise typer.Exit(code=3) from exc

    verdict = (
        "[green]adapter improved[/green]" if score.delta < 0
        else "[yellow]adapter regressed[/yellow]"
    )
    console.print(
        f"\n{verdict} on {score.n} held-out rows: "
        f"base={score.base_loss:.4f} adapter={score.adapter_loss:.4f} "
        f"delta={score.delta:+.4f}",
    )
    console.print_json(data=score.as_dict())


@app.command()
def quantize(
    config: Path = typer.Argument(...),
    checkpoint: Path = typer.Option(None, "--checkpoint", "-c"),
) -> None:
    """Quark FP8 / MXFP4 quantize the trained checkpoint."""
    from mindxtrain.deploy.quark import quark_fp8, quark_mxfp4

    cfg = load_config(config)
    ckpt = checkpoint or Path("./out/runs") / cfg.meta.run_name / "checkpoint"
    if not ckpt.exists():
        console.print(f"[red]checkpoint not found:[/red] {ckpt}")
        raise typer.Exit(code=1)
    out_dir = ckpt.parent / "quantized"
    fn = quark_fp8 if cfg.quantize.scheme == "fp8_e4m3" else quark_mxfp4
    try:
        path = fn(ckpt, out_dir)
    except RuntimeError as exc:
        console.print(f"[red]quantize failed:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    console.print(f"[green]quantized:[/green] {path}")


@app.command()
def serve(
    config: Path = typer.Argument(...),
    checkpoint: Path = typer.Option(None, "--checkpoint", "-c"),
    to: str = typer.Option(
        "vllm", "--to",
        help="Serve target: vllm (default, builds vllm-rocm launch cmd), "
             "sglang (builds sglang-rocm launch cmd), or "
             "ollama (merges LoRA + calls `ollama create`).",
    ),
    tag: str = typer.Option(
        None, "--tag",
        help="Ollama tag for `--to ollama`. Defaults to run_name when omitted.",
    ),
    ollama_bin: str = typer.Option(
        None, "--ollama-bin",
        help="Override the ollama binary path (defaults to PATH lookup).",
    ),
    register_as_fallback: bool = typer.Option(
        False, "--register-as-fallback",
        help=(
            "After --to ollama succeeds, PATCH the new tag into mindX as "
            "the local-fallback model (PATCH /v1/config/fallback-model). "
            "Best-effort — a failure logs but does NOT fail the push."
        ),
    ),
    mindx_base_url: str = typer.Option(
        None, "--mindx-base-url",
        help=(
            "Override the mindX base URL for --register-as-fallback. "
            "Defaults to MINDXTRAIN_API_BASE_URL env or "
            "https://mindx.pythai.net."
        ),
    ),
) -> None:
    """Serve the trained checkpoint locally.

    `--to vllm` (default) prints a vllm-rocm launch command against the
    quantized checkpoint — wire it into your orchestrator.

    `--to ollama` runs the local-learning loop: merges the LoRA adapter
    into the base weights, writes an ollama Modelfile, and calls
    `ollama create <tag>` so the trained model is immediately available
    on the loopback (the same backend Coach probes for its chat card).
    """
    cfg = load_config(config)

    if to == "ollama":
        from mindxtrain.deploy.ollama_push import push_to_ollama

        # The LoRA adapter is at <run_dir>/checkpoint/ — same location
        # the trl_cpu / axolotl backends save to.
        adapter_dir = checkpoint or Path("./out/runs") / cfg.meta.run_name / "checkpoint"
        if not adapter_dir.exists():
            console.print(f"[red]checkpoint not found:[/red] {adapter_dir}")
            raise typer.Exit(code=1)

        resolved_tag = tag or cfg.meta.run_name
        try:
            result = push_to_ollama(
                base_model=cfg.model.name,
                adapter_dir=adapter_dir,
                tag=resolved_tag,
                sink=lambda line: console.print(line),
                ollama_bin=ollama_bin,
                register_with_mindx=register_as_fallback,
                mindx_base_url=mindx_base_url,
            )
        except (FileNotFoundError, ImportError) as exc:
            console.print(f"[red]push-to-ollama failed:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        console.print(
            f"[green]pushed:[/green] {result.tag} "
            f"(merged: {result.merged_dir}, Modelfile: {result.modelfile})",
        )
        if result.mindx_fallback_swap:
            console.print(
                f"[green]mindX fallback swapped:[/green] "
                f"{result.mindx_fallback_swap.get('previous', '?')} -> "
                f"{result.mindx_fallback_swap.get('current', '?')}",
            )
        return

    if to not in ("vllm", "sglang"):
        console.print(f"[red]unknown serve target:[/red] {to}")
        raise typer.Exit(code=2)

    ckpt = checkpoint or Path("./out/runs") / cfg.meta.run_name / "quantized"
    if not ckpt.exists():
        console.print(f"[red]quantized checkpoint not found:[/red] {ckpt}")
        raise typer.Exit(code=1)

    if to == "sglang":
        from mindxtrain.deploy.sglang_rocm import build_sglang_command

        cmd = build_sglang_command(cfg.serve, ckpt)
        console.print(f"[green]sglang cmd:[/green] {' '.join(cmd)}")
        return

    from mindxtrain.deploy.vllm_launcher import build_vllm_command

    cmd = build_vllm_command(cfg.serve, ckpt, cfg.quantize)
    console.print(f"[green]vllm cmd:[/green] {' '.join(cmd)}")
    # Caller can pipe the cmd into their orchestrator; we don't exec by default.


# ---- dataset prep ---------------------------------------------------------


@dataset_app.command("prep")
def dataset_prep(
    config: Path = typer.Argument(..., help="path to XTrainConfig YAML"),
    out: Path = typer.Option(Path("./out/dataset"), "--out", "-o"),
) -> None:
    """Run the dataset pipeline: curate -> filter -> tokenize -> pack -> shard."""
    from mindxtrain.data.curate import load_streaming_dataset
    from mindxtrain.data.filter import quality_filter
    from mindxtrain.data.pack import emit_shards, pack_sequences
    from mindxtrain.data.tokenize import tokenize_stream

    cfg = load_config(config)
    out.mkdir(parents=True, exist_ok=True)
    try:
        rows = load_streaming_dataset(cfg.data)
        texts = (row.get("text") or row.get("content") or "" for row in rows)
        clean = quality_filter(texts)
        tokenized = tokenize_stream(clean, cfg.model.name)
        packed = pack_sequences(tokenized, cfg.data.seq_len)
        shard_dir = emit_shards(packed, out)
    except RuntimeError as exc:
        console.print(f"[red]dataset prep failed:[/red] {exc}")
        raise typer.Exit(code=3) from exc
    console.print(f"[green]shards:[/green] {shard_dir}")


# ---- publish / receipt ----------------------------------------------------


@app.command()
def publish(
    config: Path = typer.Argument(...),
    manifest: Path = typer.Option(..., "--manifest", "-m", help="path to provenance manifest.json"),
    skip_hf: bool = typer.Option(False, "--skip-hf"),
    skip_pin: bool = typer.Option(False, "--skip-pin"),
    force: bool = typer.Option(
        False, "--force",
        help="Skip the MEI promotion gate. The manifest records promotion_bypassed=true.",
    ),
) -> None:
    """Push to HF Hub + Lighthouse + register the provenance manifest with the mindX API.

    By default this verb consults the historical MEI ledger: if there's a
    score for this run_id and it doesn't pass the §8 promotion gates, the
    push is refused with the failing-gate reasons surfaced. `--force`
    skips the gate (records `promotion_bypassed=true` in the manifest).
    """
    from mindxtrain.deploy.api_client import register_with_mindx
    from mindxtrain.eval.mei import history as _mei_history
    from mindxtrain.eval.mei.score import is_promotable
    from mindxtrain.provenance.manifest import Manifest
    from mindxtrain.storage.hf_hub import publish_to_hf
    from mindxtrain.storage.lighthouse import publish_to_lighthouse

    cfg = load_config(config)
    m = Manifest.model_validate_json(manifest.read_text())
    ckpt_dir = Path("./out/runs") / cfg.meta.run_name / "checkpoint"

    # MEI promotion gate. Skip silently when there's no MEI score yet —
    # the gate is informational, not mandatory at intake (so existing
    # publish flows pre-MEI continue to work). With --force, we proceed
    # regardless and stamp the manifest so the bypass is auditable.
    mei_entries = [e for e in _mei_history.read_all() if e.run_id == m.run_id]
    if mei_entries:
        latest = mei_entries[-1]
        prior = _mei_history.currently_promoted()
        prior_score = (
            prior.score if prior is not None and prior.run_id != m.run_id else None
        )
        ok, reasons = is_promotable(latest.score, prior_promoted=prior_score)
        if ok:
            console.print(
                f"[green]MEI gate:[/green] {latest.score.composite:.3f} ≥ 0.55, "
                "all sub-indices ≥ 0.30 — promotable.",
            )
        elif force:
            console.print(
                "[yellow]MEI gate failed but --force given; "
                "marking promotion_bypassed=true in manifest:[/yellow]",
            )
            for reason in reasons:
                console.print(f"  • {reason}")
            m.promotion_bypassed = True
            m.promotion_bypass_reasons = reasons
        else:
            console.print("[red]MEI gate refused promotion:[/red]")
            for reason in reasons:
                console.print(f"  • {reason}")
            console.print(
                "Pass --force to publish anyway (the bypass is recorded "
                "in the manifest).",
            )
            raise typer.Exit(code=4)
    elif force:
        console.print(
            "[yellow]No MEI score on file; --force given. "
            "Recommend running `mindxtrain mei score <record.json>` first.[/yellow]",
        )

    hf_url = ""
    if not skip_hf and ckpt_dir.exists():
        try:
            hf_url = publish_to_hf(ckpt_dir, f"{cfg.meta.run_name}", private=False)
            m.hf_repo_id = hf_url
            console.print(f"[green]HF:[/green] {hf_url}")
        except RuntimeError as exc:
            console.print(f"[yellow]hf upload skipped:[/yellow] {exc}")

    cid = ""
    if not skip_pin and ckpt_dir.exists():
        try:
            cid = publish_to_lighthouse(ckpt_dir)
            m.lighthouse_cid = cid
            console.print(f"[green]Lighthouse:[/green] {cid}")
        except RuntimeError as exc:
            console.print(f"[yellow]lighthouse pin skipped:[/yellow] {exc}")

    try:
        receipt = register_with_mindx(run_id=m.run_id, hf_url=hf_url, cid=cid)
        console.print(f"[green]mindX:[/green] {receipt}")
    except (RuntimeError, Exception) as exc:
        console.print(f"[yellow]mindX register skipped:[/yellow] {exc}")

    manifest.write_text(m.model_dump_json(indent=2))
    console.print(f"[green]updated manifest:[/green] {manifest}")


@app.command()
def receipt(
    manifest: Path = typer.Argument(..., help="path to provenance manifest.json"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    """Verify a provenance manifest's BLAKE3 hashes against on-disk artifacts."""
    from mindxtrain.provenance.manifest import Manifest
    from mindxtrain.provenance.verify import verify_receipt

    if not manifest.is_file():
        console.print(f"[red]manifest not found:[/red] {manifest}")
        raise typer.Exit(code=1)
    m = Manifest.model_validate_json(manifest.read_text())
    console.print_json(data={"run_id": m.run_id, "blake3": m.blake3.model_dump()})

    if config is None:
        return

    cfg = load_config(config)
    run_dir = Path("./out/runs") / cfg.meta.run_name

    # A run-emitted manifest snapshots the validated config to
    # config.snapshot.yaml and persists the exact AutotunePlan bytes it hashed.
    # Prefer those when present; fall back to the user-supplied --config for
    # legacy manifests produced by `emit_receipt`.
    snapshot = run_dir / "config.snapshot.yaml"
    config_yaml_path = snapshot if snapshot.is_file() else config
    plan_path = run_dir / "autotune_plan.json"
    plan_json = plan_path.read_bytes() if plan_path.is_file() else None

    try:
        result = verify_receipt(
            m,
            config_yaml_path=config_yaml_path,
            dataset_manifest_path=run_dir / "dataset_manifest.json",
            checkpoint_dir=run_dir / "checkpoint",
            eval_json_path=run_dir / "eval/lm_eval.json",
            plan_json=plan_json,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]missing artifact:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(data=result)
    if not all(result.values()):
        raise typer.Exit(code=2)


@app.command()
def imprint(
    config: Path = typer.Argument(..., help="recipe whose checkpoint to measure"),
    out: Path = typer.Option(Path("./out/runs"), "--out", "-o"),
    max_inquiries: int = typer.Option(5, "--n", help="number of recall probes"),
    trigger_dream: bool = typer.Option(
        False, "--trigger-dream",
        help="hand the imprinted actor to mindX's machine.dream 8hr cycle",
    ),
) -> None:
    """Measure a persona imprint: recall before vs after training.

    Poses the script's own user-turns back to the actor, comparing the base
    model (before) and the trained adapter (after) against the script's
    assistant voice. Prints an ImprintReport; exit 4 if no imprint was detected.
    """
    import json as _json

    cfg = load_config(config)
    run_dir = (out / cfg.meta.run_name) if out.name == "runs" else out
    adapter_dir = run_dir / "checkpoint"
    if not adapter_dir.exists():
        console.print(f"[red]no checkpoint to measure:[/red] {adapter_dir}")
        raise typer.Exit(code=1)

    # Build inquiries (user-turns) + baseline voice (assistant-turns) from the
    # local script the actor trained on. Falls back to default probes.
    from mindxtrain.eval.imprint import default_inquiries, probe_recall, score_imprint

    inquiries: list[str] = []
    baseline: list[str] = []
    path = cfg.data.path
    if path is not None and Path(path).exists():
        files = [Path(path)] if Path(path).is_file() else sorted(Path(path).rglob("*.jsonl"))
        for f in files:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                msgs = row.get("messages", [])
                u = next((m["content"] for m in msgs if m.get("role") == "user"), None)
                a = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
                if u and len(inquiries) < max_inquiries:
                    inquiries.append(u)
                if a:
                    baseline.append(a)
    if not inquiries:
        inquiries = default_inquiries(cfg.meta.project)[:max_inquiries]

    console.print(f"[cyan]probing {len(inquiries)} inquiries (before/after)…[/cyan]")
    try:
        before = probe_recall(cfg.model.name, inquiries, force_cpu=True)
        after = probe_recall(cfg.model.name, inquiries, adapter_dir=adapter_dir, force_cpu=True)
    except RuntimeError as exc:
        console.print(f"[red]imprint probe failed:[/red] {exc}")
        raise typer.Exit(code=3) from exc

    report = score_imprint(inquiries, before, after, baseline or before)
    console.print_json(data=report.model_dump())

    if trigger_dream:
        from mindxtrain.deploy.api_client import trigger_dream_ingestion

        res = trigger_dream_ingestion(
            run_id=cfg.meta.run_name,
            adapter_dir=str(adapter_dir),
            base_model=cfg.model.name,
            persona_name=cfg.meta.project,
            imprint_delta=report.imprint_delta,
        )
        console.print(f"[green]dream trigger:[/green] {res}")

    if not report.imprinted:
        console.print("[yellow]no imprint detected (delta<=0 or no shift)[/yellow]")
        raise typer.Exit(code=4)


# ---- github / droplet (source-tree publishing + remote provision) -------


@github_app.command("push")
def github_push_cmd(
    commit_message: str = typer.Option(
        "mindXtrain initial push", "--message", "-m", help="commit message"
    ),
    force: bool = typer.Option(False, "--force", help="use --force-with-lease on push"),
) -> None:
    """Bootstrap a git repo, create the GitHub remote (via `gh`), push the working tree.

    Requires GITHUB_TOKEN + GITHUB_REPO in the environment. Reuses the same
    builders as the Coach UI's "Push to GitHub" button — output is local-shell
    rather than SSE-streamed.
    """
    import os
    import subprocess

    from mindxtrain.deploy.github_push import GithubConfig, bootstrap_steps, status_missing

    missing = status_missing()
    if missing:
        console.print(f"[red]missing:[/red] {', '.join(missing)}")
        console.print("[yellow]hint:[/yellow] set GITHUB_TOKEN and GITHUB_REPO, install gh + git")
        raise typer.Exit(code=2)

    cfg = GithubConfig(
        token=os.environ["GITHUB_TOKEN"],
        repo=os.environ["GITHUB_REPO"],
        branch=os.environ.get("GITHUB_DEFAULT_BRANCH", "main"),
        author_name=os.environ.get("GITHUB_AUTHOR_NAME", "mindXtrain bot"),
        author_email=os.environ.get("GITHUB_AUTHOR_EMAIL", "noreply@pythai.net"),
    )
    rcs: dict[str, int] = {}
    for step in bootstrap_steps(cfg, commit_message=commit_message, force=force):
        if step.predicate_step is not None:
            gate = rcs.get(step.predicate_step)
            if gate is None or gate not in step.predicate_rc_in:
                console.print(f"[dim]skip[/dim] {step.label}")
                rcs[step.label] = -1
                continue
        console.print(f"[cyan]→ {step.label}[/cyan]: {' '.join(step.cmd[:6])}…")
        proc = subprocess.run(step.cmd, env=step.env or None, check=False)
        rcs[step.label] = proc.returncode
        if proc.returncode != 0 and not step.allow_failure:
            console.print(f"[red]{step.label} failed (rc={proc.returncode}); aborting[/red]")
            raise typer.Exit(code=3)
    console.print("[green]push complete[/green]")


@droplet_app.command("provision")
def droplet_provision_cmd(
    name: str = typer.Option("mindxtrain", "--name"),
    repo: str = typer.Option(None, "--repo", help="defaults to $GITHUB_REPO"),
    branch: str = typer.Option(None, "--branch", help="defaults to $GITHUB_DEFAULT_BRANCH or 'main'"),
    container: str = typer.Option(None, "--container", help="defaults to $DROPLET_CONTAINER"),
    extras: str = typer.Option("ml,eval,data,obs", "--extras"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="poll for cloud-init bootstrap completion"),
) -> None:
    """POST a new MI300X droplet to AMD Dev Cloud + wait for cloud-init bootstrap.

    Requires AMD_DEV_CLOUD_TOKEN + AMD_DEV_CLOUD_SSH_KEY_ID. The droplet's
    `user_data` clones from GitHub and runs `mindxtrain bench` as it boots, so
    by the time SSH is reachable the autotune plan is on disk.
    """
    import os
    import time

    from mindxtrain.deploy import amd_dev_cloud as adc
    from mindxtrain.deploy.cloud_init import render

    missing = adc.missing_env()
    if missing:
        console.print(f"[red]missing:[/red] {', '.join(missing)}")
        raise typer.Exit(code=2)
    cloud_cfg = adc.from_env()
    user_data = render(
        repo=repo or os.environ.get("GITHUB_REPO", "professor-codephreak/mindXtrain"),
        branch=branch or os.environ.get("GITHUB_DEFAULT_BRANCH", "main"),
        container=container or os.environ.get("DROPLET_CONTAINER", "rocm/primus:v26.2"),
        extras=extras,
    )

    log = console.print
    with adc.AmdDevCloudClient(cloud_cfg) as client:
        droplet = client.create(name=name, user_data=user_data, log=lambda line: log(f"[cyan]{line}[/cyan]"))
        droplet_id = int(droplet["id"])
        if not wait:
            console.print(f"[green]droplet_id={droplet_id}[/green] — exiting before bootstrap (--no-wait)")
            return
        droplet = client.poll_until_active(
            droplet_id, log=lambda line: log(f"[dim]{line}[/dim]"), sleep=time.sleep, now=time.monotonic
        )
        ip = adc.extract_public_ip(droplet) or ""
        console.print(f"[green]droplet_id={droplet_id} public_ip={ip}[/green]")


@droplet_app.command("sync")
def droplet_sync_cmd(
    no_bench: bool = typer.Option(False, "--no-bench", help="rsync + provision only, skip bench"),
    no_fetch: bool = typer.Option(False, "--no-fetch", help="don't scp plan.json back"),
) -> None:
    """Rsync the working tree to $DROPLET_HOST + run bench inside rocm/primus.

    Requires DROPLET_HOST + DROPLET_USER. Reuses the same builders as the
    Coach UI's "Sync to existing droplet" button — output is local-shell.
    """
    import subprocess

    from mindxtrain.deploy.droplet import from_env, status_missing, sync_steps

    missing = status_missing()
    if missing:
        console.print(f"[red]missing:[/red] {', '.join(missing)}")
        raise typer.Exit(code=2)
    cfg = from_env()
    plan_dest = Path("./out/plan.remote.json")
    plan_dest.parent.mkdir(parents=True, exist_ok=True)
    for step in sync_steps(
        cfg,
        repo_root=Path.cwd(),
        run_bench=not no_bench,
        fetch_plan=not no_fetch,
        plan_dest=plan_dest,
    ):
        console.print(f"[cyan]→ {step.label}[/cyan]")
        proc = subprocess.run(step.cmd, env=step.env or None, check=False)
        if proc.returncode != 0 and not step.allow_failure:
            console.print(f"[red]{step.label} failed (rc={proc.returncode})[/red]")
            raise typer.Exit(code=3)
    console.print("[green]sync complete[/green]")


# ---- mei verbs --------------------------------------------------------------


@mei_app.command("score")
def mei_score(
    record: Path = typer.Argument(
        ..., help="Path to a JSON MEIRecord file (output of the measurement orchestrator).",
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Optional path to write the MEIScore JSON. Defaults to stdout.",
    ),
    append_history: bool = typer.Option(
        True, "--history/--no-history",
        help="Append the score to the historical-comparison ledger.",
    ),
) -> None:
    """Score a MEIRecord against the v0.1 anchors. Prints MEIScore JSON.

    The record JSON must conform to `mindxtrain.eval.mei.record.MEIRecord`.
    Generate one via the measurement orchestrator (Phase 1.4) or hand-craft
    against the schema for demos.
    """

    from mindxtrain.eval.mei.history import append as _hist_append
    from mindxtrain.eval.mei.record import MEIRecord
    from mindxtrain.eval.mei.score import score_record

    rec = MEIRecord.model_validate_json(record.read_text())
    sc = score_record(rec)
    out_text = sc.model_dump_json(indent=2)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(out_text + "\n")
        console.print(f"[green]wrote[/green] {out}")
    else:
        console.print(out_text)

    if append_history:
        path = _hist_append(
            sc,
            run_id=rec.model_id,
            model_id=rec.model_id,
            model_sha256=rec.model_sha256,
            promoted=False,
        )
        console.print(f"[dim]history appended → {path}[/dim]")

    # Composite headline for terminal-friendly reading.
    console.print(
        f"[bold]MEI[/bold] = [bold cyan]{sc.composite:.3f}[/bold cyan]  "
        f"Q={sc.quality:.3f} Dt={sc.decode_throughput:.3f} "
        f"Pp={sc.prefill_throughput:.3f} M={sc.memory:.3f} E={sc.energy:.3f}"
        + ("  [yellow](provisional Agentic)[/yellow]" if sc.mab_provisional else ""),
    )
    # Promotion preview (against the current ledger).
    from mindxtrain.eval.mei.history import currently_promoted
    from mindxtrain.eval.mei.score import is_promotable
    prior = currently_promoted()
    prior_score = prior.score if prior is not None else None
    ok, reasons = is_promotable(sc, prior_promoted=prior_score)
    if ok:
        console.print("[green]✓ promotable[/green] — eligible for AgenticPlace.")
    else:
        console.print("[yellow]✗ not promotable[/yellow]:")
        for r in reasons:
            console.print(f"  • {r}")


@mei_app.command("history")
def mei_history(
    last: int = typer.Option(10, "--last", "-n", help="Show the last N entries."),
    promoted_only: bool = typer.Option(
        False, "--promoted-only", help="Filter to entries promoted to AgenticPlace.",
    ),
) -> None:
    """List recent MEI scores from the historical ledger."""
    from mindxtrain.eval.mei.history import read_all

    rows = read_all()
    if promoted_only:
        rows = [r for r in rows if r.promoted]
    rows = rows[-last:] if last > 0 else rows
    if not rows:
        console.print("[dim](no MEI history yet — run `mindxtrain mei score …`)[/dim]")
        return
    for r in rows:
        mark = "[green]★[/green]" if r.promoted else "·"
        flag = " [yellow](prov)[/yellow]" if r.score.mab_provisional else ""
        console.print(
            f"{mark} {r.timestamp}  {r.model_id}  "
            f"MEI={r.score.composite:.3f}{flag}",
        )


if __name__ == "__main__":
    app()
