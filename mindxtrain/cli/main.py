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
) -> None:
    """Boot vLLM-ROCm against the quantized checkpoint."""
    from mindxtrain.deploy.vllm_launcher import build_vllm_command

    cfg = load_config(config)
    ckpt = checkpoint or Path("./out/runs") / cfg.meta.run_name / "quantized"
    if not ckpt.exists():
        console.print(f"[red]quantized checkpoint not found:[/red] {ckpt}")
        raise typer.Exit(code=1)
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
    try:
        result = verify_receipt(
            m,
            config_yaml_path=config,
            dataset_manifest_path=run_dir / "dataset_manifest.json",
            checkpoint_dir=run_dir / "checkpoint",
            eval_json_path=run_dir / "eval/lm_eval.json",
        )
    except FileNotFoundError as exc:
        console.print(f"[red]missing artifact:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(data=result)
    if not all(result.values()):
        raise typer.Exit(code=2)


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
