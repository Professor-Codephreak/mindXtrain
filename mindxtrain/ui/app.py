"""The mindXtrain UI — the whole framework as one Gradio surface.

One idea runs through it: **complexity is a dial, not a wall.** Every room has the same three
tiers, chosen once at the top and remembered:

- **Basic** — pick a recipe, press start, watch it train, read the verdict.
- **Advanced** — the knobs an operator actually turns: LoRA shape, schedule, batch, throttle,
  packing, eval split, the gate's floor, where it publishes.
- **Scientific** — the run as an experiment: every metric the trainer emits with its units and
  where it came from, the eval harness, the autotune plan, the imprint's before/after with its
  null, provenance hashes, and the receipt.

Nothing here re-implements training. Every action shells out to the real CLI (`mindxtrain …`) and
every number is parsed from what the trainer actually wrote — the log is the source of truth, so the
UI can never claim a step that did not happen.

    mindxtrain ui                      # http://127.0.0.1:7862
    mindxtrain ui --share              # a public gradio.live link
    python -m mindxtrain.ui.app
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .metrics import RunMetrics, parse_log  # noqa: F401  (parse_log re-exported for tests)
from .theme import CSS, theme

VERSION = "1.0.0"
HOME = Path(os.environ.get("MINDXTRAIN_HOME") or Path(__file__).resolve().parents[2])
RECIPES = HOME / "mindxtrain" / "train" / "recipes"
TIERS = ["Basic", "Advanced", "Scientific"]


# ── running the real CLI ──────────────────────────────────────────────────────
def cli_prefix() -> List[str]:
    """`uv run --project <home> mindxtrain` when uv is how this checkout runs, else `mindxtrain`."""
    if (HOME / "pyproject.toml").is_file() and _which("uv"):
        return ["uv", "run", "--project", str(HOME), "mindxtrain"]
    return ["mindxtrain"]


def _which(prog: str) -> Optional[str]:
    from shutil import which
    return which(prog)


class Job:
    """One CLI invocation, streamed to a log file so the UI can follow it and survive a reload."""

    def __init__(self, args: List[str], log: Path, cwd: Optional[Path] = None):
        self.args, self.log, self.cwd = args, log, cwd or HOME
        self.proc: Optional[subprocess.Popen] = None
        self.started = self.ended = None

    def start(self) -> Dict[str, Any]:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        fh = self.log.open("w", encoding="utf-8", errors="replace")
        self.started = time.time()
        try:
            self.proc = subprocess.Popen(self.args, cwd=str(self.cwd), stdout=fh, stderr=subprocess.STDOUT,
                                         text=True, start_new_session=True)
        except Exception as e:  # noqa: BLE001
            fh.write(f"[ui] failed to start: {type(e).__name__}: {e}\n"); fh.close()
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        return {"ok": True, "pid": self.proc.pid, "cmd": " ".join(shlex.quote(a) for a in self.args), "log": str(self.log)}

    @property
    def running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def stop(self) -> Dict[str, Any]:
        if not self.running:
            return {"ok": False, "reason": "not running"}
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            self.proc.terminate()
        return {"ok": True, "stopped": self.proc.pid}


JOBS: Dict[str, Job] = {}
RUNS = HOME / "out" / "ui"


def launch(kind: str, args: List[str]) -> Dict[str, Any]:
    if JOBS.get(kind) and JOBS[kind].running:
        return {"ok": False, "reason": f"a {kind} job is already running (pid {JOBS[kind].proc.pid})"}
    log = RUNS / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    j = Job(cli_prefix() + args, log)
    r = j.start()
    if r.get("ok"):
        JOBS[kind] = j
    return r


def run_sync(args: List[str], timeout: float = 120) -> Tuple[int, str]:
    try:
        p = subprocess.run(cli_prefix() + args, cwd=str(HOME), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


# ── recipes ───────────────────────────────────────────────────────────────────
def recipe_names() -> List[str]:
    return sorted(p.stem for p in RECIPES.glob("*.yaml")) if RECIPES.is_dir() else []


def read_recipe(name: str) -> str:
    p = RECIPES / f"{name}.yaml"
    return p.read_text() if p.is_file() else f"# no recipe named {name}"


def recipe_summary(name: str) -> str:
    """The five numbers that decide what a run costs, pulled from the recipe itself."""
    try:
        import yaml
        c = yaml.safe_load(read_recipe(name)) or {}
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"
    m, d, t = c.get("model") or {}, c.get("data") or {}, c.get("train") or {}
    meth, sch, bat = t.get("method") or {}, t.get("schedule") or {}, t.get("batch") or {}
    rows = [("base", m.get("name")), ("precision", t.get("precision") or m.get("torch_dtype")),
            ("method", f"{meth.get('kind')} r={meth.get('r')} α={meth.get('alpha')} → {', '.join(meth.get('target_modules') or [])}"),
            ("data", f"{d.get('source')} · seq {d.get('seq_len')} · packing {d.get('packing')} · eval split {d.get('eval_split')}"),
            ("schedule", f"{sch.get('epochs')} epochs · {sch.get('type')} · warmup {sch.get('warmup_ratio')} · lr {(t.get('optimizer') or {}).get('lr')}"),
            ("batch", f"per-device {bat.get('per_device')} × grad-accum {bat.get('grad_accum')}"),
            ("throttle", json.dumps(t.get("cpu_throttle")) if t.get("cpu_throttle") else "—")]
    return "\n".join(f"**{k}** · {v}" for k, v in rows if v)


# ── the surface ───────────────────────────────────────────────────────────────
def build() -> gr.Blocks:
    import inspect
    blocks_takes_theme = "theme" in inspect.signature(gr.Blocks.__init__).parameters
    bk = {"theme": theme(), "css": CSS} if blocks_takes_theme else {}

    def tier_vis(tier: str) -> Tuple[Any, Any]:
        return gr.update(visible=tier in ("Advanced", "Scientific")), gr.update(visible=tier == "Scientific")

    with gr.Blocks(title="mindXtrain", fill_height=True, **bk) as demo:
        gr.HTML(f"<div class='mx-head'><h1>mindXtrain</h1><div class='mx-sub'>the framework as one surface · v{VERSION} · "
                f"<code>{HOME}</code> · every action runs the real CLI, every number is parsed from the run's own log</div></div>")
        tier = gr.Radio(TIERS, value="Basic", label="complexity", info="Basic: press start. Advanced: the knobs. Scientific: the experiment.")

        with gr.Tabs():
            # ── FORGE ──
            with gr.Tab("Forge · train"):
                with gr.Row():
                    recipe = gr.Dropdown(recipe_names(), value=(recipe_names() or [None])[0], label="recipe", scale=2)
                    out_dir = gr.Textbox(value="out/runs", label="output", scale=1)
                    start_btn = gr.Button("start training", variant="primary", scale=1)
                    stop_btn = gr.Button("stop", scale=1)
                summary = gr.Markdown()
                with gr.Group(visible=False) as adv_forge:
                    gr.Markdown("**Advanced** — written into the run config before the trainer sees it.")
                    with gr.Row():
                        lora_r = gr.Slider(1, 128, value=16, step=1, label="LoRA r")
                        lora_a = gr.Slider(1, 256, value=32, step=1, label="LoRA α")
                        epochs = gr.Slider(1, 60, value=2, step=1, label="epochs")
                        lr = gr.Number(value=1e-4, label="learning rate")
                    with gr.Row():
                        seq = gr.Slider(128, 8192, value=1024, step=128, label="sequence length")
                        per_dev = gr.Slider(1, 32, value=1, step=1, label="batch per device")
                        accum = gr.Slider(1, 64, value=8, step=1, label="grad accumulation")
                        packing = gr.Checkbox(value=True, label="packing")
                    with gr.Row():
                        cpu_pct = gr.Slider(5, 100, value=33, step=1, label="CPU %")
                        cpu_nice = gr.Slider(0, 19, value=19, step=1, label="nice")
                        eval_split = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="held-out eval split")
                with gr.Group(visible=False) as sci_forge:
                    gr.Markdown("**Scientific** — the recipe verbatim. What you edit here is what the trainer reads.")
                    recipe_yaml = gr.Code(label="run.yaml", language="yaml", lines=18, interactive=True)
                    with gr.Row():
                        save_as = gr.Textbox(value="run.yaml", label="write to", scale=2)
                        save_btn = gr.Button("write config", scale=1)
                    save_state = gr.Markdown()
                gr.Markdown("### live")
                kiln = gr.HTML()
                with gr.Row():
                    loss_plot = gr.LinePlot(x="step", y="value", color="metric", title="loss · token accuracy · lr (normalised)",
                                            height=260, container=True)
                    metrics_tbl = gr.Dataframe(headers=["metric", "value", "unit", "from"], interactive=False, wrap=True)
                log_box = gr.Code(label="the run's log (tail)", lines=14, interactive=False)
                with gr.Accordion("diagnostics — the host, and what the log is telling you", open=False):
                    with gr.Row():
                        host_md = gr.Markdown()
                        diag_md = gr.Markdown()
                    diag_btn = gr.Button("refresh diagnostics")

                ticker = gr.Timer(value=6, active=False)

            # ── GATE ──
            with gr.Tab("Gate · imprint"):
                gr.Markdown("**The gate.** Recall of the corpus, measured on the frozen base first and the trained model second. "
                            "A positive delta is the only thing that makes a run count — and it proves recall, not identity.")
                with gr.Row():
                    g_cfg = gr.Textbox(value="run.yaml", label="config", scale=2)
                    g_max = gr.Slider(1, 64, value=9, step=1, label="inquiries", scale=1)
                    g_btn = gr.Button("run the gate", variant="primary", scale=1)
                with gr.Group(visible=False) as adv_gate:
                    with gr.Row():
                        g_trigger = gr.Checkbox(value=False, label="trigger a dream first (mindX node)")
                        g_out = gr.Textbox(value="out/imprint", label="output")
                with gr.Group(visible=False) as sci_gate:
                    gr.Markdown("**The null matters.** An untrained random-init adapter imprinted N times is the floor; "
                                "a delta below it is noise. Decoding is greedy with repetition_penalty 1.3 and no_repeat_ngram_size 3 — "
                                "change it and the number stops being comparable.")
                g_out_md = gr.Markdown()
                g_json = gr.Code(label="verdict", language="json", interactive=False)

            # ── MEASURE ──
            with gr.Tab("Measure · eval"):
                with gr.Row():
                    e_cfg = gr.Textbox(value="run.yaml", label="config", scale=2)
                    e_ckpt = gr.Textbox(value="", label="checkpoint (blank = the recipe's)", scale=2)
                    e_btn = gr.Button("eval", variant="primary", scale=1)
                    e_ce_btn = gr.Button("cross-entropy vs base", scale=1)
                with gr.Group(visible=False) as adv_eval:
                    with gr.Row():
                        e_jsonl = gr.Textbox(value="", label="held-out JSONL (for the CE comparison)")
                        e_max = gr.Slider(8, 2048, value=128, step=8, label="max samples")
                with gr.Group(visible=False) as sci_eval:
                    gr.Markdown("`eval` runs lm-eval-harness; `eval-checkpoint` compares **base vs base+adapter cross-entropy** on "
                                "rows the model never trained on. The second is the one that cannot be gamed by memorising the corpus.")
                e_out = gr.Code(label="result", language="json", interactive=False)

            # ── SERVE ──
            with gr.Tab("Serve"):
                with gr.Row():
                    s_cfg = gr.Textbox(value="run.yaml", label="config", scale=2)
                    s_ckpt = gr.Textbox(value="", label="checkpoint", scale=2)
                    s_to = gr.Radio(["ollama", "vllm"], value="ollama", label="to", scale=1)
                    s_tag = gr.Textbox(value="", label="tag", scale=1)
                    s_btn = gr.Button("serve", variant="primary", scale=1)
                with gr.Group(visible=False) as adv_serve:
                    with gr.Row():
                        s_fallback = gr.Checkbox(value=False, label="register as mindX's fallback model")
                        s_base = gr.Textbox(value="", label="mindX base URL (for the registration)")
                with gr.Group(visible=False) as sci_serve:
                    gr.Markdown("A LoRA has meaning only on the tensors it was trained on. Serving an adapter onto a different "
                                "architecture succeeds silently and means nothing — the tag's base must match "
                                "`adapter_config.json:base_model_name_or_path`.")
                s_out = gr.Code(label="result", language="json", interactive=False)

            # ── HUB ──
            with gr.Tab("Hub · Hugging Face"):
                gr.Markdown("The **Hugging Face extension** (`mindxtrain.hf`): who the token is, what it may write, "
                            "a finished run published with its evidence, and the corpus beside it.")
                with gr.Row():
                    who_btn = gr.Button("whoami + write scope", variant="primary")
                    h_repo = gr.Textbox(value="", label="model repo (org/name)", scale=2)
                    h_run = gr.Textbox(value="out/runs", label="run dir", scale=2)
                with gr.Row():
                    h_dry = gr.Checkbox(value=True, label="dry run (show what would upload)")
                    h_private = gr.Checkbox(value=False, label="private")
                    h_pub_btn = gr.Button("publish the run")
                with gr.Group(visible=False) as adv_hub:
                    with gr.Row():
                        h_base = gr.Textbox(value="", label="pull a base before training (model id)")
                        h_pull_btn = gr.Button("pull base")
                        h_ds = gr.Textbox(value="", label="corpus → dataset repo")
                        h_ds_path = gr.Textbox(value="", label="corpus path")
                        h_ds_btn = gr.Button("push corpus")
                with gr.Group(visible=False) as sci_hub:
                    gr.Markdown("Traps this module encodes: membership ≠ write scope · `list_repo_tree` entries carry `.path` "
                                "(only `repo_info().siblings` carry `.rfilename`) · the Hub checks a Space's ZeroGPU quota **before** "
                                "existence (402 on re-push) · a Space README `short_description` must be ≤ 60 characters · "
                                "never put a write-scoped token on a public Space.")
                    with gr.Row():
                        h_lineage_repo = gr.Textbox(value="", label="lineage of repo")
                        h_lineage_btn = gr.Button("read lineage")
                h_out = gr.Code(label="result", language="json", interactive=False)


            # ── COACH ──
            with gr.Tab("Coach · intuitive training"):
                gr.Markdown("**The coach turns a measured impression into the next run.** It reads what the last "
                            "generations actually scored, says what to change, and — when you agree — starts that run here. "
                            "`bootcamp.impression` is drill → impression; `impression.bootcamp` is impression → the next drill.")
                with gr.Row():
                    c_base = gr.Textbox(value=os.environ.get("MINDX_BASE_URL", "https://mindx.pythai.net"),
                                        label="mindX node (where the coach's measurements live)", scale=3)
                    c_read = gr.Button("read the coach", variant="primary", scale=1)
                c_verdict = gr.HTML()
                with gr.Row():
                    c_rec = gr.Code(label="the recipe the coach proposes", language="json", interactive=False, scale=2)
                    c_score = gr.Dataframe(headers=["gen", "identity", "task", "coherence", "influence Δrecall", "runs"],
                                           label="scorecards", interactive=False, scale=2)
                with gr.Row():
                    c_adopt = gr.Button("adopt it into the Forge knobs")
                    c_start = gr.Button("adopt and start the run", variant="primary")
                c_state = gr.Markdown()
                with gr.Group(visible=False) as sci_coach:
                    gr.Markdown("**What the coach is allowed to conclude.** Influence is always *after − before* on the same "
                                "probe, with the untouched base answering too. Identity is a scorer, not a vibe. Three runs with "
                                "no positive influence is `training_stalled` — the drill changes, not the compute. A rung the "
                                "ladder already rejected is never proposed again.")

            # ── BENCH ──
            with gr.Tab("Bench · autotune"):
                gr.Markdown("The 60-second AOT probe: the plan is fixed before the run starts, and JIT autotune is forbidden "
                            "inside the production loop.")
                with gr.Row():
                    b_dry = gr.Checkbox(value=True, label="dry run (no GPU)")
                    b_out = gr.Textbox(value="autotune_plan.json", label="plan out")
                    b_btn = gr.Button("bench", variant="primary")
                b_res = gr.Code(label="plan", language="json", interactive=False)

            # ── RUNS ──
            with gr.Tab("Runs · receipts"):
                with gr.Row():
                    r_refresh = gr.Button("refresh", variant="primary")
                    r_root = gr.Textbox(value="out/runs", label="runs root", scale=2)
                r_tbl = gr.Dataframe(headers=["run", "when", "steps", "train loss", "eval loss", "checkpoint", "size"],
                                     interactive=False, wrap=True)
                with gr.Group(visible=False) as sci_runs:
                    gr.Markdown("A **receipt** verifies a provenance manifest's BLAKE3 hashes against what is on disk. "
                                "A run you cannot re-hash is a story, not a result.")
                    with gr.Row():
                        r_manifest = gr.Textbox(value="", label="manifest")
                        r_receipt_btn = gr.Button("verify receipt")
                    r_receipt = gr.Code(label="receipt", language="json", interactive=False)

        gr.HTML("<div class='mx-sub' style='margin-top:12px'>mindXtrain · "
                "<a href='https://github.com/professor-codephreak/mindXtrain'>source</a> · "
                "<a href='https://mastermind.pythai.net'>orchestration</a> · "
                "<a href='https://mindx.pythai.net'>the node that runs it</a></div>")

        # ── tier wiring ──
        for adv, sci in ((adv_forge, sci_forge), (adv_gate, sci_gate), (adv_eval, sci_eval),
                         (adv_serve, sci_serve), (adv_hub, sci_hub), (adv_hub, sci_runs), (adv_hub, sci_coach)):
            tier.change(tier_vis, [tier], [adv, sci])

        # ── handlers ──
        recipe.change(lambda n: (recipe_summary(n), read_recipe(n)), [recipe], [summary, recipe_yaml])
        demo.load(lambda: (recipe_summary(recipe_names()[0]) if recipe_names() else "no recipes found",
                           read_recipe(recipe_names()[0]) if recipe_names() else ""), None, [summary, recipe_yaml])

        def do_save(text: str, where: str):
            p = (HOME / where) if not os.path.isabs(where) else Path(where)
            p.write_text(text)
            return f"wrote `{p}` ({len(text)} bytes)"
        save_btn.click(do_save, [recipe_yaml, save_as], [save_state])

        def do_train(rec, outd, t, r_, a_, ep, lr_, sq, pd, ac, pk, cp, cn, es):
            cfg = HOME / "run.ui.yaml"
            text = read_recipe(rec)
            if t in ("Advanced", "Scientific"):
                try:
                    import yaml
                    c = yaml.safe_load(text) or {}
                    tr = c.setdefault("train", {})
                    tr.setdefault("method", {}).update({"r": int(r_), "alpha": int(a_)})
                    tr.setdefault("schedule", {}).update({"epochs": int(ep)})
                    tr.setdefault("optimizer", {}).update({"lr": float(lr_)})
                    tr.setdefault("batch", {}).update({"per_device": int(pd), "grad_accum": int(ac)})
                    tr["cpu_throttle"] = {"percent": int(cp), "nice": int(cn)}
                    d = c.setdefault("data", {})
                    d.update({"seq_len": int(sq), "packing": bool(pk), "eval_split": float(es)})
                    text = yaml.safe_dump(c, sort_keys=False)
                except Exception as e:  # noqa: BLE001
                    return f"<span class='mx-bad'>config edit failed: {e}</span>", gr.Timer(active=False)
            cfg.write_text(text)
            r = launch("train", ["train", str(cfg), "--out", outd, "--cpu-percent", str(int(cp)), "--cpu-nice", str(int(cn))])
            if not r.get("ok"):
                return f"<span class='mx-bad'>{r.get('reason')}</span>", gr.Timer(active=False)
            return (f"<span class='mx-good'>started</span> · pid {r['pid']} · <code>{r['cmd']}</code>", gr.Timer(active=True))
        start_btn.click(do_train, [recipe, out_dir, tier, lora_r, lora_a, epochs, lr, seq, per_dev, accum, packing, cpu_pct, cpu_nice, eval_split],
                        [kiln, ticker])
        stop_btn.click(lambda: (json.dumps(JOBS["train"].stop() if JOBS.get("train") else {"ok": False, "reason": "no job"}), gr.Timer(active=False)),
                       None, [kiln, ticker])

        def tick(t):
            j = JOBS.get("train")
            if not j:
                return "<span class='mx-low'>no run yet</span>", [], gr.LinePlot(), "", gr.Timer(active=False)
            m = parse_log(j.log)
            head = m.headline(running=j.running, started=j.started)
            rows = m.rows(scientific=(t == "Scientific"))
            frame = m.frame()
            tail = m.tail(j.log, 60)
            return head, rows, gr.LinePlot(value=frame, x="step", y="value", color="metric"), tail, gr.Timer(active=j.running)
        ticker.tick(tick, [tier], [kiln, metrics_tbl, loss_plot, log_box, ticker])

        def do_gate(cfg, n, trigger, outp):
            args = ["imprint", "--config", cfg, "--max-inquiries", str(int(n))]
            if outp:
                args += ["--out", outp]
            if trigger:
                args += ["--trigger-dream"]
            code, text = run_sync(args, timeout=3600)
            verdict = _last_json(text)
            delta = (verdict or {}).get("delta")
            md = ("<span class='mx-good'>imprinted</span>" if (verdict or {}).get("imprinted") else "<span class='mx-low'>not imprinted</span>") \
                 + (f" · Δ recall **{delta}**" if delta is not None else "")
            return md, json.dumps(verdict or {"exit": code, "output": text[-1500:]}, indent=1)
        g_btn.click(do_gate, [g_cfg, g_max, g_trigger, g_out], [g_out_md, g_json])

        def do_eval(cfg, ck):
            code, text = run_sync(["eval", "--config", cfg] + (["--checkpoint", ck] if ck else []), timeout=3600)
            return json.dumps(_last_json(text) or {"exit": code, "output": text[-2000:]}, indent=1)
        e_btn.click(do_eval, [e_cfg, e_ckpt], [e_out])

        def do_ce(cfg, ck, jsonl, mx):
            args = ["eval-checkpoint", "--config", cfg] + (["--checkpoint", ck] if ck else [])
            if jsonl:
                args += ["--jsonl", jsonl]
            args += ["--max-samples", str(int(mx))]
            code, text = run_sync(args, timeout=3600)
            return json.dumps(_last_json(text) or {"exit": code, "output": text[-2000:]}, indent=1)
        e_ce_btn.click(do_ce, [e_cfg, e_ckpt, e_jsonl, e_max], [e_out])

        def do_serve(cfg, ck, to, tag, fb, base):
            args = ["serve", "--config", cfg, "--to", to] + (["--checkpoint", ck] if ck else []) + (["--tag", tag] if tag else [])
            if fb:
                args += ["--register-as-fallback"]
            if base:
                args += ["--mindx-base-url", base]
            code, text = run_sync(args, timeout=3600)
            return json.dumps(_last_json(text) or {"exit": code, "output": text[-2000:]}, indent=1)
        s_btn.click(do_serve, [s_cfg, s_ckpt, s_to, s_tag, s_fallback, s_base], [s_out])

        def do_bench(dry, outp):
            code, text = run_sync(["bench", "--out", outp] + (["--dry-run"] if dry else []), timeout=1800)
            return json.dumps(_last_json(text) or {"exit": code, "output": text[-2000:]}, indent=1)
        b_btn.click(do_bench, [b_dry, b_out], [b_res])

        # Hub
        def _hf():
            from mindxtrain import hf as H
            return H
        who_btn.click(lambda: json.dumps(_hf().account(), indent=1), None, [h_out])
        h_pub_btn.click(lambda repo, run, dry, priv: json.dumps(
            _hf().publish_generation(run, repo, dry_run=bool(dry), private=bool(priv)), indent=1, default=str),
            [h_repo, h_run, h_dry, h_private], [h_out])
        h_pull_btn.click(lambda mid: json.dumps(_hf().pull_base(mid), indent=1), [h_base], [h_out])
        h_ds_btn.click(lambda repo, path: json.dumps(_hf().push_dataset(path, repo), indent=1), [h_ds, h_ds_path], [h_out])
        h_lineage_btn.click(lambda repo: json.dumps(_hf().lineage(repo), indent=1), [h_lineage_repo], [h_out])


        # ── Coach ──
        def read_coach(base):
            import urllib.request
            def get(path, timeout=120):
                try:
                    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
                        return json.loads(r.read().decode("utf-8"))
                except Exception as e:  # noqa: BLE001
                    return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            c = get("/insight/hf/coach")
            if c.get("error"):
                return f"<span class='mx-bad'>{c['error']}</span>", "{}", []
            v = c.get("coach_verdict") or {}
            sc = (c.get("scorecards") or {}).get("per_generation") or {}
            rows = [[g, d.get("identity_rate"), d.get("task_score", d.get("task")), d.get("coherence"),
                     (d.get("influence") or {}).get("recall_delta") if isinstance(d.get("influence"), dict) else d.get("influence"),
                     d.get("runs")] for g, d in sorted(sc.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)
                    if isinstance(d, dict)]
            rec = c.get("recommendation") or {}
            imp = (c.get("iterations") or {})
            head = (f"<b>{v.get('verdict','—')}</b> over {v.get('n',0)} exchanges · "
                    f"Δrecall {v.get('recall_delta','—')} · Δcoherence {v.get('coherence_delta','—')} · "
                    f"Δidentity {v.get('identity_delta','—')} · persona {(c.get('personas') or {}).get('selected','—')}"
                    + (f" · iterations {imp.get('runs')}" if imp else ""))
            return head, json.dumps(rec, indent=1)[:3000], rows
        c_read.click(read_coach, [c_base], [c_verdict, c_rec, c_score])

        def adopt(rec_json):
            try:
                r = json.loads(rec_json or "{}")
            except Exception:  # noqa: BLE001
                r = {}
            p = r.get("params") or r.get("recipe") or r
            if not isinstance(p, dict) or not p:
                return ("<span class='mx-low'>no recipe to adopt — read the coach first</span>",
                        gr.update(), gr.update(), gr.update(), gr.update())
            return (f"adopted: <code>{json.dumps(p)[:200]}</code>",
                    gr.update(value=int(p.get("lora_r", 16))), gr.update(value=int(p.get("lora_alpha", 32))),
                    gr.update(value=int(p.get("epochs", 2))), gr.update(value=float(p.get("lr", 1e-4))))
        c_adopt.click(adopt, [c_rec], [c_state, lora_r, lora_a, epochs, lr])

        def adopt_and_start(rec_json, rec_name, outd, t, r_, a_, ep, lr_, sq, pd, ac, pk, cp, cn, es):
            msg, r_u, a_u, ep_u, lr_u = adopt(rec_json)
            r_ = r_u.get("value", r_) if isinstance(r_u, dict) else r_
            a_ = a_u.get("value", a_) if isinstance(a_u, dict) else a_
            ep = ep_u.get("value", ep) if isinstance(ep_u, dict) else ep
            lr_ = lr_u.get("value", lr_) if isinstance(lr_u, dict) else lr_
            head, tick_ = do_train(rec_name, outd, "Advanced", r_, a_, ep, lr_, sq, pd, ac, pk, cp, cn, es)
            return f"{msg}<br>{head}", tick_
        c_start.click(adopt_and_start,
                      [c_rec, recipe, out_dir, tier, lora_r, lora_a, epochs, lr, seq, per_dev, accum, packing, cpu_pct, cpu_nice, eval_split],
                      [c_state, ticker])

        # ── diagnostics ──
        def diagnostics():
            host = []
            try:
                import psutil
                vm = psutil.virtual_memory()
                host = [f"**CPU** {psutil.cpu_percent(interval=0.3):.0f}% of {psutil.cpu_count()} cores · load {', '.join(f'{x:.2f}' for x in os.getloadavg())}",
                        f"**RAM** {vm.used/1e9:.1f} / {vm.total/1e9:.1f} GB ({vm.percent:.0f}%)",
                        f"**disk** {psutil.disk_usage(str(HOME)).percent:.0f}% used at {HOME}"]
            except Exception:  # noqa: BLE001
                try:
                    host = [f"**load** {', '.join(f'{x:.2f}' for x in os.getloadavg())}",
                            "**RAM/CPU** install `psutil` (`uv sync --extra obs`) for the full readout"]
                except Exception:  # noqa: BLE001
                    host = ["host telemetry unavailable on this platform"]
            j = JOBS.get("train")
            if not j:
                return "\n\n".join(host), "no run to diagnose yet"
            m = parse_log(j.log)
            return "\n\n".join(host), "\n\n".join("· " + d for d in m.diagnose())
        diag_btn.click(diagnostics, None, [host_md, diag_md])

        # Runs
        def list_runs(root):
            base = (HOME / root) if not os.path.isabs(root) else Path(root)
            rows = []
            if base.is_dir():
                for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:40]:
                    if not d.is_dir():
                        continue
                    log = next((p for p in (d / "train.log", d.parent / "train.log") if p.is_file()), None)
                    m = parse_log(log) if log else None
                    ck = d / "checkpoint"
                    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    rows.append([d.name, time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
                                 (m.last_step if m else None), (m.train_loss if m else None), (m.eval_loss if m else None),
                                 "✓" if ck.is_dir() else "", f"{size/1e6:.0f} MB"])
            return rows
        r_refresh.click(list_runs, [r_root], [r_tbl])
        r_receipt_btn.click(lambda man: json.dumps(_last_json(run_sync(["receipt", "--manifest", man], 600)[1]) or {}, indent=1),
                            [r_manifest], [r_receipt])

    demo.mx_launch = {} if blocks_takes_theme else {"theme": theme(), "css": CSS}
    return demo


def _last_json(text: str) -> Optional[Dict[str, Any]]:
    """The last JSON object a CLI printed — the verbs end with one."""
    for m in reversed(list(re.finditer(r"\{.*?\}", text or "", re.S))):
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            continue
    return None


def main(host: str = "127.0.0.1", port: int = 7862, share: bool = False, mcp: bool = True) -> None:
    demo = build()
    demo.queue(default_concurrency_limit=4).launch(server_name=host, server_port=port, share=share,
                                                   mcp_server=mcp, **getattr(demo, "mx_launch", {}))


if __name__ == "__main__":
    main()
