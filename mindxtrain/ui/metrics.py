"""Diagnostics parsed from what the trainer actually wrote.

The log is the source of truth. Nothing in this module estimates a step that did not happen, and
every number carries its **unit** and **where it came from**, because a metric without provenance is
a rumour. It reads three things TRL/transformers emit and one thing mindXtrain does:

1. step dicts  — ``{'loss': .., 'grad_norm': .., 'learning_rate': .., 'entropy': ..,
                   'num_tokens': .., 'mean_token_accuracy': .., 'epoch': ..}``
2. eval dicts  — ``{'eval_loss': .., 'eval_runtime': .., 'eval_entropy': .., 'eval_num_tokens': ..,
                   'eval_mean_token_accuracy': ..}``
3. the closing dict — ``{'train_runtime': .., 'train_loss': .., 'train_samples_per_second': ..}``
4. tqdm         — ``24%|███ | 28/116 [14:55<46:01, 31.38s/it]`` (progress, ETA, seconds per step)

and it raises the warnings that actually cost runs: packing without a flash-attention backend,
a throttle that is not being honoured, a loss that stopped moving, an eval that has drifted above
train (memorising), and gradient norms that are collapsing or exploding.
"""
from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_DICT = re.compile(r"\{'(?:loss|eval_loss|train_runtime)'.*?\}")
_TQDM = re.compile(r"(?:(?P<label>[A-Za-z][A-Za-z ._-]{2,30}):\s*)?(?P<pct>\d+)%\|[^|]*\|\s*"
                   r"(?P<step>\d+)/(?P<total>\d+)\s*\[(?P<elapsed>[0-9:]+)<(?P<eta>[0-9:?]+),\s*(?P<rate>[0-9.]+)(?P<unit>s/it|it/s)")
_THROTTLE = re.compile(r"cpu_throttle overridden:\s*percent=(\d+)\s*nice=(\d+)")
_TOKENIZING = re.compile(r"Tokenizing train dataset:.*?(\d+)/(\d+)")


def _secs(hms: str) -> Optional[float]:
    if not hms or "?" in hms:
        return None
    parts = [float(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _fmt_secs(s: Optional[float]) -> str:
    if s is None:
        return "—"
    s = int(s)
    return f"{s//3600}h {s%3600//60:02d}m" if s >= 3600 else f"{s//60}m {s%60:02d}s"


@dataclass
class RunMetrics:
    steps: List[Dict[str, float]] = field(default_factory=list)
    evals: List[Dict[str, float]] = field(default_factory=list)
    final: Dict[str, float] = field(default_factory=dict)
    progress: Dict[str, Any] = field(default_factory=dict)        # the TRAINING bar
    eval_progress: Dict[str, Any] = field(default_factory=dict)   # the evaluation loop's own bar
    throttle: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    phases: List[str] = field(default_factory=list)
    dataset_rows: Optional[int] = None

    # ── the few fields other rooms read ──
    @property
    def last_step(self) -> Optional[int]:
        return self.progress.get("step") or (len(self.steps) or None)

    @property
    def train_loss(self) -> Optional[float]:
        return self.final.get("train_loss") or (self.steps[-1].get("loss") if self.steps else None)

    @property
    def eval_loss(self) -> Optional[float]:
        return self.evals[-1].get("eval_loss") if self.evals else None

    # ── derived, honestly ──
    def rate(self) -> Dict[str, Optional[float]]:
        """Throughput as the trainer reports it — seconds per step, tokens per second, ETA."""
        s_it = self.progress.get("s_per_it")
        tok = self.steps[-1].get("num_tokens") if self.steps else None
        prev = self.steps[-2].get("num_tokens") if len(self.steps) > 1 else None
        per_step = (tok - prev) if (tok is not None and prev is not None and tok >= prev) else None
        return {"s_per_step": s_it, "tokens_per_step": per_step,
                "tokens_per_s": (per_step / s_it) if (per_step and s_it) else None,
                "eta_s": self.progress.get("eta_s"), "elapsed_s": self.progress.get("elapsed_s")}

    def trend(self, key: str, window: int = 10) -> Optional[float]:
        """Change in a metric over the last `window` logged steps (late mean − early mean)."""
        vals = [s[key] for s in self.steps if key in s and isinstance(s[key], (int, float))]
        if len(vals) < 4:
            return None
        w = min(window, len(vals) // 2)
        return round(sum(vals[-w:]) / w - sum(vals[:w]) / w, 4)

    def diagnose(self) -> List[str]:
        """What an operator would want said out loud. Each line is a fact plus its consequence."""
        out = list(self.warnings)
        dl = self.trend("loss")
        if dl is not None:
            out.append(f"loss trend {dl:+.4f} over the run — " + ("falling, the run is learning" if dl < -0.005
                       else "flat: more epochs will not fix a corpus problem" if abs(dl) <= 0.005 else "RISING, stop and look"))
        da = self.trend("mean_token_accuracy")
        if da is not None:
            out.append(f"token accuracy trend {da:+.4f} — " + ("improving" if da > 0.002 else "flat"))
        if self.steps:
            gn = [s["grad_norm"] for s in self.steps if isinstance(s.get("grad_norm"), (int, float))]
            if gn:
                last = gn[-1]
                if last > 10:
                    out.append(f"grad_norm {last:.2f} — exploding; lower the learning rate")
                elif last < 1e-3:
                    out.append(f"grad_norm {last:.2e} — vanishing; the adapter has stopped moving")
        if self.evals and self.steps:
            ev, tr = self.eval_loss, self.steps[-1].get("loss")
            if ev is not None and tr is not None and ev - tr > 0.5:
                out.append(f"eval loss {ev:.3f} is {ev-tr:.2f} above train {tr:.3f} — memorising, not generalising")
        r = self.rate()
        if r["s_per_step"] and self.throttle.get("percent"):
            out.append(f"{r['s_per_step']:.1f} s/step at {self.throttle['percent']}% CPU — "
                       f"{_fmt_secs(r['eta_s'])} left at this pace")
        if not out:
            out.append("nothing anomalous in what the trainer has written so far")
        return out

    # ── rendering ──
    def headline(self, running: bool, started: Optional[float] = None) -> str:
        p = self.progress
        bar = ""
        if p.get("total"):
            pct = 100.0 * (p.get("step") or 0) / p["total"]
            done = int(pct / 4)
            bar = f"<div class='mx-bar'><i style='width:{pct:.1f}%'></i></div>" + \
                  f"<code>{'█'*done}{'░'*(25-done)} {pct:.1f}%</code> "
        state = "<span class='mx-good'>RUNNING</span>" if running else "<span class='mx-low'>stopped</span>"
        r = self.rate()
        bits = [state, f"step <b>{p.get('step','—')}/{p.get('total','—')}</b>", f"epoch {self._epoch()}",
                f"loss <b>{self._fmt(self.steps[-1].get('loss') if self.steps else None)}</b>",
                f"acc {self._fmt(self.steps[-1].get('mean_token_accuracy') if self.steps else None)}",
                f"{self._fmt(r['s_per_step'])} s/step", f"ETA {_fmt_secs(r['eta_s'])}"]
        if started:
            bits.append(f"elapsed {_fmt_secs(time.time() - started)}")
        diag = "<br>".join("· " + d for d in self.diagnose()[:4])
        return f"<div class='mx-kiln'>{bar}{' · '.join(bits)}<div class='mx-diag'>{diag}</div></div>"

    def rows(self, scientific: bool = False) -> List[List[Any]]:
        """metric · value · unit · where it came from."""
        s = self.steps[-1] if self.steps else {}
        e = self.evals[-1] if self.evals else {}
        r = self.rate()
        rows = [
            ["loss", self._fmt(s.get("loss")), "nats/token", "TRL step dict"],
            ["mean_token_accuracy", self._fmt(s.get("mean_token_accuracy")), "fraction", "TRL step dict"],
            ["learning_rate", self._fmt(s.get("learning_rate"), 8), "—", "TRL step dict"],
            ["epoch", self._epoch(), "epochs", "TRL step dict"],
            ["eval_loss", self._fmt(e.get("eval_loss")), "nats/token", "held-out split"],
            ["eval_mean_token_accuracy", self._fmt(e.get("eval_mean_token_accuracy")), "fraction", "held-out split"],
            ["s/step", self._fmt(r["s_per_step"]), "seconds", "tqdm"],
            ["tokens/step", self._fmt(r["tokens_per_step"], 0), "tokens", "num_tokens delta"],
            ["tokens/s", self._fmt(r["tokens_per_s"], 1), "tokens/s", "derived"],
            ["ETA", _fmt_secs(r["eta_s"]), "", "tqdm"],
        ]
        if scientific:
            rows += [
                ["grad_norm", self._fmt(s.get("grad_norm")), "L2", "TRL step dict"],
                ["entropy", self._fmt(s.get("entropy")), "nats", "TRL step dict"],
                ["eval_entropy", self._fmt(e.get("eval_entropy")), "nats", "held-out split"],
                ["num_tokens (cum.)", self._fmt(s.get("num_tokens"), 0), "tokens", "TRL step dict"],
                ["eval_num_tokens", self._fmt(e.get("eval_num_tokens"), 0), "tokens", "held-out split"],
                ["eval_samples_per_second", self._fmt(e.get("eval_samples_per_second")), "samples/s", "held-out split"],
                ["train_runtime", _fmt_secs(self.final.get("train_runtime")), "", "closing dict"],
                ["train_loss (final)", self._fmt(self.final.get("train_loss")), "nats/token", "closing dict"],
                ["throttle", f"{self.throttle.get('percent','—')}% nice {self.throttle.get('nice','—')}", "", "mindXtrain"],
                ["dataset rows tokenized", self.dataset_rows or "—", "rows", "datasets"],
                ["eval pass", (f"{self.eval_progress.get('step')}/{self.eval_progress.get('total')}"
                               if self.eval_progress else "—"), "batches", "tqdm (eval bar)"],
                ["phases seen", " → ".join(self.phases[-4:]) or "—", "", "log"],
            ]
        return rows

    def frame(self) -> Dict[str, List[Any]]:
        """Long-form rows for the plot: loss, token accuracy, and lr normalised to its own max."""
        xs, ms, vs = [], [], []
        lrs = [s.get("learning_rate") for s in self.steps if isinstance(s.get("learning_rate"), (int, float))]
        lr_max = max(lrs) if lrs else None
        for i, s in enumerate(self.steps, 1):
            if isinstance(s.get("loss"), (int, float)):
                xs.append(i); ms.append("loss"); vs.append(round(s["loss"], 4))
            if isinstance(s.get("mean_token_accuracy"), (int, float)):
                xs.append(i); ms.append("token accuracy"); vs.append(round(s["mean_token_accuracy"], 4))
            if lr_max and isinstance(s.get("learning_rate"), (int, float)):
                xs.append(i); ms.append("lr (÷max)"); vs.append(round(s["learning_rate"] / lr_max, 4))
        return {"step": xs, "metric": ms, "value": vs}

    @staticmethod
    def tail(log: Path, n: int = 60) -> str:
        try:
            lines = [ln.rstrip() for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001
            return ""
        return "\n".join(ln[-400:] for ln in lines[-n:])

    # ── helpers ──
    def _epoch(self) -> str:
        v = (self.steps[-1].get("epoch") if self.steps else None) or self.final.get("epoch")
        return f"{v:.2f}" if isinstance(v, (int, float)) else "—"

    @staticmethod
    def _fmt(v: Any, places: int = 4) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if isinstance(v, (int, float)):
            return f"{v:.{places}f}".rstrip("0").rstrip(".") if places else f"{int(v):,}"
        return str(v)


def parse_log(log: Optional[Path]) -> RunMetrics:
    """Read a training log into metrics. Tolerant by design: a half-written line is skipped, never fatal."""
    m = RunMetrics()
    if not log:
        return m
    try:
        text = Path(log).read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return m
    bars: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        for d in _DICT.findall(line):
            try:
                rec = ast.literal_eval(d)
            except Exception:  # noqa: BLE001
                continue
            rec = {k: (float(v) if isinstance(v, str) and re.fullmatch(r"[-+0-9.eE]+", v) else v) for k, v in rec.items()}
            if "train_runtime" in rec:
                m.final.update(rec)
            elif any(k.startswith("eval_") for k in rec):
                m.evals.append(rec)
            elif "loss" in rec:
                m.steps.append(rec)
        for tq in _TQDM.finditer(line):
            g = tq.groupdict()
            rate = float(g["rate"])
            s_it = rate if g["unit"] == "s/it" else (1.0 / rate if rate else None)
            bar = {"pct": float(g["pct"]), "step": int(g["step"]), "total": int(g["total"]),
                   "elapsed_s": _secs(g["elapsed"]), "eta_s": _secs(g["eta"]), "s_per_it": s_it}
            label = (g["label"] or "").strip()
            if label:                       # "Loading weights", "Tokenizing train dataset" — a phase, not the run
                if label not in m.phases:
                    m.phases.append(label)
                continue
            bars.append(bar)
        th = _THROTTLE.search(line)
        if th:
            m.throttle = {"percent": int(th.group(1)), "nice": int(th.group(2))}
        tk = _TOKENIZING.search(line)
        if tk:
            m.dataset_rows = int(tk.group(2))
        low = line.lower()
        if "packing" in low and "flash attention" in low:
            m.warnings.append("packing is on without a flash-attention backend — samples in one sequence can attend across "
                              "boundaries; correct on CPU eager, but do not compare these losses with a flash-attn run")
        if "loading weights" in low and "Loading weights" not in m.phases:
            m.phases.append("Loading weights")
        if "tokenizing" in low and "Tokenizing" not in m.phases:
            m.phases.append("Tokenizing")
        if "out of memory" in low or "killed" in low:
            m.warnings.append("OOM / killed in the log — the run did not finish on its own terms")
        if "traceback" in low:
            m.warnings.append("a traceback is in the log — read the tail")
    if bars:
        # A run writes ONE training bar (its total never changes) and a FRESH bar per evaluation.
        # Two wrong discriminators, both found by replaying a real log (2026-09-12): "the last bar
        # seen" reports eval progress as the run's mid-run, and "the most frequent total" flips to the
        # eval bar at the end, because tqdm redraws each eval bar many times. The one that holds is
        # the CLOCK: the training bar's elapsed runs for the whole run (1:10:00 here) while every eval
        # bar's elapsed restarts from zero.
        by_total: Dict[int, float] = {}
        for b in bars:
            by_total[b["total"]] = max(by_total.get(b["total"], 0.0), b.get("elapsed_s") or 0.0)
        train_total = max(by_total, key=lambda t: by_total[t])
        train_bars = [b for b in bars if b["total"] == train_total]
        other = [b for b in bars if b["total"] != train_total]
        if train_bars:
            m.progress.update(train_bars[-1])
        if other:
            m.eval_progress.update(other[-1])
    if m.steps and "Training" not in m.phases:
        m.phases.append("Training")
    if m.final:
        m.phases.append("Done")
    return m
