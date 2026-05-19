"""Linux terminal sampler integration + psutil-vs-ps cross-validation.

The user's framing:
    optimized using linux terminal commands when possible for minimum
    and efficiency in time.

This module is Coach's surface for the samplers defined in mindX's
`utils/cli_time_samplers.py`. It also adds a *measurement confidence*
helper that compares psutil's view of the host (the existing
`hw_diagnostics.py` path) against `ps -A`'s view. A wide divergence
flags container/cgroup/sandbox bias — exactly the failure mode that
makes raw psutil numbers misleading.

Falls back to a `"degraded"` payload when the mindX samplers module
can't be imported (e.g., Coach running standalone without mindX
checked out) so the endpoint still returns shape-stable data.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("mindxtrain.cli_diagnostics")

_MINDX_ROOT = Path(os.environ.get("MINDX_ROOT", "/home/hacker/mindX"))


def _load_samplers() -> Any:
    """Import mindX's cli_time_samplers without dragging in agents/__init__.py.

    The chronos plan put the samplers in `mindX/utils/cli_time_samplers.py`;
    Coach lives in a different project, so we add the mindX root to
    sys.path on first call and import directly. Cached at the module
    level via the import system.
    """
    samplers_path = _MINDX_ROOT / "utils" / "cli_time_samplers.py"
    if not samplers_path.exists():
        return None
    parent = str(_MINDX_ROOT)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        from utils import cli_time_samplers  # type: ignore
        return cli_time_samplers
    except ImportError as exc:
        logger.debug("cli_time_samplers unimportable: %r", exc)
        return None


def run_samplers() -> dict[str, Any]:
    """Run every sampler defined in `utils/cli_time_samplers.py`.

    Returns `{ok: bool, supported: bool, samplers: {name -> record}}` so
    callers can branch on whether the mindX samplers are reachable
    without rummaging through individual records.
    """
    mod = _load_samplers()
    if mod is None:
        return {
            "ok": False,
            "supported": False,
            "reason": "cli_time_samplers.py not found under MINDX_ROOT",
            "samplers": {},
        }
    try:
        return {
            "ok": True,
            "supported": mod.is_supported(),
            "samplers": mod.sample_all(),
        }
    except Exception as exc:
        logger.warning("sample_all failed: %r", exc)
        return {
            "ok": False, "supported": False,
            "reason": f"sample_all: {exc!r}", "samplers": {},
        }


# ---- psutil vs ps cross-check -------------------------------------------


def _psutil_totals() -> dict[str, Any]:
    """Whole-system CPU% + total RSS via psutil (existing primitive)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return {"ok": False, "reason": "psutil not installed"}
    # cpu_percent(interval=None) is non-blocking; reflects load since last call.
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    rss_used_mb = (vm.total - vm.available) / (1024 * 1024)
    return {"ok": True, "cpu_pct": float(cpu), "rss_used_mb": float(rss_used_mb)}


def _ps_totals(samplers_mod: Any) -> dict[str, Any]:
    """Whole-system CPU% + total RSS via `ps -A` (proc_snapshot sampler)."""
    if samplers_mod is None:
        return {"ok": False, "reason": "samplers unavailable"}
    rec = samplers_mod.proc_snapshot()
    if not rec["ok"]:
        return {"ok": False, "reason": rec.get("error", "proc_snapshot failed")}
    val = rec["value"]
    return {
        "ok": True,
        "cpu_pct": float(val["total_cpu_pct"]),
        "rss_used_mb": float(val["total_rss_kb"]) / 1024.0,
        "duration_us": rec["duration_us"],
    }


def _classify_band(cpu_delta_pp: float, rss_delta_mb: float) -> str:
    """Translate raw deltas into a tight | loose | divergent label.

    Thresholds picked to flag the container/cgroup measurement-bias
    case: psutil sees cgroup-clipped values while `ps` sees whole-
    system. A `divergent` band is the actionable signal — Coach should
    surface it red.
    """
    if cpu_delta_pp < 5 and rss_delta_mb < 100:
        return "tight"
    if cpu_delta_pp < 15 and rss_delta_mb < 500:
        return "loose"
    return "divergent"


def measurement_confidence() -> dict[str, Any]:
    """Compare psutil's view to `ps -A`'s view. Returns delta + band."""
    samplers_mod = _load_samplers()
    psu = _psutil_totals()
    ps = _ps_totals(samplers_mod)

    # If either source is missing, we can't compute a delta. Report
    # honestly so the UI shows a "?" rather than fabricating a band.
    if not (psu.get("ok") and ps.get("ok")):
        return {
            "ok": False,
            "psutil": psu,
            "ps": ps,
            "confidence_band": "unknown",
        }

    cpu_delta_pp = abs(psu["cpu_pct"] - ps["cpu_pct"])
    rss_delta_mb = abs(psu["rss_used_mb"] - ps["rss_used_mb"])
    return {
        "ok": True,
        "psutil_cpu_pct": round(psu["cpu_pct"], 2),
        "ps_cpu_pct": round(ps["cpu_pct"], 2),
        "cpu_delta_pp": round(cpu_delta_pp, 2),
        "psutil_rss_mb": round(psu["rss_used_mb"], 1),
        "ps_rss_mb": round(ps["rss_used_mb"], 1),
        "rss_delta_mb": round(rss_delta_mb, 1),
        "confidence_band": _classify_band(cpu_delta_pp, rss_delta_mb),
    }


__all__ = ["measurement_confidence", "run_samplers"]
