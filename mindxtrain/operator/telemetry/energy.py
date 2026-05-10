"""Energy / power telemetry — wraps `rocm-smi --showpower --json`.

Returns 0.0 W if the binary isn't on PATH (typical CPU-only dev box).
"""

from __future__ import annotations

import json
import shutil
import subprocess


def sample_power_w(gpu_index: int = 0) -> float:
    """Return current GPU power draw in watts; 0.0 if `rocm-smi` is unavailable."""
    if shutil.which("rocm-smi") is None:
        return 0.0
    try:
        out = subprocess.run(
            ["rocm-smi", "--showpower", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0.0
    if out.returncode != 0:
        return 0.0
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return 0.0
    card_key = f"card{gpu_index}"
    card = data.get(card_key) or {}
    for k, v in card.items():
        if "power" in k.lower():
            try:
                return float(str(v).replace("W", "").strip())
            except ValueError:
                continue
    return 0.0


__all__ = ["sample_power_w"]
