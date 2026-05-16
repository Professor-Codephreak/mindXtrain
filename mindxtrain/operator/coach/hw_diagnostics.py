"""Hardware diagnostics for the Coach UI — CPU / AMD / NVIDIA.

Surfaces what compute the operator actually has available so the user can
pick the right training lane:

- **CPU**: model name (Ryzen / EPYC / Intel detection), physical cores,
  current 1-minute load, total RAM, available RAM.
- **AMD GPU**: rocm-smi probe; reports each GPU's VRAM + driver if rocm
  is installed; clean "unavailable" otherwise.
- **NVIDIA GPU**: nvidia-smi probe; reports each GPU's VRAM + driver if
  the driver is installed; clean "unavailable" otherwise.

Pure stdlib + subprocess. Lazy: importing the module is cheap; probing
is a function call. Every probe has a short timeout so a misbehaving
driver tool doesn't hang Coach.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_PROBE_TIMEOUT_S = 4.0


# ---- CPU --------------------------------------------------------------------


class CPUInfo(BaseModel):
    """What `Hardware available → CPU` reports."""

    model_config = ConfigDict(extra="forbid")

    available: bool = True
    model_name: str = ""
    vendor: str = Field(
        default="",
        description="Coarse vendor tag: 'amd' / 'intel' / 'arm' / '' for unknown.",
    )
    is_ryzen: bool = Field(
        default=False,
        description="True for AMD Ryzen / EPYC / Threadripper (CCX-aware OMP affinity helps).",
    )
    cores: int = 0
    threads: int = 0
    load_avg_1m: float | None = None
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    note: str | None = None


def _parse_cpuinfo() -> tuple[str, str]:
    """Return (model_name, vendor_tag) from /proc/cpuinfo.

    On non-Linux hosts /proc/cpuinfo is absent; returns empty strings so
    the caller falls back to a generic 'CPU' label.
    """
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return "", ""
    try:
        text = path.read_text()
    except OSError:
        return "", ""
    model = ""
    vendor = ""
    for line in text.splitlines():
        if line.startswith("model name") and ":" in line and not model:
            model = line.split(":", 1)[1].strip()
        elif line.startswith("vendor_id") and ":" in line and not vendor:
            vendor = line.split(":", 1)[1].strip()
        if model and vendor:
            break
    return model, vendor


def _parse_meminfo() -> tuple[float, float]:
    """Return (total_gb, available_gb) from /proc/meminfo, (0, 0) on failure."""
    path = Path("/proc/meminfo")
    if not path.exists():
        return 0.0, 0.0
    try:
        text = path.read_text()
    except OSError:
        return 0.0, 0.0
    total_kb = 0
    avail_kb = 0
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
    return total_kb / (1024 * 1024), avail_kb / (1024 * 1024)


def probe_cpu() -> CPUInfo:
    """Build a CPUInfo for the current host."""
    cores = os.cpu_count() or 1
    model, vendor_raw = _parse_cpuinfo()
    vendor = ""
    if "AMD" in vendor_raw or "AMD" in model or "AuthenticAMD" in vendor_raw:
        vendor = "amd"
    elif "Intel" in vendor_raw or "Intel" in model or "GenuineIntel" in vendor_raw:
        vendor = "intel"
    elif "ARM" in vendor_raw or "ARM" in model:
        vendor = "arm"
    is_ryzen = bool(
        vendor == "amd"
        and re.search(r"\b(Ryzen|EPYC|Threadripper)\b", model, re.IGNORECASE),
    )

    load_1m: float | None
    try:
        load_1m = os.getloadavg()[0]
    except (OSError, AttributeError):
        load_1m = None

    total_gb, avail_gb = _parse_meminfo()

    note: str | None = None
    if not model:
        note = "/proc/cpuinfo unavailable — CPU model name not detectable."
    elif vendor == "amd" and not is_ryzen:
        note = (
            "AMD CPU detected but not the Ryzen / EPYC family. CCX-aware "
            "OMP affinity is still safe to enable; no harm if ignored."
        )

    return CPUInfo(
        available=True,
        model_name=model or "CPU",
        vendor=vendor,
        is_ryzen=is_ryzen,
        cores=cores,
        threads=cores,  # logical = physical * SMT; /proc/cpuinfo gives logical
        load_avg_1m=load_1m,
        ram_total_gb=round(total_gb, 2),
        ram_available_gb=round(avail_gb, 2),
        note=note,
    )


# ---- AMD GPU ----------------------------------------------------------------


class AMDGPU(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    vram_gb: float
    driver_version: str = ""


class AMDInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = False
    gpus: list[AMDGPU] = Field(default_factory=list)
    rocm_version: str = ""
    note: str | None = None


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def probe_amd() -> AMDInfo:
    """Probe AMD GPUs via rocm-smi.

    Detects whether ROCm is installed and what GPUs it sees. Clean
    "unavailable" return when rocm-smi isn't on PATH (the common case on
    a CPU-only laptop).
    """
    rocm_smi = _which("rocm-smi")
    if rocm_smi is None:
        return AMDInfo(available=False, note="rocm-smi not found on PATH")
    try:
        # `rocm-smi --showid --showmeminfo vram --showdriverversion --json`
        # is the JSON entrypoint; some rocm versions don't accept --json
        # so we fall back to the human-readable parser below if needed.
        result = subprocess.run(
            [rocm_smi, "--showid", "--showmeminfo", "vram",
             "--showdriverversion", "--json"],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return AMDInfo(available=False, note=f"rocm-smi failed: {exc!s}")
    if result.returncode != 0:
        return AMDInfo(
            available=False,
            note=f"rocm-smi exited rc={result.returncode}: {result.stderr.strip()[:200]}",
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        # rocm-smi versions that don't support --json print plain text;
        # surface a soft "available but indeterminate" instead of erroring.
        return AMDInfo(
            available=True,
            note="rocm-smi present but --json not supported on this version",
        )

    gpus: list[AMDGPU] = []
    rocm_version = ""
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if key.lower().startswith("system"):
            rocm_version = value.get("Driver version", "")
            continue
        # GPU entries are keyed "card0" / "card1" etc.
        name = value.get("Card series") or value.get("Card model") or key
        vram_str = value.get("VRAM Total Memory (B)") or "0"
        try:
            vram_b = int(vram_str)
        except (TypeError, ValueError):
            vram_b = 0
        gpus.append(AMDGPU(
            name=str(name),
            vram_gb=round(vram_b / (1024 ** 3), 2),
            driver_version=rocm_version,
        ))

    return AMDInfo(
        available=bool(gpus),
        gpus=gpus,
        rocm_version=rocm_version,
        note=None if gpus else "rocm-smi succeeded but reported no GPUs",
    )


# ---- NVIDIA GPU -------------------------------------------------------------


class NVIDIAGPU(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    vram_gb: float
    driver_version: str = ""
    cuda_version: str = ""


class NVIDIAInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = False
    gpus: list[NVIDIAGPU] = Field(default_factory=list)
    driver_version: str = ""
    cuda_version: str = ""
    note: str | None = None


def probe_nvidia() -> NVIDIAInfo:
    """Probe NVIDIA GPUs via nvidia-smi."""
    nvidia_smi = _which("nvidia-smi")
    if nvidia_smi is None:
        return NVIDIAInfo(available=False, note="nvidia-smi not found on PATH")
    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return NVIDIAInfo(available=False, note=f"nvidia-smi failed: {exc!s}")
    if result.returncode != 0:
        return NVIDIAInfo(
            available=False,
            note=f"nvidia-smi exited rc={result.returncode}: {result.stderr.strip()[:200]}",
        )

    gpus: list[NVIDIAGPU] = []
    driver_version = ""
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, vram_mb_str, driver = parts[0], parts[1], parts[2]
        try:
            vram_mb = float(vram_mb_str)
        except ValueError:
            vram_mb = 0.0
        gpus.append(NVIDIAGPU(
            name=name,
            vram_gb=round(vram_mb / 1024.0, 2),
            driver_version=driver,
        ))
        driver_version = driver

    # CUDA version comes from a separate nvidia-smi call (the header).
    cuda_version = ""
    try:
        header = subprocess.run(
            [nvidia_smi], capture_output=True, text=True,
            timeout=_PROBE_TIMEOUT_S, check=False,
        )
        m = re.search(r"CUDA Version:\s*([0-9.]+)", header.stdout)
        if m:
            cuda_version = m.group(1)
    except (subprocess.TimeoutExpired, OSError):
        pass

    return NVIDIAInfo(
        available=bool(gpus),
        gpus=gpus,
        driver_version=driver_version,
        cuda_version=cuda_version,
        note=None if gpus else "nvidia-smi succeeded but reported no GPUs",
    )


# ---- composite profile ------------------------------------------------------


class HardwareProfile(BaseModel):
    """What the Coach UI's `Hardware available` card consumes."""

    model_config = ConfigDict(extra="forbid")

    cpu: CPUInfo
    amd: AMDInfo
    nvidia: NVIDIAInfo
    recommended_lane: str = Field(
        description=(
            "`trl_cpu` / `axolotl_amd` / `axolotl_cuda` — picks the best "
            "training lane based on what's actually detected."
        ),
    )


def recommend_lane(cpu: CPUInfo, amd: AMDInfo, nvidia: NVIDIAInfo) -> str:
    """Pick the most capable detected training lane.

    Priority: AMD GPU (the MI300X target) → NVIDIA local GPU → CPU.
    Returns one of: 'axolotl_amd', 'axolotl_cuda', 'trl_cpu'. Defaults
    to 'trl_cpu' which is always available.
    """
    if amd.available and amd.gpus:
        return "axolotl_amd"
    if nvidia.available and nvidia.gpus:
        return "axolotl_cuda"
    _ = cpu  # CPU is the universal fallback; argument kept for signature symmetry
    return "trl_cpu"


def probe_all() -> HardwareProfile:
    """One call returns the full hardware profile + recommendation."""
    cpu = probe_cpu()
    amd = probe_amd()
    nvidia = probe_nvidia()
    return HardwareProfile(
        cpu=cpu, amd=amd, nvidia=nvidia,
        recommended_lane=recommend_lane(cpu, amd, nvidia),
    )


__all__ = [
    "AMDGPU",
    "NVIDIAGPU",
    "AMDInfo",
    "CPUInfo",
    "HardwareProfile",
    "NVIDIAInfo",
    "probe_all",
    "probe_amd",
    "probe_cpu",
    "probe_nvidia",
    "recommend_lane",
]
