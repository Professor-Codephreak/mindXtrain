"""Hardware diagnostics — CPU/AMD/NVIDIA detection.

Probes shell out to rocm-smi / nvidia-smi when available; tests
monkey-patch shutil.which and subprocess.run so the suite stays
offline and reproducible regardless of the host hardware.
"""
from __future__ import annotations

import subprocess

import pytest

from mindxtrain.operator.coach import hw_diagnostics as hw

# ---- CPU --------------------------------------------------------------------


def test_probe_cpu_returns_available():
    """Every host has a CPU; probe never returns available=False."""
    info = hw.probe_cpu()
    assert info.available is True
    assert info.cores >= 1
    assert info.threads >= 1


def test_probe_cpu_detects_ryzen_when_model_name_matches(monkeypatch):
    """Test the Ryzen detection branch with a synthetic /proc/cpuinfo."""
    monkeypatch.setattr(hw, "_parse_cpuinfo",
                        lambda: ("AMD Ryzen 5 5600U", "AuthenticAMD"))
    monkeypatch.setattr(hw, "_parse_meminfo", lambda: (16.0, 8.0))
    info = hw.probe_cpu()
    assert info.vendor == "amd"
    assert info.is_ryzen is True
    assert info.ram_total_gb == 16.0


def test_probe_cpu_detects_intel(monkeypatch):
    monkeypatch.setattr(hw, "_parse_cpuinfo",
                        lambda: ("Intel(R) Core(TM) i7-12700K", "GenuineIntel"))
    monkeypatch.setattr(hw, "_parse_meminfo", lambda: (32.0, 16.0))
    info = hw.probe_cpu()
    assert info.vendor == "intel"
    assert info.is_ryzen is False


def test_probe_cpu_unknown_vendor_no_model(monkeypatch):
    """When /proc/cpuinfo is missing, surface a friendly note."""
    monkeypatch.setattr(hw, "_parse_cpuinfo", lambda: ("", ""))
    monkeypatch.setattr(hw, "_parse_meminfo", lambda: (0.0, 0.0))
    info = hw.probe_cpu()
    assert info.model_name == "CPU"
    assert info.vendor == ""
    assert info.is_ryzen is False
    assert info.note is not None
    assert "cpuinfo" in info.note.lower()


def test_probe_cpu_amd_non_ryzen_carries_friendly_note(monkeypatch):
    """An AMD chip that isn't Ryzen / EPYC still gets a usable note."""
    monkeypatch.setattr(hw, "_parse_cpuinfo",
                        lambda: ("AMD Athlon X4 760K", "AuthenticAMD"))
    monkeypatch.setattr(hw, "_parse_meminfo", lambda: (8.0, 4.0))
    info = hw.probe_cpu()
    assert info.vendor == "amd"
    assert info.is_ryzen is False
    assert info.note is not None
    assert "amd" in info.note.lower()


# ---- AMD GPU ----------------------------------------------------------------


def test_probe_amd_unavailable_when_rocm_smi_missing(monkeypatch):
    monkeypatch.setattr(hw, "_which", lambda _: None)
    info = hw.probe_amd()
    assert info.available is False
    assert info.gpus == []
    assert info.note is not None
    assert "rocm-smi" in info.note


def test_probe_amd_parses_json_output(monkeypatch):
    """Synthesise a rocm-smi --json response with one MI300X."""
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/rocm-smi")
    fake_json = (
        '{"card0": {"Card series": "AMD Instinct MI300X", '
        '"VRAM Total Memory (B)": "206158430208"}, '
        '"system": {"Driver version": "7.2.1"}}'
    )
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=0, stdout=fake_json, stderr=""))
    info = hw.probe_amd()
    assert info.available is True
    assert len(info.gpus) == 1
    assert "MI300X" in info.gpus[0].name
    # 206158430208 bytes / 1024^3 ≈ 191.97 GB → rounds to 191.97
    assert info.gpus[0].vram_gb == pytest.approx(191.97, abs=0.05)
    assert info.rocm_version == "7.2.1"


def test_probe_amd_handles_rocm_smi_failure(monkeypatch):
    """A non-zero rc → unavailable with the stderr surfaced."""
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/rocm-smi")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=1, stdout="",
                            stderr="no AMD GPUs detected"))
    info = hw.probe_amd()
    assert info.available is False
    assert info.note is not None
    assert "rc=1" in info.note


def test_probe_amd_handles_timeout(monkeypatch):
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/rocm-smi")

    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="rocm-smi", timeout=4)
    monkeypatch.setattr(subprocess, "run", _raise)
    info = hw.probe_amd()
    assert info.available is False
    assert info.note is not None


def test_probe_amd_handles_non_json_output(monkeypatch):
    """Older rocm-smi versions print plain text — graceful soft signal."""
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/rocm-smi")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=0,
                            stdout="GPU[0]: AMD MI300X\n", stderr=""))
    info = hw.probe_amd()
    # We say "available" because rocm-smi ran cleanly; just couldn't parse.
    assert info.available is True
    assert info.note is not None
    assert "json" in info.note.lower()


# ---- NVIDIA GPU -------------------------------------------------------------


def test_probe_nvidia_unavailable_when_nvidia_smi_missing(monkeypatch):
    monkeypatch.setattr(hw, "_which", lambda _: None)
    info = hw.probe_nvidia()
    assert info.available is False
    assert info.gpus == []


def test_probe_nvidia_parses_csv_output(monkeypatch):
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/nvidia-smi")
    fake_csv = "NVIDIA H100 80GB HBM3, 81920, 560.35"
    # Two subprocess.run calls happen: one for csv, one for the header CUDA line.
    call_count = {"n": 0}

    def fake_run(*a, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return subprocess.CompletedProcess(
                args=a, returncode=0, stdout=fake_csv, stderr="",
            )
        return subprocess.CompletedProcess(
            args=a, returncode=0, stdout="CUDA Version: 12.4\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    info = hw.probe_nvidia()
    assert info.available is True
    assert len(info.gpus) == 1
    assert "H100" in info.gpus[0].name
    # 81920 MB / 1024 = 80 GB
    assert info.gpus[0].vram_gb == pytest.approx(80.0, abs=0.01)
    assert info.driver_version == "560.35"
    assert info.cuda_version == "12.4"


def test_probe_nvidia_multiple_gpus(monkeypatch):
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/nvidia-smi")
    fake_csv = (
        "NVIDIA H100 80GB HBM3, 81920, 560.35\n"
        "NVIDIA H100 80GB HBM3, 81920, 560.35\n"
    )
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=0, stdout=fake_csv, stderr=""))
    info = hw.probe_nvidia()
    assert len(info.gpus) == 2


def test_probe_nvidia_handles_rc_nonzero(monkeypatch):
    monkeypatch.setattr(hw, "_which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=9, stdout="",
                            stderr="NVIDIA-SMI has failed"))
    info = hw.probe_nvidia()
    assert info.available is False


# ---- recommended lane -------------------------------------------------------


def test_recommend_lane_prefers_amd_when_available():
    cpu = hw.CPUInfo()
    amd = hw.AMDInfo(available=True, gpus=[hw.AMDGPU(name="MI300X", vram_gb=192)])
    nv = hw.NVIDIAInfo(available=True, gpus=[hw.NVIDIAGPU(name="H100", vram_gb=80)])
    assert hw.recommend_lane(cpu, amd, nv) == "axolotl_amd"


def test_recommend_lane_falls_back_to_cuda():
    cpu = hw.CPUInfo()
    amd = hw.AMDInfo(available=False)
    nv = hw.NVIDIAInfo(available=True, gpus=[hw.NVIDIAGPU(name="H100", vram_gb=80)])
    assert hw.recommend_lane(cpu, amd, nv) == "axolotl_cuda"


def test_recommend_lane_falls_back_to_cpu():
    """The default — CPU is always the universal fallback."""
    cpu = hw.CPUInfo()
    amd = hw.AMDInfo(available=False)
    nv = hw.NVIDIAInfo(available=False)
    assert hw.recommend_lane(cpu, amd, nv) == "trl_cpu"


def test_recommend_lane_treats_empty_gpu_list_as_unavailable():
    """available=True with no GPUs should still skip the lane."""
    cpu = hw.CPUInfo()
    amd = hw.AMDInfo(available=True, gpus=[])
    nv = hw.NVIDIAInfo(available=True, gpus=[])
    assert hw.recommend_lane(cpu, amd, nv) == "trl_cpu"


# ---- probe_all (composite) --------------------------------------------------


def test_probe_all_returns_profile():
    profile = hw.probe_all()
    assert isinstance(profile, hw.HardwareProfile)
    # The CPU half is always populated on a real host.
    assert profile.cpu.available is True
    # On a CPU-only laptop, the recommendation is trl_cpu.
    assert profile.recommended_lane in {"trl_cpu", "axolotl_amd", "axolotl_cuda"}
