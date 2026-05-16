"""Coach API for hardware diagnostics."""
from __future__ import annotations

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


def test_diagnostics_returns_full_profile():
    """Endpoint returns the composite hardware profile."""
    r = client.get("/coach/api/diagnostics/hardware")
    assert r.status_code == 200
    data = r.json()
    # Shape contract: every consumer relies on these keys.
    assert "cpu" in data
    assert "amd" in data
    assert "nvidia" in data
    assert "recommended_lane" in data
    # CPU panel is always populated.
    assert data["cpu"]["available"] is True
    assert data["cpu"]["cores"] >= 1
    # Recommendation is always one of the three lanes.
    assert data["recommended_lane"] in {"trl_cpu", "axolotl_amd", "axolotl_cuda"}


def test_diagnostics_cpu_has_expected_fields():
    r = client.get("/coach/api/diagnostics/hardware")
    cpu = r.json()["cpu"]
    for key in ("model_name", "vendor", "is_ryzen", "cores", "threads",
                "ram_total_gb", "ram_available_gb"):
        assert key in cpu, f"missing CPU field: {key}"


def test_diagnostics_amd_panel_shape():
    r = client.get("/coach/api/diagnostics/hardware")
    amd = r.json()["amd"]
    assert "available" in amd
    assert "gpus" in amd
    assert isinstance(amd["gpus"], list)
    if amd["available"]:
        for gpu in amd["gpus"]:
            assert "name" in gpu
            assert "vram_gb" in gpu


def test_diagnostics_nvidia_panel_shape():
    r = client.get("/coach/api/diagnostics/hardware")
    nv = r.json()["nvidia"]
    assert "available" in nv
    assert "gpus" in nv


def test_diagnostics_is_jsonable():
    """The endpoint output must be JSON-serializable without further
    massaging (the Coach JS does JSON.parse straight to the renderer)."""
    import json
    r = client.get("/coach/api/diagnostics/hardware")
    body = r.text
    # Round-trip.
    parsed = json.loads(body)
    assert "cpu" in parsed
