"""Coach API: recipes / bench / compile / cost / health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mindxtrain.operator.app import app

client = TestClient(app)


def test_root_redirects_to_coach():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/coach/"


def test_coach_index_serves_html():
    r = client.get("/coach/")
    assert r.status_code == 200
    assert "<title>mindXtrain" in r.text
    assert "/coach/static/coach.js" in r.text


def test_coach_static_files_served():
    r_css = client.get("/coach/static/style.css")
    r_js = client.get("/coach/static/coach.js")
    assert r_css.status_code == 200
    assert r_js.status_code == 200
    assert "AMD orange" in r_css.text or "--accent" in r_css.text
    assert "loadRecipes" in r_js.text


def test_recipes_list_includes_known_recipes():
    r = client.get("/coach/api/recipes")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 12
    names = {item["name"] for item in items}
    assert "qwen3_8b_sft_lora" in names
    assert "instella_3b_lora" in names
    assert "mindx_fallback_qwen3_1_5b_sft_lora" in names
    assert "mindx_fallback_qwen3_1_5b_cpu" in names
    for item in items:
        assert "base_model" in item
        assert "method" in item
        assert "gpus" in item


def test_recipe_detail_returns_yaml_and_summary():
    r = client.get("/coach/api/recipes/qwen3_8b_sft_lora")
    assert r.status_code == 200
    data = r.json()
    assert "yaml" in data
    assert "Qwen/Qwen3-8B" in data["yaml"]
    assert data["summary"]["base_model"] == "Qwen/Qwen3-8B"
    assert data["summary"]["method"] == "lora"


def test_recipe_detail_404_for_unknown():
    r = client.get("/coach/api/recipes/does_not_exist")
    assert r.status_code == 404


def test_bench_returns_autotune_plan():
    r = client.post("/coach/api/bench")
    assert r.status_code == 200
    plan = r.json()
    assert plan["schema_version"] == "1"
    assert plan["gpu_arch"] == "gfx942"
    assert plan["attention_backend"] in ("ck", "triton")


def test_compile_returns_axolotl_yaml_and_overrides():
    r = client.post(
        "/coach/api/compile",
        json={"recipe": "qwen3_8b_sft_lora"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["recipe"] == "qwen3_8b_sft_lora"
    assert data["axolotl_yaml"]["base_model"] == "Qwen/Qwen3-8B"
    assert data["axolotl_yaml"]["adapter"] == "lora"
    assert any("attention_backend" in o for o in data["overrides"])


def test_compile_404_for_unknown_recipe():
    r = client.post("/coach/api/compile", json={"recipe": "ghost"})
    assert r.status_code == 404


def test_cost_returns_three_breakdowns():
    r = client.post("/coach/api/cost", json={"gpus": 1, "hours": 1.5})
    assert r.status_code == 200
    data = r.json()
    for key in ("mi300x", "h100", "h200"):
        assert key in data
        assert data[key]["cost_usdc"] > 0
    # MI300X must come out cheapest in this configuration.
    assert data["mi300x"]["cost_usdc"] < data["h100"]["cost_usdc"]
    assert data["speedup_vs_h100_x"] > 1.0
    assert data["mi300x"]["fits_qwen3_8b_bf16_bs8_seq4096"] is True
    assert data["h100"]["fits_qwen3_8b_bf16_bs8_seq4096"] is False


def test_cost_validates_input():
    r = client.post("/coach/api/cost", json={"gpus": 0, "hours": 1.5})
    assert r.status_code == 422


def test_health_endpoint_reports_recipes_count():
    r = client.get("/coach/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["recipes_available"] >= 12
    assert data["chat_backend_ready"] is False


def test_app_health_mentions_coach_url():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["coach_url"] == "/coach/"
