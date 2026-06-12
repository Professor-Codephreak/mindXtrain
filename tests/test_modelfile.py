"""Ollama Modelfile builder — spec → text rendering."""

from __future__ import annotations

from mindxtrain.deploy.modelfile import (
    MODELFILE_PARAMS,
    ModelfileMessage,
    ModelfileSpec,
    render_modelfile,
    write_modelfile,
)


def test_param_catalogue_covers_key_params():
    names = {p.name for p in MODELFILE_PARAMS}
    for required in ("num_ctx", "temperature", "top_k", "top_p", "repeat_penalty",
                     "seed", "num_predict", "mirostat", "min_p"):
        assert required in names
    # Every param declares a type.
    assert all(p.type in ("int", "float", "string", "bool") for p in MODELFILE_PARAMS)


def test_render_minimal():
    text = render_modelfile(ModelfileSpec(from_model="qwen3:0.6b"))
    assert text.strip() == "FROM qwen3:0.6b"


def test_render_full_instructions():
    spec = ModelfileSpec(
        from_model="HuggingFaceTB/SmolLM2-135M",
        system="You are Codephreak.",
        template="{{ .System }}\n{{ .Prompt }}",
        adapter="./out/runs/x/checkpoint",
        license="Apache-2.0",
        requires="0.5.0",
        parameters={"temperature": 0.7, "num_ctx": 4096, "top_k": 20},
        stop=["<|im_end|>", "User:"],
        messages=[ModelfileMessage(role="user", content="who are you?"),
                  ModelfileMessage(role="assistant", content="i am codephreak.")],
    )
    text = render_modelfile(spec)
    assert text.startswith("FROM HuggingFaceTB/SmolLM2-135M")
    assert "REQUIRES 0.5.0" in text
    assert "PARAMETER temperature 0.7" in text
    assert "PARAMETER num_ctx 4096" in text
    assert 'PARAMETER stop "<|im_end|>"' in text
    assert 'PARAMETER stop "User:"' in text
    assert 'SYSTEM """You are Codephreak."""' in text
    assert "TEMPLATE " in text and "{{ .Prompt }}" in text
    assert "ADAPTER ./out/runs/x/checkpoint" in text
    assert 'LICENSE """Apache-2.0"""' in text
    assert "MESSAGE user who are you?" in text
    assert "MESSAGE assistant i am codephreak." in text


def test_params_render_in_catalogue_order():
    # num_ctx precedes temperature in the catalogue → also in the output.
    text = render_modelfile(ModelfileSpec(
        from_model="m", parameters={"temperature": 0.5, "num_ctx": 1024},
    ))
    assert text.index("num_ctx") < text.index("temperature")


def test_int_param_not_rendered_as_float():
    text = render_modelfile(ModelfileSpec(from_model="m", parameters={"num_ctx": 2048.0}))
    assert "PARAMETER num_ctx 2048\n" in text
    assert "2048.0" not in text


def test_write_modelfile(tmp_path):
    p = write_modelfile(ModelfileSpec(from_model="m", system="hi"), tmp_path / "t" / "Modelfile")
    assert p.is_file()
    assert p.read_text().startswith("FROM m")
