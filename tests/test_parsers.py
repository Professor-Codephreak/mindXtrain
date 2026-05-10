"""Chat-template render + parse round-trips."""

from __future__ import annotations

from mindxtrain.models.chat_template import (
    ChatMessage,
    HermesTemplate,
    Qwen3CoderTemplate,
    Qwen3ReasoningTemplate,
    get_template,
    list_templates,
)

MSGS = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is 2 + 2?"),
]


def test_hermes_render_has_chatml_tags():
    t = HermesTemplate()
    out = t.render(MSGS)
    assert "<|im_start|>system" in out
    assert "<|im_end|>" in out
    assert out.endswith("<|im_start|>assistant\n")


def test_hermes_render_no_generation_prompt():
    t = HermesTemplate()
    out = t.render(MSGS, add_generation_prompt=False)
    assert not out.endswith("assistant\n")


def test_hermes_parse_strips_im_end():
    t = HermesTemplate()
    parsed = t.parse_response("4<|im_end|>\n<|im_start|>user\nbye")
    assert parsed == {"content": "4"}


def test_qwen3_coder_extracts_tool_call():
    t = Qwen3CoderTemplate()
    response = (
        "Sure, I'll call the tool.<tool_call>"
        '{"name":"add","arguments":{"a":2,"b":2}}</tool_call><|im_end|>'
    )
    parsed = t.parse_response(response)
    assert parsed["content"] == "Sure, I'll call the tool."
    assert "tool_call" in parsed
    assert '"name":"add"' in parsed["tool_call"]


def test_qwen3_reasoning_separates_thinking_from_content():
    t = Qwen3ReasoningTemplate()
    response = "<think>let me work this out: 2+2 is 4</think>The answer is 4.<|im_end|>"
    parsed = t.parse_response(response)
    assert parsed["content"] == "The answer is 4."
    assert parsed["thinking"] == "let me work this out: 2+2 is 4"


def test_qwen3_reasoning_no_thinking_block():
    t = Qwen3ReasoningTemplate()
    parsed = t.parse_response("Just an answer.<|im_end|>")
    assert parsed == {"content": "Just an answer."}


def test_get_template_known_names():
    assert get_template("hermes").name == "hermes"
    assert get_template("qwen3_coder").name == "qwen3_coder"
    assert get_template("qwen3").name == "qwen3_reasoning"
    assert get_template("deepseek_r1").name == "qwen3_reasoning"


def test_get_template_unknown_falls_back_to_hermes():
    assert get_template("nonsense").name == "hermes"


def test_list_templates_includes_canonical_names():
    names = list_templates()
    for required in ("hermes", "qwen3_coder", "qwen3", "qwen3_reasoning", "deepseek_r1"):
        assert required in names
