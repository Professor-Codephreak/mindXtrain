"""Held-out CE-loss scorer for an adapted checkpoint.

The heavyweight torch/transformers/peft path is mocked here — real
end-to-end coverage happens via `mindxtrain eval-checkpoint` against
a freshly trained adapter in the `_real` recipe demo run. These tests
just verify the orchestration (JSONL iteration, sink callback, both
loss invocations, summary shape) is sound.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _install_fake_ml_stack(monkeypatch, base_loss: float, adapter_loss: float) -> None:
    """Install fake `transformers` + `peft` + `torch` modules.

    The fake AutoModelForCausalLM returns sequentially `base_loss` then
    `adapter_loss` so the score function picks up a non-trivial delta.
    """
    class _FakeOutput:
        def __init__(self, loss_value: float) -> None:
            class _T:
                def detach(self_inner):
                    return self_inner
                def cpu(self_inner):
                    return self_inner
                def item(self_inner):
                    return loss_value
            self.loss = _T()

    class _FakeBaseModel:
        # Base-model branch — returns base_loss every call.
        _loss = base_loss
        def eval(self):
            return self
        def __call__(self, **_kw):
            return _FakeOutput(self.__class__._loss)

    class _FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return " ".join(m.get("content", "") for m in messages)
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=1024):
            import torch  # the fake one
            return {"input_ids": torch.tensor([[0, 1, 2]])}

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name, *a, **kw):
            return _FakeBaseModel()

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, *a, **kw):
            return _FakeTokenizer()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _FakeAutoModel
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class _FakePeftModel(_FakeBaseModel):
        # Adapter branch — overrides _loss so PeftModel calls return adapter_loss.
        _loss = adapter_loss

    class _FakePeftFactory:
        @staticmethod
        def from_pretrained(base, adapter_dir):
            return _FakePeftModel()

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = _FakePeftFactory
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    class _FakeTorch:
        @staticmethod
        def tensor(data):
            return data
        class no_grad:
            def __enter__(self): return self
            def __exit__(self, *a): return False
    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = _FakeTorch.tensor
    fake_torch.no_grad = _FakeTorch.no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_score_checkpoint_returns_summary(tmp_path, monkeypatch):
    """Happy path: 2 rows, mocked models, delta == adapter - base."""
    _install_fake_ml_stack(monkeypatch, base_loss=2.5, adapter_loss=2.1)

    from mindxtrain.eval.held_out_loss import score_checkpoint

    jsonl = tmp_path / "training.jsonl"
    _write_jsonl(jsonl, [
        {"messages": [{"role": "user", "content": "what is mindX?"},
                      {"role": "assistant", "content": "a cognitive runtime"}]},
        {"messages": [{"role": "user", "content": "dream cycle?"},
                      {"role": "assistant", "content": "phase 5b output"}]},
    ])

    score = score_checkpoint(
        adapter_dir=tmp_path / "adapter",
        base_model="HuggingFaceTB/SmolLM2-135M",
        jsonl_path=jsonl,
        max_samples=8,
    )
    assert score.n == 2
    assert score.base_loss == pytest.approx(2.5)
    assert score.adapter_loss == pytest.approx(2.1)
    assert score.delta == pytest.approx(-0.4)
    assert score.delta < 0


def test_score_checkpoint_streams_through_sink(tmp_path, monkeypatch):
    _install_fake_ml_stack(monkeypatch, base_loss=2.0, adapter_loss=2.0)
    from mindxtrain.eval.held_out_loss import score_checkpoint

    jsonl = tmp_path / "t.jsonl"
    _write_jsonl(jsonl, [{"messages": [{"role": "user", "content": "x"}]}])

    lines: list[str] = []
    score_checkpoint(
        adapter_dir=tmp_path / "adapter",
        base_model="b",
        jsonl_path=jsonl,
        max_samples=1,
        sink=lines.append,
    )
    joined = "\n".join(lines)
    # The progress prints surface key stages so the CLI / Coach UI can
    # show the user what's happening during a multi-minute scoring run.
    assert "held-out rows from" in joined
    assert "base mean loss" in joined
    assert "adapter mean loss" in joined
    assert "delta" in joined


def test_score_checkpoint_empty_jsonl_raises(tmp_path, monkeypatch):
    _install_fake_ml_stack(monkeypatch, 2.0, 2.0)
    from mindxtrain.eval.held_out_loss import score_checkpoint

    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no readable JSONL rows"):
        score_checkpoint(
            adapter_dir=tmp_path / "adapter",
            base_model="b",
            jsonl_path=jsonl,
        )


def test_score_checkpoint_skips_malformed_lines(tmp_path, monkeypatch):
    """Bad JSON / non-messages rows are silently skipped (matches iter_mindx_dreams)."""
    _install_fake_ml_stack(monkeypatch, 1.5, 1.2)
    from mindxtrain.eval.held_out_loss import score_checkpoint

    jsonl = tmp_path / "messy.jsonl"
    jsonl.write_text(
        "not-json\n"
        '{"no_messages": true}\n'
        '{"messages": [{"role": "user", "content": "real row"}]}\n',
        encoding="utf-8",
    )

    score = score_checkpoint(
        adapter_dir=tmp_path / "adapter", base_model="b", jsonl_path=jsonl,
    )
    assert score.n == 1


def test_score_checkpoint_caps_at_max_samples(tmp_path, monkeypatch):
    _install_fake_ml_stack(monkeypatch, 1.0, 0.9)
    from mindxtrain.eval.held_out_loss import score_checkpoint

    jsonl = tmp_path / "big.jsonl"
    _write_jsonl(jsonl, [
        {"messages": [{"role": "user", "content": f"row {i}"}]}
        for i in range(50)
    ])

    score = score_checkpoint(
        adapter_dir=tmp_path / "adapter", base_model="b", jsonl_path=jsonl,
        max_samples=3,
    )
    assert score.n == 3


def test_score_checkpoint_missing_extras_raises(tmp_path, monkeypatch):
    """No peft/transformers → ImportError with the install hint."""
    # Block the imports inside score_checkpoint by removing them from
    # sys.modules and shadowing the importer.
    monkeypatch.setitem(sys.modules, "peft", None)
    from mindxtrain.eval.held_out_loss import score_checkpoint
    jsonl = tmp_path / "t.jsonl"
    _write_jsonl(jsonl, [{"messages": [{"role": "user", "content": "x"}]}])

    with pytest.raises(ImportError, match="uv sync --extra ml"):
        score_checkpoint(
            adapter_dir=tmp_path / "adapter", base_model="b", jsonl_path=jsonl,
        )


def test_as_dict_round_trip(tmp_path, monkeypatch):
    _install_fake_ml_stack(monkeypatch, 2.0, 1.7)
    from mindxtrain.eval.held_out_loss import score_checkpoint

    jsonl = tmp_path / "t.jsonl"
    _write_jsonl(jsonl, [{"messages": [{"role": "user", "content": "x"}]}])

    score = score_checkpoint(
        adapter_dir=tmp_path / "adapter", base_model="b", jsonl_path=jsonl,
    )
    d = score.as_dict()
    # The CLI uses console.print_json(score.as_dict()), so it must be
    # a flat dict with the documented keys.
    assert set(d) == {"base_model", "adapter_dir", "n", "base_loss", "adapter_loss", "delta"}
    assert d["base_loss"] == pytest.approx(2.0)
    assert d["adapter_loss"] == pytest.approx(1.7)
