"""ContextManager compaction."""

from __future__ import annotations

from mindxtrain.operator.context import ContextManager, ContextManagerConfig, message_tokens


def test_message_tokens_estimate():
    m = {"role": "user", "content": "hello world"}
    assert message_tokens(m) > 0


def test_compact_under_budget_passthrough():
    cm = ContextManager(ContextManagerConfig(target_tokens=10_000_000))
    msgs = [{"role": "user", "content": "x"}]
    assert cm.compact(msgs) == msgs


def test_compact_above_budget_trims_middle():
    msgs = [{"role": "system", "content": "sys"}]
    msgs.extend({"role": "user", "content": "x" * 10_000} for _ in range(20))
    cm = ContextManager(ContextManagerConfig(target_tokens=5_000, keep_recent=4))
    out = cm.compact(msgs)
    assert out[0]["role"] == "system"
    # Should be: system + summary marker + 4 recent
    assert any("compacted" in m.get("content", "") for m in out)
    assert len(out) <= 6
