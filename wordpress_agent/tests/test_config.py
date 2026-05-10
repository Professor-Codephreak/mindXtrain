# SPDX-License-Identifier: Apache-2.0
# (c) 2026 BANKON — all rights reserved.
"""Tests for environment-driven configuration."""
from __future__ import annotations

import pytest

from wordpress_agent.config import Settings


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WP_BASE_URL", "https://rage.pythai.net")
    monkeypatch.setenv("WP_USER", "codephreak")
    monkeypatch.setenv("WP_APP_PASSWORD", "abcd-efgh-ijkl-mnop-qrst-uvwx")
    s = Settings()  # type: ignore[call-arg]
    assert s.base_url_str == "https://rage.pythai.net"
    assert s.user == "codephreak"
    assert s.app_password_value == "abcd-efgh-ijkl-mnop-qrst-uvwx"
    assert s.retry_count == 3
    assert s.timeout == 30.0


def test_settings_strip_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WP_BASE_URL", "https://rage.pythai.net/")
    monkeypatch.setenv("WP_USER", "codephreak")
    monkeypatch.setenv("WP_APP_PASSWORD", "abcd-efgh-ijkl-mnop-qrst-uvwx")
    s = Settings()  # type: ignore[call-arg]
    assert s.base_url_str == "https://rage.pythai.net"


def test_settings_secret_not_repr_leaked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WP_BASE_URL", "https://rage.pythai.net")
    monkeypatch.setenv("WP_USER", "codephreak")
    monkeypatch.setenv("WP_APP_PASSWORD", "leak-me-not")
    s = Settings()  # type: ignore[call-arg]
    assert "leak-me-not" not in repr(s)
