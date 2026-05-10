# SPDX-License-Identifier: Apache-2.0
# (c) 2026 BANKON — all rights reserved.
"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Prevent tests from picking up a developer's local .env file."""
    monkeypatch.chdir(tmp_path)
