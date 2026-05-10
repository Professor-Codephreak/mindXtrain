"""Prometheus exporter — opt-in via `--extra obs`.

If `prometheus_client` is not installed, `init_prometheus()` is a no-op.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        return False
    return True


def init_prometheus(port: int | None = None) -> bool:
    """Start the Prometheus metrics HTTP server if available; return success."""
    if not is_enabled():
        logger.info("prometheus_client not installed; skipping init_prometheus")
        return False
    try:
        from prometheus_client import start_http_server
    except ImportError:  # pragma: no cover
        return False
    bound_port = int(os.environ.get("MINDXTRAIN_PROMETHEUS_PORT", port or 9090))
    start_http_server(bound_port)
    logger.info("prometheus exporter listening on :%d/metrics", bound_port)
    return True


__all__ = ["init_prometheus", "is_enabled"]
