"""OpenTelemetry init hooks (lazy import — opt-in via `--extra obs`).

If `opentelemetry-sdk` is not installed, `init_otel()` is a graceful no-op
and `is_enabled()` returns False — calls into the package never raise.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        return False
    return True


def init_otel(service_name: str = "mindxtrain", endpoint: str | None = None) -> bool:
    """Initialize OpenTelemetry tracing if available; return whether init ran."""
    if not is_enabled():
        logger.info("opentelemetry-sdk not installed; skipping init_otel")
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover
        logger.warning("partial opentelemetry install (%s); skipping init", exc)
        return False

    endpoint = endpoint or os.environ.get("MINDXTRAIN_OTEL_ENDPOINT", "")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return True


__all__ = ["init_otel", "is_enabled"]
