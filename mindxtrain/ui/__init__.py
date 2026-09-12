"""mindxtrain.ui — the framework as one Gradio surface (Basic · Advanced · Scientific).

    from mindxtrain.ui import build, main
    main(port=7862)                     # or: mindxtrain ui
"""
from .app import VERSION, build, main  # noqa: F401
from .metrics import RunMetrics, parse_log  # noqa: F401

__all__ = ["build", "main", "VERSION", "RunMetrics", "parse_log"]
