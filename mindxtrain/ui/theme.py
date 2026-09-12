"""The UI's look: one dark palette, monospace, readable tables — Gradio 5 and 6 both.

Gradio 6 moved `theme`/`css` from `Blocks(...)` to `launch(...)` and dropped some Base tokens, so
nothing here is pinned to a major: tokens are offered only if `.set()` accepts them, and the CSS
covers what the tokens miss (dataframes and inline code render light otherwise).
"""
from __future__ import annotations

import inspect

import gradio as gr

CSS = """
:root, .gradio-container { --mt-bg:#0b0d10; --mt-panel:#12161b; --mt-line:rgba(140,160,180,.18); --mt-text:#dbe4ee;
  --mt-text2:#95a3b3; --mt-accent:#4fb3ff; --mt-good:#56d364; --mt-low:#e3b341; --mt-bad:#f85149; }
.gradio-container { background: radial-gradient(circle at 50% -10%, #151a21 0%, var(--mt-bg) 60%) !important;
  color: var(--mt-text) !important; font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; }
.mx-head h1 { font-size:1.5rem; margin:0; letter-spacing:.04em; }
.mx-sub { color: var(--mt-text2); font-size:.88rem; }
.mx-sub a, .gradio-container a { color: var(--mt-accent); }
.mx-good { color: var(--mt-good); } .mx-low { color: var(--mt-low); } .mx-bad { color: var(--mt-bad); }
.mx-kiln { font-size:.92rem; line-height:1.6; }
.mx-diag { color: var(--mt-text2); font-size:.84rem; margin-top:6px; }
.mx-bar { height:6px; background:#0c0e12; border:1px solid var(--mt-line); border-radius:5px; overflow:hidden; margin:6px 0; }
.mx-bar i { display:block; height:100%; background:var(--mt-good); }
/* Gradio 6 tables and inline code ignore the Base tokens: light rows, invisible chips. */
.gradio-container { --table-even-background-fill:#141922; --table-odd-background-fill:#10141a;
  --table-border-color: rgba(140,160,180,.18); --table-text-color:#dbe4ee; --code-background-fill: rgba(79,179,255,.10); }
.gradio-container code, .gradio-container .prose code { background: rgba(79,179,255,.10) !important; color:#dbe4ee !important;
  padding:1px 5px; border-radius:4px; border:none !important; }
.gradio-container table, .gradio-container .table-wrap { background: var(--mt-panel) !important; }
.gradio-container th, .gradio-container thead td { background:#161c24 !important; color: var(--mt-text2) !important; }
.gradio-container td { background:#10141a !important; color: var(--mt-text) !important; border-color: rgba(140,160,180,.18) !important; }
.gradio-container tbody tr:nth-child(even) td { background:#141922 !important; }
.gradio-container td *, .gradio-container th * { color: inherit !important; background: transparent !important; }
footer { display:none !important; }
"""


def _tokens() -> dict:
    want = {"body_background_fill": "#0b0d10", "body_background_fill_dark": "#0b0d10",
            "block_background_fill": "#12161b", "block_background_fill_dark": "#12161b",
            "body_text_color": "#dbe4ee", "body_text_color_dark": "#dbe4ee",
            "block_border_color": "rgba(140,160,180,.18)", "block_border_color_dark": "rgba(140,160,180,.18)",
            "button_primary_background_fill": "#18202a", "button_primary_background_fill_dark": "#18202a",
            "button_primary_text_color": "#4fb3ff", "button_primary_text_color_dark": "#4fb3ff",
            "input_background_fill": "#0c1015", "input_background_fill_dark": "#0c1015",
            "table_even_background_fill": "#141922", "table_even_background_fill_dark": "#141922",
            "table_odd_background_fill": "#10141a", "table_odd_background_fill_dark": "#10141a",
            "table_text_color": "#dbe4ee", "table_text_color_dark": "#dbe4ee",
            "code_background_fill": "rgba(79,179,255,.10)", "code_background_fill_dark": "rgba(79,179,255,.10)"}
    try:
        ok = set(inspect.signature(gr.themes.Base.set).parameters)
    except Exception:  # noqa: BLE001
        return {}
    return {k: v for k, v in want.items() if k in ok}


def theme() -> "gr.themes.Base":
    try:
        return gr.themes.Base(primary_hue="blue", neutral_hue="slate",
                              font=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"]).set(**_tokens())
    except Exception:  # noqa: BLE001
        return gr.themes.Base()
