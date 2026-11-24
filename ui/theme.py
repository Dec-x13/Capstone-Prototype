"""Theme: palette, CSS injection and one shared Plotly template. No network access, no web fonts."""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FONT_STACK = "Segoe UI, system-ui, -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif"

PALETTE = {
    "ink": "#16233A",
    "paper": "#FAF8F3",
    "shelf": "#ECEFF3",
    "teal": "#0E6B6B",
    "low": "#2F6B3F",      # moss
    "medium": "#8F5A00",   # ochre
    "high": "#9C3B2B",     # brick: muted, "worth verifying", not alarm red
}
RISK_COLORS = {"Low": PALETTE["low"], "Medium": PALETTE["medium"], "High": PALETTE["high"]}

MODES = {
    "light": {"text": PALETTE["ink"], "muted": "#4A5568", "rule": "#C9CFD8", "surface": "#FFFFFF",
              "accent": PALETTE["teal"], "sim": PALETTE["medium"], "grid": "#DDE2E8", "bar": PALETTE["teal"],
              "bar2": "#5B6B80"},
    "dark": {"text": "#ECEFF3", "muted": "#AAB4C2", "rule": "#3A4757", "surface": "#18222F",
             "accent": "#5FB8B3", "sim": "#E0A84A", "grid": "#2C3846", "bar": "#5FB8B3", "bar2": "#93A2B5"},
}
LANG_COLORS = {"Bikol": "#2F6B3F", "Tagalog": "#9C3B2B", "English": "#0E6B6B", "Ambiguous": "#7A8597"}

_CSS = (Path(__file__).resolve().parent / "theme.css").read_text(encoding="utf-8")


def mode() -> str:
    try:
        t = st.context.theme.type
    except Exception:  # bare mode / tests
        t = None
    return "dark" if t == "dark" else "light"


def colors() -> dict:
    return MODES[mode()]


def inject(presenter: bool = False) -> None:
    """Inject CSS variables for the active theme, the stylesheet, and presenter-mode sizing."""
    c = colors()
    vars_css = ":root{" + "".join(f"--bg-{k}:{v};" for k, v in c.items()) + "}"
    presenter_css = ('[data-testid="stAppViewContainer"] .main, [data-testid="stMain"] { zoom: 1.12; }'
                     if presenter else "")
    st.html(f"<style>{vars_css}{_CSS}{presenter_css}</style>")


def _template(m: str) -> go.layout.Template:
    c = MODES[m]
    return go.layout.Template(layout=dict(
        font=dict(family=FONT_STACK, size=16, color=c["text"]),
        title=dict(font=dict(size=19, color=c["text"]), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[c["bar"], PALETTE["medium"], PALETTE["low"], PALETTE["high"], c["bar2"]],
        xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"], linecolor=c["rule"], ticks="outside",
                   tickcolor=c["rule"], title=dict(font=dict(size=15))),
        yaxis=dict(gridcolor=c["grid"], zerolinecolor=c["grid"], linecolor=c["rule"],
                   title=dict(font=dict(size=15))),
        margin=dict(l=10, r=20, t=48, b=40),
        hoverlabel=dict(font=dict(family=FONT_STACK, size=15)),
        showlegend=False,
    ))


for _m in MODES:
    pio.templates[f"behaviorguard_{_m}"] = _template(_m)


def template() -> str:
    return f"behaviorguard_{mode()}"


PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def show(fig: go.Figure, key: str, height: int | None = None) -> None:
    fig.update_layout(template=template())
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, theme=None, key=key)
