"""Reusable UI pieces. Every value that goes into HTML is escaped with esc(); message text is never
passed here except already masked."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import cfg
from core.explain import fmt_hour, habits_vs_length
from core.features import GROUP_LABELS, GROUPS
from core.pipeline import WindowResult
from ui.theme import RISK_COLORS, colors

STAGES = ["Export reader", "Language tagger", "Feature extractor", "Baseline builder", "Deviation scorer",
          "Explanation generator", "Interface"]
PAGES = ["Load data", "Your baseline", "Check a window", "Simulate a takeover", "About"]
PAGE_STAGES = {
    "Load data": {"Export reader"},
    "Your baseline": {"Language tagger", "Feature extractor", "Baseline builder"},
    "Check a window": {"Deviation scorer", "Explanation generator", "Interface"},
    "Simulate a takeover": {"Deviation scorer", "Explanation generator", "Interface"},
    "About": set(STAGES),
}
SYNTH_LABEL = "Synthetic sample (for testing only)"
CUTOFF_NOTE = "Prototype cutoffs, not the validated thresholds of Section 4.2.2."
LEVEL_SEGMENTS = {"Low": 1, "Medium": 2, "High": 3}
LEVEL_PHRASE = {"Low": "looks like this user", "Medium": "somewhat unusual · worth verifying",
                "High": "unusual for this user · worth verifying"}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- header
def header(synthetic: bool) -> None:
    synth = f' &nbsp;<span class="bg-label synth">{esc(SYNTH_LABEL)}</span>' if synthetic else ""
    md(f'<div class="bg-header"><p class="bg-name">BehaviorGuard</p>'
       f'<p class="bg-tagline">Checks whether a window of Messenger messages looks like the account owner.</p>'
       f'<p class="bg-status"><b>Feasibility prototype</b> · runs offline{synth}</p></div>')


def pipeline_strip(page: str) -> None:
    active = PAGE_STAGES.get(page, set())
    parts = []
    for i, s in enumerate(STAGES):
        on = s in active
        attrs = ' class="bg-stage on" aria-current="step"' if on else ' class="bg-stage"'
        parts.append(f"<span{attrs}>{esc(s)}</span>")
        if i < len(STAGES) - 1:
            parts.append('<span class="bg-sep" aria-hidden="true">›</span>')
    md(f'<nav class="bg-pipeline" aria-label="Pipeline stages (Table 4.5)">{"".join(parts)}</nav>')


def empty(message: str) -> None:
    md(f'<div class="bg-empty">{esc(message)}</div>')


def badge(text: str, kind: str) -> str:
    return f'<span class="bg-label {esc(kind)}">{esc(text)}</span>'


def facts(rows: list[tuple[str, object]]) -> None:
    body = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    md(f'<dl class="bg-facts">{body}</dl>')


# ----------------------------------------------------------------------------- risk card
def risk_card(res: WindowResult, simulated: bool = False, compact: bool = False) -> None:
    dev, sc, level = res.deviation, res.score, res.level
    color = RISK_COLORS[level]
    segs = "".join(f'<span class="{"on" if i < LEVEL_SEGMENTS[level] else ""}"></span>' for i in range(3))
    sqs = "".join(f'<span class="bg-sq{" on" if g in dev.groups_changed else ""}" '
                  f'title="{esc(GROUP_LABELS[g])}: {"changed" if g in dev.groups_changed else "usual"}">'
                  f'{esc(GROUP_LABELS[g])}</span>' for g in GROUPS)
    ref = dev.reference
    ref_line = (f"Compared with {ref.rows.sum()} baseline windows sent to the same recipient (context-matched)"
                if ref.context_matched else f"Compared with the whole baseline ({ref.rows.sum()} windows)")
    sim = f'{badge("Simulated", "sim")}<br>' if simulated else ""
    small = " · small baseline: read the fallback score too" if sc.small_baseline else ""
    md(f'''
<section class="bg-risk{' compact' if compact else ''}" style="--bg-risk:{color}" role="group" aria-label="Risk result: {esc(level)}">
  <div>
    {sim}<p class="bg-level">{esc(level)}</p>
    <div class="bg-meter" aria-hidden="true">{segs}</div>
    <span class="bg-chip">{esc(LEVEL_PHRASE[level])}</span>
  </div>
  <div>
    <p class="bg-agree">{dev.agreement} of 6 behavior groups changed</p>
    <div class="bg-groups">{sqs}</div>
    <p class="bg-line">{esc(habits_vs_length(dev))}</p>
    <p class="bg-line">{esc(ref_line)}</p>
  </div>
  <div class="bg-foot">Score is higher than {sc.percentile:.0f}% of this owner's baseline windows
    ({esc(sc.detector)}) · largest |z| = {sc.fallback_score:.1f}{esc(small)} · {esc(CUTOFF_NOTE)}</div>
</section>''')


def reasons(res: WindowResult) -> None:
    """Feature-level reasons. The agreement and habits-vs-length lines (last two) are on the card."""
    items = res.reasons[:-2] or ["No single feature moved beyond the working threshold (|z| > 2)."]
    md('<ul class="bg-reasons">' + "".join(f"<li>{esc(r)}</li>" for r in items) + "</ul>")


def group_chart(res: WindowResult) -> go.Figure:
    scores = res.deviation.group_scores()
    thr = cfg("deviation", "z_threshold")
    c = colors()
    vals = [scores[g] for g in GROUPS]
    changed = [v > thr for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=[GROUP_LABELS[g] for g in GROUPS], orientation="h",
        marker=dict(color=[RISK_COLORS[res.level] if ch else c["bar2"] for ch in changed],
                    pattern=dict(shape=["/" if ch else "" for ch in changed], fgcolor="#ffffff", solidity=0.15)),
        text=[f"{v:.1f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: largest |z| %{x:.2f}<extra></extra>"))
    fig.add_vline(x=thr, line_dash="dash", line_width=2, line_color=c["text"],
                  annotation_text=f"|z| = {thr:g}", annotation_position="top")
    fig.update_layout(title="Deviation by behavior group", height=340,
                      xaxis=dict(title="largest |z| in the group", rangemode="tozero"),
                      yaxis=dict(autorange="reversed"))
    return fig


# ----------------------------------------------------------------------------- timeline
def timeline_chart(table: pd.DataFrame, baseline: range, window: range) -> go.Figure:
    c = colors()
    ts = table["ts"]
    y = table["hour"] + ts.dt.minute / 60
    idx = np.arange(len(table))
    in_b = (idx >= baseline.start) & (idx < baseline.stop)
    in_w = (idx >= window.start) & (idx < window.stop)
    other = ~(in_b | in_w)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ts[other], y=y[other], mode="markers", name="Other messages",
                               marker=dict(size=5, color=c["bar2"], opacity=0.45),
                               hovertemplate="%{x|%b %d %H:%M}<extra></extra>"))
    fig.add_trace(go.Scattergl(x=ts[in_b], y=y[in_b], mode="markers", name="Baseline (N messages)",
                               marker=dict(size=6, color=c["accent"], opacity=0.45),
                               hovertemplate="baseline · %{x|%b %d %H:%M}<extra></extra>"))
    fig.add_trace(go.Scattergl(x=ts[in_w], y=y[in_w], mode="markers", name="Selected window",
                               marker=dict(size=13, color=c["accent"], symbol="diamond",
                                           line=dict(width=2, color=c["text"])),
                               hovertemplate="window · %{x|%b %d %H:%M}<extra></extra>"))
    if len(ts[in_b]):
        fig.add_vrect(x0=ts[in_b].min(), x1=ts[in_w].max(), fillcolor=c["accent"], opacity=0.07, line_width=0)
    fig.update_layout(title="Your sent messages over time (dots only, no text)", height=300, showlegend=True,
                      legend=dict(orientation="h", y=-0.25, x=0),
                      yaxis=dict(title="hour of day", range=[24, 0], tickvals=[0, 6, 12, 18, 24],
                                 ticktext=["12 AM", "6 AM", "12 PM", "6 PM", "12 AM"]))
    return fig


# ----------------------------------------------------------------------------- baseline captions
def active_span(hours: pd.Series, coverage: float = 0.9) -> str:
    """Shortest circular range of hours holding `coverage` of messages, e.g. '8 AM to 11 PM'."""
    counts = np.bincount(np.asarray(hours, dtype=int), minlength=24).astype(float)
    total = counts.sum()
    if total == 0:
        return "-"
    best = None
    for length in range(1, 25):
        for start in range(24):
            cov = sum(counts[(start + i) % 24] for i in range(length)) / total
            if cov >= coverage and (best is None or cov > best[2]):
                best = (start, length, cov)
        if best:
            break
    start, length, _ = best
    end = (start + length - 1) % 24
    return f"{fmt_hour(start)} to {fmt_hour(end)}"


def bubble_phrase(words: pd.Series) -> str:
    med = float(np.median(words)) if len(words) else 0.0
    size = "short" if med < 8 else ("medium-length" if med < 15 else "long")
    return f"Usually {size} bubbles, about {med:.0f} words"
