"""Page 2: Your baseline. Derived numbers only."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.explain import fmt_hour
from core.features import LAUGH_STYLES
from ui.components import active_span, bubble_phrase, empty, facts, md
from ui.state import cached_eval, prep, selected_start, settings
from ui.theme import LANG_COLORS, colors, show


def render() -> None:
    if prep("A") is None:
        empty("Load user A's data on the Load data page to see the baseline.")
        return
    start = selected_start()
    k, n, _ = settings()
    if start is None:
        empty(f"User A has too few messages for N = {n} and k = {k}. Lower N in the sidebar or load more data.")
        return
    with st.spinner(f"Building the baseline from the {n} messages before the window..."):
        res = cached_eval(start)
    b = res.baseline
    bm = b.messages
    c = colors()

    st.subheader("Your baseline")
    st.markdown(f"**N = {n} messages** before the selected window, cut into **{b.n_windows} overlapping windows "
                f"of k = {k}** · {bm['ts'].iloc[0]:%b %d, %Y} to {bm['ts'].iloc[-1]:%b %d, %Y}")

    laugh = bm["laugh"].value_counts(normalize=True).reindex(LAUGH_STYLES, fill_value=0)
    top_laugh = laugh.drop("none").idxmax()
    laugh_line = (f"Laughs with “{top_laugh}”" if laugh[top_laugh] >= 0.03 else "Rarely laughs in text")
    lines = [f"Mostly active {active_span(bm['hour'])}", bubble_phrase(bm["n_words"]),
             f"Mostly {res.baseline_mix}", laugh_line, f"Messages {len(b.recipients)} recipients (coded)"]
    md('<p class="bg-summary-line">' + " · ".join(lines) + "</p>")

    left, right = st.columns(2, gap="large")
    with left:
        hours = bm["hour"].value_counts().reindex(range(24), fill_value=0)
        fig = go.Figure(go.Bar(x=list(range(24)), y=hours.values, marker_color=c["bar"],
                               hovertemplate="%{customdata}: %{y} messages<extra></extra>",
                               customdata=[fmt_hour(h) for h in range(24)]))
        fig.update_layout(title="When you message (hour of day)",
                          xaxis=dict(tickvals=[0, 6, 12, 18, 23], ticktext=["12 AM", "6 AM", "12 PM", "6 PM", "11 PM"]),
                          yaxis_title="messages")
        show(fig, "b_hours", 320)
    with right:
        fig = go.Figure(go.Histogram(x=bm["n_words"], xbins=dict(start=0, size=2), marker_color=c["bar"],
                                     hovertemplate="%{x} words: %{y} messages<extra></extra>"))
        fig.add_vline(x=15.5, line_dash="dot", line_color=c["text"], annotation_text="long (>15 words)")
        fig.update_layout(title="Bubble length", xaxis_title="words per message", yaxis_title="messages")
        show(fig, "b_len", 320)

    left, right = st.columns(2, gap="large")
    with left:
        langs = {"Bikol": bm["n_bikol"].sum(), "Tagalog": bm["n_tagalog"].sum(),
                 "English": bm["n_english"].sum(), "Ambiguous": bm["n_ambiguous"].sum()}
        tot = sum(langs.values()) or 1
        shares = [v / tot for v in langs.values()]
        fig = go.Figure(go.Bar(x=list(langs), y=shares, marker_color=[LANG_COLORS[x] for x in langs],
                               text=[f"{s:.0%}" for s in shares], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.0%}<extra></extra>"))
        fig.update_layout(title=f"Language mix · mostly {res.baseline_mix}", yaxis=dict(tickformat=".0%",
                          range=[0, max(shares) * 1.25 if max(shares) else 1]))
        show(fig, "b_lang", 320)
    with right:
        names = {"HAHAHA": "HAHAHA", "hahaha": "hahaha", "hehe": "hehe", "lol": "lol", "none": "no laughter"}
        fig = go.Figure(go.Bar(x=[names[s] for s in LAUGH_STYLES], y=laugh.values, marker_color=c["bar"],
                               text=[f"{v:.0%}" for v in laugh.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.0%} of messages<extra></extra>"))
        fig.update_layout(title="Laughter style (share of messages)",
                          yaxis=dict(tickformat=".0%", range=[0, max(laugh.values) * 1.25 or 1]))
        show(fig, "b_laugh", 320)

    st.markdown("#### Top writing habits")
    hw = b.windows
    left, right = st.columns(2, gap="large")
    with left:
        facts([
            ("Stretched letters (“hiiii”)", f"{hw['stretch_rate'].mean():.0%} of words"),
            ("Capital letters", f"{hw['caps_share'].mean():.0%} of letters"),
            ("? and !", f"{hw['punct_rate'].mean():.2f} per message"),
            ("Emoji", f"{hw['emoji_rate'].mean():.2f} per message"),
        ])
    with right:
        facts([
            ("Abbreviations (lng, u, pls)", f"{hw['abbrev_rate'].mean():.0%} of words"),
            ("Code-switching", f"{hw['code_switch_rate'].mean():.0%} of word pairs"),
            ("Bubbles per burst", f"{hw['bubbles_per_burst'].mean():.1f}"),
            ("Recipients", f"{len(b.recipients)} (coded)"),
        ])
    st.caption("All numbers are derived from the messages; no text is shown. "
               "The camera icon on each chart saves it as a PNG.")


__all__ = ["render", "pd"]
