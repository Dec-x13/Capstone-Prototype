"""Page 3: Check a window. The risk result card is the hero."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.anonymize import mask
from core.features import FEATURES, GROUP_LABELS, Prepared, prepare_messages
from core.pipeline import WindowResult, evaluate
from ui.components import empty, group_chart, reasons, risk_card, timeline_chart
from ui.state import cached_eval, prep, settings, ss, window_starts
from ui.theme import show


def result_block(res: WindowResult, key: str, simulated: bool = False, presenter: bool = False,
                 compact: bool = False) -> None:
    """Risk card, reasons and group chart. Used by this page and the Simulate page."""
    risk_card(res, simulated=simulated, compact=compact)
    st.markdown("**Why this looks unusual for this user**" if res.level != "Low"
                else "**What changed, if anything**")
    reasons(res)
    show(group_chart(res), f"groups_{key}")
    if not presenter:
        with st.expander("Feature details (derived numbers only)", expanded=False):
            rows = [{"group": GROUP_LABELS[g], "feature": lab, "window": res.deviation.values[f],
                     "baseline mean": float(res.deviation.reference.mean[f]), "z": res.deviation.z[f]}
                    for f, (g, lab) in FEATURES.items()]
            st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)


def masked_preview(rows: pd.DataFrame) -> None:
    with st.expander("Show this window's messages (masked)", expanded=False):
        for _, r in rows.iterrows():
            st.text(f"{r['ts']:%b %d %H:%M:%S} · {r['recipient_code']} · {mask(r['text'])}")


def render(presenter: bool) -> None:
    p = prep("A")
    if p is None:
        empty("Load user A's data on the Load data page, then come back to check a window.")
        return
    starts = window_starts()
    k, n, _ = settings()
    if not starts:
        empty(f"User A has too few messages for N = {n} and k = {k}. Lower N in the sidebar or load more data.")
        return
    mode = "Pick from the timeline"
    if not presenter:
        mode = st.radio("Window", ["Pick from the timeline", "Paste messages"], horizontal=True,
                        label_visibility="collapsed")
    if mode == "Paste messages":
        paste(p)
        return

    current = ss.get("sel_start", starts[-1])
    default_no = starts.index(current) + 1 if current in starts else len(starts)
    no = st.slider("Window number (later = more recent)", 1, len(starts), default_no, key="win_idx")
    start = starts[no - 1]
    ss["sel_start"] = start
    with st.spinner(f"Scoring this window against the baseline windows..."):
        res = cached_eval(start)
    show(timeline_chart(p.table, res.split.baseline, res.split.window), "timeline")
    t = res.window.table["ts"]
    st.markdown(f"**Window {no} of {len(starts)}** · {t.iloc[0]:%b %d, %Y %H:%M} to {t.iloc[-1]:%H:%M} · "
                f"{len(t)} messages · mostly {res.window_mix}")
    result_block(res, "check", presenter=presenter)
    masked_preview(ss["A_df"].iloc[start:start + k])


def paste(p: Prepared) -> None:
    k, n, ctx = settings()
    st.caption("Pasted text stays in memory only. One message per line.")
    pasted = st.text_area("Messages", height=160, key="pasted")
    codes = sorted(set(p.table["recipient_code"]))
    c1, c2, c3 = st.columns(3)
    rc = c1.selectbox("Recipient", codes + ["New recipient"])
    hour = c2.number_input("Hour sent (0-23)", 0, 23, 14)
    gap = c3.number_input("Seconds between messages", 1, 3600, 30)
    lines = [ln for ln in (pasted or "").splitlines() if ln.strip()]
    if not lines:
        empty("Paste one or more messages above, one per line, then press Check pasted messages.")
        return
    if len(lines) > n:
        st.error(f"Paste at most N = {n} messages, or raise N in the sidebar.")
        return
    if st.button("Check pasted messages", type="primary"):
        last = p.table["ts"].iloc[-1]
        t0 = (last + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=int(hour))
        pdf = pd.DataFrame({"ts": [t0 + pd.Timedelta(seconds=int(gap) * i) for i in range(len(lines))],
                            "recipient_code": [rc if rc != "New recipient" else "R999"] * len(lines),
                            "thread_type": "one-to-one", "text": lines})
        with st.spinner("Scoring the pasted messages against the baseline..."):
            res = evaluate(Prepared.concat([p, prepare_messages(pdf)]), len(p), len(lines), n, use_context=ctx)
        result_block(res, "paste")
        masked_preview(pdf)
