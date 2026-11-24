"""Page 4: Simulate a takeover (manuscript Section 4.1.3). Everything here is labelled Simulated."""
from __future__ import annotations

import streamlit as st

from core.pipeline import evaluate
from core.simulate import simulate
from ui.components import badge, empty, md
from ui.page_check import result_block
from ui.state import prep, settings, ss, window_starts

CONDITIONS = {
    "normal": ("Normal timing", "Uses user A's own times and recipients, so only the writing can differ."),
    "takeover": ("Takeover timing",
                 "Near-identical messages to 2 to 6 people, 5 to 90 seconds apart, at a random hour."),
}


def render(presenter: bool) -> None:
    md(f'{badge("Simulated", "sim")} &nbsp;User B\'s messages are inserted into user A\'s timeline '
       f'right after A\'s baseline (Section 4.1.3). Nothing here is a real event.')
    pa, pb = prep("A"), prep("B")
    if pa is None or pb is None:
        missing = " and ".join(x for x, p in (("user A", pa), ("user B", pb)) if p is None)
        empty(f"Load {missing} on the Load data page to run the simulation.")
        return
    k, n, ctx = settings()
    at_options = [s for s in window_starts() if s + k <= len(pa)]
    if not at_options or len(pb) < k:
        empty(f"Not enough messages for N = {n} and k = {k}. Lower N in the sidebar.")
        return

    c1, c2 = st.columns(2, gap="large")
    at_no = c1.slider("Position in user A's timeline (window number)", 1, len(at_options),
                      max(1, len(at_options) // 2), key="sim_at")
    b_no = c2.slider("User B's window number", 1, len(pb) // k, max(1, len(pb) // (2 * k)), key="sim_b")
    at, b_start = at_options[at_no - 1], (b_no - 1) * k
    cond = st.radio("Timing condition", list(CONDITIONS), horizontal=True, key="sim_cond",
                    format_func=lambda c: CONDITIONS[c][0], captions=[CONDITIONS[c][1] for c in CONDITIONS])
    if not presenter:
        ss["sim_seed"] = int(st.number_input("Random seed for takeover timing", 0, 9999, int(ss["sim_seed"])))

    b1, b2, _ = st.columns([1, 1, 3])
    run = b1.button("Run simulation", type="primary", use_container_width=True)
    again = b2.button("Run again", use_container_width=True, disabled=cond != "takeover" or "sim" not in ss,
                      help="Draws new random recipients, gaps and hour for takeover timing.")
    if again:
        ss["sim_seed"] = int(ss["sim_seed"]) + 1
    if run or again:
        with st.spinner("Inserting user B's window and scoring both windows..."):
            ss["sim"] = simulate(pa, pb, at, b_start, k, n, cond, int(ss["sim_seed"]), ctx)
            ss["sim_own"] = evaluate(pa, at, k, n, use_context=ctx)

    if "sim" not in ss:
        empty("Choose a timing condition and press Run simulation.")
        return
    sim_res, info = ss["sim"]
    st.caption(info.describe())
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Your own later messages")
        st.caption("User A's real window at the same position: natural variation.")
        result_block(ss["sim_own"], "sim_own", presenter=presenter)
    with right:
        st.markdown("### Inserted messages from user B")
        st.caption(CONDITIONS[info.condition][0] + ": " + CONDITIONS[info.condition][1])
        result_block(sim_res, "sim_b", simulated=True, presenter=presenter)
