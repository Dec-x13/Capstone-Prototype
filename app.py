"""BehaviorGuard: feasibility prototype (behavior-only pipeline, manuscript Section 4.2.3 / Table 4.5).

Runs fully offline. Raw message text lives only in this process's memory (st.session_state) and is
never written to disk, logs or caches. Anything displayed is masked.

    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from core.config import cfg
from ui import page_about, page_baseline, page_check, page_load, page_simulate
from ui.components import CUTOFF_NOTE, PAGES, SYNTH_LABEL, badge, header, md, pipeline_strip
from ui.state import any_synthetic, init, reset, ss
from ui.theme import inject

st.set_page_config(page_title="BehaviorGuard", page_icon=":material/shield:", layout="wide",
                   initial_sidebar_state="expanded")
init()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### BehaviorGuard")
    ss["presenter"] = st.toggle("Presenter mode", value=ss["presenter"],
                                help="Larger type, hides advanced controls. Keeps the labels and disclaimers.")
    presenter = ss["presenter"]
    if not presenter:
        st.markdown("**Settings**")
        ss["k"] = st.select_slider("Window size k (messages)", options=cfg("windows", "k_options"), value=ss["k"])
        ss["n"] = st.select_slider("Baseline size N (messages)", options=cfg("windows", "n_options"), value=ss["n"])
        ss["ctx"] = st.toggle("Context-matched baseline when possible", value=ss["ctx"],
                              help=f"Compare only with baseline windows to the same recipient when there are at "
                                   f"least {cfg('deviation', 'context_min_windows')} of them.")
    else:
        st.markdown(f"Window **k = {ss['k']}** · baseline **N = {ss['n']}** · context matching "
                    f"**{'on' if ss['ctx'] else 'off'}**")
    if any_synthetic():
        md(badge(SYNTH_LABEL, "synth"))
    st.button("Reset", on_click=reset, use_container_width=True, help="Clear all loaded data and start over.")
    st.caption(CUTOFF_NOTE)

inject(presenter)

# ----------------------------------------------------------------------------- navigation + header
header(any_synthetic())
choice = st.segmented_control("Page", PAGES, default=ss["page"], key="nav_w", label_visibility="collapsed")
if choice:
    ss["page"] = choice
page = ss["page"]
pipeline_strip(page)

if page == "Load data":
    page_load.render()
elif page == "Your baseline":
    page_baseline.render()
elif page == "Check a window":
    page_check.render(presenter)
elif page == "Simulate a takeover":
    page_simulate.render(presenter)
else:
    page_about.render()

st.caption(CUTOFF_NOTE + " A change in behavior is a risk signal worth verifying, not proof.")
