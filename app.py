"""BehaviorGuard - feasibility prototype (behaviour-only pipeline, manuscript Section 4.2.3 / Table 4.5).

Runs fully offline. Raw message text lives only in this process's memory (st.session_state) and is
never written to disk, logs or caches. Anything displayed is masked.

    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.anonymize import RecipientCoder, mask
from core.config import SAMPLE_DIR, cfg
from core.export_reader import UnknownLayoutError, detect_layouts, load_owner_messages, suggest_owner
from core.features import FEATURES, GROUP_LABELS, GROUPS, LAUGH_STYLES, Prepared, prepare_messages
from core.explain import fmt_hour
from core.pipeline import WindowResult, evaluate
from core.simulate import simulate
from core.windows import valid_window_starts

st.set_page_config(page_title="BehaviorGuard prototype", page_icon="🛡️", layout="wide")

SYNTH_LABEL = "Synthetic sample (for testing only)"
LEVEL_COLORS = {"Low": "#1b7f3b", "Medium": "#b26a00", "High": "#b3261e"}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 19px; }
h1 { font-size: 2.1rem !important; } h2 { font-size: 1.6rem !important; } h3 { font-size: 1.3rem !important; }
.badge { display:inline-block; padding: 0.35rem 1.1rem; border-radius: 0.5rem; color: white;
         font-weight: 700; font-size: 1.6rem; letter-spacing: 0.03em; }
.sim { background:#fff4e5; border:2px solid #b26a00; padding:0.4rem 0.8rem; border-radius:0.4rem;
       font-weight:700; color:#6b3f00; }
.synth { background:#e8f0fe; border:2px solid #1f4e9e; padding:0.3rem 0.7rem; border-radius:0.4rem;
         font-weight:700; color:#1f4e9e; }
.reason { font-size: 1.15rem; margin: 0.2rem 0; }
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("coder", RecipientCoder())
ss.setdefault("results", {})


# ----------------------------------------------------------------------------- helpers
def reset():
    for key in list(ss.keys()):
        del ss[key]


def load_user(slot: str, path: str, owner: str, synthetic: bool):
    res = load_owner_messages(path, owner, ss["coder"])
    if len(res.df) == 0:
        raise ValueError("No text messages from that owner name were found.")
    ss[f"{slot}_df"] = res.df                  # text: memory only
    ss[f"{slot}_prep"] = prepare_messages(res.df)
    ss[f"{slot}_summary"] = res.summary()
    ss[f"{slot}_synthetic"] = synthetic
    ss["results"] = {}


def source_tag(slot: str) -> str:
    return SYNTH_LABEL if ss.get(f"{slot}_synthetic") else "Researcher's own export"


def badge(level: str) -> str:
    return f'<span class="badge" style="background:{LEVEL_COLORS[level]}">{level}</span>'


def cached_eval(prep: Prepared, start: int, k: int, n: int, ctx: bool) -> WindowResult:
    key = ("A", start, k, n, ctx)
    if key not in ss["results"]:
        ss["results"][key] = evaluate(prep, start, k, n, use_context=ctx)
    return ss["results"][key]


def window_caption(res: WindowResult) -> str:
    t = res.window.table["ts"]
    return f"{t.iloc[0]:%Y-%m-%d %H:%M} → {t.iloc[-1]:%H:%M} · {len(t)} messages · mix: {res.window_mix}"


def group_chart(res: WindowResult, title: str):
    scores = res.deviation.group_scores()
    thr = cfg("deviation", "z_threshold")
    labels = [GROUP_LABELS[g] for g in GROUPS]
    vals = [scores[g] for g in GROUPS]
    colors = ["#b3261e" if v > thr else "#7a8aa0" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.1f}" for v in vals], textposition="outside"))
    fig.add_vline(x=thr, line_dash="dash", line_color="#111", annotation_text=f"|z| = {thr:g}")
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=30, t=50, b=30), font=dict(size=16),
                      xaxis_title="largest |z| in group", yaxis=dict(autorange="reversed"))
    return fig


def render_result(res: WindowResult, title: str, show_text_df: pd.DataFrame | None = None, key: str = ""):
    if res.simulated:
        st.markdown(f'<div class="sim">{res.label}</div>', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.caption(window_caption(res))
    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.markdown(f"**Estimated risk**<br>{badge(res.level)}", unsafe_allow_html=True)
    c2.metric("Agreement score", f"{res.deviation.agreement} of 6 groups")
    c3.metric("Score percentile vs own baseline", f"{res.score.percentile:.0f}")
    st.markdown("**Why (unusual for this user, worth verifying):**")
    for r in res.reasons:
        st.markdown(f'<div class="reason">• {r}</div>', unsafe_allow_html=True)
    st.plotly_chart(group_chart(res, "Deviation by behavior group"), use_container_width=True,
                    config=PLOT_CONFIG, key=f"chart_{key}")
    st.caption(f"Reference: {res.deviation.reference.label} · Detector: {res.score.detector} "
               f"(trained on {res.score.n_reference_windows} of the owner's baseline windows) · "
               f"Fallback max |z| = {res.score.fallback_score:.1f} (percentile {res.score.fallback_percentile:.0f})"
               + (" · small baseline: read the fallback too" if res.score.small_baseline else ""))
    with st.expander("Feature details (derived numbers only)"):
        rows = [{"group": GROUP_LABELS[g], "feature": lab, "window": res.deviation.values[f],
                 "baseline mean": float(res.deviation.reference.mean[f]), "z": res.deviation.z[f]}
                for f, (g, lab) in FEATURES.items()]
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
    if show_text_df is not None:
        with st.expander("Show this window's messages (masked)"):
            for _, row in show_text_df.iterrows():
                st.markdown(f"`{row['ts']:%H:%M:%S}` · {row['recipient_code']} · {mask(row['text'])}")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ BehaviorGuard")
    st.caption("Feasibility prototype · behavior-only pipeline · runs offline")
    k = st.select_slider("Window size k (messages)", options=cfg("windows", "k_options"),
                         value=cfg("windows", "k_default"))
    n = st.select_slider("Baseline size N (messages)", options=cfg("windows", "n_options"),
                         value=cfg("windows", "n_default"))
    use_ctx = st.toggle("Context-matched baseline when possible", value=True,
                        help=f"Uses only baseline windows to the same recipient if there are at least "
                             f"{cfg('deviation', 'context_min_windows')} of them.")
    if "A_prep" in ss:
        st.markdown(f'<div class="synth">{source_tag("A")}</div>' if ss.get("A_synthetic")
                    else f"**Owner data:** {source_tag('A')}", unsafe_allow_html=True)
    st.button("⟲ Reset", on_click=reset, use_container_width=True, type="primary")
    st.caption("Risk levels use prototype cutoffs, not the validated thresholds of Section 4.2.2.")

st.title("BehaviorGuard: does this window look like the account owner?")
tab_load, tab_base, tab_check, tab_sim, tab_about = st.tabs(
    ["1 · Load", "2 · Baseline", "3 · Check a window", "4 · Simulate takeover (demo)", "5 · About / limitations"])

# ----------------------------------------------------------------------------- 1. Load
with tab_load:
    def loader(slot: str, heading: str, default_sample: str):
        st.subheader(heading)
        src = st.radio("Data source", [SYNTH_LABEL, "Export folder on this computer"], key=f"{slot}_src",
                       horizontal=True)
        synthetic = src == SYNTH_LABEL
        if synthetic:
            path = str(SAMPLE_DIR / default_sample)
            st.markdown(f'<span class="synth">{SYNTH_LABEL}</span>', unsafe_allow_html=True)
        else:
            path = st.text_input("Path to the unzipped Facebook export (JSON format)", key=f"{slot}_path",
                                 placeholder=r"C:\Users\me\Downloads\facebook-export")
        if path and Path(path).exists():
            try:
                layouts = detect_layouts(path)
                if not layouts:
                    st.error("No Messenger data found in that folder.")
                    return
                if ss.get(f"{slot}_suggest_path") != path:
                    ss[f"{slot}_suggest"] = suggest_owner(path)
                    ss[f"{slot}_suggest_path"] = path
                st.caption(f"Detected layout: {', '.join(layouts)}")
                owner = st.text_input("Owner's display name (confirm)", value=ss[f"{slot}_suggest"],
                                      key=f"{slot}_owner_{path}",
                                      help="Suggested: the most frequent sender. Only this person's own sent messages are kept.")
                if st.button("Load messages", key=f"{slot}_load", type="primary"):
                    with st.spinner("Reading export and dropping other people's messages..."):
                        load_user(slot, path, owner, synthetic)
            except (UnknownLayoutError, ValueError, OSError) as e:
                st.error(str(e))
        elif path:
            st.warning("Folder not found.")
        if f"{slot}_summary" in ss:
            s = ss[f"{slot}_summary"]
            st.success(f"Loaded · {source_tag(slot)}")
            c = st.columns(4)
            c[0].metric("Messages kept", s["messages kept"])
            c[1].metric("Recipients", s["recipients"])
            c[2].metric("From", s["first message"])
            c[3].metric("To", s["last message"])
            dropped = s["other people's messages dropped"]
            st.caption(f"Threads read: {s['threads']} · other people's messages dropped in memory: "
                       f"{dropped} · owner media/calls/unsent skipped: "
                       f"{s['owner non-text items skipped']} · no message text is shown here")

    loader("A", "Account owner (user A)", "user_a")
    st.divider()
    loader("B", "Second user (user B) — only for the takeover simulation", "user_b")

# ----------------------------------------------------------------------------- shared window selection
prep_a: Prepared | None = ss.get("A_prep")
starts = list(valid_window_starts(len(prep_a), k, n)) if prep_a is not None else []

with tab_check:
    if prep_a is None:
        st.info("Load the owner's data in tab 1 first.")
    elif not starts:
        st.warning(f"Not enough messages: need more than N + k = {n + k}.")
    else:
        mode = st.radio("Window", ["Pick from the timeline", "Paste messages"], horizontal=True)
        if mode == "Pick from the timeline":
            idx = st.slider("Window number (later = more recent)", 1, len(starts), len(starts), key="win_idx")
            start = starts[idx - 1]
            ss["sel_start"] = start
            res = cached_eval(prep_a, start, k, n, use_ctx)
            render_result(res, "Owner's own real window", ss["A_df"].iloc[start:start + k], key="check")
        else:
            st.caption("Pasted text stays in memory only. One message per line.")
            pasted = st.text_area("Messages", height=160, key="pasted")
            codes = sorted(set(prep_a.table["recipient_code"]))
            c1, c2, c3 = st.columns(3)
            rc = c1.selectbox("Recipient", codes + ["New recipient"])
            hour = c2.number_input("Hour sent (0-23)", 0, 23, 14)
            gap = c3.number_input("Seconds between messages", 1, 3600, 30)
            lines = [ln for ln in (pasted or "").splitlines() if ln.strip()]
            if lines and st.button("Check pasted messages", type="primary"):
                last = prep_a.table["ts"].iloc[-1]
                t0 = (last + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=int(hour))
                pdf = pd.DataFrame({"ts": [t0 + pd.Timedelta(seconds=int(gap) * i) for i in range(len(lines))],
                                    "recipient_code": [rc if rc != "New recipient" else "R999"] * len(lines),
                                    "thread_type": "one-to-one", "text": lines})
                if len(lines) > n:
                    st.error(f"Paste at most N = {n} messages.")
                else:
                    tl = Prepared.concat([prep_a, prepare_messages(pdf)])
                    res = evaluate(tl, len(prep_a), len(lines), n, use_context=use_ctx)
                    render_result(res, "Pasted messages", pdf, key="paste")

# ----------------------------------------------------------------------------- 2. Baseline
with tab_base:
    if prep_a is None or not starts:
        st.info("Load the owner's data in tab 1 first.")
    else:
        start = ss.get("sel_start", starts[-1])
        if start not in starts:
            start = starts[-1]
        res = cached_eval(prep_a, start, k, n, use_ctx)
        b = res.baseline
        bm = b.messages
        st.subheader(f"Baseline: the {n} messages before the selected window")
        st.caption(f"{bm['ts'].iloc[0]:%Y-%m-%d} → {bm['ts'].iloc[-1]:%Y-%m-%d} · {b.n_windows} sliding windows "
                   f"of k = {k} · dominant mix: {res.baseline_mix} · derived numbers only")
        c1, c2 = st.columns(2)
        hours = bm["hour"].value_counts().reindex(range(24), fill_value=0)
        fig = go.Figure(go.Bar(x=[fmt_hour(h) for h in range(24)], y=hours.values, marker_color="#1f4e9e"))
        fig.update_layout(title="Hour of day", height=340, font=dict(size=15), margin=dict(t=50, b=40))
        c1.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key="b_hours")
        fig = go.Figure(go.Histogram(x=bm["n_words"], nbinsx=20, marker_color="#1f4e9e"))
        fig.update_layout(title="Bubble length (words per message)", height=340, font=dict(size=15),
                          margin=dict(t=50, b=40), xaxis_title="words", yaxis_title="messages")
        c2.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key="b_len")
        c3, c4 = st.columns(2)
        langs = {"Bikol": bm["n_bikol"].sum(), "Tagalog": bm["n_tagalog"].sum(),
                 "English": bm["n_english"].sum(), "Ambiguous": bm["n_ambiguous"].sum()}
        tot = sum(langs.values()) or 1
        fig = go.Figure(go.Bar(x=list(langs), y=[v / tot for v in langs.values()],
                               marker_color=["#2e7d32", "#c62828", "#1565c0", "#9e9e9e"]))
        fig.update_layout(title=f"Language mix (dominant: {res.baseline_mix})", height=340, font=dict(size=15),
                          yaxis_tickformat=".0%", margin=dict(t=50, b=40))
        c3.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key="b_lang")
        laugh = bm["laugh"].value_counts(normalize=True).reindex(LAUGH_STYLES, fill_value=0)
        fig = go.Figure(go.Bar(x=list(laugh.index), y=laugh.values, marker_color="#6a1b9a"))
        fig.update_layout(title="Laughter style (share of messages)", height=340, font=dict(size=15),
                          yaxis_tickformat=".0%", margin=dict(t=50, b=40))
        c4.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key="b_laugh")
        st.markdown("**Top habits (average over baseline windows)**")
        hw = b.windows
        habits = pd.DataFrame([
            ("Stretched letters (share of words)", hw["stretch_rate"].mean()),
            ("Capital letters (share of letters)", hw["caps_share"].mean()),
            ("? and ! per message", hw["punct_rate"].mean()),
            ("Emoji per message", hw["emoji_rate"].mean()),
            ("Abbreviations (share of words)", hw["abbrev_rate"].mean()),
            ("Code-switching rate", hw["code_switch_rate"].mean()),
            ("Words per bubble", hw["avg_words"].mean()),
            ("Bubbles per burst", hw["bubbles_per_burst"].mean()),
        ], columns=["habit", "value"])
        st.dataframe(habits.round(3), hide_index=True, use_container_width=True)
        st.caption("Tip: the camera icon on each chart saves it as a PNG for slides (derived data only).")

# ----------------------------------------------------------------------------- 4. Simulate
with tab_sim:
    st.markdown('<div class="sim">Everything on this tab is SIMULATED for the demo (manuscript Section 4.1.3).</div>',
                unsafe_allow_html=True)
    prep_b: Prepared | None = ss.get("B_prep")
    if prep_a is None or prep_b is None:
        st.info("Load both user A and user B in tab 1.")
    else:
        at_options = [s for s in starts if s + k <= len(prep_a)]
        if not at_options or len(prep_b) < k:
            st.warning("Not enough messages for this N and k.")
        else:
            c1, c2 = st.columns(2)
            at_idx = c1.slider("Insert into A's timeline at window number", 1, len(at_options),
                               min(len(at_options), max(1, len(at_options) // 2)), key="sim_at")
            at = at_options[at_idx - 1]
            b_idx = c2.slider("User B's window number", 1, len(prep_b) // k, max(1, len(prep_b) // (2 * k)),
                              key="sim_b")
            b_start = (b_idx - 1) * k
            c3, c4 = st.columns(2)
            cond = c3.radio("Timing condition", ["normal", "takeover"], horizontal=True,
                            format_func=lambda c: {"normal": "Normal timing", "takeover": "Takeover timing"}[c])
            seed = c4.number_input("Random seed (takeover)", 0, 9999, 7)
            if st.button("Run simulation", type="primary"):
                with st.spinner("Scoring..."):
                    ss["sim"] = simulate(prep_a, prep_b, at, b_start, k, n, cond, int(seed), use_ctx)
                    ss["sim_own"] = evaluate(prep_a, at, k, n, use_context=use_ctx,
                                             label="Owner's own real window at the same position")
            if "sim" in ss:
                sim_res, info = ss["sim"]
                left, right = st.columns(2)
                with left:
                    render_result(ss["sim_own"], "A's own later window (natural variation)", key="sim_own")
                with right:
                    render_result(sim_res, "B's window inserted into A's timeline", key="sim_b")

# ----------------------------------------------------------------------------- 5. About
with tab_about:
    st.subheader("About this prototype")
    st.markdown(f"""
- **What it is.** A feasibility demonstration of the behavior-only pipeline in our manuscript
  (Section 4.2.3, Table 4.5). It runs on the two researchers' own Messenger exports and on a
  clearly labeled synthetic sample.
- **What it is not.** It reports **no performance figures**. The demo does not measure how well it works.
- **A change is a risk signal, not proof.** A window can look unusual for many ordinary reasons, like a new
  topic, a new group chat, being in a hurry, or someone else typing for the owner with permission.
  The output only marks windows that are *unusual for this user* and *worth verifying*.
- **Word lists are preliminary.** The Bikol/Tagalog/English lists are seeds. Native speakers will expand and check them,
  and the tagger will later be checked against 500 hand-labeled words.
- **Cutoffs are prototype cutoffs.** Low/Medium/High compare the score's percentile with the owner's own
  baseline windows: above the {cfg('risk', 'high_percentile')}th percentile with agreement ≥
  {cfg('risk', 'high_min_agreement')} is High, above the {cfg('risk', 'medium_percentile')}th is Medium.
  These are **not** the validated thresholds of Section 4.2.2, which will be set on validation data later.
- **Detector.** Isolation Forest trained only on the owner's baseline windows. The thesis later compares
  an Extended Isolation Forest and a One-Class SVM.
- **Privacy.** Only the owner's own sent messages are kept. Other people's messages are dropped while each thread file is read.
  Contacts become random codes. Text stays in memory and is masked whenever it is shown. Nothing is sent over the network.
""")
