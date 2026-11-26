"""Page 1: Load data. Shows counts only, never message text."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.config import SAMPLE_DIR
from core.export_reader import UnknownLayoutError, detect_layouts, suggest_owner
from ui.components import SYNTH_LABEL, badge, esc, facts, md
from ui.state import load_user, ss

PANELS = {
    "A": ("User A · account owner", "user_a",
          "The person whose own messages form the baseline."),
    "B": ("User B · only for the takeover simulation", "user_b",
          "Their writing is inserted into user A's timeline on the Simulate page."),
}


def render() -> None:
    left, right = st.columns(2, gap="large")
    with left:
        panel("A")
    with right:
        panel("B")


def panel(slot: str) -> None:
    heading, sample, blurb = PANELS[slot]
    st.subheader(heading)
    st.caption(blurb)
    loaded = f"{slot}_summary" in ss
    if loaded:
        summary(slot)
    box = st.expander("Change data", expanded=False) if loaded else st.container()
    with box:
        form(slot, sample)


def summary(slot: str) -> None:
    s = ss[f"{slot}_summary"]
    src = badge(SYNTH_LABEL, "synth") if ss.get(f"{slot}_synthetic") else badge("Researcher's own export", "synth")
    md(f'{src}<p class="bg-big">{s["messages kept"]:,}</p><p class="bg-summary-line">messages kept '
       f'(the owner\'s own sent text messages)</p>')
    facts([
        ("Date range", f'{s["first message"]} to {s["last message"]}'),
        ("Recipients", f'{s["recipients"]} (shown only as codes like R07)'),
        ("Export layout", s["layouts"]),
        ("Read from", s.get("source", "unzipped folder")),
        ("Threads read", s["threads"]),
    ])
    dropped = [("other people's messages", s["other people's messages dropped"]),
               ("photos, videos, stickers and files", s["skipped media"]),
               ("unsent", s["skipped unsent"]), ("calls", s["skipped call"]),
               ("reactions and system notices", s["skipped other"])]
    st.markdown("**Dropped before analysis**")
    facts([(label.capitalize(), f"{n:,}") for label, n in dropped])
    st.caption("No message text is shown on this page.")


def form(slot: str, sample: str) -> None:
    src = st.radio("Data source", [SYNTH_LABEL, "Export folder on this computer"], key=f"{slot}_src",
                   horizontal=True)
    synthetic = src == SYNTH_LABEL
    if synthetic:
        path = str(SAMPLE_DIR / sample)
        st.caption("Two invented users made for testing. No real person's messages.")
    else:
        path = st.text_input("Facebook export (JSON format): a folder or a .zip", key=f"{slot}_path",
                             placeholder=r"C:\thesis-data\me")
        if not path:
            st.caption("Paste the unzipped export folder, the folder holding all of its .zip parts, or one .zip. "
                       "Zips are read in place; nothing is extracted.")
            return
    p = Path(path.strip().strip('"'))
    if not p.exists():
        st.error(f"Folder not found: {p}. Check the spelling, or copy the path from the File Explorer address bar.")
        return
    if p.is_file() and p.suffix.lower() != ".zip":
        st.error("That path is a file. Paste the export folder, the folder of .zip parts, or a .zip file.")
        return
    try:
        layouts = detect_layouts(p)
        if not layouts:
            st.error("No Messenger messages found here. Pick the export folder (or its .zip parts) that contains "
                     "messages/inbox or your_facebook_activity/messages. If the export is in HTML format, "
                     "request it again from Facebook as JSON.")
            return
        if ss.get(f"{slot}_suggest_path") != str(p):
            with st.spinner("Finding the most frequent sender..."):
                ss[f"{slot}_suggest"] = suggest_owner(p)
            ss[f"{slot}_suggest_path"] = str(p)
    except (UnknownLayoutError, OSError, ValueError) as e:
        st.error(f"Could not read this export: {e}")
        return
    st.caption(f"Detected layout: {', '.join(layouts)}")
    owner = st.text_input("Owner's display name", value=ss[f"{slot}_suggest"], key=f"{slot}_owner_{p}",
                          help="Suggested: the most frequent sender. Only this person's own sent messages are kept.")
    if st.button(f"Load user {slot}", key=f"{slot}_load", type="primary"):
        try:
            with st.spinner("Reading the export and dropping other people's messages..."):
                load_user(slot, str(p), owner, synthetic)
            st.rerun()
        except LookupError:
            st.error(f"No text messages from \"{owner}\" were found. Names must match exactly. "
                     f"The most frequent sender in this export is \"{ss[f'{slot}_suggest']}\".")
        except (UnknownLayoutError, OSError, ValueError) as e:
            st.error(f"Could not read this export: {e}")


__all__ = ["render", "esc"]
