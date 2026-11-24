"""Session state helpers. Raw text lives only in st.session_state (process memory), never on disk."""
from __future__ import annotations

import streamlit as st

from core.anonymize import RecipientCoder
from core.config import cfg
from core.export_reader import load_owner_messages
from core.features import Prepared, prepare_messages
from core.pipeline import WindowResult, evaluate
from core.windows import valid_window_starts

ss = st.session_state


def init() -> None:
    ss.setdefault("coder", RecipientCoder())
    ss.setdefault("results", {})
    ss.setdefault("sim_seed", 7)
    ss.setdefault("page", "Load data")
    ss.setdefault("k", cfg("windows", "k_default"))
    ss.setdefault("n", cfg("windows", "n_default"))
    ss.setdefault("ctx", True)
    ss.setdefault("presenter", False)


def reset() -> None:
    for key in list(ss.keys()):
        del ss[key]


def settings() -> tuple[int, int, bool]:
    return int(ss["k"]), int(ss["n"]), bool(ss["ctx"])


def any_synthetic() -> bool:
    return bool(ss.get("A_synthetic") or ss.get("B_synthetic"))


def load_user(slot: str, path: str, owner: str, synthetic: bool) -> None:
    res = load_owner_messages(path, owner, ss["coder"])
    if len(res.df) == 0:
        raise LookupError(owner)
    ss[f"{slot}_df"] = res.df                  # text: memory only
    ss[f"{slot}_prep"] = prepare_messages(res.df)
    ss[f"{slot}_summary"] = res.summary()
    ss[f"{slot}_synthetic"] = synthetic
    ss["results"] = {}
    ss.pop("sim", None)
    ss.pop("sel_start", None)


def prep(slot: str) -> Prepared | None:
    return ss.get(f"{slot}_prep")


def window_starts() -> list[int]:
    p = prep("A")
    k, n, _ = settings()
    return list(valid_window_starts(len(p), k, n)) if p is not None else []


def selected_start() -> int | None:
    starts = window_starts()
    if not starts:
        return None
    s = ss.get("sel_start")
    return s if s in starts else starts[-1]


def cached_eval(start: int) -> WindowResult:
    k, n, ctx = settings()
    key = (start, k, n, ctx)
    if key not in ss["results"]:
        ss["results"][key] = evaluate(prep("A"), start, k, n, use_context=ctx)
    return ss["results"][key]
