"""Headless smoke test of the Streamlit app on the synthetic sample (no browser, no network)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
BANNED = ("hacked", "scammer", "proves", "accuracy", "f1 score")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    for btn in at.button:
        if btn.key == "A_load":
            btn.click()
    at.run()
    for btn in at.button:
        if btn.key == "B_load":
            btn.click()
    at.run()
    assert not at.exception
    return at


def _all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success]
    return " ".join(str(p) for p in parts)


def test_loads_synthetic_and_labels_it(app):
    text = _all_text(app)
    assert "Synthetic sample (for testing only)" in text
    assert "A_prep" in app.session_state and "B_prep" in app.session_state
    labels = [m.label for m in app.metric]
    assert "Messages kept" in labels and "Agreement score" in labels


def test_displayed_text_is_masked(app):
    # Pick a timeline window that contains the synthetic phone number and check it is masked on screen.
    df = app.session_state["A_df"]
    k, n = 5, 100
    hits = [i for i, t in enumerate(df["text"]) if "09171234567" in t and i >= n + k]
    assert hits, "synthetic sample should contain a phone number"
    starts = list(range(n, len(df) - k + 1, k))
    win_no = next(j + 1 for j, s in enumerate(starts) if s <= hits[0] < s + k)
    app.slider(key="win_idx").set_value(win_no)
    app.run()
    text = _all_text(app)
    assert "[PHONE]" in text
    assert "09171234567" not in text
    assert "Show this window's messages (masked)" in [e.label for e in app.expander]


def test_simulation_runs_and_is_labelled(app):
    for btn in app.button:
        if btn.label == "Run simulation":
            btn.click()
    app.run()
    assert not app.exception
    assert "SIMULATED" in _all_text(app)


def test_wording_has_no_banned_terms(app):
    text = _all_text(app).lower()
    for w in BANNED:
        assert w not in text


def test_reset_clears_state(app):
    for btn in app.button:
        if btn.label == "⟲ Reset":
            btn.click()
    app.run()
    assert "A_prep" not in app.session_state
