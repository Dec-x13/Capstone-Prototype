"""Headless tests of the Streamlit app on the synthetic sample (no browser, no network)."""
import html
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
BANNED = ("hacked", "scammer", "proves", "accuracy", "f1 score")
SYNTH = "Synthetic sample (for testing only)"
CUTOFF = "Prototype cutoffs, not the validated thresholds of Section 4.2.2."


def new_app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    return at


def click(at, key=None, label=None):
    for b in at.button:
        if (key and b.key == key) or (label and b.label == label):
            b.click()
            at.run()
            assert not at.exception
            return
    raise AssertionError(f"button not found: {key or label}")


def goto(at, page):
    at.session_state["nav_w"] = page
    at.run()
    assert not at.exception


def text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success]
    parts += [str(t.value) for t in at.text] + [h.value for h in at.subheader] + [h.value for h in at.header]
    return html.unescape(" ".join(str(p) for p in parts))


@pytest.fixture(scope="module")
def app():
    at = new_app()
    click(at, key="A_load")
    click(at, key="B_load")
    return at


# ----------------------------------------------------------------------------- shell
def test_header_pipeline_and_disclaimer():
    at = new_app()
    t = text(at)
    assert "BehaviorGuard" in t and "Feasibility prototype" in t and "runs offline" in t
    for stage in ("Export reader", "Language tagger", "Feature extractor", "Baseline builder",
                  "Deviation scorer", "Explanation generator", "Interface"):
        assert stage in t
    assert 'class="bg-stage on" aria-current="step">Export reader' in t
    assert CUTOFF in t


@pytest.mark.parametrize("page,stage", [("Your baseline", "Baseline builder"), ("Check a window", "Deviation scorer"),
                                        ("Simulate a takeover", "Explanation generator"), ("About", "Interface")])
def test_pipeline_highlight_follows_page(page, stage):
    at = new_app()
    goto(at, page)
    assert f'class="bg-stage on" aria-current="step">{stage}' in text(at)


@pytest.mark.parametrize("page,hint", [("Your baseline", "Load user A's data"),
                                       ("Check a window", "Load user A's data"),
                                       ("Simulate a takeover", "Load user A and user B")])
def test_empty_states_say_what_to_do(page, hint):
    at = new_app()
    goto(at, page)
    assert hint in text(at)


def test_presenter_mode_hides_settings():
    at = new_app()
    assert any(s.label.startswith("Window size k") for s in at.select_slider)
    at.toggle[0].set_value(True)
    at.run()
    assert not at.exception
    assert not any(s.label.startswith("Window size k") for s in at.select_slider)
    assert "k = 5" in text(at) and CUTOFF in text(at)


def test_bad_folder_error_is_specific():
    at = new_app()
    at.radio(key="A_src").set_value("Export folder on this computer")
    at.run()
    at.text_input(key="A_path").set_value(r"C:\definitely\not\here")
    at.run()
    assert any("Folder not found" in e.value for e in at.error)


# ----------------------------------------------------------------------------- pages with data
def test_load_summary_replaces_form(app):
    goto(app, "Load data")
    t = text(app)
    assert SYNTH in t and "messages kept" in t and "Dropped before analysis" in t
    assert "Export layout" in t and "legacy" in t
    assert [e.label for e in app.expander].count("Change data") == 2
    assert "A_prep" in app.session_state and "B_prep" in app.session_state


def test_baseline_page(app):
    goto(app, "Your baseline")
    t = text(app)
    assert "N = 100 messages" in t and "k = 5" in t
    assert "Mostly active" in t and "Usually short bubbles" in t and "Mostly Bi-lish" in t
    assert "Top writing habits" in t


def test_check_window_card_and_masking(app):
    goto(app, "Check a window")
    df = app.session_state["A_df"]
    k, n = 5, 100
    hit = next(i for i, s in enumerate(df["text"]) if "09171234567" in s and i >= n + k)
    starts = list(range(n, len(df) - k + 1, k))
    app.slider(key="win_idx").set_value(next(j + 1 for j, s in enumerate(starts) if s <= hit < s + k))
    app.run()
    assert not app.exception
    t = text(app)
    assert 'class="bg-risk"' in t and "of 6 behavior groups changed" in t
    assert "Compared with" in t and CUTOFF in t
    assert "[PHONE]" in t and "09171234567" not in t
    labels = [e.label for e in app.expander]
    assert "Show this window's messages (masked)" in labels


def test_simulation_side_by_side_and_run_again(app):
    goto(app, "Simulate a takeover")
    click(app, label="Run simulation")
    t = text(app)
    assert "Your own later messages" in t and "Inserted messages from user B" in t
    assert t.count('class="bg-risk compact"') == 2 and "Simulated" in t
    app.radio(key="sim_cond").set_value("takeover")
    app.run()
    click(app, label="Run simulation")
    seed = app.session_state["sim_seed"]
    click(app, label="Run again")
    assert app.session_state["sim_seed"] == seed + 1
    assert "takeover timing" in text(app)


def test_about_page(app):
    goto(app, "About")
    t = text(app)
    for s in ("How to read the result", "Privacy design", "Limitations", "no performance figures",
              "A change is a risk signal, not proof", "Word lists are preliminary", "prototype cutoffs"):
        assert s.lower() in t.lower()


def test_wording_has_no_banned_terms(app):
    for page in ("Load data", "Your baseline", "Check a window", "Simulate a takeover", "About"):
        goto(app, page)
        t = text(app).lower()
        for w in BANNED:
            assert w not in t, (page, w)


def test_reset_returns_to_start(app):
    goto(app, "About")
    click(app, label="Reset")
    assert "A_prep" not in app.session_state
    assert app.session_state["page"] == "Load data"
