import math

import numpy as np
import pandas as pd
import pytest

from core.baseline import build_baseline, main_recipient
from core.deviation import (baseline_deviation_matrix, choose_reference, deviation_for_window, summarize, zscores)
from core.features import FEATURE_NAMES, Prepared
from core.windows import make_split


def test_main_recipient_tie_breaks_on_first():
    assert main_recipient(["R02", "R01", "R01", "R02"]) == "R02"
    assert main_recipient(["R05", "R01", "R01"]) == "R01"


def test_baseline_shape(prep_a):
    split = make_split(len(prep_a), 150, 5, 100)
    b = build_baseline(prep_a, split)
    assert b.n_windows == 96
    assert list(b.windows.columns) == FEATURE_NAMES
    assert b.hour_prob.sum() == pytest.approx(1.0)
    assert "text" not in b.messages.columns
    # leave-one-out keeps n-gram distance away from 0
    assert (b.windows["ngram_distance"] > 0).all()


def test_zscore_and_std_floor(prep_a):
    b = build_baseline(prep_a, make_split(len(prep_a), 150, 5, 100))
    ref = choose_reference(b, None)
    assert (ref.std >= 0.05).all()
    vals = {f: float(ref.mean[f]) for f in FEATURE_NAMES}
    z = zscores(vals, ref)
    assert all(abs(v) < 1e-9 for v in z.values() if not math.isnan(v))
    vals["avg_words"] = float(ref.mean["avg_words"] + 3 * ref.std["avg_words"])
    assert zscores(vals, ref)["avg_words"] == pytest.approx(3.0)


def test_agreement_counts_groups_not_features(prep_a):
    b = build_baseline(prep_a, make_split(len(prep_a), 150, 5, 100))
    ref = choose_reference(b, None)
    vals = {f: float(ref.mean[f]) for f in FEATURE_NAMES}
    # two features in the same group -> 1 group; one in another group -> 2 groups
    vals["avg_words"] += 5 * ref.std["avg_words"]
    vals["long_share"] += 5 * ref.std["long_share"]
    d = summarize(vals, zscores(vals, ref), ref)
    assert d.groups_changed == ["bubble"] and d.agreement == 1
    vals["stretch_rate"] -= 5 * ref.std["stretch_rate"]
    d = summarize(vals, zscores(vals, ref), ref)
    assert d.agreement == 2
    assert d.length_score > 0 and d.habits_score > 0


def test_entropy_delta_is_window_minus_mean_baseline(prep_a):
    b = build_baseline(prep_a, make_split(len(prep_a), 150, 5, 100))
    win = prep_a.take(range(150, 155))
    d = deviation_for_window(win, b, use_context=False)
    for f in ("H_hours", "H_recipients", "H_languages"):
        assert d.delta_H[f] == pytest.approx(d.values[f] - b.windows[f].mean())


def test_context_matching_switch(prep_a):
    b = build_baseline(prep_a, make_split(len(prep_a), 300, 5, 200))
    counts = pd.Series(b.window_main_recipient).value_counts()
    frequent, rare = counts.index[0], counts.index[-1]
    assert counts[frequent] >= 10
    ref = choose_reference(b, frequent)
    assert ref.context_matched and ref.rows.sum() == counts[frequent]
    if counts[rare] < 10:
        ref2 = choose_reference(b, rare)
        assert not ref2.context_matched and ref2.rows.all()
        assert "need 10" in ref2.label


def test_baseline_matrix(prep_a):
    b = build_baseline(prep_a, make_split(len(prep_a), 150, 5, 100))
    ref = choose_reference(b, None)
    X = baseline_deviation_matrix(b, ref)
    assert X.shape == (96, len(FEATURE_NAMES)) and np.isfinite(X).all()


def test_own_window_vs_other_user_window(prep_a, prep_b):
    split = make_split(len(prep_a), 200, 5, 100)
    b = build_baseline(prep_a, split)
    own = deviation_for_window(prep_a.take(split.window), b, use_context=False)
    t = prep_b.take(range(100, 105))
    # give B's window A's own timestamps/recipients so only writing can differ
    t.table["ts"] = prep_a.table["ts"].iloc[200:205].to_numpy()
    t.table["hour"] = prep_a.table["hour"].iloc[200:205].to_numpy()
    t.table["recipient_code"] = prep_a.table["recipient_code"].iloc[200:205].to_numpy()
    other = deviation_for_window(t, b, use_context=False)
    assert other.habits_score > own.habits_score
    assert other.agreement >= own.agreement
