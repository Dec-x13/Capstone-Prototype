from collections import Counter

import numpy as np
import pytest

from core.explain import explain, fmt_hour, habits_vs_length
from core.features import Prepared
from core.pipeline import evaluate
from core.scorer import DETECTORS, make_detector, percentile_of, risk_level
from core.windows import valid_window_starts

BANNED = ("hacked", "scammer", "proves", "proof", "attacker", "fraud", "accuracy", "f1")


def test_percentile_and_levels():
    ref = np.arange(100)
    assert percentile_of(50, ref) == 50.0
    assert percentile_of(1000, ref) == 100.0
    assert risk_level(100, 2) == "High"
    assert risk_level(100, 1) == "Medium"   # High needs agreement >= 2
    assert risk_level(96, 5) == "Medium"
    assert risk_level(95, 5) == "Low"


def test_detector_registry():
    assert "isolation_forest" in DETECTORS
    with pytest.raises(ValueError):
        make_detector("nope")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    d = make_detector().fit(X)
    assert d.score(np.array([[8, 8, 8, 8]]))[0] > np.median(d.score(X))


def test_detector_input_group_max():
    from core.features import FEATURE_NAMES, FEATURES, GROUPS
    from core.scorer import detector_input
    z = np.zeros(len(FEATURE_NAMES))
    z[FEATURE_NAMES.index("avg_words")] = -3.0
    z[FEATURE_NAMES.index("laugh_lol")] = 7.0
    z[FEATURE_NAMES.index("stretch_rate")] = 2.0
    g = detector_input(z, "group_max")[0]
    assert g.shape == (len(GROUPS),)
    assert g[GROUPS.index("bubble")] == 3.0 and g[GROUPS.index("habits")] == 7.0
    assert detector_input(z, "features").shape == (1, len(FEATURE_NAMES))


def test_fmt_hour():
    assert [fmt_hour(h) for h in (0, 3, 12, 15)] == ["12 AM", "3 AM", "12 PM", "3 PM"]


def _insert_b(prep_a, prep_b, b_start, at=200, k=5):
    w = prep_b.take(range(b_start, b_start + k))
    for col in ("ts", "hour", "recipient_code"):
        w.table[col] = prep_a.table[col].iloc[at:at + k].to_numpy()
    return Prepared.concat([prep_a.take(range(at)), w])


def test_own_windows_mostly_low(prep_a):
    starts = list(valid_window_starts(len(prep_a), 5, 100))[::3]
    levels = Counter(evaluate(prep_a, s, 5, 100, use_context=False).level for s in starts)
    assert levels["Low"] / sum(levels.values()) >= 0.7


def test_other_users_window_scores_higher(prep_a, prep_b):
    own = evaluate(prep_a, 200, 5, 100)
    sim = evaluate(_insert_b(prep_a, prep_b, 100), 200, 5, 100, simulated=True)
    assert sim.score.percentile >= own.score.percentile
    assert sim.deviation.agreement >= 2
    assert sim.deviation.habits_score > own.deviation.habits_score
    assert sim.level in ("Medium", "High")


def test_reasons_are_sensible_and_neutral(prep_a, prep_b):
    sim = evaluate(_insert_b(prep_a, prep_b, 100), 200, 5, 100)
    text = " | ".join(sim.reasons).lower()
    assert "of 6 behavior groups changed" in text
    assert "topic-independent habits" in text or "only in length" in text
    assert any(k in text for k in ("paragraph", "taglish", "lol", "stretched", "tagalog"))
    for word in BANNED:
        assert word not in text


def test_laughter_wording_same_style_as_usual(prep_a):
    from core.explain import _usual_laugh, phrase
    r = evaluate(prep_a, 200, 5, 100)
    d = r.deviation
    usual = _usual_laugh(d)
    assert usual != "none"  # synthetic user A laughs 'HAHAHA'
    d.z[f"laugh_{usual}"] = 5.0
    p = phrase(f"laugh_{usual}", d, r.window, r.baseline)
    assert p == f"more '{usual}' laughter than usual"
    other = "lol" if usual != "lol" else "hehe"
    d.z[f"laugh_{other}"] = 5.0
    assert phrase(f"laugh_{other}", d, r.window, r.baseline) == f"laughs with '{other}' (usually '{usual}')"


def test_habits_vs_length_wording(prep_a):
    r = evaluate(prep_a, 200, 5, 100)
    d = r.deviation
    for f in d.z:
        d.z[f] = 0.0
    assert habits_vs_length(d) == "Writing habits and message length look usual."
    d.z["avg_words"] = 5.0
    assert "only in length" in habits_vs_length(d)
    d.z["stretch_rate"] = -5.0
    assert "topic-independent" in habits_vs_length(d)
