import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from core.features import (FEATURE_NAMES, FEATURES, GROUPS, LENGTH_BASED, TOPIC_INDEPENDENT, char_ngrams,
                           cosine_distance, entropy, hour_profile, laughter_style, prepare_messages,
                           window_features, window_language_mix)
from core.language_tagger import tokenize


def make_df(texts, secs=None, recips=None, hour=10):
    base = pd.Timestamp(f"2026-06-01 {hour:02d}:00", tz="Asia/Manila")
    secs = secs or [i * 30 for i in range(len(texts))]
    recips = recips or ["R01"] * len(texts)
    return pd.DataFrame({"ts": [base + pd.Timedelta(seconds=s) for s in secs], "recipient_code": recips,
                         "thread_type": ["one-to-one"] * len(texts), "text": texts})


UNIFORM = np.full(24, 1 / 24)


def test_every_feature_has_a_group():
    assert set(g for g, _ in FEATURES.values()) == set(GROUPS)
    assert set(TOPIC_INDEPENDENT) <= set(FEATURE_NAMES) and set(LENGTH_BASED) <= set(FEATURE_NAMES)


def test_prepared_table_has_no_text():
    p = prepare_messages(make_df(["secret words here"]))
    assert "text" not in p.table.columns
    assert not any(isinstance(v, str) and "secret" in v for v in p.table.iloc[0].tolist())


@pytest.mark.parametrize("text,style", [("HAHAHA grabe", "HAHAHA"), ("hahaha ok", "hahaha"), ("hehe", "hehe"),
                                        ("lol sige", "lol"), ("ok po", "none"), ("Haha", "hahaha")])
def test_laughter_style(text, style):
    assert laughter_style(tokenize(text)) == style


def test_hour_profile_smoothing():
    p = hour_profile([10, 10, 11], alpha=1.0)
    assert p.sum() == pytest.approx(1.0)
    assert p[10] == pytest.approx(3 / 27) and p[3] == pytest.approx(1 / 27)


def test_entropy():
    assert entropy(["a", "a"]) == 0.0
    assert entropy(["a", "b"]) == pytest.approx(1.0)
    assert entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)
    assert entropy({"x": 0, "y": 0}) == 0.0


def test_cosine_distance():
    a = char_ngrams("maray man")
    assert cosine_distance(a, a) == pytest.approx(0.0)
    assert cosine_distance(a, Counter({"zz": 1})) == pytest.approx(1.0)


def test_window_features_basic_math():
    texts = ["hiiii dai pa ako", "HAHAHA", "this is a very long message with many many words in it to pass "
             "the fifteen word mark easily ok?", "ok u there?! 😂", "lng"]
    df = make_df(texts, secs=[0, 20, 40, 200, 230], recips=["R01", "R01", "R02", "R02", "R03"])
    w = prepare_messages(df)
    f = window_features(w, hour_profile([10] * 50), known_recipients={"R01", "R02"})
    assert f["long_share"] == pytest.approx(1 / 5)
    assert f["bubbles_per_burst"] == pytest.approx(5 / 2)      # bursts: [0,20,40] [200,230]
    assert f["median_gap_s"] == pytest.approx(25.0)            # gaps 20,20,160,30
    assert f["msgs_per_hour"] == pytest.approx(5 / (230 / 3600))
    assert f["laugh_HAHAHA"] == pytest.approx(0.2) and f["laugh_none"] == pytest.approx(0.8)
    assert f["punct_rate"] == pytest.approx(3 / 5)              # '?' + '?!'
    assert f["emoji_rate"] == pytest.approx(1 / 5)
    assert f["new_recipient_share"] == pytest.approx(1 / 5)
    assert f["distinct_recipients"] == 3
    assert f["H_recipients"] == pytest.approx(entropy(["R01", "R01", "R02", "R02", "R03"]))
    assert f["H_hours"] == 0.0
    assert f["stretch_rate"] > 0 and f["abbrev_rate"] > 0
    assert set(f) == set(FEATURE_NAMES)


def test_code_switch_rate_in_window():
    # 'dai meeting' -> bik,eng (1 switch / 1 pair); 'hindi talaga' -> tag,tag (0/1). pooled 1/2
    w = prepare_messages(make_df(["dai meeting", "hindi talaga"]))
    f = window_features(w, UNIFORM, set())
    assert f["code_switch_rate"] == pytest.approx(0.5)


def test_k1_timing_is_nan():
    w = prepare_messages(make_df(["hello"]))
    f = window_features(w, UNIFORM, {"R01"})
    assert math.isnan(f["median_gap_s"]) and math.isnan(f["msgs_per_hour"])


def test_repeated_sending():
    texts = ["pa-gcash man tabi 500 urgent", "pa-gcash man tabi 500 urgent!", "pa-gcash man tabi 500 urgent",
             "unrelated words entirely"]
    w = prepare_messages(make_df(texts, recips=["R01", "R02", "R03", "R04"]))
    f = window_features(w, UNIFORM, set())
    assert f["repeated_send_share"] == pytest.approx(3 / 4)
    same = prepare_messages(make_df(texts[:3], recips=["R01"] * 3))
    assert window_features(same, UNIFORM, set())["repeated_send_share"] == 0.0


def test_synthetic_users_differ_in_expected_direction(df_a, df_b):
    a = prepare_messages(df_a.iloc[:200])
    b = prepare_messages(df_b.iloc[:200])
    fa = window_features(a, UNIFORM, set())
    fb = window_features(b, UNIFORM, set())
    assert fa["avg_words"] < fb["avg_words"]
    assert fa["stretch_rate"] > 0 and fb["stretch_rate"] == 0
    assert fa["laugh_HAHAHA"] > 0 and fb["laugh_HAHAHA"] == 0
    assert fb["laugh_lol"] > 0
    assert window_language_mix(a) == "Bi-lish" and window_language_mix(b) == "Taglish"
