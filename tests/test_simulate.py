import pytest

from core.features import cosine_similarity
from core.pipeline import evaluate
from core.simulate import build_simulated_timeline, simulate


def test_normal_timing_copies_owner_time_and_recipients(prep_a, prep_b):
    tl, info = build_simulated_timeline(prep_a, prep_b, at=200, b_start=50, k=5, condition="normal")
    assert len(tl) == 205
    win, own = tl.table.iloc[200:], prep_a.table.iloc[200:205]
    for col in ("ts", "hour", "recipient_code", "thread_type"):
        assert list(win[col]) == list(own[col])
    # writing comes from B
    assert list(win["n_words"]) == list(prep_b.table["n_words"].iloc[50:55])
    assert tl.table.iloc[:200].equals(prep_a.table.iloc[:200])


@pytest.mark.parametrize("seed", range(8))
def test_takeover_timing_ranges(prep_a, prep_b, seed):
    tl, info = build_simulated_timeline(prep_a, prep_b, at=200, b_start=50, k=5, condition="takeover", seed=seed)
    win = tl.table.iloc[200:]
    assert 2 <= info.n_recipients <= 5          # 2-6, capped at k
    assert win["recipient_code"].nunique() == info.n_recipients
    assert all(5 <= g <= 90 for g in info.gaps_s)
    gaps = win["ts"].diff().dt.total_seconds().iloc[1:].tolist()
    assert gaps == info.gaps_s
    assert win["ts"].iloc[0] > tl.table["ts"].iloc[199]
    assert win["hour"].iloc[0] == info.hour or win["hour"].iloc[0] == (info.hour + 1) % 24
    # the first message goes to each recipient -> near-identical texts
    grams = tl.ngrams[200:200 + info.n_recipients]
    assert all(cosine_similarity(grams[0], g) == pytest.approx(1.0) for g in grams)


def test_takeover_is_reproducible_with_seed(prep_a, prep_b):
    _, i1 = build_simulated_timeline(prep_a, prep_b, 200, 50, 5, "takeover", seed=7)
    _, i2 = build_simulated_timeline(prep_a, prep_b, 200, 50, 5, "takeover", seed=7)
    assert i1 == i2


def test_simulated_results_are_labelled(prep_a, prep_b):
    res, info = simulate(prep_a, prep_b, 200, 50, 5, 100, condition="takeover", seed=1)
    assert res.simulated and res.label.startswith("SIMULATED")


def test_takeover_raises_recipient_signals(prep_a, prep_b):
    normal, _ = simulate(prep_a, prep_b, 200, 50, 5, 100, condition="normal")
    take, info = simulate(prep_a, prep_b, 200, 50, 5, 100, condition="takeover", seed=3)
    assert take.deviation.values["repeated_send_share"] > 0
    assert take.deviation.values["distinct_recipients"] == info.n_recipients
    assert take.deviation.agreement >= normal.deviation.agreement
    own = evaluate(prep_a, 200, 5, 100)
    assert take.score.percentile >= own.score.percentile


def test_bad_arguments(prep_a, prep_b):
    with pytest.raises(ValueError):
        build_simulated_timeline(prep_a, prep_b, 200, 50, 5, "other")
    with pytest.raises(ValueError):
        build_simulated_timeline(prep_a, prep_b, len(prep_a) - 2, 50, 5, "normal")
