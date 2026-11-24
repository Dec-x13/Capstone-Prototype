import pandas as pd
import pytest

from core.windows import burst_ids, make_split, n_bursts, sliding_starts, valid_window_starts


def times(*secs):
    base = pd.Timestamp("2026-06-01 10:00", tz="Asia/Manila")
    return pd.Series([base + pd.Timedelta(seconds=s) for s in secs])


def test_burst_boundaries_at_60_seconds():
    # gaps: 30, 60 (same burst), 61 (new), 5, 200 (new)
    ts = times(0, 30, 90, 151, 156, 356)
    assert burst_ids(ts, 60).tolist() == [0, 0, 0, 1, 1, 2]
    assert n_bursts(ts, 60) == 3


def test_burst_default_gap_from_config():
    assert n_bursts(times(0, 59, 118)) == 1
    assert n_bursts(times(0, 61, 122)) == 3


def test_burst_edge_cases():
    assert n_bursts(times()) == 0
    assert n_bursts(times(0)) == 1


def test_sliding_window_count():
    assert len(sliding_starts(100, 5)) == 96
    assert len(sliding_starts(100, 1)) == 100
    assert len(sliding_starts(4, 5)) == 0


def test_split_ranges():
    s = make_split(n_total=300, start=120, k=5, n=100)
    assert list(s.window) == [120, 121, 122, 123, 124]
    assert s.baseline == range(20, 120)
    starts = s.baseline_window_starts
    assert len(starts) == 96 and starts[0] == 20 and starts[-1] == 115
    # last baseline window ends right before the evaluated window
    assert starts[-1] + s.k == s.start


@pytest.mark.parametrize("start,k,n", [(50, 5, 100), (298, 5, 100), (120, 0, 100), (120, 10, 5)])
def test_split_rejects_invalid(start, k, n):
    with pytest.raises(ValueError):
        make_split(300, start, k, n)


def test_valid_starts():
    r = valid_window_starts(120, 5, 100)
    assert list(r) == [100, 105, 110, 115]
