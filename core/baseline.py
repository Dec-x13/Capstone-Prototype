"""Owner baseline: the N messages before the evaluated window, summarised over sliding windows of size k.

For each baseline window the features are computed the same way as for an evaluated window,
with two leave-one-out rules so a window is never compared against itself:
  * character n-gram distance is measured against the baseline profile minus that window's n-grams;
  * "new recipient" means a recipient not messaged anywhere else in the baseline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, Prepared, hour_profile, sum_ngrams, window_features
from .windows import Split


def main_recipient(codes) -> str:
    """Most frequent recipient in a window (ties: the one that appears first)."""
    c = Counter(codes)
    best = max(c.values())
    for r in codes:
        if c[r] == best:
            return r
    return ""


@dataclass
class Baseline:
    split: Split
    hour_prob: np.ndarray
    recipients: set
    ngram_profile: Counter
    windows: pd.DataFrame                       # one row per baseline window, columns = FEATURE_NAMES
    window_main_recipient: list[str]
    messages: pd.DataFrame = field(repr=False)  # derived per-message numbers for charts (no text)

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def build_baseline(prep: Prepared, split: Split) -> Baseline:
    base = prep.take(split.baseline)
    hour_prob = hour_profile(base.table["hour"])
    rec_counts = Counter(base.table["recipient_code"])
    profile = sum_ngrams(base.ngrams)

    rows, mains = [], []
    offset = split.start - split.n
    for s in split.baseline_window_starts:
        idx = range(s - offset, s - offset + split.k)
        win = base.take(idx)
        win_rc = Counter(win.table["recipient_code"])
        known = {r for r, c in rec_counts.items() if c - win_rc.get(r, 0) > 0}
        loo_profile = profile.copy()
        loo_profile.subtract(sum_ngrams(win.ngrams))
        loo_profile = +loo_profile  # drop zero/negative counts
        rows.append(window_features(win, hour_prob, known, loo_profile))
        mains.append(main_recipient(win.table["recipient_code"].tolist()))
    windows = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return Baseline(split, hour_prob, set(rec_counts), profile, windows, mains, base.table)
