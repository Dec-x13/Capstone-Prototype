"""Windowing (manuscript Section 4.2.3).

* A window is k consecutive sent messages (k in 1, 3, 5, 10; default 5).
* The baseline is the N messages immediately before the evaluated window (N in 50, 100, 150, 200; default 100).
* Baseline statistics use overlapping (sliding) windows of the same size k inside the baseline:
  N - k + 1 windows.
* A burst is a run of messages where each is sent within 60 seconds of the previous one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import cfg


def burst_ids(ts: pd.Series | list, gap_seconds: float | None = None) -> np.ndarray:
    """Burst label per message (0, 0, 1, ...). A new burst starts when the gap exceeds `gap_seconds`."""
    gap_seconds = cfg("windows", "burst_gap_seconds") if gap_seconds is None else gap_seconds
    t = pd.to_datetime(pd.Series(ts)).reset_index(drop=True)
    if len(t) == 0:
        return np.array([], dtype=int)
    gaps = t.diff().dt.total_seconds().to_numpy()
    new = np.r_[True, gaps[1:] > gap_seconds]
    return np.cumsum(new) - 1


def n_bursts(ts, gap_seconds: float | None = None) -> int:
    ids = burst_ids(ts, gap_seconds)
    return int(ids.max() + 1) if len(ids) else 0


def sliding_starts(n_messages: int, k: int) -> range:
    """Start positions of all overlapping windows of size k in a sequence of n messages."""
    return range(0, max(n_messages - k + 1, 0))


@dataclass(frozen=True)
class Split:
    """Index ranges into the owner's timeline (positions, not labels)."""
    start: int  # first message of the evaluated window
    k: int
    n: int

    @property
    def window(self) -> range:
        return range(self.start, self.start + self.k)

    @property
    def baseline(self) -> range:
        return range(self.start - self.n, self.start)

    @property
    def baseline_window_starts(self) -> list[int]:
        b0 = self.start - self.n
        return [b0 + s for s in sliding_starts(self.n, self.k)]


def make_split(n_total: int, start: int, k: int, n: int) -> Split:
    if k < 1 or n < k:
        raise ValueError("Need k >= 1 and N >= k.")
    if start < n:
        raise ValueError(f"The window must start after at least N={n} earlier messages (start={start}).")
    if start + k > n_total:
        raise ValueError("The window runs past the end of the timeline.")
    return Split(start, k, n)


def valid_window_starts(n_total: int, k: int, n: int) -> range:
    """All window starts that have a full N-message baseline before them (non-overlapping step k)."""
    return range(n, n_total - k + 1, k)
