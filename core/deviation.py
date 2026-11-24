"""Deviation of a window from the owner's baseline (manuscript Section 4.2.3).

* Numeric features:  z = (x - mean) / std over the reference baseline windows, std floored (config).
* Character n-gram profile: cosine distance from the baseline profile, then z-scored against the
  baseline windows' own (leave-one-out) distances.
* Entropy: dH = H(window) - mean H(baseline windows of size k). Its z (= dH / std) is used for the
  agreement score and the detector.
* Agreement score: number of the six behaviour groups with at least one |z| > z_threshold (config, 2.0).
* Summary scores: mean |z| over topic-independent habits and over length-based features.
* Context-matched baseline: if >= context_min_windows baseline windows share the evaluated window's
  main recipient, mean/std come from those windows only; otherwise from the whole baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baseline import Baseline, main_recipient
from .config import cfg
from .features import ENTROPY_FEATURES, FEATURE_NAMES, FEATURES, GROUPS, LENGTH_BASED, TOPIC_INDEPENDENT, Prepared, \
    window_features


@dataclass
class Reference:
    rows: np.ndarray        # boolean mask over baseline windows
    mean: pd.Series
    std: pd.Series
    label: str              # shown in the UI
    context_matched: bool


@dataclass
class Deviation:
    values: dict[str, float]
    z: dict[str, float]
    delta_H: dict[str, float]
    groups_changed: list[str]
    agreement: int
    habits_score: float
    length_score: float
    reference: Reference

    @property
    def max_abs_z(self) -> float:
        vals = [abs(v) for v in self.z.values() if not math.isnan(v)]
        return max(vals) if vals else 0.0

    def vector(self) -> np.ndarray:
        return z_vector(self.z)

    def group_scores(self) -> dict[str, float]:
        """Largest |z| per group (for the per-group chart)."""
        out = {}
        for g in GROUPS:
            vals = [abs(self.z[f]) for f, (grp, _) in FEATURES.items() if grp == g and not math.isnan(self.z[f])]
            out[g] = max(vals) if vals else 0.0
        return out


def z_vector(z: dict[str, float]) -> np.ndarray:
    return np.array([0.0 if math.isnan(z[f]) else z[f] for f in FEATURE_NAMES])


def choose_reference(baseline: Baseline, recipient: str | None) -> Reference:
    min_w = cfg("deviation", "context_min_windows")
    floor = cfg("deviation", "std_floor")
    mains = np.array(baseline.window_main_recipient)
    mask = np.ones(len(mains), dtype=bool)
    label = f"Whole baseline ({len(mains)} windows)"
    matched = False
    if recipient is not None:
        same = mains == recipient
        if same.sum() >= min_w:
            mask, matched = same, True
            label = f"Context-matched: {int(same.sum())} baseline windows to {recipient}"
        else:
            label += f" — only {int(same.sum())} windows to {recipient}, need {min_w} for context matching"
    ref = baseline.windows[mask]
    mean = ref.mean(skipna=True)
    std = ref.std(ddof=0, skipna=True).fillna(0.0).clip(lower=floor)
    return Reference(mask, mean, std, label, matched)


def zscores(values: dict[str, float], ref: Reference) -> dict[str, float]:
    out = {}
    for f in FEATURE_NAMES:
        x, m = values.get(f, float("nan")), ref.mean.get(f, float("nan"))
        out[f] = float((x - m) / ref.std[f]) if not (math.isnan(x) or math.isnan(m)) else float("nan")
    return out


def summarize(values: dict[str, float], z: dict[str, float], ref: Reference) -> Deviation:
    thr = cfg("deviation", "z_threshold")
    changed = [g for g in GROUPS
               if any(abs(z[f]) > thr for f, (grp, _) in FEATURES.items() if grp == g and not math.isnan(z[f]))]
    delta_H = {f: values[f] - float(ref.mean[f]) for f in ENTROPY_FEATURES}

    def mean_abs(names):
        v = [abs(z[n]) for n in names if not math.isnan(z[n])]
        return float(np.mean(v)) if v else 0.0

    return Deviation(values, z, delta_H, changed, len(changed), mean_abs(TOPIC_INDEPENDENT), mean_abs(LENGTH_BASED),
                     ref)


def deviation_for_window(win: Prepared, baseline: Baseline, use_context: bool = True) -> Deviation:
    values = window_features(win, baseline.hour_prob, baseline.recipients, baseline.ngram_profile)
    recipient = main_recipient(win.table["recipient_code"].tolist()) if use_context else None
    ref = choose_reference(baseline, recipient)
    return summarize(values, zscores(values, ref), ref)


def baseline_deviation_matrix(baseline: Baseline, ref: Reference) -> np.ndarray:
    """Deviation vectors of the reference baseline windows (detector training data)."""
    rows = baseline.windows[ref.rows]
    return np.vstack([z_vector(zscores(r.to_dict(), ref)) for _, r in rows.iterrows()])


def baseline_agreements(baseline: Baseline, ref: Reference) -> np.ndarray:
    rows = baseline.windows[ref.rows]
    return np.array([summarize(r.to_dict(), zscores(r.to_dict(), ref), ref).agreement for _, r in rows.iterrows()])
