"""Window scoring and prototype risk levels.

Detector: Isolation Forest trained ONLY on the owner's baseline windows' deviation vectors.
The thesis later compares an Extended Isolation Forest and a One-Class SVM; detectors are
pluggable through DETECTORS / register_detector, so those can be added without touching callers.

Detector input (config [scorer].detector_input): by default each window is summarised as the largest
|z| in each of the six behaviour groups; "features" feeds all per-feature z-scores instead.

Fallback score (for small baselines): the largest |z| in the window.

Risk levels (PROTOTYPE CUTOFFS, not the validated thresholds of Section 4.2.2):
  percentile of the window's score among the owner's own baseline windows' scores
  > high_percentile (99) and agreement >= high_min_agreement (2)  -> High
  > medium_percentile (95)                                         -> Medium
  otherwise                                                        -> Low
The level always comes from the detector's percentile. The fallback score and its percentile are
reported alongside; when the reference baseline has fewer than small_baseline_windows windows the
UI flags the baseline as small so the fallback can be read next to the detector score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest

from .baseline import Baseline
from .config import cfg
from .deviation import Deviation, baseline_deviation_matrix
from .features import FEATURE_NAMES, FEATURES, GROUPS


class Detector(Protocol):
    name: str

    def fit(self, X: np.ndarray) -> "Detector": ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more unusual for this owner."""
        ...


DETECTORS: dict[str, type] = {}


def register_detector(key: str):
    def deco(cls):
        DETECTORS[key] = cls
        return cls
    return deco


@register_detector("isolation_forest")
class IsolationForestDetector:
    name = "Isolation Forest"

    def __init__(self, n_estimators: int | None = None, random_state: int | None = None):
        self.model = IsolationForest(
            n_estimators=n_estimators or cfg("scorer", "n_estimators"),
            random_state=cfg("scorer", "random_state") if random_state is None else random_state,
            contamination="auto",
        )

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


# Placeholders for the thesis comparison (not implemented in the prototype):
#   @register_detector("extended_isolation_forest")  class ExtendedIsolationForestDetector: ...
#   @register_detector("one_class_svm")               class OneClassSVMDetector: ...  (sklearn.svm.OneClassSVM)


def make_detector(key: str | None = None) -> Detector:
    key = key or cfg("scorer", "detector")
    if key not in DETECTORS:
        raise ValueError(f"Unknown detector '{key}'. Available: {sorted(DETECTORS)}")
    return DETECTORS[key]()


def percentile_of(value: float, reference: np.ndarray) -> float:
    """Share (0-100) of reference values strictly below `value`."""
    reference = np.asarray(reference, dtype=float)
    return float(100.0 * np.mean(reference < value)) if len(reference) else 0.0


def risk_level(percentile: float, agreement: int) -> str:
    if percentile > cfg("risk", "high_percentile") and agreement >= cfg("risk", "high_min_agreement"):
        return "High"
    if percentile > cfg("risk", "medium_percentile"):
        return "Medium"
    return "Low"


@dataclass
class ScoreResult:
    detector: str
    score: float
    percentile: float
    fallback_score: float          # max |z|
    fallback_percentile: float
    small_baseline: bool
    level: str
    n_reference_windows: int

    @property
    def level_basis(self) -> str:
        return "max |z| fallback (small baseline)" if self.small_baseline else self.detector


_GROUP_IDX = [[i for i, f in enumerate(FEATURE_NAMES) if FEATURES[f][0] == g] for g in GROUPS]


def detector_input(Z: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Map z-vectors (rows) to what the detector sees (config [scorer].detector_input)."""
    mode = mode or cfg("scorer", "detector_input")
    Z = np.atleast_2d(Z)
    if mode == "features":
        return Z
    if mode == "group_max":
        return np.column_stack([np.abs(Z[:, ix]).max(axis=1) for ix in _GROUP_IDX])
    raise ValueError(f"Unknown detector_input '{mode}'")


def score_window(dev: Deviation, baseline: Baseline, detector: str | None = None,
                 agreement: int | None = None) -> ScoreResult:
    """`dev` supplies the deviation vector and reference windows for the detector.
    `agreement` (default: dev.agreement) is the agreement score used for the High rule."""
    Z = baseline_deviation_matrix(baseline, dev.reference)
    X = detector_input(Z)
    det = make_detector(detector).fit(X)
    base_scores = det.score(X)
    s = float(det.score(detector_input(dev.vector()))[0])
    pct = percentile_of(s, base_scores)
    fb = dev.max_abs_z
    fb_base = np.abs(Z).max(axis=1)
    fb_pct = percentile_of(fb, fb_base)
    small = len(X) < cfg("scorer", "small_baseline_windows")
    level = risk_level(pct, dev.agreement if agreement is None else agreement)
    return ScoreResult(det.name, s, pct, fb, fb_pct, small, level, len(X))
