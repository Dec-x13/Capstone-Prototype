"""End-to-end evaluation of one window: baseline -> deviation -> score -> reasons (Table 4.5 order)."""
from __future__ import annotations

from dataclasses import dataclass

from .baseline import Baseline, build_baseline
from .deviation import Deviation, deviation_for_window
from .explain import explain
from .features import Prepared, window_language_mix
from .scorer import ScoreResult, score_window
from .windows import Split, make_split


@dataclass
class WindowResult:
    split: Split
    baseline: Baseline
    window: Prepared
    deviation: Deviation
    score: ScoreResult
    reasons: list[str]
    window_mix: str
    baseline_mix: str
    simulated: bool = False
    label: str = ""

    @property
    def level(self) -> str:
        return self.score.level


def evaluate(timeline: Prepared, start: int, k: int, n: int, use_context: bool = True,
             simulated: bool = False, label: str = "", detector: str | None = None) -> WindowResult:
    split = make_split(len(timeline), start, k, n)
    baseline = build_baseline(timeline, split)
    win = timeline.take(split.window)
    dev = deviation_for_window(win, baseline, use_context=use_context)
    # The detector is always trained on the whole baseline (N-k+1 windows). A context-matched
    # reference can be as small as 10 heavily overlapping windows, too few to train it on.
    # Context-matched deviations still drive the agreement score, the reasons and the charts.
    dev_all = dev if not dev.reference.context_matched else deviation_for_window(win, baseline, use_context=False)
    score = score_window(dev_all, baseline, detector, agreement=dev.agreement)
    reasons = explain(dev, win, baseline)
    return WindowResult(split, baseline, win, dev, score, reasons, window_language_mix(win),
                        window_language_mix(Prepared(baseline.messages, [])), simulated, label)
