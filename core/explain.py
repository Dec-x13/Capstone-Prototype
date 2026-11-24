"""Plain-language reasons for a window's deviation. Wording is neutral: changes are described as
'unusual for this user' and 'worth verifying', never as proof of anything."""
from __future__ import annotations

import math

from .baseline import Baseline
from .config import cfg
from .deviation import Deviation
from .features import FEATURES, GROUP_LABELS, LAUGH_STYLES, LENGTH_BASED, TOPIC_INDEPENDENT, Prepared, \
    window_language_mix
from .language_tagger import LANGS


def fmt_hour(h: int) -> str:
    h = int(h) % 24
    return f"{12 if h % 12 == 0 else h % 12} {'AM' if h < 12 else 'PM'}"


def _usual_laugh(dev: Deviation) -> str:
    m = dev.reference.mean
    styles = [s for s in LAUGH_STYLES if s != "none"]
    best = max(styles, key=lambda s: m[f"laugh_{s}"])
    return best if m[f"laugh_{best}"] > 0.02 else "none"


def _often(x: float, hi: float = 0.15, lo: float = 0.03) -> str:
    return "frequent" if x >= hi else ("occasional" if x >= lo else "rare")


def phrase(feature: str, dev: Deviation, win: Prepared, baseline: Baseline) -> str | None:
    z, x, m = dev.z[feature], dev.values[feature], float(dev.reference.mean[feature])
    up = z > 0
    t = win.table
    if feature == "hour_prob":
        if up:
            return "messaging at this user's busiest hours (usually more spread out)"
        h = min(t["hour"], key=lambda hh: baseline.hour_prob[int(hh)])
        return f"messaging at {fmt_hour(h)} (rarely active then)"
    if feature == "msgs_per_hour":
        return "many messages in a short time (usually a slower pace)" if up else "a slower pace than usual"
    if feature == "median_gap_s":
        return (f"longer pauses between messages (usually about {m:.0f} s)" if up
                else f"messages only seconds apart (usually about {m:.0f} s)")
    if feature == "avg_words":
        if up:
            return ("long paragraphs (usually short bubbles)" if m < 10
                    else f"longer messages than usual (about {x:.0f} words, usually {m:.0f})")
        return f"short bubbles (usually about {m:.0f} words per message)"
    if feature == "long_share":
        return "more long messages (usually rare)" if up else "fewer long messages than usual"
    if feature == "bubbles_per_burst":
        return ("many bubbles in a row (usually fewer)" if up
                else f"fewer bubbles per burst (usually about {m:.1f} in a row)")
    if feature.startswith("share_"):
        lang = feature.split("_", 1)[1]
        name = "ambiguous (shared) words" if lang == "ambiguous" else f"{lang.capitalize()} words"
        return f"more {name} than usual" if up else f"fewer {name} than usual"
    if feature == "code_switch_rate":
        return "switches language more often than usual" if up else "switches language less often than usual"
    if feature == "stretch_rate":
        if up:
            return "stretched letters like 'hiiii' (usually rare)"
        return f"{'no' if x == 0 else 'fewer'} stretched letters (usually {_often(m)})"
    if feature.startswith("laugh_"):
        style = feature.split("_", 1)[1]
        usual = _usual_laugh(dev)
        if style == "none":
            return "laughs more than usual" if up else None
        if up:
            return f"laughs with '{style}' (usually {repr(usual) if usual != 'none' else 'no laughter'})"
        return f"no '{style}' laughter (usually {_often(m, 0.1, 0.02)})" if x == 0 else f"less '{style}' laughter than usual"
    if feature == "punct_rate":
        return "more ? and ! than usual" if up else "fewer ? and ! than usual"
    if feature == "caps_share":
        return "more capital letters than usual" if up else "fewer capital letters than usual"
    if feature == "emoji_rate":
        return "more emoji than usual" if up else ("no emoji (usually uses them)" if x == 0 else "fewer emoji than usual")
    if feature == "abbrev_rate":
        return ("more abbreviations like 'lng', 'u', 'pls' than usual" if up
                else "fewer abbreviations than usual (e.g. 'lng', 'u', 'pls')")
    if feature == "ngram_distance":
        return "character patterns unlike this user's usual writing" if up else None
    if feature == "new_recipient_share":
        new = sorted({r for r in t["recipient_code"] if r not in baseline.recipients})
        if not up or not new:
            return None
        mins = max((t["ts"].iloc[-1] - t["ts"].iloc[0]).total_seconds() / 60, 1)
        span = f"within {mins:.0f} minutes" if mins < 120 else "in one window"
        return f"messages to {len(new)} new recipient{'s' if len(new) > 1 else ''} {span}"
    if feature == "distinct_recipients":
        return (f"messages to {x:.0f} different people in one window (usually about {m:.0f})" if up
                else f"fewer recipients than usual ({x:.0f}, usually about {m:.0f})")
    if feature == "repeated_send_share":
        return "near-identical messages sent to several people" if up else None
    if feature in ("H_hours", "H_recipients", "H_languages"):
        what = {"H_hours": "active hours", "H_recipients": "recipients", "H_languages": "languages"}[feature]
        return f"more varied {what} than usual" if up else f"less varied {what} than usual"
    return None


def explain(dev: Deviation, win: Prepared, baseline: Baseline, max_reasons: int = 5) -> list[str]:
    thr = cfg("deviation", "z_threshold")
    ranked = sorted((f for f in FEATURES if not math.isnan(dev.z[f]) and abs(dev.z[f]) > thr),
                    key=lambda f: -abs(dev.z[f]))
    reasons: list[str] = []

    # dominant language mix, if it changed
    win_mix = window_language_mix(win)
    base_mix = window_language_mix(Prepared(baseline.messages, []))
    if win_mix not in (base_mix, "Unknown") and any(FEATURES[f][0] == "language" for f in ranked):
        reasons.append(f"mostly {win_mix} (usually {base_mix})")

    for f in ranked:
        if len(reasons) >= max_reasons:
            break
        p = phrase(f, dev, win, baseline)
        if p and p not in reasons:
            reasons.append(p)

    # agreement line
    top_labels = []
    for f in ranked:
        lab = FEATURES[f][1]
        if lab not in top_labels:
            top_labels.append(lab)
        if len(top_labels) == 2:
            break
    if dev.agreement:
        inc = f", including {' and '.join(top_labels)}" if top_labels else ""
        groups = ", ".join(GROUP_LABELS[g] for g in dev.groups_changed)
        reasons.append(f"{dev.agreement} of 6 behavior groups changed{inc} ({groups})")
    else:
        reasons.append(f"0 of 6 behavior groups changed beyond the working threshold (|z| > {thr:g})")

    reasons.append(habits_vs_length(dev))
    return reasons


def habits_vs_length(dev: Deviation) -> str:
    thr = cfg("deviation", "z_threshold")
    habit_changed = any(abs(dev.z[f]) > thr for f in TOPIC_INDEPENDENT if not math.isnan(dev.z[f]))
    length_changed = any(abs(dev.z[f]) > thr for f in LENGTH_BASED if not math.isnan(dev.z[f]))
    if habit_changed:
        extra = " as well as message length" if length_changed else ""
        return f"The change includes topic-independent habits (how the user writes){extra}."
    if length_changed:
        return "The change is only in length (how much was written); writing habits look usual."
    return "Writing habits and message length look usual."


__all__ = ["explain", "habits_vs_length", "phrase", "fmt_hour", "LANGS"]
