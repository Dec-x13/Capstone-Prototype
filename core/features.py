"""Per-window behaviour features, grouped as in Table 4.3 of the manuscript.

Two stages keep raw text short-lived:
  1. prepare_messages(df) turns each message into numbers (word counts, tags, habits) plus a
     character n-gram Counter. After this, the raw `text` column is no longer needed.
  2. window_features(...) aggregates those numbers over a window of k messages, using context
     from the baseline (hour-of-day profile, recipients seen in the baseline).

Notes on choices the manuscript leaves open (documented here and in README):
  * Bubbles per burst is computed inside the window: k / number of bursts among the window's messages.
  * Messages per hour = k / span of the window in hours, with the span floored at 1 minute.
    Undefined for k = 1 (NaN, skipped by deviation).
  * ? and ! rate = number of '?' and '!' characters per message.
  * Code-switching pairs are counted within each message and pooled over the window.
  * Repeated sending = share of the window's messages that are near-identical (char n-gram cosine >=
    config value) to a message sent to a different recipient within the same window.
  * Language entropy uses the Bikol/Tagalog/English word distribution.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import cfg
from .language_tagger import LANGS, is_emoji, is_laughter, load_lexicon, tag_token, tokenize
from .windows import n_bursts

GROUPS = ("timing", "bubble", "language", "habits", "recipients", "entropy")
GROUP_LABELS = {
    "timing": "Timing", "bubble": "Bubble pattern", "language": "Language mix",
    "habits": "Writing habits", "recipients": "Recipients", "entropy": "Entropy",
}
LAUGH_STYLES = ("HAHAHA", "hahaha", "hehe", "lol", "none")

# name -> (group, human label)
FEATURES: dict[str, tuple[str, str]] = {
    "hour_prob": ("timing", "typical hour of day"),
    "msgs_per_hour": ("timing", "messages per hour"),
    "median_gap_s": ("timing", "median gap between messages"),
    "avg_words": ("bubble", "bubble length"),
    "long_share": ("bubble", "share of long bubbles"),
    "bubbles_per_burst": ("bubble", "bubbles per burst"),
    "share_bikol": ("language", "Bikol share"),
    "share_tagalog": ("language", "Tagalog share"),
    "share_english": ("language", "English share"),
    "share_ambiguous": ("language", "ambiguous-word share"),
    "code_switch_rate": ("language", "code-switching rate"),
    "stretch_rate": ("habits", "stretched letters"),
    **{f"laugh_{s}": ("habits", f"laughter style '{s}'") for s in LAUGH_STYLES},
    "punct_rate": ("habits", "? and ! use"),
    "caps_share": ("habits", "capital letters"),
    "emoji_rate": ("habits", "emoji use"),
    "abbrev_rate": ("habits", "abbreviations"),
    "ngram_distance": ("habits", "character n-gram profile"),
    "new_recipient_share": ("recipients", "messages to new recipients"),
    "distinct_recipients": ("recipients", "distinct recipients"),
    "repeated_send_share": ("recipients", "near-identical messages to several recipients"),
    "H_hours": ("entropy", "spread of active hours"),
    "H_recipients": ("entropy", "spread of recipients"),
    "H_languages": ("entropy", "spread of languages"),
}
FEATURE_NAMES = list(FEATURES)
TOPIC_INDEPENDENT = ["stretch_rate", *[f"laugh_{s}" for s in LAUGH_STYLES], "punct_rate", "caps_share",
                     "emoji_rate", "abbrev_rate"]
LENGTH_BASED = ["avg_words", "long_share", "bubbles_per_burst"]
ENTROPY_FEATURES = ["H_hours", "H_recipients", "H_languages"]

_STRETCH = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)


# ----------------------------------------------------------------------------- stage 1: per message
def laughter_style(tokens: list[str]) -> str:
    for t in tokens:
        if is_emoji(t) or not is_laughter(t):
            continue
        low = t.lower()
        if "l" in low or "x" in low or "m" in low or "f" in low:
            return "lol"
        if "e" in low or "i" in low:
            return "hehe"
        return "HAHAHA" if t.isupper() else "hahaha"
    return "none"


def char_ngrams(text: str, nmin: int | None = None, nmax: int | None = None) -> Counter:
    nmin = cfg("features", "ngram_min") if nmin is None else nmin
    nmax = cfg("features", "ngram_max") if nmax is None else nmax
    s = f" {' '.join((text or '').lower().split())} "
    c: Counter = Counter()
    for n in range(nmin, nmax + 1):
        c.update(s[i:i + n] for i in range(len(s) - n + 1))
    return c


@dataclass
class Prepared:
    """Per-message derived values for a timeline. Contains no message text."""
    table: pd.DataFrame
    ngrams: list[Counter]

    def __len__(self):
        return len(self.table)

    def take(self, idx) -> "Prepared":
        idx = list(idx)
        return Prepared(self.table.iloc[idx].reset_index(drop=True), [self.ngrams[i] for i in idx])

    @staticmethod
    def concat(parts: list["Prepared"]) -> "Prepared":
        return Prepared(pd.concat([p.table for p in parts], ignore_index=True),
                        [c for p in parts for c in p.ngrams])


def prepare_messages(df: pd.DataFrame) -> Prepared:
    """Numbers per message from (ts, recipient_code, thread_type, text). The text is not kept."""
    _, abbr = load_lexicon()
    long_words = cfg("features", "long_bubble_words")
    rows, grams = [], []
    for ts, rc, tt, text in zip(df["ts"], df["recipient_code"], df["thread_type"], df["text"]):
        toks = tokenize(text)
        tags = [tag_token(t) for t in toks]
        words = [t for t in toks if not is_emoji(t)]
        plain_words = [w for w in words if not is_laughter(w)]
        switches = pairs = 0
        for a, b in zip(tags, tags[1:]):
            if a in LANGS and b in LANGS:
                pairs += 1
                switches += a != b
        letters = [ch for ch in text if ch.isalpha()]
        rows.append({
            "ts": ts, "recipient_code": rc, "thread_type": tt, "hour": ts.hour,
            "n_words": len(words), "is_long": int(len(words) > long_words),
            "n_plain_words": len(plain_words),
            "n_stretched": sum(bool(_STRETCH.search(w)) for w in plain_words),
            "laugh": laughter_style(toks),
            "n_punct": text.count("?") + text.count("!"),
            "n_upper": sum(ch.isupper() for ch in letters), "n_letters": len(letters),
            "n_emoji": sum(is_emoji(t) for t in toks),
            "n_abbr": sum(w.lower() in abbr for w in plain_words),
            **{f"n_{lang}": tags.count(lang) for lang in LANGS + ("ambiguous",)},
            "cs_switches": switches, "cs_pairs": pairs,
        })
        grams.append(char_ngrams(text))
    table = pd.DataFrame(rows)
    if len(table) == 0:
        table = pd.DataFrame(columns=["ts", "recipient_code", "thread_type", "hour"])
    return Prepared(table, grams)


# ----------------------------------------------------------------------------- helpers
def entropy(values) -> float:
    """Shannon entropy in bits of the empirical distribution of `values` (or of a count dict)."""
    counts = list(values.values()) if isinstance(values, dict) else list(Counter(values).values())
    n = sum(counts)
    if n <= 0:
        return 0.0
    return float(-sum((c / n) * math.log2(c / n) for c in counts if c > 0))


def cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * big.get(g, 0) for g, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def cosine_distance(a: Counter, b: Counter) -> float:
    return 1.0 - cosine_similarity(a, b)


def hour_profile(hours, alpha: float | None = None) -> np.ndarray:
    """p_h = (count in hour h + alpha) / (M + 24 * alpha)."""
    alpha = cfg("features", "hour_smoothing_alpha") if alpha is None else alpha
    counts = np.bincount(np.asarray(list(hours), dtype=int), minlength=24).astype(float)
    return (counts + alpha) / (counts.sum() + 24 * alpha)


def sum_ngrams(counters) -> Counter:
    total: Counter = Counter()
    for c in counters:
        total.update(c)
    return total


# ----------------------------------------------------------------------------- stage 2: per window
def window_features(win: Prepared, hour_prob: np.ndarray, known_recipients: set,
                    ngram_reference: Counter | None = None) -> dict[str, float]:
    """Features for one window. `hour_prob` and `known_recipients` come from the baseline.
    `ngram_reference` is the baseline n-gram profile the window is compared against."""
    t = win.table
    k = len(t)
    f: dict[str, float] = {}
    # timing
    f["hour_prob"] = float(np.mean(hour_prob[t["hour"].to_numpy(dtype=int)]))
    if k >= 2:
        secs = t["ts"].diff().dt.total_seconds().to_numpy()[1:]
        span_h = max((t["ts"].iloc[-1] - t["ts"].iloc[0]).total_seconds(), 60.0) / 3600.0
        f["msgs_per_hour"] = k / span_h
        f["median_gap_s"] = float(np.median(secs))
    else:
        f["msgs_per_hour"] = float("nan")
        f["median_gap_s"] = float("nan")
    # bubble pattern
    f["avg_words"] = float(t["n_words"].mean())
    f["long_share"] = float(t["is_long"].mean())
    f["bubbles_per_burst"] = k / max(n_bursts(t["ts"]), 1)
    # language mix
    counts = {lang: int(t[f"n_{lang}"].sum()) for lang in LANGS + ("ambiguous",)}
    n_lang = sum(counts.values())
    for lang in LANGS + ("ambiguous",):
        f[f"share_{lang}"] = counts[lang] / n_lang if n_lang else 0.0
    pairs = int(t["cs_pairs"].sum())
    f["code_switch_rate"] = int(t["cs_switches"].sum()) / pairs if pairs else 0.0
    # writing habits
    pw = int(t["n_plain_words"].sum())
    f["stretch_rate"] = int(t["n_stretched"].sum()) / pw if pw else 0.0
    for s in LAUGH_STYLES:
        f[f"laugh_{s}"] = float((t["laugh"] == s).mean())
    f["punct_rate"] = float(t["n_punct"].mean())
    letters = int(t["n_letters"].sum())
    f["caps_share"] = int(t["n_upper"].sum()) / letters if letters else 0.0
    f["emoji_rate"] = float(t["n_emoji"].mean())
    f["abbrev_rate"] = int(t["n_abbr"].sum()) / pw if pw else 0.0
    profile = sum_ngrams(win.ngrams)
    f["ngram_distance"] = cosine_distance(profile, ngram_reference) if ngram_reference is not None else float("nan")
    # recipients
    rc = t["recipient_code"].tolist()
    f["new_recipient_share"] = float(np.mean([r not in known_recipients for r in rc]))
    f["distinct_recipients"] = float(len(set(rc)))
    f["repeated_send_share"] = repeated_send_share(win)
    # entropy
    f["H_hours"] = entropy(t["hour"].tolist())
    f["H_recipients"] = entropy(rc)
    f["H_languages"] = entropy({lang: counts[lang] for lang in LANGS})
    return f


def repeated_send_share(win: Prepared, threshold: float | None = None) -> float:
    threshold = cfg("features", "near_identical_similarity") if threshold is None else threshold
    rc = win.table["recipient_code"].tolist()
    k = len(rc)
    if k < 2:
        return 0.0
    flagged = set()
    for i in range(k):
        for j in range(i + 1, k):
            if rc[i] != rc[j] and cosine_similarity(win.ngrams[i], win.ngrams[j]) >= threshold:
                flagged.update((i, j))
    return len(flagged) / k


def window_language_mix(win: Prepared) -> str:
    from .language_tagger import dominant_mix
    t = win.table
    counts = {lang: int(t[f"n_{lang}"].sum()) for lang in LANGS}
    n = sum(counts.values())
    return dominant_mix({k: (v / n if n else 0.0) for k, v in counts.items()})
