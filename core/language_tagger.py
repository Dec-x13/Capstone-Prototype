"""Rule-based token language tagger for code-switched Bikol / Tagalog / English chat.

Tags: bikol, tagalog, english, ambiguous, other.

Steps for each token:
  1. lowercase, strip punctuation (hyphenated words are split: "pa-gcash" -> "pa", "gcash")
  2. laughter (haha, hehe, lol, ...), emoji and numbers -> other
  3. abbreviation -> its expansion for lookup (wordlists/abbreviations.txt)
  4. lookup as is; else shorten stretched letters (3+ repeats -> 1, then -> 2) and look up again
  5. found in more than one list -> ambiguous; in no list -> other

Dominant mix of a window (documented rule, threshold in config.toml [language].mix_second_share):
  Look only at Bikol/Tagalog/English word shares (ambiguous and other excluded).
  If the second most common language reaches the threshold, the mix is the pair:
      Bikol+English = Bi-lish, Tagalog+Bikol = Tag-col, Tagalog+English = Taglish.
  Otherwise the mix is the single top language. No known words -> "Unknown".
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import WORDLIST_DIR, cfg

LANGS = ("bikol", "tagalog", "english")
TAGS = LANGS + ("ambiguous", "other")

_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿️‍❤]")
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|" + _EMOJI.pattern)
_LAUGH = re.compile(r"^(?:a?(?:h+a+){2,}h*|(?:h+e+){2,}h*|(?:h+i+){2,}h*|(?:j+a+){2,}|l+o+l+|lmao+|rofl|xd+|wahaha\w*)$",
                    re.IGNORECASE)
_STRETCH3 = re.compile(r"(\w)\1{2,}")
_STRETCH2 = re.compile(r"(\w)\1{1,}")


def _read_list(path: Path) -> set[str]:
    out = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            out.add(line.split()[0])
    return out


@lru_cache(maxsize=4)
def load_lexicon(wordlist_dir: str | None = None) -> tuple[dict[str, frozenset], dict[str, str]]:
    d = Path(wordlist_dir) if wordlist_dir else WORDLIST_DIR
    lex = {lang: frozenset(_read_list(d / f"{lang}.txt")) for lang in LANGS}
    abbr = {}
    p = d / "abbreviations.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                abbr[parts[0].lower()] = parts[1].lower()
    return lex, abbr


def tokenize(text: str) -> list[str]:
    """Words (hyphen parts split) and individual emoji, in order. Punctuation is dropped."""
    return _TOKEN.findall(text or "")


def is_emoji(tok: str) -> bool:
    return bool(_EMOJI.fullmatch(tok))


def is_laughter(tok: str) -> bool:
    return bool(_LAUGH.match(tok))


def destretch(word: str) -> tuple[str, str]:
    """(3+ repeats -> 1, 3+ repeats -> 2). 'hiiii' -> ('hi', 'hii'); 'dakooool' -> ('dakol', 'dakool')."""
    return _STRETCH3.sub(r"\1", word), _STRETCH3.sub(r"\1\1", word)


def _langs_of(word: str, lex) -> set[str]:
    return {lang for lang in LANGS if word in lex[lang]}


def tag_token(tok: str, wordlist_dir: str | None = None) -> str:
    lex, abbr = load_lexicon(wordlist_dir)
    if is_emoji(tok):
        return "other"
    w = re.sub(r"[\W_]", "", tok.lower())  # strip punctuation
    if not w or w.isdigit() or is_laughter(w):
        return "other"
    candidates = [w]
    one, two = destretch(w)
    if one != w:
        candidates += [one, two]
    for c in candidates:
        c = abbr.get(c, c)
        langs = _langs_of(c, lex)
        if langs:
            return langs.pop() if len(langs) == 1 else "ambiguous"
    return "other"


def tag_text(text: str, wordlist_dir: str | None = None) -> list[tuple[str, str]]:
    return [(t, tag_token(t, wordlist_dir)) for t in tokenize(text)]


def language_shares(tags: list[str]) -> dict[str, float]:
    """Share of bikol/tagalog/english/ambiguous among word tokens that are not 'other'."""
    words = [t for t in tags if t != "other"]
    n = len(words)
    return {k: (words.count(k) / n if n else 0.0) for k in LANGS + ("ambiguous",)}


def code_switch_count(tags: list[str]) -> tuple[int, int]:
    """(switches, eligible pairs): adjacent token pairs where both have a known language."""
    pairs = switches = 0
    for a, b in zip(tags, tags[1:]):
        if a in LANGS and b in LANGS:
            pairs += 1
            switches += a != b
    return switches, pairs


def code_switch_rate(tags: list[str]) -> float:
    s, p = code_switch_count(tags)
    return s / p if p else 0.0


MIX_NAMES = {
    frozenset({"bikol", "english"}): "Bi-lish",
    frozenset({"tagalog", "bikol"}): "Tag-col",
    frozenset({"tagalog", "english"}): "Taglish",
}
SINGLE_NAMES = {"bikol": "Bikol", "tagalog": "Tagalog", "english": "English"}


def dominant_mix(shares: dict[str, float], second_share: float | None = None) -> str:
    second_share = cfg("language", "mix_second_share") if second_share is None else second_share
    known = {k: shares.get(k, 0.0) for k in LANGS}
    total = sum(known.values())
    if total <= 0:
        return "Unknown"
    ranked = sorted(known, key=lambda k: known[k], reverse=True)
    top, second = ranked[0], ranked[1]
    if known[second] / total >= second_share:
        return MIX_NAMES[frozenset({top, second})]
    return SINGLE_NAMES[top]
