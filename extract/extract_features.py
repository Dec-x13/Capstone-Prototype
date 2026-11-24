"""Offline participant extraction script: features, timestamps, coded recipients and hashed counts only.

Usage (no internet needed):
    python extract_features.py <export_path> --owner "Name" --out features.json
    python extract_features.py <export_path> --owner "Name" --preview      # show what would be shared, save nothing

What the output contains:
  * per-message timestamps, recipient codes (R07, ...) and thread type (direct / group);
  * per-window behaviour features from core/features.py (numbers only);
  * per-window hashed word counts and hashed character n-gram counts.

Hashing uses SHA-256 with a random salt that is created for this run and never saved, so the
hashes cannot be looked up against a word list and words cannot be read back. Hashes are
consistent within one output file, so counts can still be compared across windows.
Message text, names, and the salt are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.anonymize import RecipientCoder  # noqa: E402
from core.export_reader import load_owner_messages  # noqa: E402
from core.features import Prepared, hour_profile, prepare_messages, sum_ngrams, window_features  # noqa: E402
from core.language_tagger import tokenize  # noqa: E402

SCHEMA = "behaviorguard-features/1"
THREAD_TYPES = {"one-to-one": "direct", "group": "group"}
# Output-safe names for laughter features (style names like 'lol' are themselves chat tokens).
RENAME = {"laugh_HAHAHA": "laugh_upper_ha", "laugh_hahaha": "laugh_lower_ha", "laugh_hehe": "laugh_he_style",
          "laugh_lol": "laugh_acronym", "laugh_none": "laugh_absent"}


def make_hasher(salt: bytes):
    def h(token: str) -> str:
        return hashlib.sha256(salt + token.encode("utf-8")).hexdigest()[:16]
    return h


def hashed_counts(texts: list[str], h) -> tuple[dict, dict]:
    words: Counter = Counter()
    grams: Counter = Counter()
    for t in texts:
        words.update(h(w.lower()) for w in tokenize(t))
    for g, c in sum_ngrams(_ngrams(texts)).items():
        grams[h(g)] += c
    return dict(words), dict(grams)


def _ngrams(texts):
    from core.features import char_ngrams
    return [char_ngrams(t) for t in texts]


def extract(export_path: str, owner: str, k: int = 5, n: int = 100) -> dict:
    salt = secrets.token_bytes(32)          # memory only, never written
    h = make_hasher(salt)
    res = load_owner_messages(export_path, owner, RecipientCoder())
    df = res.df
    prep: Prepared = prepare_messages(df)
    texts = df["text"].tolist()

    messages = [{"ts": ts.isoformat(), "recipient": rc, "thread_type": THREAD_TYPES[tt]}
                for ts, rc, tt in zip(df["ts"], df["recipient_code"], df["thread_type"])]

    windows = []
    for start in range(n, len(prep) - k + 1, k):
        base = prep.take(range(start - n, start))
        win = prep.take(range(start, start + k))
        feats = window_features(win, hour_profile(base.table["hour"]), set(base.table["recipient_code"]),
                                sum_ngrams(base.ngrams))
        feats = {RENAME.get(key, key): (None if v != v else round(float(v), 6)) for key, v in feats.items()}
        words, grams = hashed_counts(texts[start:start + k], h)
        windows.append({"first_message": start, "k": k, "baseline_n": n, "features": feats,
                        "hashed_word_counts": words, "hashed_char_ngram_counts": grams})
    del texts, df  # drop text as soon as features exist

    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layouts": res.layouts,
        "counts": {"messages": len(messages), "windows": len(windows), "recipients": len(set(m["recipient"] for m in messages)),
                   "other_peoples_messages_dropped": res.n_other_dropped},
        "hashing": "sha256(per-run random salt, not saved) truncated to 16 hex characters",
        "messages": messages,
        "windows": windows,
    }


def summary(out: dict) -> str:
    c = out["counts"]
    first = out["messages"][0]["ts"][:10] if out["messages"] else "-"
    last = out["messages"][-1]["ts"][:10] if out["messages"] else "-"
    lines = [
        "=" * 64,
        "WHAT WILL BE SHARED (nothing else is in the file)",
        "=" * 64,
        f"  Messages: {c['messages']} timestamps from {first} to {last}",
        f"  Recipients: {c['recipients']} random codes (e.g. R07), names are not included",
        f"  Windows: {c['windows']}, each with {len(out['windows'][0]['features']) if out['windows'] else 0} numeric features",
        "  Hashed word counts and hashed character n-gram counts per window",
        "     (salted SHA-256; the salt is discarded, so words cannot be read back)",
        "  NOT included: message text, your name, contact names, links, phone numbers, photos",
        f"  Other people's messages were dropped while reading ({c['other_peoples_messages_dropped']}).",
    ]
    if out["windows"]:
        w = out["windows"][0]
        sample = {k: v for k, v in list(w["features"].items())[:6]}
        hw = list(w["hashed_word_counts"].items())[:3]
        lines += ["", "  Example window features: " + json.dumps(sample),
                  "  Example hashed word counts: " + json.dumps(dict(hw))]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract behaviour features (no message text) from a Messenger export.")
    ap.add_argument("export_path")
    ap.add_argument("--owner", required=True, help="Your display name exactly as in the export")
    ap.add_argument("--out", default="features.json")
    ap.add_argument("--k", type=int, default=5, choices=[1, 3, 5, 10])
    ap.add_argument("--n", type=int, default=100, choices=[50, 100, 150, 200])
    ap.add_argument("--preview", action="store_true", help="Show what would be shared and save nothing")
    args = ap.parse_args(argv)

    out = extract(args.export_path, args.owner, args.k, args.n)
    print(summary(out))
    if args.preview:
        print("\nPreview only: nothing was saved.")
        return 0
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True)
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
