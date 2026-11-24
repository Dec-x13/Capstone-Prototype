"""Recipient coding and display masking (privacy design, see README).

* RecipientCoder turns contact/thread names into stable random codes (R07, R42, ...).
  The salt lives only in memory, so codes are stable within a session but cannot be
  mapped back to names after the session ends. The name->code table is never written to disk.
* mask() replaces phone numbers, emails, URLs and long digit runs with placeholders.
  Apply it to every piece of text that is ever displayed.
"""
from __future__ import annotations

import random
import re
import secrets


class RecipientCoder:
    def __init__(self, salt: bytes | None = None, prefix: str = "R"):
        self._salt = salt if salt is not None else secrets.token_bytes(16)
        pool = list(range(1, 1000))
        random.Random(self._salt).shuffle(pool)
        # Two-digit codes first so typical exports get short codes like R07.
        self._pool = [n for n in pool if n < 100] + [n for n in pool if n >= 100]
        self._codes: dict[str, str] = {}
        self._prefix = prefix

    def code(self, name: str) -> str:
        if name not in self._codes:
            n = self._pool[len(self._codes)]
            self._codes[name] = f"{self._prefix}{n:02d}"
        return self._codes[name]

    def __len__(self) -> int:
        return len(self._codes)

    def __repr__(self) -> str:  # never reveal names
        return f"RecipientCoder({len(self._codes)} recipients)"


_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|net|org|ph|io|me|ly|co)(?:/\S*)?\b")
_EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?63[\s-]?|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
_DIGITS = re.compile(r"(?<!\d)\d(?:[\s-]?\d){5,}(?!\d)")


def mask(text: str) -> str:
    """Mask URLs, emails, PH phone numbers (09xx / +63) and digit runs of 6 or more."""
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _DIGITS.sub("[NUMBER]", text)
    return text
