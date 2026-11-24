"""Read a Facebook "Download Your Information" Messenger export (JSON) into owner-only messages.

Privacy: messages written by anyone other than the owner are dropped while each thread file is
parsed, before anything is collected. The returned `text` column must stay in memory only.

Layouts are pluggable. Each layout is a small class registered with @register_layout that
(1) says whether it recognises a folder and (2) yields Thread records. To support a new
Facebook layout, add one class; nothing else changes.

Supported now (verified against the SYNTHETIC sample only; re-check on a real export):
  legacy      messages/inbox/<thread>/message_N.json
  dyi_2024    your_facebook_activity/messages/<inbox|archived_threads|...>/<thread>/message_N.json
  e2ee        separate end-to-end-encrypted chats download (PROVISIONAL: key names assumed,
              to be confirmed on a real export before the defense)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from .anonymize import RecipientCoder
from .config import TIMEZONE

THREAD_SUBFOLDERS = ("inbox", "archived_threads", "filtered_threads", "message_requests", "e2ee_cutover")
MEDIA_KEYS = ("photos", "videos", "audio_files", "files", "gifs", "sticker")
SKIP_TYPES = {"Call", "Subscribe", "Unsubscribe"}
# System placeholders Facebook writes into `content`.
_PLACEHOLDER = re.compile(
    r"^(?:.+ (?:unsent a message|sent an attachment|sent a photo|sent a video|sent a sticker"
    r"|sent a voice message|sent a GIF|called you|missed your call|missed a call"
    r"|set the nickname.*|named the group.*|changed the group photo|left the group"
    r"|added .+ to the group|removed .+ from the group)\.?"
    r"|Reacted .+ to your message|You unsent a message\.?|This message was unsent\.?)$",
    re.IGNORECASE,
)


def fix_encoding(s: str) -> str:
    """Undo Facebook's bug (UTF-8 bytes stored as latin-1 code points). Safe on clean text."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


@dataclass
class RawMessage:
    sender: str
    timestamp_ms: int
    content: str | None
    is_text: bool  # False for media, calls, unsent, placeholders, reactions-only


@dataclass
class Thread:
    thread_id: str
    title: str
    participants: list[str]
    messages: list[RawMessage] = field(default_factory=list)

    @property
    def is_group(self) -> bool:
        return len(set(self.participants)) > 2


# ----------------------------------------------------------------------------- layout registry
LAYOUTS: list[type["Layout"]] = []


def register_layout(cls):
    LAYOUTS.append(cls)
    return cls


class Layout:
    name = "base"

    @classmethod
    def detect(cls, root: Path) -> bool:
        raise NotImplementedError

    @classmethod
    def threads(cls, root: Path) -> Iterator[Thread]:
        raise NotImplementedError


def _thread_dirs(messages_dir: Path) -> Iterator[Path]:
    for sub in THREAD_SUBFOLDERS:
        base = messages_dir / sub
        if base.is_dir():
            for d in sorted(p for p in base.iterdir() if p.is_dir()):
                if any(d.glob("message_*.json")):
                    yield d


def _parse_classic_message(m: dict) -> RawMessage:
    sender = fix_encoding(m.get("sender_name", ""))
    content = m.get("content")
    content = fix_encoding(content) if isinstance(content, str) else None
    is_text = bool(content and content.strip())
    if m.get("is_unsent") or m.get("type") in SKIP_TYPES or "call_duration" in m:
        is_text = False
    if any(k in m for k in MEDIA_KEYS):
        is_text = False
    if is_text and _PLACEHOLDER.match(content.strip()):
        is_text = False
    return RawMessage(sender, int(m.get("timestamp_ms", 0)), content, is_text)


def _read_classic_thread(folder: Path) -> Thread:
    """Facebook's classic per-thread JSON (participants/messages/title/thread_path)."""
    participants: list[str] = []
    title = ""
    msgs: list[RawMessage] = []
    for f in sorted(folder.glob("message_*.json")):
        with open(f, encoding="utf-8") as fh:
            doc = json.load(fh)
        participants = participants or [fix_encoding(p.get("name", "")) for p in doc.get("participants", [])]
        title = title or fix_encoding(doc.get("title", ""))
        msgs.extend(_parse_classic_message(m) for m in doc.get("messages", []))
    return Thread(folder.name, title, participants, msgs)


@register_layout
class LegacyLayout(Layout):
    name = "legacy"

    @classmethod
    def _messages_dir(cls, root: Path) -> Path | None:
        for cand in (root / "messages", root):
            if (cand / "inbox").is_dir():
                return cand
        return None

    @classmethod
    def detect(cls, root: Path) -> bool:
        md = cls._messages_dir(root)
        return md is not None and not (root / "your_facebook_activity").exists()

    @classmethod
    def threads(cls, root: Path) -> Iterator[Thread]:
        for d in _thread_dirs(cls._messages_dir(root)):
            yield _read_classic_thread(d)


@register_layout
class DYI2024Layout(Layout):
    name = "dyi_2024"

    @classmethod
    def detect(cls, root: Path) -> bool:
        return (root / "your_facebook_activity" / "messages").is_dir()

    @classmethod
    def threads(cls, root: Path) -> Iterator[Thread]:
        for d in _thread_dirs(root / "your_facebook_activity" / "messages"):
            yield _read_classic_thread(d)


@register_layout
class E2EELayout(Layout):
    """PROVISIONAL. The separate encrypted-chats download is one JSON per chat with camelCase keys.
    Key names below are assumptions until we inspect a real file; adjust _KEYS if they differ."""
    name = "e2ee"
    _KEYS = {"participants": "participants", "messages": "messages", "title": "threadName",
             "sender": "senderName", "ts": "timestamp", "text": "text", "unsent": "isUnsent",
             "media": "media"}

    @classmethod
    def _files(cls, root: Path) -> list[Path]:
        out = []
        for f in sorted(root.rglob("*.json")):
            if f.name.startswith("message_"):
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    head = fh.read(4096)
            except OSError:
                continue
            if f'"{cls._KEYS["sender"]}"' in head or f'"{cls._KEYS["title"]}"' in head:
                out.append(f)
        return out

    @classmethod
    def detect(cls, root: Path) -> bool:
        return not LegacyLayout.detect(root) and not DYI2024Layout.detect(root) and bool(cls._files(root))

    @classmethod
    def threads(cls, root: Path) -> Iterator[Thread]:
        k = cls._KEYS
        for f in cls._files(root):
            with open(f, encoding="utf-8") as fh:
                doc = json.load(fh)
            parts = [fix_encoding(p if isinstance(p, str) else p.get("name", "")) for p in doc.get(k["participants"], [])]
            msgs = []
            for m in doc.get(k["messages"], []):
                text = m.get(k["text"])
                text = fix_encoding(text) if isinstance(text, str) else None
                ts = int(m.get(k["ts"], 0))
                ts = ts * 1000 if ts < 10**11 else ts  # seconds -> ms if needed
                is_text = bool(text and text.strip()) and not m.get(k["unsent"]) and not m.get(k["media"])
                if is_text and _PLACEHOLDER.match(text.strip()):
                    is_text = False
                msgs.append(RawMessage(fix_encoding(m.get(k["sender"], "")), ts, text, is_text))
            yield Thread(f.stem, fix_encoding(doc.get(k["title"], f.stem)), parts, msgs)


# ----------------------------------------------------------------------------- public API
class UnknownLayoutError(ValueError):
    pass


def detect_layouts(root: str | Path) -> list[str]:
    """All layouts recognised under `root` (an export can hold DYI and E2EE parts side by side)."""
    root = Path(root)
    found = [L.name for L in LAYOUTS if L.detect(root)]
    # An E2EE download unzipped next to the main export: look one level down too.
    if not found:
        for child in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            found += [f"{L.name}:{child.name}" for L in LAYOUTS if L.detect(child)]
    return found


def iter_threads(root: str | Path) -> Iterator[tuple[str, Thread]]:
    root = Path(root)
    layouts = detect_layouts(root)
    if not layouts:
        raise UnknownLayoutError(
            "No Messenger data found. Expected messages/inbox/... or your_facebook_activity/messages/...")
    by_name = {L.name: L for L in LAYOUTS}
    for spec in layouts:
        name, _, sub = spec.partition(":")
        for t in by_name[name].threads(root / sub if sub else root):
            yield spec, t


def suggest_owner(root: str | Path) -> str:
    """Most frequent sender across all threads (only a suggestion; the user confirms in the UI)."""
    counts: Counter = Counter()
    for _, t in iter_threads(root):
        counts.update(m.sender for m in t.messages)
    if not counts:
        raise UnknownLayoutError("Export contains no messages.")
    return counts.most_common(1)[0][0]


@dataclass
class LoadResult:
    df: pd.DataFrame          # ts, recipient_code, thread_type, text  (text: memory only)
    layouts: list[str]
    n_threads: int
    n_owner_text: int
    n_other_dropped: int
    n_owner_skipped: int      # owner items without usable text (media, calls, unsent, ...)

    def summary(self) -> dict:
        """Counts only, safe to display."""
        df = self.df
        return {
            "layouts": ", ".join(self.layouts),
            "threads": self.n_threads,
            "messages kept": self.n_owner_text,
            "other people's messages dropped": self.n_other_dropped,
            "owner non-text items skipped": self.n_owner_skipped,
            "recipients": int(df["recipient_code"].nunique()) if len(df) else 0,
            "first message": df["ts"].min().strftime("%Y-%m-%d") if len(df) else "-",
            "last message": df["ts"].max().strftime("%Y-%m-%d") if len(df) else "-",
        }


def load_owner_messages(root: str | Path, owner: str, coder: RecipientCoder | None = None) -> LoadResult:
    """Owner's own text messages, oldest first. Everyone else's messages are discarded immediately."""
    coder = coder or RecipientCoder()
    owner = owner.strip()
    rows = []
    n_threads = n_other = n_skip = 0
    layouts = detect_layouts(root)
    for _, thread in iter_threads(root):
        n_threads += 1
        others = [p for p in thread.participants if p != owner]
        if thread.is_group:
            recipient = f"group:{thread.thread_id}"
        else:
            recipient = others[0] if others else f"thread:{thread.thread_id}"
        code = coder.code(recipient)
        ttype = "group" if thread.is_group else "one-to-one"
        for m in thread.messages:
            if m.sender != owner:
                n_other += 1
                continue
            if not m.is_text:
                n_skip += 1
                continue
            rows.append((m.timestamp_ms, code, ttype, m.content))
        thread.messages.clear()  # release other people's text as early as possible
    df = pd.DataFrame(rows, columns=["ts", "recipient_code", "thread_type", "text"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(TIMEZONE)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return LoadResult(df, layouts, n_threads, len(df), n_other, n_skip)
