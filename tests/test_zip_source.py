"""Reading exports straight from .zip parts, all thread subfolders, and the system-line filter."""
import shutil
import zipfile
from pathlib import Path

import pytest

from conftest import OWNER_A, OWNER_B, USER_A, USER_B
from core.export_reader import (UnknownLayoutError, _parse_classic_message, detect_layouts, load_owner_messages,
                                suggest_owner)


def make_zips(src: Path, dest: Path, parts: int, prefix: str = "") -> list[Path]:
    """Split `src` into `parts` zips the way Facebook does: same folder tree, files spread across parts."""
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src.rglob("*") if f.is_file())
    zips = [dest / f"facebook-synthetic-part{i}.zip" for i in range(parts)]
    handles = [zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) for z in zips]
    for i, f in enumerate(files):
        handles[i % parts].write(f, prefix + f.relative_to(src).as_posix())
    for h in handles:
        h.writestr("facebook_000.zip", b"")  # Facebook adds empty nested zips at the top level
        h.close()
    return zips


def same_messages(a, b):
    assert list(a.df["ts"]) == list(b.df["ts"])
    assert list(a.df["text"]) == list(b.df["text"])
    assert list(a.df["thread_type"]) == list(b.df["thread_type"])
    assert a.df["recipient_code"].nunique() == b.df["recipient_code"].nunique()
    assert (a.n_threads, a.n_other_dropped, a.skipped_by_kind) == (b.n_threads, b.n_other_dropped, b.skipped_by_kind)


def test_folder_of_zip_parts_matches_unzipped_folder(tmp_path):
    make_zips(USER_B, tmp_path / "parts", 3)
    assert detect_layouts(tmp_path / "parts") == ["dyi_2024"]
    assert suggest_owner(tmp_path / "parts") == OWNER_B
    z = load_owner_messages(tmp_path / "parts", OWNER_B)
    same_messages(z, load_owner_messages(USER_B, OWNER_B))
    assert z.source == "3 .zip files (read in place)"
    assert z.summary()["source"] == z.source
    assert load_owner_messages(USER_B, OWNER_B).source == "unzipped folder"


def test_single_zip_legacy_layout(tmp_path):
    (z,) = make_zips(USER_A, tmp_path, 1)
    assert detect_layouts(z) == ["legacy"]
    same_messages(load_owner_messages(z, OWNER_A), load_owner_messages(USER_A, OWNER_A))


def test_zip_with_wrapper_folder(tmp_path):
    make_zips(USER_B, tmp_path / "parts", 2, prefix="facebook-ben/")
    assert detect_layouts(tmp_path / "parts") == ["dyi_2024:facebook-ben"]
    assert len(load_owner_messages(tmp_path / "parts", OWNER_B).df) == len(load_owner_messages(USER_B, OWNER_B).df)


def test_nothing_is_extracted(tmp_path):
    make_zips(USER_B, tmp_path, 2)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    load_owner_messages(tmp_path, OWNER_B)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_broken_zip_gives_clear_error(tmp_path):
    (tmp_path / "part1.zip").write_bytes(b"not a zip")
    with pytest.raises(UnknownLayoutError):
        detect_layouts(tmp_path)


def test_unknown_thread_subfolders_are_read(tmp_path):
    root = tmp_path / "ben"
    shutil.copytree(USER_B, root)
    msgs = root / "your_facebook_activity" / "messages"
    threads = sorted(p for p in (msgs / "inbox").iterdir() if p.is_dir())
    for sub, th in (("e2ee_cutover_archived", threads[0]), ("brand_new_folder", threads[1])):
        (msgs / sub).mkdir()
        shutil.move(str(th), str(msgs / sub / th.name))
    (msgs / "photos").mkdir()  # a media-only folder is not a thread folder
    (msgs / "photos" / "x.jpg").write_bytes(b"\xff\xd8")
    moved, orig = load_owner_messages(root, OWNER_B), load_owner_messages(USER_B, OWNER_B)
    assert moved.n_threads == orig.n_threads
    assert sorted(moved.df["text"]) == sorted(orig.df["text"])


@pytest.mark.parametrize("line", [
    "You joined the call.", "You joined the video call", "Maria joined the video chat.", "You pinned a message.",
    "You started a video call", "You started sharing video.", "You set the nickname for R to Bes.",
    "You set the quick reaction to 😂.", "You changed the theme to Love.", "You changed the chat name to Barkada.",
    "You named the group Tropa.", "You turned on disappearing messages.", "You can now message and call each other.",
    "You are now connected on Messenger.", "Juan left the group.", "You added Ana to the group.",
])
def test_system_lines_are_not_text(line):
    m = _parse_classic_message({"sender_name": "A", "timestamp_ms": 1, "content": line})
    assert not m.is_text and m.kind == "other"


@pytest.mark.parametrize("line", [
    "You can do it!", "You are now my favorite haha", "joined the call ba ka?", "I pinned a message for you",
    "You started it lol", "You changed a lot", "tara call tayo", "You set me up",
])
def test_ordinary_messages_stay_text(line):
    assert _parse_classic_message({"sender_name": "A", "timestamp_ms": 1, "content": line}).is_text
