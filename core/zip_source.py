"""Read a Facebook export straight from its .zip parts, without extracting anything to disk.

Facebook splits a large "Download Your Information" export into many .zip files. Each part holds a slice of the
same folder tree (`your_facebook_activity/messages/...`), mostly photos and videos. `ZipTree` merges the parts
into one read-only, path-like view so the layout readers in `export_reader` work unchanged. JSON files are read
into memory on demand; media is never read, and nothing is written to disk.

Only the small subset of `pathlib.Path` that the readers use is implemented.
"""
from __future__ import annotations

import fnmatch
import io
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator


class _Index:
    """Merged table of contents of several zip files. Zip handles stay open while the index is cached."""

    def __init__(self, zips: tuple[Path, ...]):
        self.zips = [zipfile.ZipFile(z) for z in zips]
        self.files: dict[str, tuple[zipfile.ZipFile, zipfile.ZipInfo]] = {}
        self.children: dict[str, set[str]] = {"": set()}
        for zf in self.zips:
            for info in zf.infolist():
                name = info.filename.rstrip("/")
                if not name:
                    continue
                if not info.is_dir():
                    self.files.setdefault(name, (zf, info))
                parts = name.split("/")
                for i in range(len(parts)):
                    parent, child = "/".join(parts[:i]), "/".join(parts[: i + 1])
                    self.children.setdefault(parent, set()).add(child)
                    if i < len(parts) - 1 or info.is_dir():
                        self.children.setdefault(child, set())

    def is_dir(self, key: str) -> bool:
        return key in self.children and key not in self.files

    def read(self, key: str) -> bytes:
        zf, info = self.files[key]
        return zf.read(info)


@lru_cache(maxsize=4)
def _index(zips: tuple[Path, ...], stamp: tuple) -> _Index:  # stamp: sizes and mtimes, so edits re-index
    return _Index(zips)


class ZipTree:
    """Path-like node inside a merged set of zip files."""

    def __init__(self, index: _Index, key: str = "", label: str = ""):
        self._ix, self._key, self._label = index, key, label

    # --- construction
    @classmethod
    def from_zips(cls, zips: list[Path], label: str = "") -> "ZipTree":
        zips = tuple(sorted(Path(z).resolve() for z in zips))
        stamp = tuple((z.stat().st_size, z.stat().st_mtime_ns) for z in zips)
        return cls(_index(zips, stamp), "", label or (zips[0].parent.name if zips else ""))

    # --- Path-like API used by the layout readers
    def __truediv__(self, other: str) -> "ZipTree":
        other = str(other).strip("/")
        return ZipTree(self._ix, f"{self._key}/{other}" if self._key else other, self._label)

    def __repr__(self) -> str:
        return f"ZipTree({self._label}:{self._key or '/'})"

    def __lt__(self, other: "ZipTree") -> bool:
        return self._key < other._key

    def __eq__(self, other) -> bool:
        return isinstance(other, ZipTree) and other._ix is self._ix and other._key == self._key

    def __hash__(self) -> int:
        return hash((id(self._ix), self._key))

    @property
    def name(self) -> str:
        return self._key.rsplit("/", 1)[-1] if self._key else self._label

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    def exists(self) -> bool:
        return self._key in self._ix.children or self._key in self._ix.files

    def is_dir(self) -> bool:
        return self._ix.is_dir(self._key)

    def is_file(self) -> bool:
        return self._key in self._ix.files

    def iterdir(self) -> Iterator["ZipTree"]:
        for c in sorted(self._ix.children.get(self._key, ())):
            yield ZipTree(self._ix, c, self._label)

    def glob(self, pattern: str) -> Iterator["ZipTree"]:
        for c in self.iterdir():
            if fnmatch.fnmatchcase(c.name, pattern):
                yield c

    def rglob(self, pattern: str) -> Iterator["ZipTree"]:
        prefix = f"{self._key}/" if self._key else ""
        for key in sorted(self._ix.files):
            if key.startswith(prefix) and fnmatch.fnmatchcase(key.rsplit("/", 1)[-1], pattern):
                yield ZipTree(self._ix, key, self._label)

    def open(self, mode: str = "r", encoding: str | None = "utf-8"):
        if mode not in ("r", "rt", "rb"):
            raise ValueError("ZipTree is read-only")
        data = self._ix.read(self._key)
        return io.BytesIO(data) if mode == "rb" else io.StringIO(data.decode(encoding or "utf-8"))


def zip_parts(p: Path) -> list[Path]:
    """The .zip parts at `p`: the file itself, or the .zip files directly inside a folder."""
    if p.is_file() and p.suffix.lower() == ".zip":
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".zip")
    return []
