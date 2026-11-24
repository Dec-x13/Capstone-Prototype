import json
import re
import sys
from pathlib import Path

import pytest

from conftest import OWNER_A, USER_A
from core.export_reader import iter_threads
from core.language_tagger import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extract"))
import extract_features as ef  # noqa: E402

ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?[+-]\d{2}:\d{2}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
CODE = re.compile(r"^R\d{2,3}$")


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    out = tmp_path_factory.mktemp("ext") / "features.json"
    assert ef.main([str(USER_A), "--owner", OWNER_A, "--out", str(out)]) == 0
    return out


def _all_original_tokens():
    """Every token any participant wrote, plus every participant name and thread title."""
    toks = set()
    for _, t in iter_threads(USER_A):
        for m in t.messages:
            if m.content:
                toks.update(tokenize(m.content))
        for name in t.participants + [t.title]:
            toks.update(tokenize(name))
    return {x for x in toks if any(ch.isalpha() for ch in x)}


def test_no_original_token_anywhere_in_output(output):
    raw = output.read_text(encoding="utf-8")
    leaks = [t for t in _all_original_tokens()
             if re.search(r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])", raw, re.IGNORECASE)]
    assert leaks == []


def test_leak_check_detects_an_injected_token(output, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(output.read_text(encoding="utf-8").replace('"direct"', '"maray"', 1), encoding="utf-8")
    raw = bad.read_text(encoding="utf-8")
    assert "maray" in _all_original_tokens()
    assert re.search(r"(?<![A-Za-z0-9_])maray(?![A-Za-z0-9_])", raw)


def test_values_are_only_allowed_shapes(output):
    doc = json.loads(output.read_text(encoding="utf-8"))
    allowed_text = {ef.SCHEMA, "legacy", "dyi_2024", "e2ee", "direct", "group", doc["hashing"], doc["created_utc"]}

    def walk(x, in_hash=False):
        if isinstance(x, dict):
            for k, v in x.items():
                if in_hash:
                    assert HEX16.match(k)
                walk(v, in_hash=k in ("hashed_word_counts", "hashed_char_ngram_counts"))
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            assert x in allowed_text or ISO.match(x) or CODE.match(x), x
    walk(doc)


def test_structure(output):
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["counts"]["messages"] == len(doc["messages"]) >= 400
    w = doc["windows"][0]
    assert set(w) == {"first_message", "k", "baseline_n", "features", "hashed_word_counts",
                      "hashed_char_ngram_counts"}
    assert "laugh_upper_ha" in w["features"] and "laugh_HAHAHA" not in w["features"]
    assert sum(w["hashed_word_counts"].values()) > 0


def test_hashes_salted_per_run():
    h1, h2 = ef.make_hasher(b"a"), ef.make_hasher(b"b")
    assert h1("dai") != h2("dai") and h1("dai") == h1("dai")


def test_preview_saves_nothing(tmp_path, capsys):
    out = tmp_path / "should_not_exist.json"
    assert ef.main([str(USER_A), "--owner", OWNER_A, "--out", str(out), "--preview"]) == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "WHAT WILL BE SHARED" in printed and "nothing was saved" in printed
    for t in ("Ana", "Synthetica", "hiiii"):
        assert t not in printed
