import json

import pytest

from conftest import OWNER_A, OWNER_B, USER_A, USER_B
from core.export_reader import (UnknownLayoutError, detect_layouts, fix_encoding, load_owner_messages,
                                suggest_owner)


def mangle(s):  # what Facebook writes
    return s.encode("utf-8").decode("latin-1")


@pytest.mark.parametrize("text", [
    "Señor, salamat po 🙏",
    "HAHAHA 😂😂 dai ko aram",
    "Paañuhon? ñ Ñ ¿",
    "❤️ ingat",
    "plain ascii text",
])
def test_fix_encoding_roundtrip(text):
    assert fix_encoding(mangle(text)) == text


@pytest.mark.parametrize("text", ["😂 already fine", "café already fine", "ñ"])
def test_fix_encoding_falls_back_safely_on_clean_text(text):
    assert fix_encoding(text) == text


def test_fix_encoding_non_string():
    assert fix_encoding(None) is None


def test_detect_both_layouts():
    assert detect_layouts(USER_A) == ["legacy"]
    assert detect_layouts(USER_B) == ["dyi_2024"]


def test_detect_nested_one_level(tmp_path):
    # export unzipped inside an extra folder
    import shutil
    shutil.copytree(USER_A, tmp_path / "facebook-ana")
    assert detect_layouts(tmp_path) == ["legacy:facebook-ana"]


def test_unknown_layout(tmp_path):
    with pytest.raises(UnknownLayoutError):
        load_owner_messages(tmp_path, "x")


def test_suggest_owner():
    assert suggest_owner(USER_A) == OWNER_A
    assert suggest_owner(USER_B) == OWNER_B


def test_columns_and_timezone(df_a):
    assert list(df_a.columns) == ["ts", "recipient_code", "thread_type", "text"]
    assert str(df_a["ts"].dt.tz) == "Asia/Manila"
    assert df_a["ts"].is_monotonic_increasing


def test_at_least_400_owner_messages(load_a, load_b):
    assert load_a.n_owner_text >= 400
    assert load_b.n_owner_text >= 400


def test_other_people_and_non_text_dropped(load_a, df_a):
    assert load_a.n_other_dropped > 0
    assert load_a.n_owner_skipped > 0
    others = {"ok", "sige", "see you", "thanks!", "nice", "haha", "where na", "omw"}
    assert not set(df_a["text"]) & others  # synthetic other-party lines never kept
    assert not df_a["text"].str.contains("called you").any()
    assert (df_a["text"].str.strip() != "").all()


def test_encoding_fixed_in_loaded_text(df_b):
    assert df_b["text"].str.contains("Señor").any()
    assert not df_b["text"].str.contains("Ã").any()


def test_recipient_codes_and_thread_types(df_a):
    assert df_a["recipient_code"].str.fullmatch(r"R\d{2,3}").all()
    assert set(df_a["thread_type"]) == {"one-to-one", "group"}
    assert df_a["recipient_code"].nunique() == 8


def test_skipped_counts_by_kind(load_a):
    by = load_a.skipped_by_kind
    assert set(by) <= {"media", "unsent", "call", "other"}
    assert sum(by.values()) == load_a.n_owner_skipped
    # the synthetic generator writes photos, stickers and unsent placeholders (calls are rare; see precedence test)
    assert by.get("media", 0) > 0 and by.get("unsent", 0) > 0
    s = load_a.summary()
    assert s["skipped media"] == by["media"] and s["skipped call"] == by.get("call", 0)


def test_skip_kind_precedence():
    from core.export_reader import _parse_classic_message
    assert _parse_classic_message({"sender_name": "x", "timestamp_ms": 1, "is_unsent": True,
                                   "photos": [{}]}).kind == "unsent"
    assert _parse_classic_message({"sender_name": "x", "timestamp_ms": 1, "call_duration": 5,
                                   "content": "x called you."}).kind == "call"
    assert _parse_classic_message({"sender_name": "x", "timestamp_ms": 1, "sticker": {}}).kind == "media"
    assert _parse_classic_message({"sender_name": "x", "timestamp_ms": 1,
                                   "content": "Reacted 😂 to your message"}).kind == "other"
    assert _parse_classic_message({"sender_name": "x", "timestamp_ms": 1, "content": "dai pa"}).kind == "text"


def test_summary_has_counts_only(load_a):
    s = load_a.summary()
    assert s["messages kept"] == len(load_a.df)
    assert "text" not in s


def test_e2ee_provisional_layout(tmp_path):
    doc = {"participants": ["Ana Synthetica", "Carla Inventa"], "threadName": "Carla Inventa",
           "messages": [
               {"senderName": "Ana Synthetica", "timestamp": 1780000000000, "text": mangle("hiiii 😂")},
               {"senderName": "Carla Inventa", "timestamp": 1780000001000, "text": "hello"},
               {"senderName": "Ana Synthetica", "timestamp": 1780000002000, "text": "", "media": [{"uri": "x"}]},
               {"senderName": "Ana Synthetica", "timestamp": 1780000003, "text": "tara na", "isUnsent": False},
           ]}
    (tmp_path / "carla_1.json").write_text(json.dumps(doc), encoding="utf-8")
    assert detect_layouts(tmp_path) == ["e2ee"]
    res = load_owner_messages(tmp_path, "Ana Synthetica")
    assert list(res.df["text"]) == ["hiiii 😂", "tara na"]
    assert res.n_other_dropped == 1 and res.n_owner_skipped == 1
