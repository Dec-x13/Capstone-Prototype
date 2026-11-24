import pytest

from core.language_tagger import (code_switch_count, code_switch_rate, destretch, dominant_mix, language_shares,
                                  tag_text, tag_token, tokenize)


@pytest.mark.parametrize("tok,tag", [
    ("dai", "bikol"), ("DAI", "bikol"), ("maray,", "bikol"),
    ("hindi", "tagalog"), ("talaga!", "tagalog"),
    ("meeting", "english"), ("Thanks", "english"),
    ("ako", "ambiguous"), ("man", "ambiguous"), ("iyo", "ambiguous"),
    ("haha", "other"), ("HAHAHA", "other"), ("hehehe", "other"), ("lol", "other"), ("LOL", "other"),
    ("😂", "other"), ("200", "other"), ("zxqv", "other"),
])
def test_tag_token(tok, tag):
    assert tag_token(tok) == tag


def test_destretch_rules():
    assert destretch("dakooool")[0] == "dakol"
    assert destretch("hiiii")[0] == "hi"
    assert destretch("good") == ("good", "good")  # 2 repeats untouched


@pytest.mark.parametrize("tok,tag", [("dakooool", "bikol"), ("hiiii", "english"), ("sigeeee", "tagalog"),
                                     ("goooood", "english")])  # 'good' found only via the 2-letter fallback
def test_stretched_lookup(tok, tag):
    assert tag_token(tok) == tag


@pytest.mark.parametrize("tok,tag", [("lng", "tagalog"), ("u", "english"), ("pls", "english"), ("nmn", "tagalog")])
def test_abbreviations_use_expansion(tok, tag):
    assert tag_token(tok) == tag


def test_tokenize_splits_hyphen_and_emoji():
    assert tokenize("pa-gcash tabi😂!!") == ["pa", "gcash", "tabi", "😂"]
    assert tokenize("") == []


def test_code_switching_math():
    # bik eng | eng tag | tag (amb skipped) eng
    tags = ["bikol", "english", "english", "tagalog", "ambiguous", "english", "other", "bikol"]
    s, p = code_switch_count(tags)
    # eligible pairs: (bik,eng) (eng,eng) (eng,tag) -> 3 pairs, 2 switches
    assert (s, p) == (2, 3)
    assert code_switch_rate(tags) == pytest.approx(2 / 3)
    assert code_switch_rate(["other", "ambiguous"]) == 0.0


def test_language_shares_exclude_other():
    sh = language_shares(["bikol", "bikol", "english", "ambiguous", "other", "other"])
    assert sh["bikol"] == pytest.approx(0.5)
    assert sh["english"] == pytest.approx(0.25)
    assert sh["ambiguous"] == pytest.approx(0.25)
    assert sum(sh.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("shares,mix", [
    ({"bikol": .6, "english": .4, "tagalog": 0}, "Bi-lish"),
    ({"bikol": .3, "english": 0, "tagalog": .7}, "Tag-col"),
    ({"bikol": 0, "english": .5, "tagalog": .5}, "Taglish"),
    ({"bikol": .9, "english": .1, "tagalog": 0}, "Bikol"),
    ({"bikol": 0, "english": 0, "tagalog": 1}, "Tagalog"),
    ({"bikol": .05, "english": .95, "tagalog": 0}, "English"),
    ({"bikol": 0, "english": 0, "tagalog": 0}, "Unknown"),
])
def test_dominant_mix(shares, mix):
    assert dominant_mix(shares, second_share=0.2) == mix


def test_synthetic_users_have_expected_mix(df_a, df_b):
    def mix_of(texts):
        tags = [t for txt in texts for _, t in tag_text(txt)]
        return dominant_mix(language_shares(tags))
    assert mix_of(df_a["text"]) == "Bi-lish"
    assert mix_of(df_b["text"]) == "Taglish"
