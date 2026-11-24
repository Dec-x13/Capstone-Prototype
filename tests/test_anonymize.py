import pytest

from core.anonymize import RecipientCoder, mask


def test_codes_stable_within_session():
    c = RecipientCoder(salt=b"s1")
    a1, b1, a2 = c.code("Carla"), c.code("Dino"), c.code("Carla")
    assert a1 == a2 and a1 != b1
    assert len(c) == 2


def test_codes_random_across_sessions():
    names = [f"n{i}" for i in range(10)]
    c1, c2 = RecipientCoder(salt=b"one"), RecipientCoder(salt=b"two")
    assert [c1.code(n) for n in names] != [c2.code(n) for n in names]


def test_code_format_and_unique():
    c = RecipientCoder()
    codes = [c.code(f"p{i}") for i in range(150)]
    assert len(set(codes)) == 150
    assert all(x.startswith("R") for x in codes)
    assert all(len(x) == 3 for x in codes[:99])


def test_repr_hides_names():
    c = RecipientCoder()
    c.code("Secret Name")
    assert "Secret" not in repr(c)


@pytest.mark.parametrize("raw,expected", [
    ("text me 09171234567 tabi", "text me [PHONE] tabi"),
    ("0917-123-4567", "[PHONE]"),
    ("+63 917 123 4567", "[PHONE]"),
    ("+639171234567", "[PHONE]"),
    ("email ana@example.com po", "email [EMAIL] po"),
    ("see https://example.com/x?y=1 ok", "see [URL] ok"),
    ("www.sample.ph", "[URL]"),
    ("acct 1234 5678 9012", "acct [NUMBER]"),
    ("ref 123456", "ref [NUMBER]"),
    ("pa-gcash 200 tabi", "pa-gcash 200 tabi"),
    ("12345 not masked", "12345 not masked"),
    ("", ""),
])
def test_mask(raw, expected):
    assert mask(raw) == expected
