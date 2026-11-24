import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.anonymize import RecipientCoder  # noqa: E402
from core.config import SAMPLE_DIR  # noqa: E402
from core.export_reader import load_owner_messages  # noqa: E402

USER_A = SAMPLE_DIR / "user_a"
USER_B = SAMPLE_DIR / "user_b"
OWNER_A = "Ana Synthetica"
OWNER_B = "Ben Synthetico"


@pytest.fixture(scope="session")
def coder():
    return RecipientCoder(salt=b"test-salt")


@pytest.fixture(scope="session")
def load_a(coder):
    return load_owner_messages(USER_A, OWNER_A, coder)


@pytest.fixture(scope="session")
def load_b(coder):
    return load_owner_messages(USER_B, OWNER_B, coder)


@pytest.fixture(scope="session")
def df_a(load_a):
    return load_a.df


@pytest.fixture(scope="session")
def df_b(load_b):
    return load_b.df
