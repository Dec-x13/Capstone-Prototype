"""Load working values from config.toml at the repository root."""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.toml"
WORDLIST_DIR = REPO_ROOT / "wordlists"
SAMPLE_DIR = REPO_ROOT / "data" / "sample" / "SYNTHETIC"
TIMEZONE = "Asia/Manila"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def cfg(section: str, key: str):
    return load_config()[section][key]
