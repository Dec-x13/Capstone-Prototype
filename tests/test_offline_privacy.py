"""Static checks for the offline and privacy rules (see README, Privacy design)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [*ROOT.glob("core/*.py"), ROOT / "app.py", ROOT / "extract" / "extract_features.py"]
NETWORK = re.compile(r"^\s*(import|from)\s+(socket|requests|urllib|http\.client|httpx|aiohttp|openai|anthropic)\b",
                     re.MULTILINE)
WRITES_TEXT = re.compile(r"(to_csv|to_json|to_parquet|to_pickle|pickle\.dump|logging\.)")


def test_no_network_imports():
    for f in SOURCES:
        assert not NETWORK.search(f.read_text(encoding="utf-8")), f


def test_no_dataframe_dumps_or_logging_in_pipeline():
    for f in SOURCES:
        assert not WRITES_TEXT.search(f.read_text(encoding="utf-8")), f


def test_cache_data_never_used_on_text():
    # The app keeps text in session memory only; st.cache_data is not used at all.
    assert "cache_data" not in (ROOT / "app.py").read_text(encoding="utf-8")


def test_streamlit_offline_config():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "headless = true" in cfg and "gatherUsageStats = false" in cfg


def test_no_performance_claims_in_docs_and_ui():
    banned = re.compile(r"\b(accuracy|precision|recall|F1|detects \d+%|hacked|scammer|proves)\b", re.IGNORECASE)
    allowed_context = re.compile(r"(no|not|never|without)[^.\n]{0,60}$", re.IGNORECASE)
    for name in ("README.md", "DEMO.md", "app.py"):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            m = banned.search(line)
            if m:
                # allowed only when negated, e.g. "no accuracy figures"
                assert allowed_context.search(line[:m.start()]), f"{name}: {line.strip()}"
