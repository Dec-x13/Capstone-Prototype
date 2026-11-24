#!/usr/bin/env bash
# BehaviorGuard prototype - macOS/Linux launcher. Runs fully offline after the first setup.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating Python 3.11 virtual environment..."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi

echo "Open http://localhost:8501 in your browser. Press Ctrl+C here to stop."
exec .venv/bin/python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false \
  --server.address localhost --server.port 8501
