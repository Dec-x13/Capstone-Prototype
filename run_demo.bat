@echo off
REM BehaviorGuard prototype - Windows launcher. Runs fully offline after the first setup.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv venv --python 3.11 .venv || goto :error
        uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :error
    ) else (
        py -3.11 -m venv .venv || goto :error
        .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
    )
)

echo.
echo Open http://localhost:8501 in your browser. Press Ctrl+C here to stop.
echo.
.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --browser.gatherUsageStats false --server.address localhost --server.port 8501
goto :eof

:error
echo Setup failed. Install Python 3.11 (or uv) and try again.
pause
