@echo off
echo ═══════════════════════════════════════════
echo   ROCHE_OS_V1 - Cognitive Prosthetic
echo ═══════════════════════════════════════════
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

REM Install playwright browsers (first run only)
playwright install chromium --with-deps 2>nul

REM Run the app
echo.
echo Starting ROCHE_OS...
echo.
streamlit run app.py --theme.base dark

pause
