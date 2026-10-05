@echo off
rem Generic VidSage launcher - works from any folder. Double-click to start the app.
cd /d "%~dp0"

rem Use the project virtual environment if one exists, otherwise system Python.
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH"^), then run this file again.
    pause
    exit /b 1
)

"%PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements - first run only, this can take a few minutes...
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 ( echo Install failed. See SETUP.md. & pause & exit /b 1 )
)

where ffmpeg >nul 2>&1
if errorlevel 1 echo WARNING: ffmpeg not found. Run: winget install ffmpeg  ^(then restart this window^)

echo Starting VidSage - your browser should open; if not, use the Local URL shown below...
"%PY%" -m streamlit run app.py
pause
