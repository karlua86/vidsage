@echo off
rem Generic VidSage launcher - works from any folder. Double-click to start the app.
cd /d "%~dp0"

rem 1. Prefer a Python that ALREADY has streamlit: project venv, then .venv, then system Python.
set "PY="
if exist "venv\Scripts\python.exe" ( "venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1 && set "PY=venv\Scripts\python.exe" )
if not defined PY if exist ".venv\Scripts\python.exe" ( ".venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1 && set "PY=.venv\Scripts\python.exe" )
if not defined PY ( python -c "import streamlit" >nul 2>&1 && set "PY=python" )
if defined PY goto :run

rem 2. None has streamlit yet: install into the venv if one exists, otherwise system Python.
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not exist "venv\Scripts\python.exe" if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH"^), then run this file again.
    pause
    exit /b 1
)
echo Installing requirements - first run only, this can take a few minutes (PyTorch is ~2 GB)...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed. See SETUP.md.
    pause
    exit /b 1
)

:run
where ffmpeg >nul 2>&1
if errorlevel 1 echo WARNING: ffmpeg not found. Run: winget install ffmpeg  ^(then restart this window^)

echo Starting VidSage with %PY% - your browser should open; if not, use the Local URL shown below.
"%PY%" -m streamlit run app.py
pause
