@echo off
REM ─────────────────────────────────────────────────────────────
REM  run_app.bat  –  Launch FreshVision AI inside the project venv
REM  Double-click this file OR run it from the terminal.
REM ─────────────────────────────────────────────────────────────

REM Activate the virtual environment
call "%~dp0..\venv\Scripts\activate.bat"

REM Run Streamlit using the venv's streamlit executable
"%~dp0..\venv\Scripts\streamlit.exe" run "%~dp0app.py"

pause
