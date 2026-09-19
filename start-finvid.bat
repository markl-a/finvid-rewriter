@echo off
REM finvid one-click start (Windows): double-click this file.
REM Finds Python 3.11+, creates .venv, installs, checks ffmpeg, opens the dashboard.
setlocal
cd /d "%~dp0"
set "PYEXE="
for %%v in (3.12 3.13 3.11) do (
  if not defined PYEXE (
    py -%%v -c "import sys" >nul 2>&1 && set "PYEXE=py -%%v"
  )
)
if not defined PYEXE (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  echo [start] Python 3.11+ not found. Install it first:  winget install Python.Python.3.12
  echo         then double-click start-finvid.bat again.
  pause
  exit /b 1
)
%PYEXE% start.py
if errorlevel 1 (
  echo.
  echo [start] something went wrong - see the messages above.
  pause
)
endlocal
