#!/bin/bash
# finvid one-click start (macOS: double-click in Finder; Linux: ./start-finvid.command)
# Finds Python 3.11+, creates .venv, installs, checks ffmpeg, opens the dashboard.
cd "$(dirname "$0")" || exit 1
for c in python3.12 python3.13 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
    PYEXE="$c"; break
  fi
done
if [ -z "$PYEXE" ]; then
  echo "[start] Python 3.11+ not found. macOS: brew install python@3.12   Ubuntu: sudo apt install python3.12 python3.12-venv"
  read -r -p "Press Enter to close..." _
  exit 1
fi
"$PYEXE" start.py
status=$?
if [ $status -ne 0 ]; then
  echo; echo "[start] something went wrong - see the messages above."
  read -r -p "Press Enter to close..." _
fi
exit $status
