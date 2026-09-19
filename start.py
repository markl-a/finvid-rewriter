"""One-click bootstrap: create the venv, install the project, check ffmpeg, seed .env, open the
dashboard. Run by start-finvid.bat (Windows) / start-finvid.command (macOS/Linux); safe to run repeatedly -
every step is skipped when already done. Needs only a Python 3.11+ interpreter to start.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WIN = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
FINVID = VENV / ("Scripts/finvid.exe" if WIN else "bin/finvid")


def say(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def main() -> int:
    os.chdir(ROOT)
    if sys.version_info < (3, 11):
        say(f"Python 3.11+ is required, this is {sys.version.split()[0]}. "
            f"Install 3.12 from python.org (Windows: `winget install Python.Python.3.12`, macOS: `brew install python@3.12`).")
        return 1

    if not PY.exists():
        say(f"creating virtual environment at {VENV} ...")
        venv.EnvBuilder(with_pip=True, upgrade_deps=False).create(VENV)
    else:
        say("virtual environment exists")

    marker = VENV / ".finvid-installed"
    pyproject_mtime = (ROOT / "pyproject.toml").stat().st_mtime
    if not FINVID.exists() or not marker.exists() or float(marker.read_text() or 0) < pyproject_mtime:
        say("installing the project (first time takes ~2 minutes) ...")
        r = subprocess.run([str(PY), "-m", "pip", "install", "-q", "--disable-pip-version-check", "-e", ".[dev]"])
        if r.returncode != 0:
            say("pip install failed - see the messages above")
            return r.returncode
        marker.write_text(str(pyproject_mtime))
    else:
        say("project already installed")

    if shutil.which("ffmpeg") is None and not (WIN and _winget_ffmpeg_present()):
        say("ffmpeg not found on PATH. Install it and run this again:")
        say("  Windows: winget install Gyan.FFmpeg     macOS: brew install ffmpeg     Ubuntu: sudo apt install ffmpeg")
        say("(the dashboard still opens so you can fill in keys; step 1/4 need ffmpeg)")
    else:
        say("ffmpeg found")

    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists() and example.exists():
        shutil.copy(example, env)
        say("created .env from .env.example - fill your keys in the dashboard's 設定 panel")
    else:
        say(".env exists (keys are read from it; the dashboard can edit it)")

    say("starting the dashboard at http://127.0.0.1:8000 (Ctrl+C / close this window to stop)")
    try:
        return subprocess.run([str(FINVID), "serve", "--open"]).returncode
    except KeyboardInterrupt:
        return 0


def _winget_ffmpeg_present() -> bool:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    return any(base.glob("Gyan.FFmpeg*/**/ffmpeg.exe")) if base.exists() else False


if __name__ == "__main__":
    sys.exit(main())
