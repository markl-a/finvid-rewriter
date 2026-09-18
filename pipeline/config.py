"""Settings loaded from environment / .env. All knobs that affect cost live here."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="FINVID_",
        extra="ignore",
    )

    # keys (no prefix)
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    # step 2
    stt_provider: str = "openai"  # openai | local
    stt_model: str = "gpt-4o-mini-transcribe"
    local_whisper_model: str = "small"

    # step 3
    llm_cheap_model: str = "gpt-5-mini"
    llm_strong_model: str = "gpt-5"
    hook_threshold: int = 3
    plagiarism_ngram: int = 6
    plagiarism_max_overlap: float = 0.15  # share of script n-grams found in transcript
    plagiarism_max_lcs: int = 12  # longest common substring, in characters

    # step 4
    tts_provider: str = "edge"  # edge | openai
    tts_voice: str = "zh-TW-HsiaoChenNeural"
    openai_tts_model: str = "gpt-4o-mini-tts"
    video_width: int = 1080
    video_height: int = 1920

    # guardrails
    max_budget_usd: float = 1.0
    max_clips: int = 3

    # step 1 preprocessing
    remove_silence: bool = True
    trim_head_sec: float = 0.0
    trim_tail_sec: float = 0.0
    speedup: float = 1.0
    sample_rate: int = 16000

    ffmpeg: str | None = None

    def ffmpeg_bin(self) -> str:
        return _find_bin("ffmpeg", self.ffmpeg)

    def ffprobe_bin(self) -> str:
        if self.ffmpeg:
            p = Path(self.ffmpeg)
            cand = p.with_name("ffprobe" + p.suffix)
            if cand.exists():
                return str(cand)
        return _find_bin("ffprobe", None)


def _find_bin(name: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which(name)
    if found:
        return found
    # winget install location on Windows (PATH may not be refreshed in the current shell)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if base.exists():
            hits = sorted(base.rglob(f"{name}.exe"))
            if hits:
                return str(hits[-1])
    raise FileNotFoundError(
        f"{name} not found. Install ffmpeg (winget install Gyan.FFmpeg / brew install ffmpeg) "
        f"or set FINVID_FFMPEG in .env"
    )


def get_settings() -> Settings:
    return Settings()
