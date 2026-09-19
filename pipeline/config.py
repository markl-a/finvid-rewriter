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
    font: str | None = None  # path to a TTF/TTC with Traditional Chinese glyphs (auto-detected if unset)

    # step 4: AI-generated opening shot per clip. none = static card only (default, $0, seconds)
    ai_video: str = "none"  # none | comfy
    ai_shot_seconds: float = 5.0
    ai_shots_per_clip: int = 2  # 1 = one shot ping-pong looped under the whole clip; 2+ = more variety, linear GPU cost
    ai_shot_width: int = 576   # 9:16, multiples of 32 for LTX
    ai_shot_height: int = 1024
    ai_shot_fps: int = 24
    comfy_url: str = "http://127.0.0.1:8188"
    comfy_workflow: str | None = None  # API-format workflow JSON; default: bundled ltxv_t2v.json
    comfy_checkpoint: str = "ltxv-2b-0.9.8-distilled.safetensors"
    comfy_text_encoder: str = "t5xxl_fp8_e4m3fn_scaled.safetensors"
    comfy_steps: int = 8       # distilled model: 8 steps, cfg 1
    comfy_cfg: float = 1.0
    comfy_timeout_sec: float = 1800
    comfy_est_gpu_seconds: float = 90  # dry-run estimate per shot: measured 79-107 s on a Radeon 8060S iGPU

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


class ConfigError(RuntimeError):
    """A .env / settings problem that should stop the run before any download or API call."""


def check_openai_key(s: Settings) -> None:
    """Fail fast, with a message that points at .env, instead of downloading 15 minutes of audio
    and then dying inside the OpenAI client with an obscure UnicodeEncodeError."""
    key = (s.openai_api_key or "").strip()
    hint = "Copy .env.example to .env and put your key in OPENAI_API_KEY (or export it)."
    if not key:
        raise ConfigError(f"OPENAI_API_KEY is not set. {hint}")
    if not key.isascii() or " " in key:
        raise ConfigError(f"OPENAI_API_KEY looks like a placeholder ({key[:12]!r}...). {hint}")


def get_settings() -> Settings:
    return Settings()
