"""Stage 1: YouTube -> preprocessed 16 kHz mono WAV. Costs $0 but decides how many STT minutes we buy.

Preprocessing (all ffmpeg, all free):
  - audio-only download (bestaudio, never the video stream)
  - 16 kHz mono 16-bit PCM      -> what STT models resample to anyway; small file, fast upload
  - silenceremove                -> drops pauses > 0.7 s; talk shows typically lose 5-15 % of seconds
  - optional head/tail trim      -> skip intro jingle / outro CTA
  - optional atempo speed-up     -> off by default (accuracy trade-off documented in ANALYSIS.md)
Every second removed here is a second we do not pay the STT vendor for.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..config import Settings
from ..context import RunContext
from ..manifest import StageResult
from ..models import AudioInfo

STAGE = "s1_download"

RAW_STEM = "01_raw"
AUDIO_FILE = "01_audio.wav"
INFO_FILE = "01_info.json"

# silenceremove parameters: any internal silence longer than this, quieter than this, is cut.
SILENCE_MIN_SEC = 0.7
SILENCE_THRESHOLD_DB = -35
# dry-run assumption: silence removal saves roughly this share of a talk-show's seconds
SILENCE_SAVING_RATIO = 0.03  # dry-run guess only; measured 0.2% on a show with a music bed, 5-15% on plain talk

DOWNLOAD_HELP = (
    "YouTube download failed. Options: (1) retry later; (2) export browser cookies: "
    "`yt-dlp --cookies-from-browser chrome -f bestaudio -o data/<video_id>/01_raw.%(ext)s <url>`; "
    "(3) place a `01_audio.wav` (16 kHz mono) plus `01_info.json` manually in the workdir and re-run."
)


# ----------------------------------------------------------------------------- config
def stage_config(ctx: RunContext) -> dict[str, Any]:
    s = ctx.settings
    return {
        "video_id": ctx.video_id,
        "sample_rate": s.sample_rate,
        "remove_silence": s.remove_silence,
        "trim_head_sec": s.trim_head_sec,
        "trim_tail_sec": s.trim_tail_sec,
        "speedup": s.speedup,
    }


# ----------------------------------------------------------------------------- helpers
def probe_duration(settings: Settings, path: Path) -> float:
    """Duration in seconds via ffprobe."""
    out = subprocess.run(
        [settings.ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def build_audio_filter(settings: Settings) -> str:
    """The ffmpeg -af chain; exposed so tests/docs can show exactly what is applied."""
    filters: list[str] = []
    if settings.remove_silence:
        filters.append(
            f"silenceremove=stop_periods=-1:stop_duration={SILENCE_MIN_SEC}"
            f":stop_threshold={SILENCE_THRESHOLD_DB}dB"
        )
    if settings.speedup and abs(settings.speedup - 1.0) > 1e-6:
        # atempo accepts 0.5..100 per instance; the values we use (1.0-1.5) fit in one
        filters.append(f"atempo={settings.speedup}")
    return ",".join(filters)


def preprocess_audio(settings: Settings, src: Path, dst: Path,
                     original_duration: float | None = None) -> float:
    """Convert any audio file to a 16 kHz (settings.sample_rate) mono 16-bit PCM WAV,
    trimming head/tail and removing silence / speeding up per settings.
    Returns the processed duration in seconds (measured with ffprobe)."""
    if original_duration is None:
        original_duration = probe_duration(settings, src)
    cmd = [settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin"]
    if settings.trim_head_sec > 0:
        cmd += ["-ss", f"{settings.trim_head_sec:.3f}"]
    if settings.trim_tail_sec > 0 and original_duration > settings.trim_tail_sec:
        cmd += ["-to", f"{original_duration - settings.trim_tail_sec:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", str(settings.sample_rate)]
    af = build_audio_filter(settings)
    if af:
        cmd += ["-af", af]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return probe_duration(settings, dst)


def _fetch_metadata(url: str) -> dict[str, Any]:
    from yt_dlp import YoutubeDL

    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False) or {}
    return {
        "title": info.get("title") or "",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": float(info.get("duration") or 0.0),
    }


def _download_audio(url: str, workdir: Path) -> Path:
    from yt_dlp import YoutubeDL

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(workdir / f"{RAW_STEM}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    hits = sorted(workdir.glob(f"{RAW_STEM}.*"))
    if not hits:
        raise RuntimeError("yt-dlp reported success but no 01_raw.* file was written")
    return hits[0]


def _estimate_processed(settings: Settings, original: float) -> float:
    est = max(0.0, original - settings.trim_head_sec - settings.trim_tail_sec)
    if settings.remove_silence:
        est *= 1.0 - SILENCE_SAVING_RATIO
    if settings.speedup and settings.speedup > 0:
        est /= settings.speedup
    return round(est, 1)


# ----------------------------------------------------------------------------- execute
def execute(ctx: RunContext) -> StageResult:
    s = ctx.settings
    audio_path = ctx.path(AUDIO_FILE)
    info_path = ctx.path(INFO_FILE)

    # Manual fallback: user dropped the wav + info in place (e.g. yt-dlp blocked). Reuse as-is - but only
    # if it was made with the SAME preprocessing we are about to record as this stage's config; otherwise
    # a changed FINVID_SPEEDUP / trim / silence setting would be recorded while the old audio stays.
    if not ctx.dry_run and not ctx.force and audio_path.exists() and info_path.exists():
        info = AudioInfo.model_validate_json(info_path.read_text(encoding="utf-8"))
        # a hand-placed wav has no preprocessing record: trust it; one we made ourselves must match
        same = not info.preprocessing or all(
            info.preprocessing.get(k) == v for k, v in stage_config(ctx).items() if k != "video_id")
        if not same:
            ctx.log(f"[{STAGE}] existing {AUDIO_FILE} was made with different preprocessing "
                    f"({ {k: info.preprocessing.get(k) for k in ('remove_silence', 'trim_head_sec', 'trim_tail_sec', 'speedup')} }) "
                    f"-> redoing it")
    if not ctx.dry_run and not ctx.force and audio_path.exists() and info_path.exists() and same:
        ctx.log(f"[{STAGE}] reusing existing {AUDIO_FILE} ({info.processed_duration_sec:.0f}s) - no download")
        return StageResult(
            outputs={"audio": AUDIO_FILE, "info": INFO_FILE}, costs=[],
            meta={"title": info.title, "channel": info.channel,
                  "original_duration_sec": info.original_duration_sec,
                  "processed_duration_sec": info.processed_duration_sec,
                  "seconds_saved": round(info.original_duration_sec - info.processed_duration_sec, 1),
                  "reused_existing": True},
        )

    try:
        md = _fetch_metadata(ctx.video_url)
    except Exception as e:  # noqa: BLE001 - yt-dlp raises many types
        raise RuntimeError(f"{DOWNLOAD_HELP}\n(metadata error: {e})") from e
    original = md["duration"]
    ctx.log(f"[{STAGE}] {md['title']!r} by {md['channel']!r}, {original:.0f}s ({original / 60:.1f} min)")

    if ctx.dry_run:
        est = _estimate_processed(s, original)
        ctx.log(f"[{STAGE}] dry-run: no download. Expected after preprocessing ~{est:.0f}s "
                f"({(1 - est / original) * 100 if original else 0:.0f}% fewer STT seconds)")
        return StageResult(
            outputs={}, costs=[],
            meta={"title": md["title"], "channel": md["channel"],
                  "original_duration_sec": original, "estimated_processed_duration_sec": est},
        )

    try:
        raw = _download_audio(ctx.video_url, ctx.workdir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{DOWNLOAD_HELP}\n(download error: {e})") from e
    raw_bytes = raw.stat().st_size
    raw_duration = probe_duration(s, raw) or original
    if original <= 0:
        original = raw_duration
    ctx.log(f"[{STAGE}] downloaded audio-only {raw.name}: {raw_bytes / 1e6:.1f} MB, {raw_duration:.0f}s")

    processed = preprocess_audio(s, raw, audio_path, original_duration=raw_duration)
    wav_bytes = audio_path.stat().st_size
    raw.unlink(missing_ok=True)  # keep disk small; the wav is the only thing later stages need

    saved = round(original - processed, 1)
    pct = (saved / original * 100) if original else 0.0
    ctx.log(f"[{STAGE}] preprocessing: {original:.0f}s -> {processed:.0f}s "
            f"(saved {saved:.0f}s = {pct:.1f}% of STT minutes), wav {wav_bytes / 1e6:.1f} MB")

    info = AudioInfo(
        video_id=ctx.video_id, title=md["title"], channel=md["channel"], url=ctx.video_url,
        original_duration_sec=round(original, 3), processed_duration_sec=round(processed, 3),
        audio_path=AUDIO_FILE,
        preprocessing={
            "sample_rate": s.sample_rate, "mono": True, "remove_silence": s.remove_silence,
            "silence_min_sec": SILENCE_MIN_SEC, "silence_threshold_db": SILENCE_THRESHOLD_DB,
            "trim_head_sec": s.trim_head_sec, "trim_tail_sec": s.trim_tail_sec,
            "speedup": s.speedup, "raw_bytes": raw_bytes, "wav_bytes": wav_bytes,
            "ffmpeg_filter": build_audio_filter(s),
        },
    )
    info_path.write_text(json.dumps(info.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    return StageResult(
        outputs={"audio": AUDIO_FILE, "info": INFO_FILE}, costs=[],
        meta={"title": md["title"], "channel": md["channel"],
              "original_duration_sec": round(original, 1), "processed_duration_sec": round(processed, 1),
              "seconds_saved": saved, "percent_saved": round(pct, 1),
              "raw_bytes": raw_bytes, "wav_bytes": wav_bytes},
    )
