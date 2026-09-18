"""Stage 2: preprocessed WAV -> timestamped Traditional-Chinese transcript.

Cost controls, in order:
  1. pre-flight estimate from the *processed* duration (stage 1 already shaved the silence)
  2. ctx.charge() budget guard BEFORE the first byte hits the API
  3. chunking (<= 600 s per request) so a failure only re-sends one chunk, and we stay
     under the vendor's per-request size limit
  4. bill the *actual* seconds sent (ffprobe on each chunk), not the estimate
  5. provider "local" (faster-whisper) is the $0 fallback with the same output contract
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .. import pricing
from ..context import RunContext
from ..manifest import StageResult
from ..models import CostEntry, Transcript, TranscriptSegment
from . import s1_download as s1

STAGE = "s2_transcribe"

TRANSCRIPT_FILE = "02_transcript.json"
CHUNK_DIR = "02_chunks"
CHUNK_SECONDS = 600  # 10 min; well under OpenAI's 25 MB / 1500 s limits at 16 kHz mono PCM
LANGUAGE = "zh"

# models that only return plain text (no segment timestamps)
_TEXT_ONLY_MODELS = ("gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe")
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")


# ----------------------------------------------------------------------------- config
def stage_config(ctx: RunContext) -> dict[str, Any]:
    s = ctx.settings
    model = s.local_whisper_model if s.stt_provider == "local" else s.stt_model
    return {
        "video_id": ctx.video_id,
        "stt_provider": s.stt_provider,
        "stt_model": model,
        "language": LANGUAGE,
        "chunk_seconds": CHUNK_SECONDS,
        "opencc": "s2twp",
        # a changed stage-1 config (e.g. speedup) produces different audio -> re-transcribe
        "s1_config_hash": ctx.manifest.data["stages"].get(s1.STAGE, {}).get("config_hash"),
    }


# ----------------------------------------------------------------------------- helpers
def _processed_seconds(ctx: RunContext) -> float:
    info_path = ctx.path(s1.INFO_FILE)
    if info_path.exists():
        return float(json.loads(info_path.read_text(encoding="utf-8"))["processed_duration_sec"])
    r = ctx.stage_results.get(s1.STAGE)
    if r is not None:
        m = r.meta
        return float(m.get("processed_duration_sec") or m.get("estimated_processed_duration_sec") or 0.0)
    raise RuntimeError(f"{s1.INFO_FILE} missing; run stage 1 first")


def split_into_chunks(ctx: RunContext, src: Path,
                      chunk_seconds: int | None = None) -> list[tuple[Path, float, float]]:
    """ffmpeg segment muxer -> [(path, start_sec, duration_sec), ...] in order."""
    chunk_seconds = chunk_seconds or CHUNK_SECONDS
    out_dir = ctx.path(CHUNK_DIR)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    subprocess.run(
        [ctx.settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-i", str(src),
         "-f", "segment", "-segment_time", str(chunk_seconds), "-c", "copy",
         str(out_dir / "chunk_%03d.wav")],
        check=True, capture_output=True, text=True,
    )
    chunks: list[tuple[Path, float, float]] = []
    t = 0.0
    for p in sorted(out_dir.glob("chunk_*.wav")):
        d = s1.probe_duration(ctx.settings, p)
        chunks.append((p, t, d))
        t += d
    return chunks


def _to_tw(text: str) -> str:
    from opencc import OpenCC

    global _cc
    if _cc is None:
        _cc = OpenCC("s2twp")
    return _cc.convert(text)


_cc = None


def text_to_segments(text: str, start: float, end: float) -> list[TranscriptSegment]:
    """Split plain text on sentence punctuation; assign times proportionally to char count.
    Used for models that do not return timestamps (gpt-4o-*-transcribe)."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    if not parts:
        return []
    total = sum(len(p) for p in parts) or 1
    span = max(end - start, 0.0)
    segs: list[TranscriptSegment] = []
    t = start
    for p in parts:
        dur = span * len(p) / total
        segs.append(TranscriptSegment(start=round(t, 2), end=round(t + dur, 2), text=p))
        t += dur
    return segs


def _postprocess(segs: list[TranscriptSegment]) -> list[TranscriptSegment]:
    out: list[TranscriptSegment] = []
    for s in segs:
        txt = _to_tw(s.text).strip()
        if txt:
            out.append(TranscriptSegment(start=s.start, end=s.end, text=txt))
    return out


# ----------------------------------------------------------------------------- providers
def _transcribe_openai(ctx: RunContext, chunks: list[tuple[Path, float, float]],
                       model: str) -> list[TranscriptSegment]:
    from ..llm import _openai

    client = _openai(ctx.settings)
    text_only = any(model.startswith(m) for m in _TEXT_ONLY_MODELS)
    fmt = "json" if text_only else "verbose_json"
    segs: list[TranscriptSegment] = []
    for i, (path, start, dur) in enumerate(chunks):
        ctx.log(f"[{STAGE}] chunk {i + 1}/{len(chunks)}: {dur:.0f}s -> {model} ({fmt})")
        with path.open("rb") as fh:
            resp = client.audio.transcriptions.create(
                model=model, file=fh, language=LANGUAGE, response_format=fmt,
            )
        if text_only:
            segs.extend(text_to_segments(getattr(resp, "text", "") or "", start, start + dur))
        else:
            raw_segs = getattr(resp, "segments", None) or []
            if not raw_segs:  # some SDK versions return text only
                segs.extend(text_to_segments(getattr(resp, "text", "") or "", start, start + dur))
            for rs in raw_segs:
                g = (lambda k: rs.get(k)) if isinstance(rs, dict) else (lambda k: getattr(rs, k, None))
                segs.append(TranscriptSegment(
                    start=round(start + float(g("start") or 0.0), 2),
                    end=round(start + float(g("end") or 0.0), 2),
                    text=str(g("text") or ""),
                ))
    return segs


def _transcribe_local(ctx: RunContext, audio: Path, model_name: str) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            'FINVID_STT_PROVIDER=local needs faster-whisper: pip install -e ".[local-stt]"'
        ) from e
    ctx.log(f"[{STAGE}] local faster-whisper model={model_name} (first run downloads the model)")
    wm = WhisperModel(model_name, device="cpu", compute_type="int8")
    it, _info = wm.transcribe(str(audio), language=LANGUAGE, vad_filter=True, beam_size=1)
    return [TranscriptSegment(start=round(s.start, 2), end=round(s.end, 2), text=s.text) for s in it]


# ----------------------------------------------------------------------------- execute
def execute(ctx: RunContext) -> StageResult:
    s = ctx.settings
    provider = s.stt_provider
    model = s.local_whisper_model if provider == "local" else s.stt_model
    price_model = "faster-whisper" if provider == "local" else model
    seconds = _processed_seconds(ctx)

    # ---- pre-flight estimate + budget guard (before any network) ----
    est = pricing.stt_cost(STAGE, provider, price_model, seconds, estimated=True)
    ctx.log(f"[{STAGE}] pre-flight: {seconds / 60:.1f} min x ${est.unit_price_usd}/min "
            f"({provider}/{price_model}) = ~${est.usd:.4f}")
    ctx.charge(est.usd, "STT")
    if ctx.dry_run:
        return StageResult(outputs={}, costs=[est],
                           meta={"provider": provider, "model": price_model,
                                 "seconds_estimated": seconds,
                                 "chunks": int(seconds // CHUNK_SECONDS) + 1})

    audio = ctx.path(s1.AUDIO_FILE)
    if not audio.exists():
        raise RuntimeError(f"{audio} missing; run stage 1 first")

    costs: list[CostEntry] = []
    if provider == "openai":
        chunks = split_into_chunks(ctx, audio)
        billed = sum(d for _, _, d in chunks)
        try:
            segs = _transcribe_openai(ctx, chunks, model)
        finally:
            shutil.rmtree(ctx.path(CHUNK_DIR), ignore_errors=True)
        costs.append(pricing.stt_cost(STAGE, provider, model, billed, estimated=False))
        n_chunks = len(chunks)
    elif provider == "local":
        segs = _transcribe_local(ctx, audio, model)
        billed = seconds
        costs.append(pricing.stt_cost(STAGE, "local", "faster-whisper", billed, estimated=False))
        n_chunks = 1
    else:
        raise ValueError(f"unknown FINVID_STT_PROVIDER={provider!r} (openai | local)")

    segs = _postprocess(segs)
    if not segs:
        raise RuntimeError("STT returned no text")
    tr = Transcript(video_id=ctx.video_id, language="zh-TW", provider=provider, model=price_model,
                    duration_sec=round(seconds, 3), segments=segs)
    ctx.path(TRANSCRIPT_FILE).write_text(
        json.dumps(tr.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    chars = len(tr.text)
    ctx.log(f"[{STAGE}] {len(segs)} segments, {chars} chars, billed {billed / 60:.2f} min "
            f"-> ${sum(c.usd for c in costs):.4f}")
    return StageResult(
        outputs={"transcript": TRANSCRIPT_FILE}, costs=costs,
        meta={"provider": provider, "model": price_model, "chunks": n_chunks,
              "seconds_billed": round(billed, 1), "segments": len(segs), "chars": chars},
    )
