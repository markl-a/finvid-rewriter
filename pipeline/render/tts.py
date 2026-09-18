"""Text-to-speech, one file per script line so subtitles get exact per-line timing.

Providers:
  edge   - edge-tts (Microsoft Edge neural voices). Free, needs internet, no key. Default.
  openai - gpt-4o-mini-tts via the OpenAI API. ~$0.003 per 30 s clip; only used if selected.
"""
from __future__ import annotations

import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Awaitable, Callable, TypeVar

from ..config import Settings, check_openai_key
from ..models import CostEntry
from ..pricing import tts_cost

STAGE = "s4_render"
GAP_SEC = 0.25  # silence inserted after each line when concatenating

T = TypeVar("T")


def _run_async(coro: Awaitable[T]) -> T:
    """asyncio.run, but tolerate being called from inside a running loop (e.g. the web UI)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()  # type: ignore[arg-type]


def probe_duration(settings: Settings, path: Path) -> float:
    out = subprocess.run(
        [settings.ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def provider_model(settings: Settings) -> tuple[str, str]:
    if settings.tts_provider == "openai":
        return "openai", settings.openai_tts_model
    return "edge", "edge-tts"


def tts_cost_entry(settings: Settings, chars: int, estimated: bool = False, note: str = "") -> CostEntry:
    prov, model = provider_model(settings)
    entry = tts_cost(STAGE, prov, model, chars, estimated=estimated)
    if note:
        entry.note = note
    return entry


def synthesize(settings: Settings, text: str, out_mp3: Path) -> float:
    """Synthesize `text` to `out_mp3`; returns duration in seconds (ffprobe)."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if settings.tts_provider == "openai":
        _synth_openai(settings, text, out_mp3)
    elif settings.tts_provider == "edge":
        _synth_edge(settings, text, out_mp3)
    else:
        raise ValueError(f"unknown FINVID_TTS_PROVIDER={settings.tts_provider!r} (edge | openai)")
    if not out_mp3.exists() or out_mp3.stat().st_size == 0:
        raise RuntimeError(f"TTS produced no audio for: {text[:30]!r}")
    return probe_duration(settings, out_mp3)


def _synth_edge(settings: Settings, text: str, out_mp3: Path) -> None:
    import edge_tts

    async def go() -> None:
        await edge_tts.Communicate(text, settings.tts_voice).save(str(out_mp3))

    try:
        _run_async(go())
    except Exception as e:  # noqa: BLE001 - network / service errors
        raise RuntimeError(
            f"edge-tts failed ({type(e).__name__}: {e}). edge-tts needs internet access; "
            f"alternatively set FINVID_TTS_PROVIDER=openai."
        ) from e


def _synth_openai(settings: Settings, text: str, out_mp3: Path) -> None:
    check_openai_key(settings)  # FINVID_TTS_PROVIDER=openai
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.audio.speech.create(model=settings.openai_tts_model, voice="alloy", input=text)
    data = resp.content if hasattr(resp, "content") else resp.read()
    out_mp3.write_bytes(data)


Synth = Callable[[Settings, str, Path], float]


def synthesize_lines(settings: Settings, texts: list[str], out_dir: Path, stem: str,
                     synth: Synth | None = None) -> list[tuple[str, Path, float]]:
    """One audio file per line -> [(text, path, duration_sec)]. `synth` is injectable for tests."""
    fn: Synth = synth or synthesize
    out: list[tuple[str, Path, float]] = []
    for i, text in enumerate(texts):
        p = out_dir / f"{stem}_line_{i:02d}.mp3"
        dur = fn(settings, text, p)
        out.append((text, p, dur))
    return out


def concat_audio(settings: Settings, parts: list[Path], out_wav: Path, gap_sec: float = GAP_SEC) -> float:
    """Concatenate the per-line files into one 24 kHz mono WAV with `gap_sec` of silence after each
    line (apad). One ffmpeg call, no intermediate files. Returns the measured duration."""
    if not parts:
        raise ValueError("nothing to concatenate")
    cmd = [settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin"]
    for p in parts:
        cmd += ["-i", str(p)]
    chain = "".join(
        f"[{i}:a]aresample=24000,aformat=sample_fmts=s16:channel_layouts=mono,"
        f"apad=pad_dur={gap_sec}[a{i}];"
        for i in range(len(parts))
    )
    chain += "".join(f"[a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[out]"
    cmd += ["-filter_complex", chain, "-map", "[out]", "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return probe_duration(settings, out_wav)


def line_timings(durations: list[float], gap_sec: float = GAP_SEC) -> list[tuple[float, float]]:
    """(start, end) per line in the concatenated audio; the subtitle stays up through the gap."""
    t = 0.0
    out: list[tuple[float, float]] = []
    for d in durations:
        out.append((round(t, 3), round(t + d + gap_sec, 3)))
        t += d + gap_sec
    return out
