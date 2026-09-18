"""Compose one vertical clip: PIL background + chart card + per-line subtitle PNGs + one ffmpeg call.

Layout (1080x1920, scaled proportionally for other sizes):
  top     clip title (bold) + accent rule
  middle  chart card (re-drawn with matplotlib) or, without chart data, the hook as a key-message card
  y~1500  subtitle band, one PNG per script line, shown for exactly that line's audio
  bottom  attribution  "資料來源：<source>"  +  "本影片為重新整理改寫之原創內容"
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..config import Settings
from ..models import ScriptClip
from . import tts
from .fonts import find_cjk_font, find_cjk_font_bold

# palette (matches chart.py)
NAVY_TOP = "#0F172A"
NAVY_BOTTOM = "#1E3A5F"
CARD = "#1E293B"
ACCENT = "#2563EB"
EMPHASIS = "#FBBF24"
WHITE = "#FFFFFF"
MUTED = "#94A3B8"
STROKE = "#0F172A"

# reference geometry at 1080x1920
REF_W, REF_H = 1080, 1920
TITLE_Y = 150
TITLE_SIZE = 72
TITLE_CHARS = 12
CHART_W = 960
MID_TOP, MID_BOTTOM = 520, 1400
SUB_Y = 1500
SUB_H = 220
SUB_SIZE = 56
SUB_CHARS = 18
BAND_Y = 1760
ATTR_SIZE = 36
NOTE_SIZE = 30
REWRITE_NOTE = "本影片為重新整理改寫之原創內容"

_CLOSING = set("，。、！？：；」』）》…,.!?:;)")


def wrap_cjk(text: str, max_chars: int) -> list[str]:
    """Fixed-width wrap for CJK; never starts a line with closing punctuation."""
    text = text.strip()
    if not text:
        return []
    lines: list[str] = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if len(cur) >= max_chars and ch not in _CLOSING:
            lines.append(cur)
            cur = ""
        cur += ch
    if cur:
        lines.append(cur)
    return lines


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], cx: int, y: int, font: ImageFont.FreeTypeFont,
                fill: str, line_h: int, stroke_width: int = 0, stroke_fill: str | None = None) -> int:
    """Draw centred lines starting at y; returns y after the last line."""
    for ln in lines:
        draw.text((cx, y), ln, font=font, fill=fill, anchor="ma",
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        y += line_h
    return y


def _gradient(w: int, h: int) -> Image.Image:
    top = Image.new("RGB", (w, h), NAVY_TOP)
    bottom = Image.new("RGB", (w, h), NAVY_BOTTOM)
    mask = Image.linear_gradient("L").resize((w, h))
    return Image.composite(bottom, top, mask)


def build_background(settings: Settings, clip: ScriptClip, chart_png: Path | None, source_name: str,
                     out_png: Path) -> Path:
    W, H = settings.video_width, settings.video_height
    k = W / REF_W
    reg, bold = find_cjk_font(), find_cjk_font_bold()
    img = _gradient(W, H)
    d = ImageDraw.Draw(img)
    cx = W // 2

    # --- title
    tsize = int(TITLE_SIZE * k)
    tfont = _font(bold, tsize)
    tlines = wrap_cjk(clip.title, TITLE_CHARS)[:3]
    y = _draw_lines(d, tlines, cx, int(TITLE_Y * k), tfont, WHITE, int(tsize * 1.35))
    rule_w = int(120 * k)
    d.rounded_rectangle([cx - rule_w // 2, y + int(10 * k), cx + rule_w // 2, y + int(18 * k)],
                        radius=int(4 * k), fill=ACCENT)

    # --- middle: chart card or key-message card
    mid_top, mid_bottom = int(MID_TOP * k), int(MID_BOTTOM * k)
    if chart_png is not None and Path(chart_png).exists():
        chart = Image.open(chart_png).convert("RGB")
        cw = int(CHART_W * k)
        ch = int(chart.height * cw / chart.width)
        max_h = mid_bottom - mid_top
        if ch > max_h:  # keep it inside the middle band
            ch = max_h
            cw = int(chart.width * ch / chart.height)
        chart = chart.resize((cw, ch), Image.LANCZOS)
        x0 = cx - cw // 2
        y0 = mid_top + (max_h - ch) // 2
        mask = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw - 1, ch - 1], radius=int(28 * k), fill=255)
        # soft shadow
        d.rounded_rectangle([x0 + 6, y0 + 10, x0 + cw + 6, y0 + ch + 10], radius=int(28 * k), fill="#0B1220")
        img.paste(chart, (x0, y0), mask)
    else:
        pad = int(60 * k)
        x0, x1 = pad, W - pad
        hsize = int(60 * k)
        hfont = _font(bold, hsize)
        hlines = wrap_cjk(clip.hook, 13)[:5]
        lh = int(hsize * 1.45)
        block_h = len(hlines) * lh + int(140 * k)
        y0 = mid_top + (mid_bottom - mid_top - block_h) // 2
        d.rounded_rectangle([x0, y0, x1, y0 + block_h], radius=int(28 * k), fill=CARD)
        d.rounded_rectangle([x0, y0 + int(40 * k), x0 + int(10 * k), y0 + block_h - int(40 * k)],
                            radius=int(5 * k), fill=ACCENT)
        lab = _font(reg, int(32 * k))
        d.text((cx, y0 + int(36 * k)), "重點", font=lab, fill=MUTED, anchor="ma")
        _draw_lines(d, hlines, cx, y0 + int(100 * k), hfont, WHITE, lh)

    # --- bottom band: attribution
    by = int(BAND_Y * k)
    d.line([(int(80 * k), by), (W - int(80 * k), by)], fill="#334155", width=max(1, int(2 * k)))
    afont = _font(reg, int(ATTR_SIZE * k))
    nfont = _font(reg, int(NOTE_SIZE * k))
    attribution = f"資料來源：{source_name}" if source_name else clip.attribution
    d.text((cx, by + int(28 * k)), attribution, font=afont, fill=WHITE, anchor="ma")
    d.text((cx, by + int(28 * k) + int(ATTR_SIZE * 1.5 * k)), REWRITE_NOTE, font=nfont, fill=MUTED, anchor="ma")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def build_subtitle(settings: Settings, text: str, out_png: Path, emphasis: bool = False) -> Path:
    W = settings.video_width
    k = W / REF_W
    h = int(SUB_H * k)
    size = int(SUB_SIZE * k)
    font = _font(find_cjk_font_bold(), size)
    lines = wrap_cjk(text, SUB_CHARS)[:3]
    lh = int(size * 1.3)
    img = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    total = len(lines) * lh
    y = max(0, (h - total) // 2)
    _draw_lines(d, lines, W // 2, y, font, EMPHASIS if emphasis else WHITE, lh,
                stroke_width=max(2, int(4 * k)), stroke_fill=STROKE)
    img.save(out_png)
    return out_png


def compose_clip(settings: Settings, clip: ScriptClip, line_audio: list[tuple[str, Path, float]],
                 chart_png: Path | None, out_mp4: Path, source_name: str) -> float:
    """Render clip -> out_mp4. `line_audio` = [(text, audio_path, duration)] in playback order
    (hook first, then clip.lines). Returns the measured duration in seconds."""
    if not line_audio:
        raise ValueError("compose_clip needs at least one line of audio")
    emphasis = [False] + [ln.emphasis for ln in clip.lines]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="finvid_") as td:
        tmp = Path(td)
        bg = build_background(settings, clip, chart_png, source_name, tmp / "bg.png")
        wav = tmp / "voice.wav"
        total = tts.concat_audio(settings, [p for _, p, _ in line_audio], wav)
        timings = tts.line_timings([dur for _, _, dur in line_audio])
        subs: list[Path] = []
        for i, (text, _, _) in enumerate(line_audio):
            subs.append(build_subtitle(settings, text, tmp / f"sub_{i:02d}.png",
                                       emphasis=emphasis[i] if i < len(emphasis) else False))

        cmd = [settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin",
               "-loop", "1", "-framerate", "30", "-i", str(bg),
               "-i", str(wav)]
        for s in subs:
            cmd += ["-i", str(s)]
        sub_y = int(SUB_Y * settings.video_width / REF_W)
        chain: list[str] = []
        prev = "[0:v]"
        for i, (start, end) in enumerate(timings):
            if i == len(timings) - 1:
                end = max(end, total + 1.0)  # keep the last subtitle up until the video ends
            out = f"[v{i + 1}]"
            chain.append(f"{prev}[{i + 2}:v]overlay=(W-w)/2:{sub_y}:enable='between(t,{start},{end})'{out}")
            prev = out
        chain.append(f"{prev}format=yuv420p[vout]")
        cmd += ["-filter_complex", ";".join(chain), "-map", "[vout]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-tune", "stillimage",
                "-r", "30", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-t", f"{total:.3f}", "-shortest", "-movflags", "+faststart", str(out_mp4)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    return tts.probe_duration(settings, out_mp4)
