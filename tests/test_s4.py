"""Stage 4 (render) tests. No network: TTS is replaced by ffmpeg-generated silence.
Needs ffmpeg/ffprobe and a system CJK font (Windows/macOS ship one; Linux: fonts-noto-cjk).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.config import Settings
from pipeline.context import RunContext, run_stage
from pipeline.models import (ChartSeries, ChartSpec, DataPoint, RenderOutput, ScriptClip, ScriptLine,
                             ScriptsOutput, TopicSegment)
from pipeline.render import tts
from pipeline.render.chart import render_chart
from pipeline.render.compose import compose_clip, wrap_cjk
from pipeline.stages import s4_render as s4

VID = "KjAI9r8tnOs"
URL = f"https://www.youtube.com/watch?v={VID}"
SILENT_SEC = 1.2


def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, tts_provider="edge", max_budget_usd=1.0, max_clips=3)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _silent_audio(settings: Settings, text: str, out: Path, seconds: float = SILENT_SEC) -> float:
    """Stand-in for TTS: `seconds` of silence, 24 kHz mono."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(seconds), str(out)],
        check=True, capture_output=True, text=True,
    )
    return tts.probe_duration(settings, out)


def _streams(settings: Settings, path: Path) -> list[str]:
    out = subprocess.run(
        [settings.ffprobe_bin(), "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return out


def _chart() -> ChartSpec:
    return ChartSpec(
        type="bar", title="台北市房價所得比", y_label="倍",
        series=[ChartSeries(name="房價所得比", points=[
            DataPoint(label="2019", value=13.9, unit="倍"),
            DataPoint(label="2022", value=15.8, unit="倍"),
            DataPoint(label="2024", value=16.9, unit="倍"),
        ])],
    )


def _clip(segment_id: int = 1, chart: ChartSpec | None = None) -> ScriptClip:
    return ScriptClip(
        segment_id=segment_id, title="利率降了，房價卻更高？",
        hook="利息變輕，房價反而更貴，為什麼？",
        lines=[ScriptLine(text="根據 TVBS《健康2.0》報導指出，降息讓月付減少。"),
               ScriptLine(text="但買方預算放大，賣方順勢加價。", emphasis=True),
               ScriptLine(text="結果總價愈墊愈高，負擔並沒有真的變輕。")],
        chart=chart, attribution="根據 TVBS《健康2.0》節目報導指出", est_seconds=28,
    )


# ----------------------------------------------------------------------------- unit
def test_wrap_cjk_keeps_punctuation_off_line_start():
    lines = wrap_cjk("一二三四五六七八九十，十一十二", 10)
    assert lines[0] == "一二三四五六七八九十，"
    assert all(len(l) <= 11 for l in lines)


def test_line_timings_include_gap():
    t = tts.line_timings([1.0, 2.0], gap_sec=0.25)
    assert t == [(0.0, 1.25), (1.25, 3.5)]


def test_chart_renders_cjk_png(tmp_path):
    out = render_chart(_chart(), tmp_path / "chart.png")
    assert out.exists() and out.stat().st_size > 5_000
    from PIL import Image
    w, h = Image.open(out).size
    assert w == 960 and h == 720


def test_chart_line_two_series(tmp_path):
    spec = ChartSpec(type="line", title="兩條線", series=[
        ChartSeries(name="A", points=[DataPoint(label="1月", value=1), DataPoint(label="2月", value=3)]),
        ChartSeries(name="B", points=[DataPoint(label="1月", value=2), DataPoint(label="2月", value=1)]),
    ])
    out = render_chart(spec, tmp_path / "line.png")
    assert out.stat().st_size > 5_000


def test_compose_silent_clip(tmp_path):
    s = _settings()
    clip = _clip()
    texts = [clip.hook] + [l.text for l in clip.lines]
    line_audio = tts.synthesize_lines(s, texts, tmp_path, "c", synth=_silent_audio)
    assert len(line_audio) == 4
    # 3 lines requested by the spec of this test: drop the hook to keep timing = 3 x 1.2 + gaps
    line_audio = line_audio[1:]
    mp4 = tmp_path / "out.mp4"
    dur = compose_clip(s, clip, line_audio, None, mp4, "TVBS《健康2.0》")
    assert mp4.exists() and mp4.stat().st_size > 10_000
    assert 3.6 <= dur <= 4.5, dur
    streams = _streams(s, mp4)
    assert "video" in streams and "audio" in streams


def test_compose_with_chart(tmp_path):
    s = _settings()
    clip = _clip(chart=_chart())
    png = render_chart(clip.chart, tmp_path / "chart.png")
    line_audio = tts.synthesize_lines(s, [clip.hook, clip.lines[0].text], tmp_path, "c", synth=_silent_audio)
    dur = compose_clip(s, clip, line_audio, png, tmp_path / "out.mp4", "TVBS《健康2.0》")
    assert 2.4 <= dur <= 3.2, dur


# ----------------------------------------------------------------------------- stage
def _scripts_json(workdir: Path) -> None:
    seg = lambda i: TopicSegment(id=i, start=0, end=30, topic=f"t{i}", summary="s", hook_score=4, selected=True)
    out = ScriptsOutput(video_id=VID, source_name="TVBS《健康2.0》", segments=[seg(1), seg(2)],
                        clips=[_clip(1, chart=_chart()), _clip(2, chart=None)])
    (workdir / "03_scripts.json").write_text(json.dumps(out.model_dump(), ensure_ascii=False), encoding="utf-8")


def _ctx(tmp_path: Path, settings: Settings, **kw) -> RunContext:
    logs: list[str] = []
    ctx = RunContext.create(settings, URL, data_dir=tmp_path, log=logs.append, **kw)
    ctx._logs = logs  # type: ignore[attr-defined]
    return ctx


def test_execute_renders_two_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "synthesize", _silent_audio)
    s = _settings()
    ctx = _ctx(tmp_path, s)
    _scripts_json(ctx.workdir)

    res = run_stage(ctx, s4.STAGE, s4.stage_config(ctx), s4.execute)

    assert res.outputs["render"] == "04_render.json"
    out = RenderOutput.model_validate_json(ctx.path("04_render.json").read_text(encoding="utf-8"))
    assert [c.segment_id for c in out.clips] == [1, 2]
    for c in out.clips:
        assert ctx.path(c.video_path).exists()
        assert c.duration_sec > 4
    assert out.clips[0].chart_path and ctx.path(out.clips[0].chart_path).exists()
    assert out.clips[1].chart_path is None
    assert res.outputs["clip_1"] == "04_clips/clip_01.mp4"
    # per-line mp3s cleaned up, chart + mp4 kept
    assert not list(ctx.path("04_clips").glob("*_line_*.mp3"))

    refs = [c for c in res.costs if c.provider == "reference"]
    assert len(refs) == 2 and all(c.estimated for c in refs) and all(c.usd > 0 for c in refs)
    real = [c for c in res.costs if not c.estimated]
    assert len(real) == 2 and all(c.provider == "edge" and c.usd == 0 for c in real)
    assert all(c.quantity > 0 for c in real)  # chars recorded even though edge is free
    assert res.usd == 0.0
    assert res.meta["clips_rendered"] == 2
    assert res.meta["reference_ai_video_usd"] == pytest.approx(sum(c.usd for c in refs))

    # second run: cache hit, nothing re-rendered
    ctx2 = _ctx(tmp_path, s)
    calls: list[str] = []
    monkeypatch.setattr(tts, "synthesize", lambda *a, **k: calls.append("x") or 1.0)
    res2 = run_stage(ctx2, s4.STAGE, s4.stage_config(ctx2), s4.execute)
    assert res2.cached and not calls


def test_execute_respects_max_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "synthesize", _silent_audio)
    ctx = _ctx(tmp_path, _settings(), max_clips=1)
    _scripts_json(ctx.workdir)
    res = run_stage(ctx, s4.STAGE, s4.stage_config(ctx), s4.execute)
    assert res.meta["clips_rendered"] == 1 and "clip_2" not in res.outputs


def test_dry_run_calls_no_tts(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("TTS must not be called in dry-run")

    monkeypatch.setattr(tts, "synthesize", boom)
    ctx = _ctx(tmp_path, _settings(tts_provider="openai"), dry_run=True)
    ctx.workdir.mkdir(parents=True, exist_ok=True)  # simulate s3 having run earlier (dry-run creates nothing)
    _scripts_json(ctx.workdir)
    res = run_stage(ctx, s4.STAGE, s4.stage_config(ctx), s4.execute)
    assert res.outputs == {}
    assert all(c.estimated for c in res.costs)
    tts_est = next(c for c in res.costs if c.provider == "openai")
    assert tts_est.quantity == sum(len(c.full_text) for c in [_clip(1), _clip(2)])
    assert tts_est.usd > 0
    assert next(c for c in res.costs if c.provider == "reference").usd == pytest.approx(56 * 0.25)
    assert not ctx.path("04_render.json").exists()


def _fake_shot(settings: Settings, out: Path, seconds: float = 2.0) -> Path:
    """Stand-in for an AI-generated shot: a 24 fps 576x1024 test pattern, no audio."""
    subprocess.run([settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin",
                    "-f", "lavfi", "-i", f"testsrc2=size=576x1024:rate=24", "-t", str(seconds),
                    "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True, text=True)
    return out


def test_compose_with_ai_footage(tmp_path):
    """Generated shots run under the whole clip (ping-pong looped per scene window); the chart is
    overlaid as a card from the first body line; title/attribution layer on top throughout."""
    from PIL import Image
    s = _settings()
    clip = _clip(chart=_chart())
    png = render_chart(clip.chart, tmp_path / "chart.png")
    line_audio = tts.synthesize_lines(s, [clip.hook, clip.lines[0].text, clip.lines[1].text, clip.lines[2].text],
                                      tmp_path, "c", synth=_silent_audio)
    shots = [_fake_shot(s, tmp_path / "shot0.mp4", seconds=0.8),  # shorter than its window: must loop
             _fake_shot(s, tmp_path / "shot1.mp4", seconds=0.8)]
    mp4 = tmp_path / "out.mp4"
    dur = compose_clip(s, clip, line_audio, png, mp4, "TVBS《健康2.0》", shots=shots)
    assert 4.8 <= dur <= 5.8, dur
    streams = _streams(s, mp4)
    assert "video" in streams and "audio" in streams

    def px(t: float, xy: tuple[int, int]) -> tuple[int, int, int]:
        out = tmp_path / f"f{t}.png"
        subprocess.run([s.ffmpeg_bin(), "-y", "-v", "error", "-ss", str(t), "-i", str(mp4),
                        "-frames:v", "1", str(out)], check=True, capture_output=True, text=True)
        return Image.open(out).convert("RGB").getpixel(xy)

    W, H = s.video_width, s.video_height
    def colourful(c): return max(c) - min(c) > 40
    # hook: footage in the middle, no chart yet
    assert colourful(px(0.5, (W // 2, H // 2)))
    # body: chart card (white) in the middle band, footage still visible at the card's sides
    card_pts = [(W // 2 + dx, int(H * 0.36)) for dx in (-300, -150, 0, 150, 300)]  # top strip of the card
    assert any(min(px(3.0, pt)) > 200 for pt in card_pts), "chart card should be overlaid during body lines"
    assert not any(min(px(0.5, pt)) > 200 for pt in card_pts), "no chart card during the hook"
    assert colourful(px(3.0, (30, H // 2))), "footage must keep playing beside the chart card"
    assert colourful(px(dur - 0.3, (30, H // 2))), "footage runs to the very end"


def test_scene_windows_split():
    from pipeline.render.compose import _scene_windows
    t = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
    assert _scene_windows(t, 8.0, 1) == [(0.0, 9.0)]
    assert _scene_windows(t, 8.0, 2) == [(0.0, 2.0), (2.0, 9.0)]
    assert _scene_windows(t, 8.0, 3) == [(0.0, 2.0), (2.0, 4.0), (4.0, 9.0)]
    assert len(_scene_windows(t, 8.0, 10)) == 4  # never more windows than lines


def test_shot_prompts_use_visual_keywords():
    from pipeline.models import ScriptLine
    from pipeline.stages.s4_render import shot_prompts
    c = _clip()
    c.ai_shot = "Slow push-in over Taipei rooftops at dusk"
    c.lines[1] = ScriptLine(text=c.lines[1].text, visual="mortgage papers on a kitchen table")
    ps = shot_prompts(c, 3)
    assert len(ps) == 3 and ps[0].startswith("Slow push-in") and "mortgage papers" in ps[1] or "mortgage papers" in ps[2]
    assert all("no text" in p for p in ps)
    assert shot_prompts(_clip(), 1)[0].startswith("Cinematic vertical b-roll for a finance news short")
