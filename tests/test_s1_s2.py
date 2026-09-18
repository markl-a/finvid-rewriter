"""Stage 1 (preprocessing) + stage 2 (transcription) tests. No network, no API key.

Stage 2 uses a fake OpenAI client injected via pipeline.llm._client.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline import llm
from pipeline.config import Settings
from pipeline.context import BudgetExceeded, RunContext, extract_video_id, run_stage
from pipeline.models import AudioInfo
from pipeline.stages import s1_download as s1
from pipeline.stages import s2_transcribe as s2

VID = "KjAI9r8tnOs"
URL = f"https://www.youtube.com/watch?v={VID}"


# ----------------------------------------------------------------------------- helpers
def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, stt_provider="openai", stt_model="gpt-4o-mini-transcribe",
                remove_silence=True, speedup=1.0, trim_head_sec=0.0, trim_tail_sec=0.0,
                sample_rate=16000, max_budget_usd=1.0)
    base.update(kw)
    return Settings(_env_file=None, **base)


def _make_tone_silence_tone(settings: Settings, dst: Path) -> None:
    """5 s 440 Hz tone + 3 s silence + 5 s tone = 13 s, 16 kHz mono."""
    subprocess.run(
        [settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]", "-map", "[out]",
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True, text=True,
    )


class FakeTranscriptions:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kw):
        kw["file"] = getattr(kw.get("file"), "name", None)
        self.calls.append(kw)
        return type("Resp", (), {"text": self.text})()


class FakeClient:
    def __init__(self, text: str):
        self.audio = type("Audio", (), {})()
        self.audio.transcriptions = FakeTranscriptions(text)

    @property
    def calls(self):
        return self.audio.transcriptions.calls


@pytest.fixture
def settings() -> Settings:
    return _settings()


@pytest.fixture
def ctx_with_audio(tmp_path: Path, settings: Settings) -> RunContext:
    """A RunContext whose workdir already has a stage-1 output (13 s synthetic wav)."""
    ctx = RunContext.create(settings, URL, data_dir=tmp_path, log=lambda *_: None)
    _make_tone_silence_tone(settings, ctx.path(s1.AUDIO_FILE))
    dur = s1.probe_duration(settings, ctx.path(s1.AUDIO_FILE))
    info = AudioInfo(video_id=VID, title="t", channel="c", url=URL, original_duration_sec=dur,
                     processed_duration_sec=dur, audio_path=s1.AUDIO_FILE)
    ctx.path(s1.INFO_FILE).write_text(json.dumps(info.model_dump()), encoding="utf-8")
    return ctx


# ----------------------------------------------------------------------------- video id
@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://www.youtube.com/watch?v={VID}&t=42s",
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    VID,
])
def test_extract_video_id(url):
    assert extract_video_id(url) == VID


def test_extract_video_id_rejects_garbage():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/nothing")


# ----------------------------------------------------------------------------- stage 1
def test_s1_silence_removal_shortens_audio(tmp_path: Path, settings: Settings):
    src = tmp_path / "src.wav"
    _make_tone_silence_tone(settings, src)
    original = s1.probe_duration(settings, src)
    assert 12.5 <= original <= 13.5

    with_sr = s1.preprocess_audio(settings, src, tmp_path / "sr.wav")
    assert original - with_sr >= 2.0, f"silenceremove should cut ~3 s: {original} -> {with_sr}"

    no_sr = s1.preprocess_audio(_settings(remove_silence=False), src, tmp_path / "nosr.wav")
    assert abs(no_sr - original) < 0.2


def test_s1_speedup_and_trim(tmp_path: Path):
    st = _settings(remove_silence=False, speedup=1.25, trim_head_sec=1.0, trim_tail_sec=1.0)
    src = tmp_path / "src.wav"
    _make_tone_silence_tone(st, src)
    out = s1.preprocess_audio(st, src, tmp_path / "out.wav")
    # (13 - 2) / 1.25 = 8.8
    assert abs(out - 8.8) < 0.3
    assert "atempo=1.25" in s1.build_audio_filter(st)


def test_s1_stage_config_tracks_cost_knobs(tmp_path: Path):
    a = RunContext.create(_settings(), URL, data_dir=tmp_path, log=lambda *_: None)
    cfg = s1.stage_config(a)
    assert set(cfg) == {"video_id", "sample_rate", "remove_silence", "trim_head_sec",
                        "trim_tail_sec", "speedup"}


def test_s1_dry_run_estimate(monkeypatch, tmp_path: Path, settings: Settings):
    monkeypatch.setattr(s1, "_fetch_metadata",
                        lambda url: {"title": "T", "channel": "C", "duration": 1000.0})
    monkeypatch.setattr(s1, "_download_audio",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")))
    ctx = RunContext.create(settings, URL, dry_run=True, data_dir=tmp_path, log=lambda *_: None)
    r = run_stage(ctx, s1.STAGE, s1.stage_config(ctx), s1.execute)
    assert r.outputs == {} and r.costs == []
    assert r.meta["original_duration_sec"] == 1000.0
    assert r.meta["estimated_processed_duration_sec"] == pytest.approx(920.0)
    assert s1.STAGE not in ctx.manifest.data["stages"]  # dry-run records nothing


def test_s1_reuses_manually_placed_audio(ctx_with_audio: RunContext, monkeypatch):
    monkeypatch.setattr(s1, "_fetch_metadata",
                        lambda url: (_ for _ in ()).throw(AssertionError("no network expected")))
    r = run_stage(ctx_with_audio, s1.STAGE, s1.stage_config(ctx_with_audio), s1.execute)
    assert r.outputs["audio"] == s1.AUDIO_FILE and r.meta.get("reused_existing")


# ----------------------------------------------------------------------------- stage 2
def test_s2_text_to_segments_interpolates_time():
    segs = s2.text_to_segments("你好。今天很好！再見？", 10.0, 20.0)
    assert [s.text for s in segs] == ["你好。", "今天很好！", "再見？"]
    assert segs[0].start == 10.0 and segs[-1].end == pytest.approx(20.0)
    assert all(a.end == b.start for a, b in zip(segs, segs[1:]))


def test_s2_openai_fake_client(ctx_with_audio: RunContext, monkeypatch):
    fake = FakeClient("房价今年涨了百分之十。专家说明年会跌。")
    monkeypatch.setattr(llm, "_client", fake)
    ctx = ctx_with_audio
    r = run_stage(ctx, s2.STAGE, s2.stage_config(ctx), s2.execute)

    out = ctx.path(s2.TRANSCRIPT_FILE)
    assert out.exists()
    tr = json.loads(out.read_text(encoding="utf-8"))
    assert tr["language"] == "zh-TW" and tr["model"] == "gpt-4o-mini-transcribe"
    assert len(tr["segments"]) == 2
    assert tr["segments"][0]["text"] == "房價今年漲了百分之十。"  # s2twp applied
    assert "专家" not in json.dumps(tr, ensure_ascii=False)

    # one 13 s chunk, billed on actual seconds
    assert len(fake.calls) == 1
    assert fake.calls[0]["response_format"] == "json" and fake.calls[0]["language"] == "zh"
    assert len(r.costs) == 1
    c = r.costs[0]
    assert not c.estimated and c.unit == "minute"
    assert c.quantity == pytest.approx(13 / 60, abs=0.01)
    assert c.usd == pytest.approx(13 / 60 * 0.003, abs=1e-5)
    assert ctx.spent_usd == pytest.approx(c.usd)
    assert not ctx.path(s2.CHUNK_DIR).exists()  # temp chunks cleaned


def test_s2_chunks_long_audio(ctx_with_audio: RunContext, monkeypatch):
    fake = FakeClient("第一句。第二句。")
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(s2, "CHUNK_SECONDS", 5)  # 13 s -> 3 chunks
    ctx = ctx_with_audio
    r = s2.execute(ctx)
    assert len(fake.calls) == 3 and r.meta["chunks"] == 3
    tr = json.loads(ctx.path(s2.TRANSCRIPT_FILE).read_text(encoding="utf-8"))
    starts = [s["start"] for s in tr["segments"]]
    assert starts == sorted(starts) and starts[2] == pytest.approx(5.0, abs=0.1)  # offset by chunk start
    assert r.costs[0].quantity == pytest.approx(13 / 60, abs=0.01)


def test_s2_dry_run_estimates_without_calling(ctx_with_audio: RunContext, monkeypatch):
    fake = FakeClient("x")
    monkeypatch.setattr(llm, "_client", fake)
    ctx = ctx_with_audio
    ctx.dry_run = True
    r = run_stage(ctx, s2.STAGE, s2.stage_config(ctx), s2.execute)
    assert fake.calls == [] and r.outputs == {}
    assert len(r.costs) == 1 and r.costs[0].estimated
    assert r.estimated_usd == pytest.approx(13 / 60 * 0.003, abs=1e-5)


def test_s2_budget_guard_blocks_before_call(ctx_with_audio: RunContext, monkeypatch):
    fake = FakeClient("x")
    monkeypatch.setattr(llm, "_client", fake)
    ctx = ctx_with_audio
    ctx.settings = _settings(max_budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        run_stage(ctx, s2.STAGE, s2.stage_config(ctx), s2.execute)
    assert fake.calls == []


def test_s2_run_stage_caches(ctx_with_audio: RunContext, monkeypatch):
    fake = FakeClient("测试。")
    monkeypatch.setattr(llm, "_client", fake)
    ctx = ctx_with_audio
    first = run_stage(ctx, s2.STAGE, s2.stage_config(ctx), s2.execute)
    assert not first.cached and len(fake.calls) == 1

    # fresh context, same workdir + same config -> cache hit, no API call, $0 this run
    ctx2 = RunContext.create(ctx.settings, URL, data_dir=ctx.workdir.parent, log=lambda *_: None)
    second = run_stage(ctx2, s2.STAGE, s2.stage_config(ctx2), s2.execute)
    assert second.cached and len(fake.calls) == 1
    assert ctx2.spent_usd == 0.0 and ctx2.stages_cached == [s2.STAGE]
    assert second.outputs == first.outputs

    # changing a cost knob (model) invalidates the cache
    ctx3 = RunContext.create(_settings(stt_model="whisper-1"), URL, data_dir=ctx.workdir.parent,
                             log=lambda *_: None)
    assert ctx3.manifest.cached(s2.STAGE, s2.stage_config(ctx3)) is None
