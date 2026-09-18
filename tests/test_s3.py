"""Stage 3 tests. No network: every test patches pipeline.stages.s3_script.chat_json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import plagiarism, pricing
from pipeline.config import Settings
from pipeline.context import RunContext, run_stage
from pipeline.models import DataPoint, TopicSegment
from pipeline.stages import s3_script

VIDEO_ID = "testvideo01"
ATTR = s3_script.attribution_for("TVBS《健康2.0》")  # "根據TVBS《健康2.0》報導指出"

# ------------------------------------------------------------------ fixture data

TRANSCRIPT_SEGMENTS = [
    (0, 20, "各位觀眾大家好，歡迎收看健康2.0，今天我們要來談台灣的房價到底還會不會漲。"),
    (20, 40, "根據內政部最新統計，台北市的房價所得比已經來到15.7倍，新北市也有12.3倍。"),
    (40, 60, "也就是說一個家庭要不吃不喝將近16年才買得起一間房子，這個數字真的很驚人。"),
    (60, 80, "再來看利率，央行今年已經連續升息，重貼現率從1.875%調到2%。"),
    (80, 100, "假設貸款一千萬、三十年期，每個月的房貸支出大概會增加六百多塊。"),
    (100, 120, "如果利率再升半碼，也就是0.125%，每月大概再多三百塊左右。"),
    (120, 135, "主持人：哈哈，我們先進一段廣告，回來繼續聊。"),
    (135, 150, "歡迎回來，剛剛講到利率，接下來講買房的時機。"),
    (150, 170, "我建議首購族要先算清楚自己的負擔能力，房貸不要超過家庭月收入的三分之一。"),
    (170, 190, "另外自備款至少準備兩成，也就是一千萬的房子要有兩百萬的頭期款。"),
    (190, 210, "不要因為怕漲就急著追高，市場永遠有機會。"),
    (210, 230, "回到房價所得比，台北市15.7倍這個數字其實比去年的15.2倍又更高了。"),
    (230, 260, "新北市從11.9倍上升到12.3倍，桃園則是8.5倍。"),
    (260, 280, "最後看租金，台北市的平均租金今年漲了3.2%，新北漲了2.8%。"),
    (280, 300, "租金漲幅其實比房價漲幅還快，這對租屋族的壓力會越來越大。"),
]

PASS_A_RESPONSE = {"segments": [
    {"id": 1, "start": 0, "end": 60, "topic": "房價所得比創新高",
     "summary": "台北市房價所得比15.7倍、新北12.3倍，家庭要不吃不喝16年",
     "key_points": ["台北15.7倍", "新北12.3倍", "不吃不喝16年"],
     "data_points": [{"label": "台北市", "value": 15.7, "unit": "倍"},
                     {"label": "新北市", "value": 12.3, "unit": "倍"},
                     {"label": "不吃不喝", "value": "16", "unit": "年"}],
     "has_chart_data": True, "hook_score": 5},
    {"id": 2, "start": 60, "end": 120, "topic": "央行升息影響房貸",
     "summary": "重貼現率1.875%升到2%，千萬房貸月增六百元，再升半碼再多三百",
     "key_points": ["重貼現率2%", "月增600元", "半碼再多300"],
     "data_points": [{"label": "升息前", "value": 1.875, "unit": "%"},
                     {"label": "升息後", "value": 2.0, "unit": "%"},
                     {"label": "月增", "value": 600, "unit": "元"},
                     {"label": "半碼", "value": 0.125, "unit": "%"}],
     "has_chart_data": True, "hook_score": 4},
    {"id": 3, "start": 120, "end": 150, "topic": "廣告與寒暄", "summary": "主持人進廣告後回來",
     "key_points": ["廣告"], "data_points": [], "has_chart_data": False, "hook_score": 1},
    {"id": 4, "start": 150, "end": 210, "topic": "首購族買房建議",
     "summary": "房貸不超過月收入三分之一，自備款至少兩成，不要追高",
     "key_points": ["房貸不超過收入1/3", "自備款兩成"],
     "data_points": [{"label": "自備款", "value": 20, "unit": "%"}],
     "has_chart_data": False, "hook_score": 4},
    {"id": 5, "start": 210, "end": 260, "topic": "房價所得比創新高",
     "summary": "台北市房價所得比15.7倍、新北12.3倍，比去年更高",
     "key_points": ["台北15.7倍", "去年15.2倍"],
     "data_points": [{"label": "台北市今年", "value": 15.7, "unit": "倍"},
                     {"label": "台北市去年", "value": 15.2, "unit": "倍"}],
     "has_chart_data": True, "hook_score": 4},
    {"id": 6, "start": 260, "end": 300, "topic": "租金漲幅超越房價",
     "summary": "台北租金漲3.2%、新北2.8%，漲幅比房價快",
     "key_points": ["台北租金+3.2%", "新北+2.8%"],
     "data_points": [{"label": "台北市", "value": 3.2, "unit": "%"},
                     {"label": "新北市", "value": 2.8, "unit": "%"}],
     "has_chart_data": True, "hook_score": 3},
    # garbage entry: must be dropped, not crash
    {"id": "x", "topic": "", "hook_score": "high"},
]}

PASS_B_GOOD = {
    1: {"segment_id": 1, "title": "台北買房要16年", "hook": "不吃不喝16年才買得起房？",
        "lines": [
            {"text": "台北買房到底有多難", "emphasis": True},
            {"text": ATTR, "emphasis": False},
            {"text": "台北房價所得比高達15.7倍", "emphasis": True},
            {"text": "新北也來到12.3倍", "emphasis": False},
            {"text": "等於整個家庭收入全部存下來", "emphasis": False},
            {"text": "還要存超過15年才能買房", "emphasis": False},
            {"text": "這就是年輕人買不起房的真相", "emphasis": False},
        ],
        "chart": {"type": "bar", "title": "房價所得比", "y_label": "倍",
                  "series": [{"name": "2024", "points": [
                      {"label": "台北市", "value": 15.7, "unit": "倍"},
                      {"label": "新北市", "value": 12.3, "unit": "倍"}]}]},
        "attribution": ATTR, "est_seconds": 36},
    2: {"segment_id": 2, "title": "升息後房貸多繳多少", "hook": "升息了，你的房貸多繳多少？",
        "lines": [
            {"text": "央行把重貼現率由1.875%拉高到2%", "emphasis": True},
            {"text": "以千萬房貸、30年期來算", "emphasis": False},
            {"text": "每個月要多付大約600元", "emphasis": True},
            {"text": "若再升半碼0.125%", "emphasis": False},
            {"text": "每月又會再增加約300元", "emphasis": False},
            {"text": "算清楚再決定要不要進場", "emphasis": False},
        ],
        # attribution deliberately missing here: the stage must inject it
        "chart": {"type": "line", "title": "重貼現率", "y_label": "%",
                  "series": [{"name": "利率", "points": [
                      {"label": "升息前", "value": 1.875, "unit": "%"},
                      {"label": "升息後", "value": 2.0, "unit": "%"}]}]},
        "attribution": "", "est_seconds": 34},
    6: {"segment_id": 6, "title": "租金漲得比房價快", "hook": "房租比房價漲得更兇？",
        "lines": [
            {"text": ATTR, "emphasis": False},
            {"text": "台北今年租金平均上漲3.2%", "emphasis": True},
            {"text": "新北也漲了2.8%", "emphasis": False},
            {"text": "漲幅甚至超過房價本身", "emphasis": True},
            {"text": "租屋族的負擔只會越來越重", "emphasis": False},
            {"text": "現在該不該咬牙買房？", "emphasis": False},
        ],
        "chart": {"type": "bar", "title": "租金年漲幅", "y_label": "%",
                  "series": [{"name": "2024", "points": [
                      {"label": "台北市", "value": 3.2, "unit": "%"},
                      {"label": "新北市", "value": 2.8, "unit": "%"}]}]},
        "attribution": ATTR, "est_seconds": 33},
}

# first Pass B answer for segment 2: a verbatim copy of the transcript -> must trigger rewrite
PASS_B_COPY_2 = {
    "segment_id": 2, "title": "央行升息", "hook": "央行今年已經連續升息",
    "lines": [{"text": t} for _, _, t in TRANSCRIPT_SEGMENTS[3:6]],
    "chart": None, "attribution": ATTR, "est_seconds": 30,
}


def make_settings(**over) -> Settings:
    kw = {"max_budget_usd": 1.0, "max_clips": 3, **over}
    return Settings(_env_file=None, **kw)


def make_ctx(tmp_path: Path, *, dry_run: bool = False, with_transcript: bool = True,
             **settings_over) -> tuple[RunContext, list[str]]:
    logs: list[str] = []
    ctx = RunContext.create(make_settings(**settings_over), VIDEO_ID, dry_run=dry_run,
                            log=logs.append, data_dir=tmp_path)
    if with_transcript:
        ctx.workdir.mkdir(parents=True, exist_ok=True)  # a dry-run ctx no longer creates it
        info = {"video_id": VIDEO_ID, "title": "房價還會漲嗎？", "channel": "健康2.0",
                "url": f"https://www.youtube.com/watch?v={VIDEO_ID}", "original_duration_sec": 300,
                "processed_duration_sec": 300, "audio_path": "01_audio.wav", "preprocessing": {}}
        transcript = {"video_id": VIDEO_ID, "language": "zh-TW", "provider": "openai",
                      "model": "gpt-4o-mini-transcribe", "duration_sec": 300,
                      "segments": [{"start": a, "end": b, "text": t} for a, b, t in TRANSCRIPT_SEGMENTS]}
        ctx.path("01_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        ctx.path("02_transcript.json").write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    return ctx, logs


class FakeChat:
    """Stand-in for llm.chat_json: canned replies + realistic CostEntry list."""

    def __init__(self):
        self.calls: list[dict] = []
        self.pass_b_seen: dict[int, int] = {}

    def __call__(self, settings, model, system, user, *, stage, note="", **kw):
        self.calls.append({"model": model, "system": system, "user": user, "stage": stage, "note": note})
        if note.startswith("pass A"):
            payload = PASS_A_RESPONSE
        else:
            sid = s3_script.segment_id_from_prompt(user)
            assert sid is not None, "pass B prompt must carry segment_id"
            n = self.pass_b_seen.get(sid, 0)
            self.pass_b_seen[sid] = n + 1
            if sid == 2 and n == 0:
                payload = PASS_B_COPY_2
            else:
                payload = PASS_B_GOOD[sid]
        costs = pricing.llm_cost(stage, "openai", model, 2000, 800, note=note)
        return json.loads(json.dumps(payload)), costs


# ------------------------------------------------------------------ plagiarism

def test_plagiarism_verbatim_copy_fails():
    src = "".join(t for _, _, t in TRANSCRIPT_SEGMENTS)
    script = TRANSCRIPT_SEGMENTS[3][2] + TRANSCRIPT_SEGMENTS[4][2]
    ok, overlap, lcs = plagiarism.check(script, src, n=6, max_overlap=0.15, max_lcs=12)
    assert not ok
    assert overlap == pytest.approx(1.0)
    assert lcs >= len(plagiarism.normalise(script))


def test_plagiarism_paraphrase_with_same_numbers_passes():
    src = "".join(t for _, _, t in TRANSCRIPT_SEGMENTS[3:6])
    script = "升息了，你的房貸多繳多少？央行把重貼現率由1.875%拉高到2%以千萬房貸、30年期來算每個月要多付大約600元若再升半碼0.125%每月又會再增加約300元"
    ok, overlap, lcs = plagiarism.check(script, src, n=6, max_overlap=0.15, max_lcs=12)
    assert ok, (overlap, lcs)
    assert overlap < 0.15
    assert lcs <= 12


def test_longest_common_substring_known_pair():
    assert plagiarism.longest_common_substring("abc房價所得比xyz", "qq房價所得比zz") == 5
    # punctuation and whitespace are ignored
    assert plagiarism.longest_common_substring("房價，所得 比！", "房價所得比") == 5
    assert plagiarism.longest_common_substring("", "abc") == 0
    assert plagiarism.longest_common_substring_text("xx台北市15.7倍yy", "台北市15.7倍") == "台北市157倍"


def test_ngram_overlap_edge_cases():
    assert plagiarism.ngram_overlap("短", "任何來源文字", n=6) == 0.0
    assert plagiarism.ngram_overlap("台北市的房價所得比", "台北市的房價所得比很高", n=6) == pytest.approx(1.0)
    assert plagiarism.ngram_overlap("完全不同的一段文字內容", "台北市的房價所得比很高", n=6) == 0.0


# ------------------------------------------------------------------ gate

def _seg(i, score, chart, topic, summary, start=None):
    dps = [DataPoint(label="a", value=i * 10), DataPoint(label="b", value=i * 10 + 1)] if chart else []
    return TopicSegment(id=i, start=start if start is not None else i * 30.0, end=(i + 1) * 30.0,
                        topic=topic, summary=summary, data_points=dps, has_chart_data=chart,
                        hook_score=score)


def test_gate_threshold_dedup_and_chart_preference():
    segs = [
        _seg(1, 5, True, "房價所得比", "台北15.7倍新北12.3倍家庭要16年"),
        _seg(2, 2, True, "節目開場", "主持人問候觀眾介紹來賓"),
        _seg(3, 4, False, "買房建議", "房貸不超過收入三分之一自備款兩成"),
        _seg(4, 5, True, "房價所得比", "台北15.7倍新北12.3倍比去年更高"),   # dup of 1 (text)
        _seg(5, 1, False, "廣告", "進廣告休息一下"),
        _seg(6, 3, True, "租金漲幅", "台北租金漲3.2%新北2.8%"),
        _seg(7, 4, True, "央行升息", "重貼現率1.875%升到2%月增600元"),
        _seg(8, 5, False, "結語", "感謝收看下次再見記得訂閱"),
    ]
    chosen = s3_script.select_segments(segs, threshold=3, max_clips=3)
    ids = {s.id for s in chosen}
    assert len(chosen) == 3
    assert ids == {1, 7, 6}, ids  # chart-bearing ones preferred over score-5 no-chart #8
    by_id = {s.id: s for s in segs}
    assert by_id[2].skip_reason == "hook_score 2 < threshold 3"
    assert by_id[5].skip_reason == "hook_score 1 < threshold 3"
    assert by_id[4].skip_reason == "duplicate of segment 1 (text similarity)"
    assert by_id[8].skip_reason == "over max_clips budget"
    assert by_id[3].skip_reason == "over max_clips budget"
    assert all(s.selected for s in chosen)
    assert all(not s.selected and s.skip_reason for s in segs if s.id not in ids)
    # returned in transcript order
    assert [s.id for s in chosen] == [1, 6, 7]


def test_gate_respects_max_clips_and_threshold_variants():
    summaries = ["央行升息半碼房貸族每月多繳", "台北租金年漲三個百分點", "首購族自備款兩成才安全",
                 "桃園房價所得比八點五倍", "節目結尾感謝收看記得訂閱"]
    segs = [_seg(i, 5, False, f"主題{i}", summaries[i - 1]) for i in range(1, 6)]
    chosen = s3_script.select_segments(segs, threshold=3, max_clips=2)
    assert len(chosen) == 2
    assert sum(1 for s in segs if s.skip_reason == "over max_clips budget") == 3
    chosen = s3_script.select_segments(segs, threshold=5, max_clips=10)
    assert len(chosen) == 5


# ------------------------------------------------------------------ end-to-end

def test_execute_end_to_end(tmp_path, monkeypatch):
    fake = FakeChat()
    monkeypatch.setattr("pipeline.stages.s3_script.chat_json", fake)
    ctx, logs = make_ctx(tmp_path)

    result = run_stage(ctx, s3_script.STAGE, s3_script.stage_config(ctx), s3_script.execute)

    out_path = ctx.path("03_scripts.json")
    assert out_path.exists()
    assert result.outputs == {"scripts": "03_scripts.json"}
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["video_id"] == VIDEO_ID
    assert data["source_name"] == "TVBS《健康2.0》"

    # Pass A: garbage entry dropped, 6 valid segments kept
    assert len(data["segments"]) == 6
    assert result.meta["segments_total"] == 6

    # gate: exactly max_clips selected, rest carry a reason
    selected = [s for s in data["segments"] if s["selected"]]
    assert {s["id"] for s in selected} == {1, 2, 6}
    for s in data["segments"]:
        if not s["selected"]:
            assert s["skip_reason"], s
    reasons = {s["id"]: s["skip_reason"] for s in data["segments"]}
    assert reasons[3].startswith("hook_score 1 <")
    assert reasons[5] .startswith("duplicate of segment 1")
    assert reasons[4] == "over max_clips budget"

    # clips
    clips = data["clips"]
    assert len(clips) == ctx.max_clips == 3
    assert data["rejected_clips"] == []
    for clip in clips:
        assert clip["attribution"] == ATTR
        assert any(ATTR in line["text"] for line in clip["lines"]), clip["title"]
        assert clip["plagiarism_ok"] is True
        assert clip["plagiarism_overlap"] <= 0.15
        assert clip["plagiarism_lcs"] <= 12
        assert clip["chart"] is not None
    by_seg = {c["segment_id"]: c for c in clips}
    assert by_seg[2]["rewrite_attempts"] == 1
    assert by_seg[1]["rewrite_attempts"] == 0
    assert by_seg[6]["rewrite_attempts"] == 0
    # attribution was injected right after the hook line for the clip whose model forgot it
    assert by_seg[2]["lines"][1]["text"] == ATTR
    assert by_seg[2]["chart"]["type"] == "line"

    # LLM traffic: 1 pass A (cheap) + 3 pass B + 1 rewrite (strong)
    assert len(fake.calls) == 5
    assert fake.calls[0]["model"] == ctx.settings.llm_cheap_model
    assert all(c["model"] == ctx.settings.llm_strong_model for c in fake.calls[1:])
    assert all(c["stage"] == "s3_script" for c in fake.calls)
    rewrite_call = [c for c in fake.calls if c["note"].endswith("rewrite")]
    assert len(rewrite_call) == 1 and "重寫要求" in rewrite_call[0]["user"]
    # the plagiarism source window and the rewrite prompt both quote the transcript
    assert "重貼現率" in rewrite_call[0]["user"]

    # costs: real, stage-tagged, recorded to the manifest and settled into the run budget
    assert result.costs and all(c.stage == "s3_script" and not c.estimated for c in result.costs)
    assert result.usd > 0
    assert ctx.spent_usd == pytest.approx(result.usd)
    assert ctx.manifest.data["stages"]["s3_script"]["outputs"] == {"scripts": "03_scripts.json"}
    assert result.meta["segments_selected"] == 3
    assert result.meta["clips_accepted"] == 3
    assert result.meta["clips_rejected"] == 0
    assert result.meta["pass_b_skipped_saved_usd_est"] > 0
    assert any("Pass B skipped for 3/6" in line for line in logs)
    assert any("asking for a rewrite" in line for line in logs)

    # idempotent: a second run is a cache hit and makes no LLM calls
    n_calls = len(fake.calls)
    again = run_stage(ctx, s3_script.STAGE, s3_script.stage_config(ctx), s3_script.execute)
    assert again.cached and len(fake.calls) == n_calls


def test_execute_rejects_clip_that_stays_copied(tmp_path, monkeypatch):
    """If the rewrite is still a copy, the clip goes to rejected_clips instead of clips."""
    fake = FakeChat()

    def stubborn(settings, model, system, user, *, stage, note="", **kw):
        payload, costs = fake(settings, model, system, user, stage=stage, note=note, **kw)
        if note.startswith("pass B seg 2"):
            payload = json.loads(json.dumps(PASS_B_COPY_2))
        return payload, costs

    monkeypatch.setattr("pipeline.stages.s3_script.chat_json", stubborn)
    ctx, logs = make_ctx(tmp_path)
    result = s3_script.execute(ctx)
    data = json.loads(ctx.path("03_scripts.json").read_text(encoding="utf-8"))
    assert {c["segment_id"] for c in data["clips"]} == {1, 6}
    assert len(data["rejected_clips"]) == 1
    rej = data["rejected_clips"][0]
    assert rej["segment_id"] == 2 and rej["plagiarism_ok"] is False and rej["rewrite_attempts"] == 1
    assert rej["plagiarism_lcs"] > 12
    assert result.meta["clips_rejected"] == 1
    assert any("rejected: not rewritten enough" in line for line in logs)


def test_budget_guard_fires_before_pass_b(tmp_path, monkeypatch):
    from pipeline.context import BudgetExceeded

    fake = FakeChat()
    monkeypatch.setattr("pipeline.stages.s3_script.chat_json", fake)
    # enough for Pass A on the cheap model, not for a strong-model call
    ctx, _ = make_ctx(tmp_path, max_budget_usd=0.012)
    with pytest.raises(BudgetExceeded):
        s3_script.execute(ctx)
    assert len(fake.calls) == 1  # aborted before the first Pass B call
    assert not ctx.path("03_scripts.json").exists()


def test_dry_run_estimates_only(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry-run must not call the LLM")

    monkeypatch.setattr("pipeline.stages.s3_script.chat_json", boom)

    # with a transcript on disk
    ctx, logs = make_ctx(tmp_path, dry_run=True)
    result = run_stage(ctx, s3_script.STAGE, s3_script.stage_config(ctx), s3_script.execute)
    assert result.outputs == {}
    assert result.costs and all(c.estimated for c in result.costs)
    assert result.estimated_usd > 0 and result.usd == 0
    assert {c.model for c in result.costs} == {ctx.settings.llm_cheap_model, ctx.settings.llm_strong_model}
    assert not ctx.path("03_scripts.json").exists()
    assert "s3_script" not in ctx.manifest.data["stages"]
    assert result.meta["transcript_chars_basis"] == "02_transcript.json"

    # without a transcript (s1/s2 were dry-run too): falls back gracefully
    ctx2, _ = make_ctx(tmp_path / "b", dry_run=True, with_transcript=False)
    r2 = s3_script.execute(ctx2)
    assert r2.outputs == {} and all(c.estimated for c in r2.costs)
    assert r2.meta["transcript_chars"] == s3_script.FALLBACK_TRANSCRIPT_CHARS


def test_stage_config_tracks_inputs(tmp_path):
    ctx, _ = make_ctx(tmp_path, with_transcript=False)
    cfg = s3_script.stage_config(ctx)
    for key in ("prompt_version", "llm_cheap_model", "llm_strong_model", "hook_threshold", "max_clips",
                "plagiarism_ngram", "plagiarism_max_overlap", "plagiarism_max_lcs", "s2_config_hash"):
        assert key in cfg
    assert cfg["s2_config_hash"] is None
    ctx.manifest.data["stages"]["s2_transcribe"] = {"config_hash": "abc"}
    assert s3_script.stage_config(ctx)["s2_config_hash"] == "abc"


def test_source_name_rules():
    from pipeline.models import AudioInfo

    def info(channel):
        return AudioInfo(video_id="x", title="t", channel=channel, url="u", original_duration_sec=1,
                         processed_duration_sec=1, audio_path="a")

    assert s3_script.source_name_for(info("健康2.0")) == "TVBS《健康2.0》"
    assert s3_script.source_name_for(info("TVBS HEALTH 2.0")) == "TVBS《健康2.0》"
    assert s3_script.source_name_for(info("財經頻道")) == "財經頻道"
    assert s3_script.source_name_for(None) == "原始節目"
    assert s3_script.attribution_for("TVBS《健康2.0》") == ATTR


def test_format_transcript_merges_tiny_segments():
    from pipeline.models import TranscriptSegment

    segs = [TranscriptSegment(start=i * 2.0, end=i * 2.0 + 2, text="短句") for i in range(30)]
    text = s3_script.format_transcript(segs)
    lines = text.splitlines()
    assert 1 < len(lines) < 30
    assert lines[0].startswith("[00:00] ")


# ------------------------------------------------------------------ number clean-up

def test_humanize_numbers_taiwanese_units():
    h = s3_script.humanize_numbers
    assert h("示範貸款1e+07元") == "示範貸款1000萬元"
    assert h("開價23800000元") == "開價2380萬元"
    assert h("突破400000元/坪") == "突破40萬元/坪"
    assert h("案量580000000000元") == "案量5800億元"
    assert h("月付15000元") == "月付15000元"  # small numbers untouched
    assert h("總價2000萬、利率2.5%") == "總價2000萬、利率2.5%"


def test_unify_chart_units_drops_minority_unit():
    pts = [DataPoint(label="原", value=30, unit="萬/坪"), DataPoint(label="新", value=40, unit="萬/坪"),
           DataPoint(label="個案", value=2380, unit="萬")]
    kept = s3_script.unify_chart_units(pts)
    assert [p.label for p in kept] == ["原", "新"]
    # if nothing shares a unit, no chart is better than a wrong chart
    assert s3_script.unify_chart_units([DataPoint(label="a", value=1, unit="x"),
                                        DataPoint(label="b", value=2, unit="y")]) == []
