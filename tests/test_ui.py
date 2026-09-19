"""Web UI tests: run against a fixture data dir, never against the real pipeline."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.ui import server

VID = "KjAI9r8tnOs"


def _cost(stage, provider, model, unit, qty, price, usd, estimated=False, note=""):
    return dict(stage=stage, provider=provider, model=model, unit=unit, quantity=qty,
                unit_price_usd=price, usd=usd, estimated=estimated, note=note)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    wd = tmp_path / VID
    (wd / "04_clips").mkdir(parents=True)
    manifest = {
        "video_id": VID, "url": f"https://www.youtube.com/watch?v={VID}",
        "created": "2026-09-18T10:00:00", "updated": "2026-09-18T10:30:00",
        "stages": {
            "s2_transcribe": {
                "config_hash": "abc", "config": {}, "outputs": {"transcript": "02_transcript.json"},
                "meta": {"seconds_billed": 840, "segments": 120, "chars": 5000, "provider": "openai",
                         "model": "gpt-4o-mini-transcribe"},
                "finished_at": "2026-09-18T10:05:00",
                "costs": [_cost("s2_transcribe", "openai", "gpt-4o-mini-transcribe", "minute", 14, 0.003, 0.042)],
            },
            "s3_script": {
                "config_hash": "def", "config": {}, "outputs": {"scripts": "03_scripts.json"},
                "meta": {}, "finished_at": "2026-09-18T10:10:00",
                "costs": [
                    _cost("s3_script", "openai", "gpt-5-mini", "input_tokens", 8000, 0.25e-6, 0.002, note="pass A"),
                    _cost("s3_script", "openai", "gpt-5", "output_tokens", 1500, 10e-6, 0.015, note="pass B"),
                    _cost("s3_script", "openai", "gpt-5", "output_tokens", 1500, 10e-6, 0.999, estimated=True, note="pass B dry-run"),
                    _cost("s4_render", "reference", "ai-video-api", "second", 60, 0.25, 15.0, estimated=True),
                ],
            },
        },
        "runs": [
            {"started": 1.0, "dry_run": False, "stages_run": ["s2_transcribe", "s3_script"], "stages_cached": [], "usd": 0.059},
            {"started": 2.0, "dry_run": False, "stages_run": ["s3_script"], "stages_cached": ["s2_transcribe"], "usd": 0.017},
        ],
    }
    (wd / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (wd / "01_info.json").write_text(json.dumps({
        "video_id": VID, "title": "測試影片", "channel": "測試頻道", "url": manifest["url"],
        "original_duration_sec": 960, "processed_duration_sec": 840, "audio_path": "01_audio.wav",
        "preprocessing": {"remove_silence": True},
    }, ensure_ascii=False), encoding="utf-8")
    seg = lambda i, sel, reason="": dict(id=i, start=i * 100.0, end=i * 100.0 + 90, topic=f"主題{i}", summary="摘要",
                                          key_points=[], data_points=[], has_chart_data=i == 1,
                                          hook_score=5 if sel else 2, selected=sel, skip_reason=reason)
    (wd / "03_scripts.json").write_text(json.dumps({
        "video_id": VID, "source_name": "TVBS",
        "segments": [seg(1, True), seg(2, False, "hook_score 2 < 3"), seg(3, False, "duplicate topic")],
        "clips": [dict(segment_id=1, title="標題", hook="鉤子", lines=[{"text": "台詞一"}], chart=None,
                       attribution="根據 TVBS 報導", plagiarism_overlap=0.05, plagiarism_lcs=6,
                       plagiarism_ok=True, rewrite_attempts=0)],
        "rejected_clips": [],
    }, ensure_ascii=False), encoding="utf-8")
    (wd / "04_render.json").write_text(json.dumps({
        "video_id": VID,
        "clips": [dict(segment_id=1, title="標題", video_path="04_clips/clip_01.mp4", duration_sec=31.5, chart_path=None)],
    }), encoding="utf-8")
    (wd / "04_clips" / "clip_01.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42dummy")
    # a directory without a manifest must be ignored by the listing
    (tmp_path / "junk").mkdir()
    monkeypatch.setenv("FINVID_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(data_dir):
    server._runs.clear()
    with TestClient(server.app) as c:
        yield c


def test_index_and_list(client):
    r = client.get("/")
    assert r.status_code == 200 and "成本" in r.text
    r = client.get("/api/videos")
    assert r.status_code == 200
    vids = r.json()
    assert [v["video_id"] for v in vids] == [VID]
    v = vids[0]
    assert v["title"] == "測試影片"
    assert v["stages_done"] == ["s2_transcribe", "s3_script"]
    assert v["spent_usd"] == pytest.approx(0.042 + 0.002 + 0.015)


def test_detail_cost_summary(client):
    r = client.get(f"/api/videos/{VID}")
    assert r.status_code == 200
    d = r.json()
    cs = d["cost_summary"]
    # spent excludes estimated (0.999) and reference (15.0)
    assert cs["spent_usd"] == pytest.approx(0.059)
    assert cs["estimated_usd"] == pytest.approx(0.999)
    assert cs["reference_ai_video_usd"] == pytest.approx(15.0)
    assert cs["last_run_usd"] == pytest.approx(0.017)
    # last run cached s2 -> saved its recorded real cost
    assert cs["saved_by_cache_usd"] == pytest.approx(0.042)
    assert cs["per_stage"]["s2_transcribe"]["cached_last_run"] is True
    assert cs["per_stage"]["s2_transcribe"]["usd"] == pytest.approx(0.042)
    assert cs["per_stage"]["s3_script"]["cached_last_run"] is False
    assert cs["per_stage"]["s3_script"]["usd"] == pytest.approx(0.017)
    assert cs["per_stage"]["s1_download"]["present"] is False
    # naive = reference + pass B scaled to all 3 segments (1 selected): 15 + 0.015*3
    nb = cs["naive_breakdown"]
    assert nb["segments_total"] == 3 and nb["segments_selected"] == 1
    assert nb["pass_b_actual_usd"] == pytest.approx(0.015)
    assert cs["naive_alternative_usd"] == pytest.approx(15.0 + 0.045)
    assert len(cs["ledger"]) == 5
    assert d["info"]["title"] == "測試影片"
    assert len(d["scripts"]["segments"]) == 3
    assert d["render"]["clips"][0]["video_path"] == "04_clips/clip_01.mp4"
    assert d["transcript"] is None  # 02_transcript.json missing -> defensive None


def test_detail_missing(client):
    assert client.get("/api/videos/nope_nope_1").status_code == 404
    assert client.get("/api/videos/bad%20id").status_code == 400


def test_files_serve_and_traversal(client):
    r = client.get(f"/files/{VID}/04_clips/clip_01.mp4")
    assert r.status_code == 200 and r.content.startswith(b"\x00\x00\x00\x18ftyp")
    assert client.get(f"/files/{VID}/04_clips/nope.mp4").status_code == 404
    for bad in (f"/files/{VID}/..%2F..%2Fmanifest.json", f"/files/{VID}/../junk", f"/files/{VID}/04_clips/..%2F..%2F..%2Fpyproject.toml"):
        r = client.get(bad)
        assert r.status_code in (400, 404), bad
    assert client.get("/files/..%2F/manifest.json").status_code in (400, 404)


def test_run_records_logs(client, monkeypatch):
    seen = {}

    def fake(url, *, dry_run, force, max_clips, until, log, data_dir):
        seen.update(url=url, dry_run=dry_run, max_clips=max_clips, until=until, data_dir=data_dir)
        log("line one")
        log("[bold red]BUDGET GUARD:[/] not really")
        log("line three")

    monkeypatch.setattr(server, "run_pipeline", fake)
    r = client.post("/api/runs", json={"url": f"https://youtu.be/{VID}", "dry_run": True, "force": False,
                                        "max_clips": 2, "until": "all"})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    assert r.json()["video_id"] == VID
    for _ in range(50):
        rec = client.get(f"/api/runs/{run_id}").json()
        if rec["status"] != "running":
            break
        time.sleep(0.05)
    assert rec["status"] == "done"
    assert rec["logs"] == ["line one", "BUDGET GUARD: not really", "line three"]
    assert rec["error"] is None and rec["finished"] is not None
    assert seen["dry_run"] is True and seen["max_clips"] == 2 and seen["until"] is None
    assert Path(seen["data_dir"]).resolve() == Path(client.get("/api/videos").json()[0] and server.data_dir())


def test_run_error_and_budget(client, monkeypatch):
    from pipeline.context import BudgetExceeded

    def boom(url, **kw):
        kw["log"]("starting")
        raise BudgetExceeded("would cost too much")

    monkeypatch.setattr(server, "run_pipeline", boom)
    run_id = client.post("/api/runs", json={"url": VID}).json()["run_id"]
    for _ in range(50):
        rec = client.get(f"/api/runs/{run_id}").json()
        if rec["status"] != "running":
            break
        time.sleep(0.05)
    assert rec["status"] == "budget_exceeded" and "too much" in rec["error"]
    assert client.post("/api/runs", json={"url": "not a url"}).status_code == 400
    assert client.post("/api/runs", json={"url": VID, "until": "s9"}).status_code == 400
    assert client.get("/api/runs/zzz").status_code == 404


def test_concurrent_run_guard(client, monkeypatch):
    release = threading.Event()

    def slow(url, **kw):
        kw["log"]("waiting")
        release.wait(5)

    monkeypatch.setattr(server, "run_pipeline", slow)
    r1 = client.post("/api/runs", json={"url": VID})
    assert r1.status_code == 202
    r2 = client.post("/api/runs", json={"url": f"https://www.youtube.com/watch?v={VID}"})
    assert r2.status_code == 409
    assert "duplicate" in r2.json()["detail"]
    # a different video is not blocked
    r3 = client.post("/api/runs", json={"url": "https://www.youtube.com/watch?v=abcdefghijk"})
    assert r3.status_code == 202
    release.set()
    for _ in range(50):
        if client.get(f"/api/runs/{r1.json()['run_id']}").json()["status"] == "done":
            break
        time.sleep(0.05)
    assert client.get(f"/api/runs/{r1.json()['run_id']}").json()["status"] == "done"
    # once finished, the same video may run again
    release.set()
    assert client.post("/api/runs", json={"url": VID}).status_code == 202


def test_settings_api_reads_masked_and_writes_env(tmp_path, monkeypatch, client):
    """The dashboard's 設定 panel: stored values are shown (localhost only), writes merge into .env
    (placeholder comment lines become real assignments), unknown keys are refused."""
    from pipeline.ui import settings_api as sa
    env = tmp_path / ".env"
    env.write_text("# 說明\n# HF_TOKEN=hf_...（選填）\nOPENAI_API_KEY=\nFINVID_MAX_CLIPS=3\n", encoding="utf-8")
    monkeypatch.setattr(sa, "ENV_PATH", env)
    monkeypatch.setattr(sa, "ENV_EXAMPLE", tmp_path / "nope")

    d = client.get("/api/settings").json()
    assert d["secrets"]["OPENAI_API_KEY"] == {"set": False, "hint": "", "value": ""} and d["plain"]["FINVID_MAX_CLIPS"] == "3"

    r = client.post("/api/settings", json={"values": {"OPENAI_API_KEY": "sk-test-1234567890", "HF_TOKEN": "hf_abcdefghijkl",
                                                       "FINVID_AI_VIDEO": "hf,pixazo", "FINVID_BROLL": "pexels"}})
    assert r.status_code == 200 and sorted(r.json()["saved"]) == ["FINVID_AI_VIDEO", "FINVID_BROLL", "HF_TOKEN", "OPENAI_API_KEY"]
    text = env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-test-1234567890" in text and "HF_TOKEN=hf_abcdefghijkl" in text
    assert "# 說明" in text and "# HF_TOKEN=" not in text  # comment kept, placeholder replaced
    assert "FINVID_AI_VIDEO=hf,pixazo" in text and "FINVID_MAX_CLIPS=3" in text

    d = client.get("/api/settings").json()
    assert d["secrets"]["OPENAI_API_KEY"] == {"set": True, "hint": "…7890", "value": "sk-test-1234567890"}  # shown: localhost-only panel
    assert d["plain"]["FINVID_AI_VIDEO"] == "hf,pixazo"

    assert client.post("/api/settings", json={"values": {"EVIL": "x"}}).status_code == 400
    assert client.post("/api/settings", json={"values": {"OPENAI_API_KEY": "請在此填入"}}).status_code == 400
    assert client.post("/api/settings/probe", json={"what": "nope"}).status_code == 400


def test_settings_probe_comfy_reports_missing_models(monkeypatch, client):
    from pipeline.ui import settings_api as sa
    import httpx

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/system_stats":
            return httpx.Response(200, json={"devices": [{"name": "fake gpu", "vram_total": 8e9}]})
        if "CheckpointLoaderSimple" in req.url.path:
            return httpx.Response(200, json={"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["other.safetensors"]]}}}})
        if "CLIPLoader" in req.url.path:
            return httpx.Response(200, json={"CLIPLoader": {"input": {"required": {"clip_name": [["t5xxl_fp8_e4m3fn_scaled.safetensors"]]}}}})
        return httpx.Response(404)

    real_client = httpx.Client
    monkeypatch.setattr(sa.httpx, "Client", lambda base_url, timeout: real_client(base_url=base_url, transport=httpx.MockTransport(handler)))
    res = sa.probe_comfy("http://comfy.test")
    assert res["ok"] is False and "缺模型檔" in res["message"] and "ltxv-2b" in res["message"] and res["device"] == "fake gpu"
