"""ComfyUI provider tests against a fake server (httpx MockTransport). No GPU, no network."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from pipeline.config import Settings
from pipeline.render.aivideo import make_provider
from pipeline.render.aivideo.comfy import WORKFLOWS_DIR, ComfyUIError, ComfyUIProvider


def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, ai_video="comfy")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _webm(settings: Settings, out: Path) -> bytes:
    subprocess.run([settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc2=size=576x1024:rate=24", "-t", "0.5", "-c:v", "libvpx-vp9", str(out)],
                   check=True, capture_output=True, text=True)
    return out.read_bytes()


class FakeComfy:
    """Minimal ComfyUI: /system_stats, /prompt, /history/<id>, /view. Records what it was sent."""

    def __init__(self, video_bytes: bytes, polls_until_done: int = 2, fail: bool = False):
        self.video = video_bytes
        self.polls_until_done = polls_until_done
        self.fail = fail
        self.prompts: list[dict] = []
        self.polls = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"devices": [{"name": "AMD Radeon 8060S (fake)"}]})
        if path == "/prompt":
            body = json.loads(req.content)
            self.prompts.append(body["prompt"])
            return httpx.Response(200, json={"prompt_id": "pid-1", "number": 1})
        if path.startswith("/history/"):
            self.polls += 1
            if self.polls < self.polls_until_done:
                return httpx.Response(200, json={})
            if self.fail:
                return httpx.Response(200, json={"pid-1": {"outputs": {}, "status": {
                    "status_str": "error", "messages": [["execution_error", {"exception_message": "OOM (fake)"}]]}}})
            return httpx.Response(200, json={"pid-1": {"outputs": {"11": {"gifs": [
                {"filename": "finvid_ai_05_00001.webm", "subfolder": "finvid", "type": "output"}]}},
                "status": {"status_str": "success"}}})
        if path == "/view":
            assert req.url.params["filename"] == "finvid_ai_05_00001.webm"
            return httpx.Response(200, content=self.video, headers={"content-type": "video/webm"})
        return httpx.Response(404)


def _provider(settings: Settings, fake: FakeComfy) -> ComfyUIProvider:
    client = httpx.Client(base_url="http://comfy.test", transport=httpx.MockTransport(fake.handler))
    p = ComfyUIProvider.from_settings(settings)
    p.client = client
    return p


def test_make_provider_none_and_unknown():
    assert make_provider(_settings(ai_video="none")) is None
    with pytest.raises(ValueError):
        make_provider(_settings(ai_video="kling"))


def test_workflow_template_is_filled_and_typed():
    s = _settings()
    p = ComfyUIProvider.from_settings(s)
    wf = p.build_workflow("a city at dusk", seconds=5, seed=7, prefix="ai_05")
    assert "_comment" not in wf
    assert wf["3"]["inputs"]["text"] == "a city at dusk"
    assert wf["5"]["inputs"] == {"width": 576, "height": 1024, "length": 121, "batch_size": 1}  # 5s*24 -> 8k+1
    assert wf["9"]["inputs"]["noise_seed"] == 7 and wf["9"]["inputs"]["cfg"] == 1.0
    assert wf["7"]["inputs"]["steps"] == 8
    assert wf["1"]["inputs"]["ckpt_name"] == s.comfy_checkpoint
    assert wf["11"]["inputs"]["filename_prefix"] == "finvid/ai_05"
    assert json.dumps(wf).count("<<") == 0
    assert p.frames_for(1.0) == 25 and p.frames_for(0.1) == 9


def test_generate_polls_downloads_converts_and_caches(tmp_path, monkeypatch):
    s = _settings()
    fake = FakeComfy(_webm(s, tmp_path / "src.webm"), polls_until_done=3)
    p = _provider(s, fake)
    monkeypatch.setattr("pipeline.render.aivideo.comfy.time.sleep", lambda _s: None)
    logs: list[str] = []

    out = tmp_path / "ai_05.mp4"
    r = p.generate("skyline", out, seconds=2, seed=5, log=logs.append)
    assert r.path == out and out.exists() and out.stat().st_size > 1000
    assert not r.cached and fake.polls == 3 and len(fake.prompts) == 1
    assert fake.prompts[0]["5"]["inputs"]["length"] == 49  # 2s*24=48 -> 49
    assert (tmp_path / "ai_05.json").exists()
    assert len(r.costs) == 1 and r.costs[0].usd == 0 and r.costs[0].unit == "gpu_second" and not r.costs[0].estimated
    assert "AMD Radeon 8060S (fake)" in r.costs[0].note
    assert not (tmp_path / "ai_05.raw").exists()  # temp download cleaned up

    # same prompt/params -> cache hit, server not called again
    r2 = p.generate("skyline", out, seconds=2, seed=5, log=logs.append)
    assert r2.cached and len(fake.prompts) == 1 and r2.costs[0].quantity == 0

    # different prompt -> regenerate
    p.generate("harbour", out, seconds=2, seed=5, log=logs.append)
    assert len(fake.prompts) == 2


def test_generate_surfaces_server_errors(tmp_path, monkeypatch):
    s = _settings()
    monkeypatch.setattr("pipeline.render.aivideo.comfy.time.sleep", lambda _s: None)
    p = _provider(s, FakeComfy(b"", fail=True))
    with pytest.raises(ComfyUIError, match="OOM"):
        p.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)

    down = ComfyUIProvider.from_settings(s)
    down.client = httpx.Client(base_url="http://comfy.test",
                               transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused"))))
    with pytest.raises(ComfyUIError, match="not reachable"):
        down.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)


def test_estimate_is_free_but_counts_gpu_seconds():
    p = ComfyUIProvider.from_settings(_settings(comfy_est_gpu_seconds=120))
    est = p.estimate(3, 5)
    assert len(est) == 1 and est[0].estimated and est[0].usd == 0 and est[0].quantity == 360


def test_bundled_workflow_exists_and_parses():
    wf = json.loads((WORKFLOWS_DIR / "ltxv_t2v.json").read_text(encoding="utf-8"))
    assert {"1", "3", "5", "9", "11"} <= set(wf)
