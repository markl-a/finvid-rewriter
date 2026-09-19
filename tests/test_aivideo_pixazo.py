"""Pixazo (free LTX tier) provider against a fake gateway (httpx MockTransport). No key, no network."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from pipeline.config import Settings
from pipeline.render.aivideo import FallbackProvider, make_provider
from pipeline.render.aivideo.pixazo import ENDPOINT, PixazoError, PixazoProvider


def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, ai_video="pixazo", pixazo_api_key="px-test-key")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _mp4(settings: Settings, out: Path) -> bytes:
    subprocess.run([settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc2=size=576x1024:rate=24", "-t", "0.5", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, text=True)
    return out.read_bytes()


class FakePixazo:
    """text-to-video -> QUEUED + polling_url; status polls -> IN_PROGRESS until done; media download."""

    POLL = "https://gateway.pixazo.ai/v2/requests/status/req-1"
    MEDIA = "https://cdn.pixazo.test/out/req-1.mp4"

    def __init__(self, video: bytes, polls_until_done: int = 2, *, sync: bool = False, fail: bool = False,
                 submit_status: int = 200):
        self.video, self.polls_until_done, self.sync, self.fail = video, polls_until_done, sync, fail
        self.submit_status = submit_status
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        self.polls = 0
        self.requests = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests += 1
        url = str(req.url)
        if url == ENDPOINT:
            assert req.method == "POST"
            self.headers.append(dict(req.headers))
            self.bodies.append(json.loads(req.content))
            if self.submit_status != 200:
                return httpx.Response(self.submit_status, json={"message": "nope"})
            if self.sync:
                return httpx.Response(200, json={"request_id": "req-1", "status": "COMPLETED",
                                                 "output": {"media_url": [self.MEDIA], "media_type": "video/mp4"}})
            return httpx.Response(200, json={"request_id": "req-1", "status": "QUEUED", "polling_url": self.POLL})
        if url == self.POLL:
            assert req.headers.get("Ocp-Apim-Subscription-Key") == "px-test-key"
            self.polls += 1
            if self.polls < self.polls_until_done:
                return httpx.Response(200, json={"request_id": "req-1", "status": "IN_PROGRESS"})
            if self.fail:
                return httpx.Response(200, json={"request_id": "req-1", "status": "FAILED",
                                                 "error": "content policy (fake)"})
            return httpx.Response(200, json={"request_id": "req-1", "status": "COMPLETED",
                                             "output": {"media_url": [self.MEDIA], "media_type": "video/mp4"}})
        if url == self.MEDIA:
            assert "Ocp-Apim-Subscription-Key" not in req.headers  # never leak the key to the CDN
            return httpx.Response(200, content=self.video, headers={"content-type": "video/mp4"})
        return httpx.Response(404, text=f"unexpected {url}")


def _provider(settings: Settings, fake: FakePixazo, monkeypatch) -> PixazoProvider:
    monkeypatch.setattr("pipeline.render.aivideo.pixazo.time.sleep", lambda _s: None)
    p = PixazoProvider.from_settings(settings)
    p.client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return p


def test_make_provider_pixazo_and_chain():
    p = make_provider(_settings())
    assert isinstance(p, PixazoProvider) and p.api_key == "px-test-key" and p.timeout_sec == 600
    chain = make_provider(_settings(ai_video="hf,pixazo,comfy"))
    assert isinstance(chain, FallbackProvider) and len(chain.providers) == 3
    assert [x.name for x in chain.providers] == ["hf-zerogpu", "pixazo", "comfyui"]
    with pytest.raises(ValueError, match="pixazo"):
        make_provider(_settings(ai_video="kling"))


def test_body_is_free_tier_ltx_shape():
    p = PixazoProvider.from_settings(_settings())
    body = p.build_body("a harbour at dawn", seconds=5, seed=7)
    assert body["prompt"] == "a harbour at dawn" and body["seed"] == 7 and body["negative"]
    assert body["width"] == 576 and body["height"] == 1024 and body["frame_rate"] == 24
    assert body["num_frames"] == 121 and (body["num_frames"] - 1) % 8 == 0  # 5s*24 -> 8k+1
    assert body["steps"] == 8 and body["cfg"] == 1
    assert p.frames_for(2) == 49 and p.frames_for(0.1) == 9
    assert "resolution" not in body and "duration" not in body  # paid-tier fields not sent on free tier


def test_generate_polls_until_completed_downloads_and_caches(tmp_path, monkeypatch):
    s = _settings()
    fake = FakePixazo(_mp4(s, tmp_path / "src.mp4"), polls_until_done=3)
    p = _provider(s, fake, monkeypatch)
    logs: list[str] = []

    out = tmp_path / "ai_05.mp4"
    r = p.generate("skyline", out, seconds=5, seed=5, log=logs.append)
    assert r.path == out and out.exists() and out.stat().st_size > 1000 and not r.cached
    assert fake.polls == 3 and len(fake.bodies) == 1 and fake.requests == 5  # submit + 3 polls + download
    assert fake.headers[0]["ocp-apim-subscription-key"] == "px-test-key"
    assert fake.bodies[0]["num_frames"] == 121 and fake.bodies[0]["seed"] == 5
    assert (tmp_path / "ai_05.json").exists() and not (tmp_path / "ai_05.raw").exists()
    side = json.loads((tmp_path / "ai_05.json").read_text(encoding="utf-8"))
    assert side["request_id"] == "req-1" and side["polls"] == 3

    c = r.costs[0]
    assert c.provider == "pixazo" and c.model == "ltx-video" and c.unit == "second"
    assert c.quantity == 5 and c.unit_price_usd == 0 and c.usd == 0 and not c.estimated
    assert "ai_05.mp4" in c.note and "s on pixazo" in c.note  # wall time in the note
    assert any("pixazo: text-to-video" in m for m in logs)

    # same prompt/params -> cache hit, zero requests
    before = fake.requests
    r2 = p.generate("skyline", out, seconds=5, seed=5, log=logs.append)
    assert r2.cached and fake.requests == before and r2.costs[0].quantity == 0 and r2.costs[0].usd == 0

    # different seed -> new request
    p.generate("skyline", out, seconds=5, seed=6, log=logs.append)
    assert len(fake.bodies) == 2


def test_synchronous_response_skips_polling(tmp_path, monkeypatch):
    s = _settings()
    fake = FakePixazo(_mp4(s, tmp_path / "src.mp4"), sync=True)
    p = _provider(s, fake, monkeypatch)
    r = p.generate("skyline", tmp_path / "ai.mp4", seconds=2, seed=1, log=lambda _m: None)
    assert r.path.exists() and fake.polls == 0 and fake.requests == 2  # submit + download
    assert r.meta["polls"] == 0 and r.costs[0].quantity == 2


def test_failure_status_and_http_errors(tmp_path, monkeypatch):
    s = _settings()
    p = _provider(s, FakePixazo(b"", fail=True), monkeypatch)
    with pytest.raises(PixazoError, match="content policy"):
        p.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)
    assert not (tmp_path / "ai.json").exists()  # nothing cached on failure

    p = _provider(s, FakePixazo(b"", submit_status=401), monkeypatch)
    with pytest.raises(PixazoError, match="PIXAZO_API_KEY"):
        p.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)

    p = _provider(s, FakePixazo(b"", submit_status=429), monkeypatch)
    with pytest.raises(PixazoError, match="60 requests/min"):
        p.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)

    p = _provider(s, FakePixazo(b"", polls_until_done=10 ** 6), monkeypatch)
    p.timeout_sec = 0  # deadline already passed -> no poll happens
    with pytest.raises(PixazoError, match="did not finish"):
        p.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)

    down = PixazoProvider.from_settings(s)
    down.client = httpx.Client(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused"))))
    with pytest.raises(PixazoError, match="not reachable"):
        down.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)


def test_missing_key_is_actionable(tmp_path, monkeypatch):
    s = _settings(pixazo_api_key=None)
    fake = FakePixazo(b"")
    p = _provider(s, fake, monkeypatch)
    with pytest.raises(PixazoError, match="PIXAZO_API_KEY not set.*pixazo.ai"):
        p.generate("x", tmp_path / "ai.mp4", seconds=1, seed=1, log=lambda _m: None)
    assert fake.requests == 0  # fails before any call
    assert make_provider(_settings(pixazo_api_key="  ")).api_key is None


def test_estimate_is_free_in_video_seconds():
    est = PixazoProvider.from_settings(_settings(pixazo_est_seconds=45)).estimate(3, 5)
    assert len(est) == 1 and est[0].estimated and est[0].usd == 0 and est[0].unit_price_usd == 0
    assert est[0].unit == "second" and est[0].quantity == 15 and "~45s wall" in est[0].note
