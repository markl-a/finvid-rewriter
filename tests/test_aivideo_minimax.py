"""MiniMax Hailuo (paid) provider against a fake API (httpx MockTransport). No key, no network, no charge."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from pipeline.config import Settings
from pipeline.render.aivideo import FallbackProvider, make_provider
from pipeline.render.aivideo.minimax import PRICES_USD, MiniMaxError, MiniMaxProvider, duration_for, price_for


def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, ai_video="minimax", minimax_api_key="mm-test-key")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _mp4(settings: Settings, out: Path) -> bytes:
    # MiniMax T2V returns 16:9; compose crops to 9:16, the provider just passes it through ffmpeg
    subprocess.run([settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc2=size=1366x768:rate=25", "-t", "0.5", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, text=True)
    return out.read_bytes()


class FakeMiniMax:
    """video_generation -> task_id; query -> Queueing/Processing then Success; files/retrieve -> download_url."""

    DL = "https://cdn.minimax.test/files/f-1.mp4"

    def __init__(self, video: bytes, polls_until_done: int = 3, *, fail: bool = False,
                 submit_code: int = 0, submit_msg: str = "success", http_status: int = 200):
        self.video, self.polls_until_done, self.fail = video, polls_until_done, fail
        self.submit_code, self.submit_msg, self.http_status = submit_code, submit_msg, http_status
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        self.polls = 0
        self.requests = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests += 1
        path = req.url.path
        ok = {"status_code": 0, "status_msg": "success"}
        if path == "/v1/video_generation":
            assert req.method == "POST"
            self.headers.append(dict(req.headers))
            self.bodies.append(json.loads(req.content))
            if self.http_status != 200:
                return httpx.Response(self.http_status, json={"message": "nope"})
            if self.submit_code:
                return httpx.Response(200, json={"task_id": "", "base_resp": {
                    "status_code": self.submit_code, "status_msg": self.submit_msg}})
            return httpx.Response(200, json={"task_id": "task-1", "base_resp": ok})
        if path == "/v1/query/video_generation":
            assert req.url.params["task_id"] == "task-1"
            assert req.headers.get("authorization") == "Bearer mm-test-key"
            self.polls += 1
            if self.polls < self.polls_until_done:
                status = "Queueing" if self.polls == 1 else "Processing"
                return httpx.Response(200, json={"task_id": "task-1", "status": status, "file_id": "", "base_resp": ok})
            if self.fail:
                return httpx.Response(200, json={"task_id": "task-1", "status": "Fail", "file_id": "",
                                                 "base_resp": {"status_code": 0, "status_msg": "sensitive content (fake)"}})
            return httpx.Response(200, json={"task_id": "task-1", "status": "Success", "file_id": "f-1", "base_resp": ok})
        if path == "/v1/files/retrieve":
            assert req.url.params["file_id"] == "f-1"
            return httpx.Response(200, json={"file": {"file_id": "f-1", "download_url": self.DL}, "base_resp": ok})
        if str(req.url) == self.DL:
            assert "authorization" not in req.headers  # never leak the key to the CDN
            return httpx.Response(200, content=self.video, headers={"content-type": "video/mp4"})
        return httpx.Response(404, text=f"unexpected {req.url}")


def _provider(settings: Settings, fake: FakeMiniMax, monkeypatch) -> MiniMaxProvider:
    monkeypatch.setattr("pipeline.render.aivideo.minimax.time.sleep", lambda _s: None)
    p = MiniMaxProvider.from_settings(settings)
    p.client = httpx.Client(base_url="https://api.minimax.io/v1", transport=httpx.MockTransport(fake.handler))
    return p


def test_make_provider_minimax_defaults_and_chain():
    p = make_provider(_settings())
    assert isinstance(p, MiniMaxProvider) and p.model == "MiniMax-Hailuo-02" and p.resolution == "768P"
    assert p.unit == "video" and p.unit_price_usd == 0.27 and p.duration == 6 and p.timeout_sec == 900
    assert make_provider(_settings(minimax_resolution="512P")).unit_price_usd == 0.08
    assert make_provider(_settings(minimax_resolution="1080P")).unit_price_usd == 0.54
    chain = make_provider(_settings(ai_video="hf,pixazo,comfy"))
    assert isinstance(chain, FallbackProvider) and [x.name for x in chain.providers] == ["hf-zerogpu", "pixazo", "comfyui"]
    paid_last = make_provider(_settings(ai_video="hf,minimax"))
    assert isinstance(paid_last, FallbackProvider) and paid_last.providers[1].name == "minimax"


def test_price_table_and_duration_rounding():
    assert PRICES_USD[("MiniMax-Hailuo-02", "768P", 6)] == 0.27
    assert duration_for(5) == 6 and duration_for(6) == 6 and duration_for(6.5) == 10 and duration_for(10) == 10
    assert price_for("MiniMax-Hailuo-02", "512P", 6) == 0.08
    with pytest.raises(MiniMaxError, match="no price on file.*PRICES_USD"):
        price_for("MiniMax-Hailuo-02", "768P", 10)
    with pytest.raises(MiniMaxError, match="no price on file"):  # unknown combos refuse to start, not charge blind
        make_provider(_settings(minimax_model="MiniMax-Hailuo-99"))


def test_estimate_is_the_real_price_so_the_budget_guard_bites():
    est = MiniMaxProvider.from_settings(_settings()).estimate(2, 5)
    assert len(est) == 1 and est[0].estimated and est[0].unit == "video"
    assert est[0].quantity == 2 and est[0].unit_price_usd == 0.27 and est[0].usd == pytest.approx(0.54)
    assert "paid" in est[0].note and "6s 768P" in est[0].note
    assert MiniMaxProvider.from_settings(_settings(minimax_resolution="1080P")).estimate(1, 5)[0].usd == pytest.approx(0.54)


def test_generate_polls_retrieves_downloads_and_caches(tmp_path, monkeypatch):
    s = _settings()
    fake = FakeMiniMax(_mp4(s, tmp_path / "src.mp4"), polls_until_done=3)
    p = _provider(s, fake, monkeypatch)
    logs: list[str] = []

    out = tmp_path / "ai_05.mp4"
    r = p.generate("skyline at dusk", out, seconds=5, seed=5, log=logs.append)
    assert r.path == out and out.exists() and out.stat().st_size > 1000 and not r.cached
    assert fake.polls == 3 and len(fake.bodies) == 1
    assert fake.requests == 6  # submit + 3 polls + retrieve + download
    assert fake.headers[0]["authorization"] == "Bearer mm-test-key"
    body = fake.bodies[0]
    assert body == {"model": "MiniMax-Hailuo-02", "prompt": "skyline at dusk", "duration": 6,
                    "resolution": "768P", "prompt_optimizer": True}
    assert (tmp_path / "ai_05.json").exists() and not (tmp_path / "ai_05.raw").exists()
    side = json.loads((tmp_path / "ai_05.json").read_text(encoding="utf-8"))
    assert side["task_id"] == "task-1" and side["file_id"] == "f-1" and side["polls"] == 3 and "16:9" in side["aspect"]

    c = r.costs[0]
    assert c.provider == "minimax" and c.model == "MiniMax-Hailuo-02" and c.unit == "video"
    assert c.quantity == 1 and c.unit_price_usd == 0.27 and c.usd == pytest.approx(0.27) and not c.estimated
    assert "ai_05.mp4" in c.note
    assert any("16:9" in m for m in logs)

    # same prompt/params -> cache hit, zero requests, zero dollars
    before = fake.requests
    r2 = p.generate("skyline at dusk", out, seconds=5, seed=5, log=logs.append)
    assert r2.cached and fake.requests == before and r2.costs[0].usd == 0 and r2.costs[0].quantity == 0

    # a different prompt is a new (billed) generation
    p.generate("harbour", out, seconds=5, seed=5, log=logs.append)
    assert len(fake.bodies) == 2


def test_long_requests_round_to_10s(monkeypatch):
    s = _settings(ai_shot_seconds=8)
    with pytest.raises(MiniMaxError, match="10s"):  # no 10 s price on file yet -> refuse, do not guess
        MiniMaxProvider.from_settings(s)
    monkeypatch.setitem(PRICES_USD, ("MiniMax-Hailuo-02", "768P", 10), 0.45)
    p = MiniMaxProvider.from_settings(s)
    assert p.duration == 10 and p.unit_price_usd == 0.45 and p.build_body("x", seconds=8)["duration"] == 10


def test_failures_are_minimax_errors(tmp_path, monkeypatch):
    s = _settings()
    p = _provider(s, FakeMiniMax(b"", fail=True), monkeypatch)
    with pytest.raises(MiniMaxError, match="task task-1 failed: sensitive content"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)
    assert not (tmp_path / "ai.json").exists()  # nothing cached on failure

    p = _provider(s, FakeMiniMax(b"", submit_code=1008, submit_msg="insufficient balance"), monkeypatch)
    with pytest.raises(MiniMaxError, match="MiniMax error 1008: insufficient balance"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)

    p = _provider(s, FakeMiniMax(b"", http_status=401), monkeypatch)
    with pytest.raises(MiniMaxError, match="MINIMAX_API_KEY"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)

    p = _provider(s, FakeMiniMax(b"", polls_until_done=10 ** 6), monkeypatch)
    p.timeout_sec = 0
    with pytest.raises(MiniMaxError, match="did not finish"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)

    down = MiniMaxProvider.from_settings(s)
    down.client = httpx.Client(base_url="https://api.minimax.io/v1", transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused"))))
    with pytest.raises(MiniMaxError, match="not reachable"):
        down.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)


def test_missing_key_is_actionable(tmp_path, monkeypatch):
    fake = FakeMiniMax(b"")
    p = _provider(_settings(minimax_api_key=None), fake, monkeypatch)
    with pytest.raises(MiniMaxError, match="MINIMAX_API_KEY not set.*platform.minimax.io"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)
    assert fake.requests == 0  # fails before any call, so nothing is billed
    assert MiniMaxProvider.from_settings(_settings(minimax_api_key=None)).estimate(1, 5)[0].usd == pytest.approx(0.27)
