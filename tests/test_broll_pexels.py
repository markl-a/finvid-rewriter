"""Pexels b-roll backend against a fake API (httpx MockTransport). No key, no network."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from pipeline.config import ConfigError, Settings
from pipeline.render import tts
from pipeline.render.broll import BrollError, make_broll
from pipeline.render.broll.pexels import DEFAULT_QUERY, PexelsBroll

QUERY = "mortgage papers kitchen table"
VIDEO_ID = 3571264


def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, broll="pexels", pexels_api_key="test-key")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _mp4(settings: Settings, out: Path, seconds: float = 12.0) -> bytes:
    """A 12 s portrait test pattern - longer than broll_max_clip_seconds so the trim is observable."""
    subprocess.run([settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc2=size=288x512:rate=24", "-t", str(seconds), "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, text=True)
    return out.read_bytes()


def _video(vid: int = VIDEO_ID) -> dict:
    cdn = f"https://videos.pexels.com/video-files/{vid}"
    return {"id": vid, "duration": 14, "width": 2160, "height": 3840, "video_files": [
        {"id": 1, "quality": "hd", "file_type": "video/mp4", "width": 1920, "height": 1080, "link": f"{cdn}/land.mp4"},
        {"id": 2, "quality": "uhd", "file_type": "video/mp4", "width": 2160, "height": 3840, "link": f"{cdn}/4k.mp4"},
        {"id": 3, "quality": "hd", "file_type": "video/mp4", "width": 1080, "height": 1920, "link": f"{cdn}/hd.mp4"},
        {"id": 4, "quality": "sd", "file_type": "video/mp4", "width": 540, "height": 960, "link": f"{cdn}/sd.mp4"},
        {"id": 5, "quality": "hd", "file_type": "video/webm", "width": 1080, "height": 1920, "link": f"{cdn}/hd.webm"},
    ]}


class FakePexels:
    """Minimal Pexels: /videos/search + CDN downloads. Records every request it sees."""

    def __init__(self, video_bytes: bytes, videos: list[dict] | None = None, status: int = 200):
        self.video, self.videos, self.status = video_bytes, videos if videos is not None else [_video()], status
        self.requests: list[httpx.Request] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        if req.url.host == "api.pexels.com" and req.url.path == "/videos/search":
            if req.headers.get("Authorization") != "test-key":
                return httpx.Response(401, json={"error": "Unauthorized"})
            if self.status != 200:
                return httpx.Response(self.status, json={"error": "Too many requests"})
            return httpx.Response(200, json={"page": 1, "per_page": 5, "total_results": len(self.videos),
                                             "videos": self.videos})
        if req.url.host == "videos.pexels.com":
            return httpx.Response(200, content=self.video, headers={"content-type": "video/mp4"})
        return httpx.Response(404)

    @property
    def searches(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == "/videos/search"]


def _broll(fake: FakePexels, cache_dir: Path, **kw) -> PexelsBroll:
    s = _settings(**kw)
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), base_url="https://api.pexels.com")
    return PexelsBroll(s.pexels_api_key, cache_dir, max_clip_seconds=s.broll_max_clip_seconds,
                       ffmpeg_bin=s.ffmpeg_bin(), client=client)


def test_make_broll():
    assert make_broll(_settings(broll="none")) is None
    b = make_broll(_settings())
    assert isinstance(b, PexelsBroll) and b.api_key == "test-key" and b.max_clip_seconds == 8.0
    assert b.cache_dir.name == "_broll"
    with pytest.raises(ConfigError, match="FINVID_BROLL"):
        make_broll(_settings(broll="shutterstock"))


def test_pick_prefers_portrait_under_1920():
    f = PexelsBroll.pick(_video())
    assert f["width"] == 1080 and f["height"] == 1920 and f["link"].endswith("/hd.mp4")
    v = _video()
    v["video_files"] = [x for x in v["video_files"] if x["height"] <= x["width"] or x["file_type"] != "video/mp4"]
    assert PexelsBroll.pick(v)["link"].endswith("/land.mp4")  # landscape is acceptable, compose crops
    v["video_files"] = [x for x in v["video_files"] if x["file_type"] != "video/mp4"]
    assert PexelsBroll.pick(v) is None


def test_fetch_searches_visual_downloads_portrait_and_trims(tmp_path):
    s = _settings()
    fake = FakePexels(_mp4(s, tmp_path / "src.mp4"))
    b = _broll(fake, tmp_path / "_broll")
    logs: list[str] = []
    out = b.fetch(QUERY, logs.append)

    assert out == tmp_path / "_broll" / f"{VIDEO_ID}_8s.mp4" and out.exists()
    srch = fake.searches
    assert len(srch) == 1 and srch[0].url.params["query"] == QUERY
    assert srch[0].url.params["orientation"] == "portrait" and srch[0].url.params["per_page"] == "5"
    assert srch[0].headers["Authorization"] == "test-key"
    dl = [r for r in fake.requests if r.url.host == "videos.pexels.com"]
    assert len(dl) == 1 and dl[0].url.path.endswith("/hd.mp4")  # portrait 1080x1920, not the 4K or landscape file
    assert (tmp_path / "_broll" / f"{VIDEO_ID}.mp4").exists()  # untrimmed original kept for other lengths
    assert (tmp_path / "_broll" / b.search_cache_name(QUERY)).exists()
    assert tts.probe_duration(s, out) <= 8.05
    assert "video" in subprocess.run([s.ffprobe_bin(), "-v", "error", "-show_entries", "stream=codec_type",
                                      "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert b.requests == 1 and b.downloads == 1
    assert any("downloaded video" in l for l in logs)


def test_fetch_is_cached_across_instances(tmp_path):
    s = _settings()
    fake = FakePexels(_mp4(s, tmp_path / "src.mp4"))
    b = _broll(fake, tmp_path / "_broll")
    first = b.fetch(QUERY, lambda _m: None)
    n = len(fake.requests)
    assert b.fetch(QUERY, lambda _m: None) == first and len(fake.requests) == n  # zero HTTP requests

    # a new run (new instance, same cache dir) - search result and download both cached
    b2 = _broll(fake, tmp_path / "_broll")
    assert b2.fetch(QUERY, lambda _m: None) == first
    assert len(fake.requests) == n and b2.requests == 0 and b2.downloads == 0
    assert b2.cost_entry().quantity == 0

    # a cached search whose video is already downloaded but a different trim length -> ffmpeg only
    b3 = _broll(fake, tmp_path / "_broll", broll_max_clip_seconds=4)
    p = b3.fetch(QUERY, lambda _m: None)
    assert p.name == f"{VIDEO_ID}_4s.mp4" and len(fake.requests) == n and tts.probe_duration(s, p) <= 4.05


def test_query_normalised_and_default(tmp_path):
    s = _settings()
    fake = FakePexels(_mp4(s, tmp_path / "src.mp4"))
    b = _broll(fake, tmp_path / "_broll")
    b.fetch("  mortgage   papers kitchen table ", lambda _m: None)
    assert fake.searches[-1].url.params["query"] == QUERY
    b.fetch("", lambda _m: None)
    assert fake.searches[-1].url.params["query"] == DEFAULT_QUERY


def test_empty_results_return_none_and_are_cached(tmp_path):
    s = _settings()
    fake = FakePexels(b"", videos=[])
    b = _broll(fake, tmp_path / "_broll")
    logs: list[str] = []
    assert b.fetch("xyzzy nothing", logs.append) is None
    assert b.fetch("xyzzy nothing", logs.append) is None
    assert len(fake.searches) == 1 and b.requests == 1 and b.downloads == 0
    assert any("nothing usable" in l for l in logs)
    # a video with no mp4 rendition is skipped the same way
    v = _video(); v["video_files"] = [f for f in v["video_files"] if f["file_type"] != "video/mp4"]
    fake2 = FakePexels(b"", videos=[v])
    assert _broll(fake2, tmp_path / "_broll2").fetch("q", logs.append) is None


def test_missing_or_bad_key_and_rate_limit(tmp_path):
    fake = FakePexels(b"")
    b = _broll(fake, tmp_path / "_broll", pexels_api_key=None)
    with pytest.raises(BrollError, match="PEXELS_API_KEY not set.*pexels.com/api"):
        b.fetch(QUERY, lambda _m: None)
    with pytest.raises(BrollError, match="PEXELS_API_KEY not set"):
        b.require_key()
    assert not fake.requests  # nothing sent without a key

    b = _broll(fake, tmp_path / "_broll", pexels_api_key="wrong")
    with pytest.raises(BrollError, match="401"):
        b.fetch(QUERY, lambda _m: None)

    b = _broll(FakePexels(b"", status=429), tmp_path / "_broll")
    with pytest.raises(BrollError, match="200 requests/hour"):
        b.fetch(QUERY, lambda _m: None)
    assert not list((tmp_path / "_broll").glob("search_*.json"))  # failures are not cached


def test_cost_entry_counts_requests_made(tmp_path):
    s = _settings()
    fake = FakePexels(_mp4(s, tmp_path / "src.mp4"))
    b = _broll(fake, tmp_path / "_broll")
    b.fetch("a", lambda _m: None)
    b.fetch("b", lambda _m: None)
    b.fetch("a", lambda _m: None)  # cached
    e = b.cost_entry()
    assert e.quantity == 2 == len(fake.searches) == b.requests
    assert e.provider == "pexels" and e.model == "videos/search" and e.unit == "request"
    assert e.usd == 0 and e.unit_price_usd == 0 and not e.estimated and e.stage == "s4_render"
    assert b.downloads == 1  # same Pexels id both times -> one download
    est = b.estimate(7)
    assert est[0].estimated and est[0].quantity == 7 and est[0].usd == 0
    cache = json.loads((tmp_path / "_broll" / b.search_cache_name("a")).read_text(encoding="utf-8"))
    assert cache["query"] == "a" and cache["videos"][0]["id"] == VIDEO_ID
