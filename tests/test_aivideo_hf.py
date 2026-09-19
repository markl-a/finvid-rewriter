"""Hugging Face ZeroGPU provider against a fake gradio_client. No network."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline.config import Settings
from pipeline.render.aivideo import make_provider
from pipeline.render.aivideo.hf_space import HFSpaceError, HFSpaceProvider


def _settings(**kw) -> Settings:
    base = dict(openai_api_key=None, ai_video="hf")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _mp4(settings: Settings, out: Path) -> Path:
    subprocess.run([settings.ffmpeg_bin(), "-y", "-v", "error", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc2=size=576x1024:rate=30", "-t", "0.5", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True, text=True)
    return out


class FakeGradio:
    def __init__(self, video: Path, error: Exception | None = None):
        self.video, self.error, self.calls = video, error, []

    def predict(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return ({"video": str(self.video), "subtitles": None}, kw["seed_ui"])


def test_make_provider_hf_and_defaults():
    p = make_provider(_settings())
    assert isinstance(p, HFSpaceProvider) and p.space == "Lightricks/ltx-video-distilled" and p.token is None
    assert make_provider(_settings(hf_token="hf_x", hf_space="me/space")).token == "hf_x"


def test_hf_generate_calls_space_and_caches(tmp_path):
    s = _settings()
    fake = FakeGradio(_mp4(s, tmp_path / "space_out.mp4"))
    p = HFSpaceProvider.from_settings(s)
    p._client = fake
    out = tmp_path / "ai_05.mp4"
    logs: list[str] = []
    r = p.generate("a harbour at dawn", out, seconds=5, seed=5, log=logs.append)
    assert out.exists() and out.stat().st_size > 1000 and not r.cached
    kw = fake.calls[0]
    assert kw["api_name"] == "/text_to_video" and kw["prompt"] == "a harbour at dawn"
    assert kw["height_ui"] == 1024 and kw["width_ui"] == 576 and kw["duration_ui"] == 5 and kw["seed_ui"] == 5
    assert kw["mode"] == "text-to-video" and kw["randomize_seed"] is False
    assert r.costs[0].provider == "hf-zerogpu" and r.costs[0].usd == 0 and r.costs[0].unit == "gpu_second"
    assert (tmp_path / "ai_05.json").exists() and not (tmp_path / "ai_05.raw").exists()
    assert any("without HF_TOKEN" in l for l in logs)

    r2 = p.generate("a harbour at dawn", out, seconds=5, seed=5, log=logs.append)
    assert r2.cached and len(fake.calls) == 1

    # a different space or size is a different cache key
    p2 = HFSpaceProvider("other/space", None, width=576, height=1024, fps=24, ffmpeg_bin=s.ffmpeg_bin())
    assert p2.cache_key("a harbour at dawn", seconds=5, seed=5) != p.cache_key("a harbour at dawn", seconds=5, seed=5)


def test_hf_quota_and_generic_errors(tmp_path):
    s = _settings()
    p = HFSpaceProvider.from_settings(s)
    p._client = FakeGradio(tmp_path / "x.mp4", error=RuntimeError("You have exceeded your GPU quota (60s requested vs 12s left)"))
    with pytest.raises(HFSpaceError, match="quota exhausted"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)
    p._client = FakeGradio(tmp_path / "x.mp4", error=RuntimeError("boom"))
    with pytest.raises(HFSpaceError, match="failed: boom"):
        p.generate("x", tmp_path / "ai.mp4", seconds=5, seed=1, log=lambda _m: None)
    assert not (tmp_path / "ai.json").exists()  # nothing cached on failure


def test_hf_estimate_free():
    est = HFSpaceProvider.from_settings(_settings(hf_est_seconds=30)).estimate(6, 5)
    assert est[0].usd == 0 and est[0].quantity == 180 and est[0].estimated


def test_fallback_chain_switches_on_provider_error_only(tmp_path):
    from pipeline.render.aivideo import FallbackProvider
    from pipeline.render.aivideo.base import AIVideoError
    from pipeline.render.aivideo import ShotResult

    class Flaky:
        name, model = "flaky", "m"
        def __init__(self, fail_with): self.fail_with, self.calls = fail_with, 0
        def estimate(self, n, s): return []
        def generate(self, prompt, out, *, seconds, seed, log):
            self.calls += 1
            if self.fail_with: raise self.fail_with
            out.write_bytes(b"x"); return ShotResult(path=out)

    a, b = Flaky(AIVideoError("quota exhausted")), Flaky(None)
    chain = FallbackProvider([a, b])
    r = chain.generate("p", tmp_path / "1.mp4", seconds=5, seed=1, log=lambda m: None)
    assert r.path.exists() and a.calls == 1 and b.calls == 1 and chain.name == "flaky"
    chain.generate("p", tmp_path / "2.mp4", seconds=5, seed=2, log=lambda m: None)
    assert a.calls == 1 and b.calls == 2  # stays on the fallback, does not retry the dead one

    with pytest.raises(AIVideoError):  # last provider failing propagates
        FallbackProvider([Flaky(AIVideoError("down"))]).generate("p", tmp_path / "3.mp4", seconds=5, seed=1, log=lambda m: None)
    with pytest.raises(RuntimeError, match="bug"):  # non-provider errors are not swallowed
        FallbackProvider([Flaky(RuntimeError("bug")), b]).generate("p", tmp_path / "4.mp4", seconds=5, seed=1, log=lambda m: None)

    from pipeline.render.aivideo import make_provider
    assert isinstance(make_provider(_settings(ai_video="hf,comfy")), FallbackProvider)
    assert make_provider(_settings(ai_video="none")) is None


def test_fallback_chain_preflights_worst_case_and_never_rerenders_other_providers_shot(tmp_path):
    """QA findings: (1) estimate() must be the worst case of the remaining chain so ctx.charge()
    guards a paid tail; (2) a shot rendered by provider A must be a cache hit for the chain even
    after it fell back to B (never pay B to redo A's work)."""
    from pipeline.models import CostEntry
    from pipeline.render.aivideo import FallbackProvider, ShotResult
    from pipeline.render.aivideo.base import AIVideoError, CachedShotProvider

    class Free(CachedShotProvider):
        name, model = "free", "m"
        def __init__(self, fail=False):
            super().__init__(width=576, height=1024, fps=24, ffmpeg_bin="ffmpeg", est_seconds=10)
            self.fail, self.calls = fail, 0
        def _render(self, prompt, raw_out, *, seconds, seed, log):
            self.calls += 1
            if self.fail: raise AIVideoError("quota")
            raw_out.write_bytes(b"x"); return {}
        def _to_mp4(self, src, out): out.write_bytes(b"mp4")

    class Paid(Free):
        name, model, unit, unit_price_usd = "paid", "p", "video", 0.27
        def estimate(self, n, s): return [self.entry(n, estimated=True, note="paid")]
        def quantity(self, *, wall, seconds): return 1

    a, b = Free(), Paid()
    chain = FallbackProvider([a, b])
    est = chain.estimate(3, 5)
    assert sum(c.usd for c in est) == pytest.approx(0.81) and "worst case" in est[0].note  # (1)

    out = tmp_path / "ai_01.mp4"
    r = chain.generate("p", out, seconds=5, seed=1, log=lambda m: None)
    assert not r.cached and a.calls == 1 and b.calls == 0
    a.fail = True                                  # free tier exhausted before a second run
    r2 = chain.generate("p", out, seconds=5, seed=1, log=lambda m: None)
    assert r2.cached and b.calls == 0 and "free" in r2.costs[0].note  # (2): A's shot reused, B not billed
    r3 = chain.generate("p", tmp_path / "ai_02.mp4", seconds=5, seed=2, log=lambda m: None)
    assert b.calls == 1 and r3.costs[0].usd == 0.27  # only the genuinely missing shot goes to the paid tail
    assert sum(c.usd for c in chain.estimate(1, 5)) == 0.27  # now on the paid provider, estimate says so


def test_kling_jwt_and_flow(tmp_path, monkeypatch):
    import base64, json as _json
    import httpx
    from pipeline.render.aivideo.kling import KlingError, KlingProvider, sign_jwt
    tok = sign_jwt("AK", "SK", now=1_700_000_000)
    h, p, sig = tok.split(".")
    pad = lambda x: x + "=" * (-len(x) % 4)  # noqa: E731
    assert _json.loads(base64.urlsafe_b64decode(pad(h))) == {"alg": "HS256", "typ": "JWT"}
    assert _json.loads(base64.urlsafe_b64decode(pad(p))) == {"iss": "AK", "exp": 1_700_001_800, "nbf": 1_699_999_995}

    video = _mp4(_settings(), tmp_path / "k.mp4").read_bytes()
    calls = {"post": 0, "poll": 0, "auth": set()}
    def handler(req: httpx.Request) -> httpx.Response:
        calls["auth"].add(req.headers.get("authorization", "")[:7])
        if req.url.path == "/v1/videos/text2video" and req.method == "POST":
            calls["post"] += 1
            body = _json.loads(req.content)
            assert body["aspect_ratio"] == "9:16" and body["duration"] == "5" and body["model_name"] == "kling-v1"
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "t1"}})
        if req.url.path.startswith("/v1/videos/text2video/"):
            calls["poll"] += 1
            status = "processing" if calls["poll"] < 2 else "succeed"
            return httpx.Response(200, json={"code": 0, "data": {"task_status": status,
                                                                  "task_result": {"videos": [{"url": "http://cdn.test/v.mp4"}]}}})
        if req.url.host == "cdn.test":
            return httpx.Response(200, content=video)
        return httpx.Response(404)
    monkeypatch.setattr("pipeline.render.aivideo.kling.time.sleep", lambda _s: None)
    kp = KlingProvider("AK", "SK", width=576, height=1024, fps=24, ffmpeg_bin=_settings().ffmpeg_bin(),
                       client=httpx.Client(base_url="http://kling.test", transport=httpx.MockTransport(handler)))
    r = kp.generate("harbour", tmp_path / "ai_k.mp4", seconds=5, seed=1, log=lambda m: None)
    assert r.path.exists() and calls["post"] == 1 and calls["poll"] == 2 and calls["auth"] == {"Bearer ", ""}  # signed API calls, bare CDN download
    assert r.costs[0].usd == 0.18 and r.costs[0].unit == "video" and not r.costs[0].estimated
    assert kp.estimate(3, 5)[0].usd == pytest.approx(0.54)
    with pytest.raises(KlingError, match="KLING_ACCESS_KEY"):
        KlingProvider(None, None, width=576, height=1024, fps=24).headers()
    with pytest.raises(KlingError, match="no price on file"):
        KlingProvider("a", "b", model="kling-v9", width=576, height=1024, fps=24)
