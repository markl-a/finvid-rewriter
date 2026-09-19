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
