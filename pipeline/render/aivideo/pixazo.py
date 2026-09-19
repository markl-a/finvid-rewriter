"""Pixazo backend: their hosted LTX-Video text-to-video endpoint (free during preview, no card).

    POST https://gateway.pixazo.ai/ltx-video/v1/text-to-video     header Ocp-Apim-Subscription-Key
         -> {"request_id", "status": "QUEUED", "polling_url"}    (or {"output": {...}} straight away)
    GET  <polling_url>   same header, until status == "COMPLETED" -> {"output": {"media_url": [".../x.mp4"]}}
    GET  media_url       download

Free tier: 60 requests/min, LTX parameters (num_frames / width / height / frame_rate / steps / cfg /
negative). Paid tiers take resolution / duration / aspect_ratio instead - `build_body()` is the one
place to change for those. Same LTX-Video family as `hf` and `comfy`, so the shots look alike; the
difference is where the GPU minutes come from: Pixazo's preview quota instead of ZeroGPU's or yours.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from ...config import Settings
from .base import NEGATIVE, AIVideoError, CachedShotProvider

ENDPOINT = "https://gateway.pixazo.ai/ltx-video/v1/text-to-video"
KEY_HINT = "PIXAZO_API_KEY not set - free key at https://www.pixazo.ai (LTX free tier)"


class PixazoError(AIVideoError):
    pass


class PixazoProvider(CachedShotProvider):
    name = "pixazo"
    model = "ltx-video"
    unit = "second"  # video seconds requested; $0 on the free tier, wall time goes in the note
    unit_price_usd = 0.0

    def __init__(self, api_key: str | None, *, width: int, height: int, fps: int, steps: int = 8, cfg: float = 1.0,
                 timeout_sec: float = 600, est_seconds: float = 60.0, ffmpeg_bin: str = "ffmpeg",
                 client: httpx.Client | None = None):
        super().__init__(width=width, height=height, fps=fps, ffmpeg_bin=ffmpeg_bin, est_seconds=est_seconds)
        self.api_key = (api_key or "").strip() or None
        self.steps, self.cfg = steps, cfg
        self.timeout_sec = timeout_sec
        self.poll_sec = 3.0
        self.client = client or httpx.Client(timeout=120.0, follow_redirects=True)

    @classmethod
    def from_settings(cls, s: Settings) -> "PixazoProvider":
        return cls(s.pixazo_api_key, width=s.ai_shot_width, height=s.ai_shot_height, fps=s.ai_shot_fps,
                   timeout_sec=s.pixazo_timeout_sec, est_seconds=s.pixazo_est_seconds, ffmpeg_bin=s.ffmpeg_bin())

    # ------------------------------------------------------------------ request
    def headers(self) -> dict:
        if not self.api_key:
            raise PixazoError(KEY_HINT)
        return {"Ocp-Apim-Subscription-Key": self.api_key}

    def frames_for(self, seconds: float) -> int:
        n = int(round(seconds * self.fps))
        return max(9, ((n - 1 + 7) // 8) * 8 + 1)  # LTX wants 8k+1 frames; round up

    def build_body(self, prompt: str, *, seconds: float, seed: int) -> dict:
        """Free-tier (LTX-native) body. Paid tiers: replace num_frames/width/height/frame_rate with
        resolution/duration/aspect_ratio here and nothing else has to change."""
        return {"prompt": prompt, "negative": NEGATIVE, "width": self.width, "height": self.height,
                "num_frames": self.frames_for(seconds), "frame_rate": self.fps,
                "steps": self.steps, "cfg": self.cfg, "seed": seed}

    def cache_fields(self, prompt: str, *, seconds: float, seed: int) -> dict:
        return {"provider": self.name, "model": self.model, "body": self.build_body(prompt, seconds=seconds, seed=seed)}

    # ------------------------------------------------------------------ ledger
    def estimate(self, n_shots: int, seconds_each: float):
        return [self.entry(n_shots * seconds_each, estimated=True,
                           note=f"{n_shots} AI shot(s) x {seconds_each:.0f}s via {self.name} free tier, "
                                f"~{self.est_seconds:.0f}s wall each")]

    def quantity(self, *, wall: float, seconds: float) -> float:
        return seconds

    # ------------------------------------------------------------------ generate
    def _render(self, prompt: str, raw_out: Path, *, seconds: float, seed: int, log) -> dict:
        headers = self.headers()
        body = self.build_body(prompt, seconds=seconds, seed=seed)
        log(f"[s4] pixazo: text-to-video {self.width}x{self.height} {body['num_frames']} frames "
            f"@{self.fps}fps, {self.steps} steps")
        r = self._call("POST", ENDPOINT, headers=headers, json=body)
        data = r.json()
        polls = 0
        if not data.get("output"):  # async: poll until COMPLETED
            data = self._wait(data, headers)
            polls = data.pop("_polls", 0)
        url = self._media_url(data)
        dl = self._call("GET", url, headers={})
        raw_out.write_bytes(dl.content)
        return {"request_id": data.get("request_id"), "polls": polls, "media_url": url}

    def _call(self, method: str, url: str, **kw) -> httpx.Response:
        try:
            r = self.client.request(method, url, **kw)
        except httpx.HTTPError as e:
            raise PixazoError(f"Pixazo not reachable ({url}): {e}") from e
        if r.status_code in (401, 403):
            raise PixazoError(f"Pixazo rejected PIXAZO_API_KEY ({r.status_code}): {r.text[:200]}. "
                              f"Check the key at https://www.pixazo.ai.")
        if r.status_code == 429:
            raise PixazoError("Pixazo rate limit hit (free tier: 60 requests/min). Shots already rendered "
                              "are cached; re-run in a minute or chain another backend in FINVID_AI_VIDEO.")
        if r.status_code >= 400:
            raise PixazoError(f"Pixazo {method} {url} failed ({r.status_code}): {r.text[:500]}")
        return r

    def _wait(self, first: dict, headers: dict) -> dict:
        polling_url = first.get("polling_url")
        if not polling_url:
            raise PixazoError(f"Pixazo returned neither output nor polling_url: {first!r}"[:400])
        deadline = time.time() + self.timeout_sec
        polls = 0
        while time.time() < deadline:
            time.sleep(self.poll_sec)
            polls += 1
            data = self._call("GET", polling_url, headers=headers).json()
            status = str(data.get("status", "")).upper()
            if status == "COMPLETED" or data.get("output"):
                data.setdefault("request_id", first.get("request_id"))
                data["_polls"] = polls
                return data
            if status in ("FAILED", "ERROR", "CANCELLED"):
                detail = data.get("error") or data.get("message") or data.get("detail") or status
                raise PixazoError(f"Pixazo generation failed ({status}): {str(detail)[:400]}")
        raise PixazoError(f"Pixazo did not finish within {self.timeout_sec:.0f}s "
                          f"(request {first.get('request_id')}); raise FINVID_PIXAZO_TIMEOUT_SEC or retry")

    @staticmethod
    def _media_url(data: dict) -> str:
        out = data.get("output") or {}
        urls = out.get("media_url") or out.get("video_url") or []
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            raise PixazoError(f"Pixazo completed but returned no media_url: {out!r}"[:400])
        return urls[0]
