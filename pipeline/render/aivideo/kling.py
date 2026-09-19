"""Kling (Kuaishou) backend: the second paid option, native 9:16 output.

Auth is not a static key: Kling issues an Access Key + Secret Key pair (kling.ai/dev/api-key) and
every request carries a 30-minute JWT signed with the secret (HS256, iss = access key). We sign
it with the standard library so no extra dependency is needed.

    POST {base}/v1/videos/text2video   Authorization: Bearer <jwt>
         {"model_name", "prompt", "negative_prompt", "cfg_scale", "mode", "aspect_ratio": "9:16",
          "duration": "5"|"10"}                       -> {"code": 0, "data": {"task_id"}}
    GET  {base}/v1/videos/text2video/<task_id>       -> {"data": {"task_status": submitted|processing|succeed|failed,
                                                                  "task_result": {"videos": [{"url"}]}}}
Per-video prices below are what the ledger charges and what the budget guard pre-flights.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx

from ...config import Settings
from .base import NEGATIVE, AIVideoError, CachedShotProvider

BASE_URL = "https://api-singapore.klingai.com"  # international endpoint; mainland accounts use api.klingai.com
KEY_HINT = ("KLING_ACCESS_KEY / KLING_SECRET_KEY not set - create the pair at https://kling.ai/dev/api-key "
            "(API use is paid per video)")

# USD per generated video, keyed by (model_name, mode, duration_seconds). Third-party price lists seen
# 2026-09-19 (Segmind/Renderful); Kling bills in prepaid "resource units", so check kling.ai/dev/pricing
# before relying on these. Missing combination -> the provider refuses to start (see minimax.py).
PRICES_USD: dict[tuple[str, str, int], float] = {
    ("kling-v1", "std", 5): 0.18,
    ("kling-v1", "std", 10): 0.36,
    ("kling-v2-5-turbo", "std", 5): 0.31,
    ("kling-v2-5-turbo", "std", 10): 0.62,
}


class KlingError(AIVideoError):
    pass


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_jwt(access_key: str, secret_key: str, *, now: float | None = None, ttl_sec: int = 1800) -> str:
    """Kling's documented token: HS256, {"iss": ak, "exp": now+1800, "nbf": now-5}."""
    now = int(now if now is not None else time.time())
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps({"iss": access_key, "exp": now + ttl_sec, "nbf": now - 5},
                                 separators=(",", ":")).encode())
    sig = hmac.new(secret_key.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


def duration_for(seconds: float) -> int:
    return 5 if seconds <= 5 else 10


def price_for(model: str, mode: str, duration: int) -> float:
    key = (model, mode, duration)
    if key in PRICES_USD:
        return PRICES_USD[key]
    known = ", ".join(f"{m} {md} {d}s" for m, md, d in PRICES_USD)
    raise KlingError(f"no price on file for Kling {model} {mode} {duration}s; the budget guard needs one. "
                     f"Known: {known}. Add a row to PRICES_USD in pipeline/render/aivideo/kling.py "
                     f"(https://kling.ai/dev/pricing) or change FINVID_KLING_MODEL/MODE.")


class KlingProvider(CachedShotProvider):
    name = "kling"
    unit = "video"

    def __init__(self, access_key: str | None, secret_key: str | None, *, model: str = "kling-v1",
                 mode: str = "std", width: int, height: int, fps: int, seconds: float = 5.0,
                 timeout_sec: float = 900, est_seconds: float = 120.0, base_url: str = BASE_URL,
                 ffmpeg_bin: str = "ffmpeg", client: httpx.Client | None = None):
        super().__init__(width=width, height=height, fps=fps, ffmpeg_bin=ffmpeg_bin, est_seconds=est_seconds)
        self.access_key = (access_key or "").strip() or None
        self.secret_key = (secret_key or "").strip() or None
        self.model, self.mode = model, mode
        self.duration = duration_for(seconds)
        self.unit_price_usd = price_for(model, mode, self.duration)
        self.timeout_sec = timeout_sec
        self.poll_sec = 5.0
        self.client = client or httpx.Client(base_url=base_url, timeout=120.0, follow_redirects=True)

    @classmethod
    def from_settings(cls, s: Settings) -> "KlingProvider":
        return cls(s.kling_access_key, s.kling_secret_key, model=s.kling_model, mode=s.kling_mode,
                   width=s.ai_shot_width, height=s.ai_shot_height, fps=s.ai_shot_fps, seconds=s.ai_shot_seconds,
                   timeout_sec=s.kling_timeout_sec, base_url=s.kling_base_url, ffmpeg_bin=s.ffmpeg_bin())

    # ------------------------------------------------------------------ request
    def headers(self) -> dict:
        if not (self.access_key and self.secret_key):
            raise KlingError(KEY_HINT)
        return {"Authorization": f"Bearer {sign_jwt(self.access_key, self.secret_key)}",
                "Content-Type": "application/json"}

    def build_body(self, prompt: str, *, seconds: float) -> dict:
        return {"model_name": self.model, "prompt": prompt[:2500], "negative_prompt": NEGATIVE[:2500],
                "cfg_scale": 0.5, "mode": self.mode, "aspect_ratio": "9:16",
                "duration": str(duration_for(seconds))}

    def cache_fields(self, prompt: str, *, seconds: float, seed: int) -> dict:
        return {"provider": self.name, "body": self.build_body(prompt, seconds=seconds), "seed": seed}

    # ------------------------------------------------------------------ ledger
    def estimate(self, n_shots: int, seconds_each: float):
        return [self.entry(n_shots, estimated=True,
                           note=f"{n_shots} AI shot(s) x {duration_for(seconds_each)}s {self.mode} via "
                                f"{self.name}/{self.model} at ${self.unit_price_usd:.2f}/video (paid)")]

    def quantity(self, *, wall: float, seconds: float) -> float:
        return 1

    # ------------------------------------------------------------------ generate
    def _render(self, prompt: str, raw_out: Path, *, seconds: float, seed: int, log) -> dict:
        body = self.build_body(prompt, seconds=seconds)
        log(f"[s4] kling: {self.model} {self.mode} {body['duration']}s 9:16 (${self.unit_price_usd:.2f})")
        data = self._call("POST", "/v1/videos/text2video", json=body)
        task_id = (data.get("data") or {}).get("task_id")
        if not task_id:
            raise KlingError(f"Kling accepted the request but returned no task_id: {data!r}"[:400])
        url, polls = self._wait(task_id)
        dl = self._raw("GET", url, headers={})
        raw_out.write_bytes(dl.content)
        return {"task_id": task_id, "polls": polls, "duration": body["duration"], "mode": self.mode}

    def _raw(self, method: str, url: str, **kw) -> httpx.Response:
        try:
            r = self.client.request(method, url, **kw)
        except httpx.HTTPError as e:
            raise KlingError(f"Kling not reachable ({url}): {e}") from e
        if r.status_code in (401, 403):
            raise KlingError(f"Kling rejected the signed token ({r.status_code}): {r.text[:200]}. "
                             f"Check KLING_ACCESS_KEY / KLING_SECRET_KEY and the base URL ({self.client.base_url}).")
        if r.status_code == 429:
            raise KlingError("Kling rate/concurrency limit hit; rendered shots are cached, retry shortly.")
        if r.status_code >= 400:
            raise KlingError(f"Kling {method} {url} failed ({r.status_code}): {r.text[:500]}")
        return r

    def _call(self, method: str, path: str, **kw) -> dict:
        r = self._raw(method, path, headers=self.headers(), **kw)
        try:
            data = r.json()
        except ValueError as e:
            raise KlingError(f"Kling returned non-JSON for {path}: {r.text[:200]}") from e
        if data.get("code", 0) != 0:
            code, msg = data.get("code"), data.get("message", "")
            hint = " (account balance / resource package?)" if code in (1102, 1103, 1120) else ""
            raise KlingError(f"Kling error {code}: {msg}{hint}")
        return data

    def _wait(self, task_id: str) -> tuple[str, int]:
        deadline = time.time() + self.timeout_sec
        polls = 0
        while time.time() < deadline:
            data = self._call("GET", f"/v1/videos/text2video/{task_id}")
            polls += 1
            d = data.get("data") or {}
            status = d.get("task_status")
            if status == "succeed":
                videos = (d.get("task_result") or {}).get("videos") or []
                url = videos[0].get("url") if videos else None
                if not url:
                    raise KlingError(f"Kling task {task_id} succeeded without a video url: {d!r}"[:400])
                return url, polls
            if status == "failed":
                raise KlingError(f"Kling task {task_id} failed: {d.get('task_status_msg', 'no detail')}")
            time.sleep(self.poll_sec)
        raise KlingError(f"Kling task {task_id} did not finish within {self.timeout_sec:.0f}s")
