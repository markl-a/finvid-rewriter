"""MiniMax Hailuo backend: the one paid option, so the $0-but-slow vs $0.27-in-20-s trade-off in the
ledger is a real comparison and not a guess.

    POST https://api.minimax.io/v1/video_generation                Authorization: Bearer MINIMAX_API_KEY
         {"model", "prompt", "duration": 6|10, "resolution", "prompt_optimizer"} -> {"task_id", "base_resp"}
    GET  https://api.minimax.io/v1/query/video_generation?task_id=  -> {"status": Preparing|Queueing|Processing|
                                                                        Success|Fail, "file_id"}
    GET  https://api.minimax.io/v1/files/retrieve?file_id=          -> {"file": {"download_url"}}
    GET  download_url                                                download

Every response carries base_resp.status_code (0 = ok); anything else becomes a MiniMaxError with the
vendor's status_msg. Billing is per video, not per second: the price table below is what the ledger
charges, and `estimate()` returns the same price so ctx.charge() in s4 pre-flights it against
FINVID_MAX_BUDGET_USD before the call - the same guard every OpenAI call goes through.

Limitation: Hailuo text-to-video outputs 16:9 (no documented aspect_ratio parameter). compose.py
scales-to-cover and center-crops to 9:16, so the sides of each shot are lost; prompts that keep the
subject centred work best.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from ...config import Settings
from .base import AIVideoError, CachedShotProvider

BASE_URL = "https://api.minimax.io/v1"
KEY_HINT = "MINIMAX_API_KEY not set - paid key at https://platform.minimax.io (Hailuo video)"

# USD per generated video, keyed by (model, resolution, duration_seconds). MiniMax sells "points"
# in packages; 1 point ~= $0.27 at the package price seen 2026-09-19 on platform.minimax.io/pricing.
# Add a row when you enable a combination that is not here: without a price the budget guard
# cannot protect you, so the provider refuses to start instead of charging blind.
PRICES_USD: dict[tuple[str, str, int], float] = {
    ("MiniMax-Hailuo-02", "512P", 6): 0.08,
    ("MiniMax-Hailuo-02", "768P", 6): 0.27,
    ("MiniMax-Hailuo-02", "1080P", 6): 0.54,
}


class MiniMaxError(AIVideoError):
    pass


def duration_for(seconds: float) -> int:
    """MiniMax accepts 6 or 10 s only: 6 if the request fits, else 10."""
    return 6 if seconds <= 6 else 10


def price_for(model: str, resolution: str, duration: int) -> float:
    key = (model, resolution, duration)
    if key in PRICES_USD:
        return PRICES_USD[key]
    known = ", ".join(f"{m} {r} {d}s" for m, r, d in PRICES_USD)
    raise MiniMaxError(f"no price on file for MiniMax {model} {resolution} {duration}s; the budget guard needs one. "
                       f"Known: {known}. Add a row to PRICES_USD in pipeline/render/aivideo/minimax.py "
                       f"(check https://platform.minimax.io/pricing) or change FINVID_MINIMAX_MODEL/RESOLUTION.")


class MiniMaxProvider(CachedShotProvider):
    name = "minimax"
    unit = "video"  # one generation = one unit; unit_price_usd is the per-video price from PRICES_USD

    def __init__(self, api_key: str | None, *, model: str = "MiniMax-Hailuo-02", resolution: str = "768P",
                 width: int, height: int, fps: int, seconds: float = 5.0, timeout_sec: float = 900,
                 est_seconds: float = 40.0, ffmpeg_bin: str = "ffmpeg", client: httpx.Client | None = None):
        super().__init__(width=width, height=height, fps=fps, ffmpeg_bin=ffmpeg_bin, est_seconds=est_seconds)
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.resolution = resolution
        self.duration = duration_for(seconds)
        self.unit_price_usd = price_for(model, resolution, self.duration)
        self.timeout_sec = timeout_sec
        self.poll_sec = 5.0
        self.client = client or httpx.Client(base_url=BASE_URL, timeout=120.0, follow_redirects=True)

    @classmethod
    def from_settings(cls, s: Settings) -> "MiniMaxProvider":
        return cls(s.minimax_api_key, model=s.minimax_model, resolution=s.minimax_resolution,
                   width=s.ai_shot_width, height=s.ai_shot_height, fps=s.ai_shot_fps, seconds=s.ai_shot_seconds,
                   timeout_sec=s.minimax_timeout_sec, ffmpeg_bin=s.ffmpeg_bin())

    # ------------------------------------------------------------------ request
    def headers(self) -> dict:
        if not self.api_key:
            raise MiniMaxError(KEY_HINT)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def build_body(self, prompt: str, *, seconds: float) -> dict:
        return {"model": self.model, "prompt": prompt, "duration": duration_for(seconds),
                "resolution": self.resolution, "prompt_optimizer": True}

    def cache_fields(self, prompt: str, *, seconds: float, seed: int) -> dict:
        # no seed parameter in the API: the request body is the whole key (seed still varies our prompt choice)
        return {"provider": self.name, "body": self.build_body(prompt, seconds=seconds), "seed": seed}

    # ------------------------------------------------------------------ ledger
    def estimate(self, n_shots: int, seconds_each: float):
        return [self.entry(n_shots, estimated=True,
                           note=f"{n_shots} AI shot(s) x {duration_for(seconds_each)}s {self.resolution} via "
                                f"{self.name}/{self.model} at ${self.unit_price_usd:.2f}/video (paid)")]

    def quantity(self, *, wall: float, seconds: float) -> float:
        return 1

    # ------------------------------------------------------------------ generate
    def _render(self, prompt: str, raw_out: Path, *, seconds: float, seed: int, log) -> dict:
        headers = self.headers()
        body = self.build_body(prompt, seconds=seconds)
        log(f"[s4] minimax: {self.model} {self.resolution} {body['duration']}s (${self.unit_price_usd:.2f}, "
            f"16:9 output, cropped to 9:16 at compose)")
        data = self._call("POST", "/video_generation", headers=headers, json=body)
        task_id = data.get("task_id")
        if not task_id:
            raise MiniMaxError(f"MiniMax accepted the request but returned no task_id: {data!r}"[:400])
        file_id, polls = self._wait(task_id, headers)
        meta = self._call("GET", "/files/retrieve", headers=headers, params={"file_id": file_id})
        url = (meta.get("file") or {}).get("download_url")
        if not url:
            raise MiniMaxError(f"MiniMax file {file_id} has no download_url: {meta!r}"[:400])
        dl = self._raw("GET", url, headers={})
        raw_out.write_bytes(dl.content)
        return {"task_id": task_id, "file_id": file_id, "polls": polls, "duration": body["duration"],
                "resolution": self.resolution, "aspect": "16:9 (cropped to 9:16 at compose)"}

    def _raw(self, method: str, url: str, **kw) -> httpx.Response:
        try:
            r = self.client.request(method, url, **kw)
        except httpx.HTTPError as e:
            raise MiniMaxError(f"MiniMax not reachable ({url}): {e}") from e
        if r.status_code in (401, 403):
            raise MiniMaxError(f"MiniMax rejected MINIMAX_API_KEY ({r.status_code}): {r.text[:200]}. "
                               f"Check the key at https://platform.minimax.io.")
        if r.status_code == 429:
            raise MiniMaxError("MiniMax rate limit hit; shots already rendered are cached, retry in a minute.")
        if r.status_code >= 400:
            raise MiniMaxError(f"MiniMax {method} {url} failed ({r.status_code}): {r.text[:500]}")
        return r

    def _call(self, method: str, url: str, **kw) -> dict:
        """JSON call with MiniMax's in-body status: base_resp.status_code != 0 is an error."""
        data = self._raw(method, url, **kw).json()
        resp = data.get("base_resp") or {}
        code = resp.get("status_code", 0)
        if code:
            msg = resp.get("status_msg") or "unknown"
            hint = ""
            if code == 1004 or "auth" in str(msg).lower() or "token" in str(msg).lower():
                hint = " (check MINIMAX_API_KEY)"
            elif code == 1008 or "balance" in str(msg).lower() or "insufficient" in str(msg).lower():
                hint = " (top up at https://platform.minimax.io or switch FINVID_AI_VIDEO)"
            raise MiniMaxError(f"MiniMax error {code}: {msg}{hint}")
        return data

    def _wait(self, task_id: str, headers: dict) -> tuple[str, int]:
        deadline = time.time() + self.timeout_sec
        polls = 0
        while time.time() < deadline:
            time.sleep(self.poll_sec)
            polls += 1
            data = self._call("GET", "/query/video_generation", headers=headers, params={"task_id": task_id})
            status = str(data.get("status", ""))
            if status == "Success":
                file_id = data.get("file_id")
                if not file_id:
                    raise MiniMaxError(f"MiniMax task {task_id} succeeded without a file_id: {data!r}"[:400])
                return file_id, polls
            if status == "Fail":
                detail = (data.get("base_resp") or {}).get("status_msg") or data.get("error") or "no detail"
                raise MiniMaxError(f"MiniMax task {task_id} failed: {detail}")
        raise MiniMaxError(f"MiniMax task {task_id} did not finish within {self.timeout_sec:.0f}s; "
                           f"raise FINVID_MINIMAX_TIMEOUT_SEC or retry (the task is not re-billed by us)")
