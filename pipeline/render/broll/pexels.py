"""Pexels Video API backend: free stock footage, commercial use allowed, no attribution required.

    GET /videos/search?query=..&orientation=portrait&size=medium&per_page=5   (Authorization: <key>)
      -> {"videos": [{"id", "duration", "width", "height",
                      "video_files": [{"quality", "file_type", "width", "height", "link"}]}]}

Free tier: 200 requests/hour, 20,000/month. The search calls are what the limit counts, so they
are cached per query (search_<sha1>.json) and every downloaded video is cached by its Pexels id,
shared across videos. A second run over the same scripts makes zero HTTP requests.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import httpx

from ...config import DATA_DIR, Settings
from ...models import CostEntry
from . import BrollError

STAGE = "s4_render"
API_URL = "https://api.pexels.com"
CACHE_DIRNAME = "_broll"
DEFAULT_QUERY = "taiwan city apartment buildings"  # when a line has no `visual` keywords
MIN_SHORT_SIDE = 1080  # smallest rendition that still fills a 1080-wide frame
MAX_SOURCE_SECONDS = 30  # prefer sources at most this long (download size), longer ones are a fallback
MAX_HEIGHT = 1920  # 4K renditions cost download time for nothing: compose scales to 1080x1920
KEY_HINT = "PEXELS_API_KEY not set - free key at https://www.pexels.com/api/"
RATE_HINT = "Pexels free tier allows 200 requests/hour (20,000/month)"


class PexelsBroll:
    name = "pexels"
    model = "videos/search"
    unit = "request"
    unit_price_usd = 0.0

    def __init__(self, api_key: str | None, cache_dir: Path, *, max_clip_seconds: float = 8.0,
                 per_page: int = 5, ffmpeg_bin: str = "ffmpeg", client: httpx.Client | None = None):
        self.api_key = (api_key or "").strip() or None
        self.cache_dir = cache_dir
        self.max_clip_seconds = max_clip_seconds
        self.per_page = per_page
        self.ffmpeg_bin = ffmpeg_bin
        self.client = client or httpx.Client(base_url=API_URL, timeout=60.0, follow_redirects=True)
        self.requests = 0   # API calls actually made (what the rate limit counts)
        self.downloads = 0  # CDN fetches: not rate limited, cached all the same

    @classmethod
    def from_settings(cls, s: Settings, cache_dir: Path | None = None) -> "PexelsBroll":
        return cls(s.pexels_api_key, cache_dir or DATA_DIR / CACHE_DIRNAME,
                   max_clip_seconds=s.broll_max_clip_seconds, ffmpeg_bin=s.ffmpeg_bin())

    # ---- ledger --------------------------------------------------------------------------------
    def entry(self, qty: float, *, estimated: bool, note: str) -> CostEntry:
        return CostEntry(stage=STAGE, provider=self.name, model=self.model, unit=self.unit,
                         quantity=qty, unit_price_usd=self.unit_price_usd, usd=0.0,
                         estimated=estimated, note=note)

    def estimate(self, n_queries: int) -> list[CostEntry]:
        return [self.entry(n_queries, estimated=True,
                           note=f"~{n_queries} Pexels search(es), $0; cached queries make none")]

    def cost_entry(self, note: str = "") -> CostEntry:
        return self.entry(self.requests, estimated=False,
                          note=note or f"{self.requests} search(es), {self.downloads} download(s)")

    # ---- API -----------------------------------------------------------------------------------
    def require_key(self) -> None:
        if not self.api_key:
            raise BrollError(KEY_HINT)

    @staticmethod
    def search_cache_name(query: str) -> str:
        return f"search_{hashlib.sha1(query.encode('utf-8')).hexdigest()[:12]}.json"

    def search(self, query: str) -> list[dict]:
        """Candidate videos for `query`, from the per-query cache or one API call."""
        query = " ".join(query.split())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir / self.search_cache_name(query)
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8")).get("videos", [])
            except ValueError:
                pass
        self.require_key()
        r = self.client.get("/videos/search", headers={"Authorization": self.api_key},
                            params={"query": query, "orientation": "portrait", "size": "medium",
                                    "per_page": self.per_page})
        self.requests += 1
        if r.status_code == 401:
            raise BrollError(f"Pexels rejected PEXELS_API_KEY (401). {KEY_HINT}")
        if r.status_code == 429:
            raise BrollError(f"Pexels rate limit hit (429): {RATE_HINT}. Wait an hour or re-run later - "
                             f"searches and downloads cached under {self.cache_dir} cost no request.")
        if r.status_code != 200:
            raise BrollError(f"Pexels search failed ({r.status_code}): {r.text[:300]}")
        videos = r.json().get("videos", [])
        cache.write_text(json.dumps({"query": query, "videos": videos}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        return videos

    @staticmethod
    def pick(video: dict) -> dict | None:
        """The mp4 rendition to download: portrait first (landscape is acceptable - compose
        scales-and-crops to the frame), then the SMALLEST rendition whose short side is still
        >= 1080 px; we only keep 8 s of it, so a 4K master (100+ MB) is wasted bandwidth."""
        files = [f for f in video.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("link")]
        if not files:
            return None
        portrait = [f for f in files if (f.get("height") or 0) > (f.get("width") or 0)]
        pool = portrait or files
        short = lambda f: min(f.get("width") or 0, f.get("height") or 0)  # noqa: E731
        pixels = lambda f: (f.get("width") or 0) * (f.get("height") or 0)  # noqa: E731
        enough = [f for f in pool if short(f) >= MIN_SHORT_SIDE]
        return min(enough, key=pixels) if enough else max(pool, key=pixels)

    def download(self, video_id: int, link: str, log) -> Path:
        out = self.cache_dir / f"{video_id}.mp4"
        if out.exists() and out.stat().st_size > 0:
            return out
        part = out.with_suffix(".part")
        with self.client.stream("GET", link) as r:
            if r.status_code != 200:
                raise BrollError(f"Pexels download failed ({r.status_code}) for video {video_id}")
            with part.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        part.replace(out)
        self.downloads += 1
        log(f"[s4] pexels: downloaded video {video_id} ({out.stat().st_size / 1e6:.1f} MB) -> {out.name}")
        return out

    def trim(self, src: Path, video_id: int) -> Path:
        """First `max_clip_seconds` only, no audio, h264; the ping-pong loop in compose fills the rest."""
        out = self.cache_dir / f"{video_id}_{self.max_clip_seconds:g}s.mp4"
        if out.exists() and out.stat().st_size > 0:
            return out
        subprocess.run([self.ffmpeg_bin, "-y", "-v", "error", "-nostdin", "-i", str(src),
                        "-t", f"{self.max_clip_seconds:.3f}", "-an", "-c:v", "libx264", "-preset", "fast",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                       check=True, capture_output=True, text=True)
        return out

    def fetch(self, query: str, log) -> Path | None:
        """One trimmed clip for `query` (a line's `visual` keywords), or None when Pexels has nothing.
        Never raises on empty results: the caller reuses the previous scene's clip instead."""
        query = " ".join((query or "").split()) or DEFAULT_QUERY
        # shorter source videos first: we trim to a few seconds anyway, and a 60 s master is 10x the download
        for video in sorted(self.search(query), key=lambda v: (v.get("duration") or 0) > MAX_SOURCE_SECONDS):
            f = self.pick(video)
            if f is None:
                continue
            raw = self.download(video["id"], f["link"], log)
            return self.trim(raw, video["id"])
        log(f"[s4] pexels: nothing usable for {query!r}")
        return None
