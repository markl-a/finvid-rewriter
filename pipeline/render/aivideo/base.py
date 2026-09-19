"""Shared plumbing for AI-shot providers: prompt-hash cache + sidecar, and the final h264 pass.

A provider only implements `_render(prompt, raw_out, seconds, seed, log) -> meta`; everything
around it (skip if the same prompt was already rendered, normalise to our fps/codec, write the
.json sidecar, build the ledger row) is identical whether the shot came from a local GPU or a
free cloud tier.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from ...models import CostEntry
from . import ShotResult

STAGE = "s4_render"
NEGATIVE = ("low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
            "text, watermark, logo, subtitles, blurry")


class AIVideoError(RuntimeError):
    pass


class CachedShotProvider:
    name = "base"
    model = ""
    unit = "gpu_second"
    unit_price_usd = 0.0

    def __init__(self, *, width: int, height: int, fps: int, ffmpeg_bin: str, est_seconds: float):
        self.width, self.height, self.fps = width, height, fps
        self.ffmpeg_bin = ffmpeg_bin
        self.est_seconds = est_seconds

    # ---- what subclasses provide -------------------------------------------------------------
    def cache_fields(self, prompt: str, *, seconds: float, seed: int) -> dict:
        """Everything that changes the output. Subclasses extend with model/workflow params."""
        return {"provider": self.name, "model": self.model, "prompt": prompt, "seconds": seconds,
                "seed": seed, "w": self.width, "h": self.height, "fps": self.fps}

    def _render(self, prompt: str, raw_out: Path, *, seconds: float, seed: int, log) -> dict:
        raise NotImplementedError

    # ---- ledger --------------------------------------------------------------------------------
    def entry(self, qty: float, *, estimated: bool, note: str) -> CostEntry:
        return CostEntry(stage=STAGE, provider=self.name, model=self.model, unit=self.unit,
                         quantity=round(qty, 1), unit_price_usd=self.unit_price_usd,
                         usd=round(qty * self.unit_price_usd, 6), estimated=estimated, note=note)

    def estimate(self, n_shots: int, seconds_each: float) -> list[CostEntry]:
        return [self.entry(self.est_seconds * n_shots, estimated=True,
                           note=f"{n_shots} AI shot(s) x ~{seconds_each:.0f}s via {self.name}, "
                                f"~{self.est_seconds:.0f}s each")]

    # ---- generate with cache -------------------------------------------------------------------
    def cache_key(self, prompt: str, *, seconds: float, seed: int) -> str:
        raw = json.dumps(self.cache_fields(prompt, seconds=seconds, seed=seed), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def generate(self, prompt: str, out_mp4: Path, *, seconds: float, seed: int, log) -> ShotResult:
        key = self.cache_key(prompt, seconds=seconds, seed=seed)
        side = out_mp4.with_suffix(".json")
        if out_mp4.exists() and side.exists():
            try:
                if json.loads(side.read_text(encoding="utf-8")).get("key") == key:
                    return ShotResult(path=out_mp4, cached=True,
                                      costs=[self.entry(0, estimated=False, note=f"{out_mp4.name}: cache hit")])
            except ValueError:
                pass
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        raw = out_mp4.parent / f"{out_mp4.stem}.raw"
        t0 = time.time()
        meta = self._render(prompt, raw, seconds=seconds, seed=seed, log=log)
        self._to_mp4(raw, out_mp4)
        raw.unlink(missing_ok=True)
        wall = time.time() - t0
        side.write_text(json.dumps({"key": key, "provider": self.name, "model": self.model, "prompt": prompt,
                                    "seconds": seconds, "seed": seed, "wall_seconds": round(wall, 1), **meta},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        where = meta.get("device") or meta.get("space") or self.name
        return ShotResult(path=out_mp4, wall_seconds=wall, meta=meta,
                          costs=[self.entry(wall, estimated=False, note=f"{out_mp4.name}: {wall:.0f}s on {where}")])

    def _to_mp4(self, src: Path, out_mp4: Path) -> None:
        """Whatever came back (webm/mp4/webp, any fps) -> h264 mp4 at our fps, no audio."""
        subprocess.run([self.ffmpeg_bin, "-y", "-v", "error", "-nostdin", "-i", str(src),
                        "-an", "-r", str(self.fps), "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)],
                       check=True, capture_output=True, text=True)
