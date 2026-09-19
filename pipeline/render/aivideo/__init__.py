"""AI-generated opening shot for each clip ("step 4 generation").

One provider interface, several backends. `comfy` runs a local ComfyUI server (free, your GPU,
minutes per shot); cloud backends cost cents-to-dollars per shot and seconds of wall time.
Every backend goes through the same gate: only clips that survived the s3 selection +
plagiarism check get a shot, one shot per clip, budget-guarded, cached by prompt hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ...config import Settings
from ...models import CostEntry


@dataclass
class ShotResult:
    path: Path
    costs: list[CostEntry] = field(default_factory=list)
    cached: bool = False
    wall_seconds: float = 0.0
    meta: dict = field(default_factory=dict)


class AIVideoProvider(Protocol):
    name: str
    model: str

    def estimate(self, n_shots: int, seconds_each: float) -> list[CostEntry]:
        """Dry-run entries (estimated=True) for n shots; free backends report qty in GPU seconds."""

    def generate(self, prompt: str, out_mp4: Path, *, seconds: float, seed: int, log) -> ShotResult:
        """Produce a vertical mp4 of ~`seconds` at out_mp4 (no audio needed)."""


def make_provider(settings: Settings) -> AIVideoProvider | None:
    kind = (settings.ai_video or "none").lower()
    if kind == "none":
        return None
    if kind == "comfy":
        from .comfy import ComfyUIProvider

        return ComfyUIProvider.from_settings(settings)
    raise ValueError(f"unknown FINVID_AI_VIDEO={settings.ai_video!r} (none | comfy)")
