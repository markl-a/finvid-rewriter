"""AI-generated opening shot for each clip ("step 4 generation").

One provider interface, several backends. `comfy` runs a local ComfyUI server (free, your GPU,
~90 s per shot); `hf` calls a public Hugging Face ZeroGPU Space (free daily quota, ~25 s per
shot, no key needed); `pixazo` calls Pixazo's hosted LTX endpoint (free preview tier, free key,
~60 s per shot); `minimax` is MiniMax Hailuo, the one paid backend ($0.27 per 768P shot, ~20-60 s),
kept so the cost comparison in the ledger is real. Any of them can be chained with commas
(`hf,pixazo,comfy`): free quotas first, then the local GPU, and a paid API only if you list it.
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


class FallbackProvider:
    """`FINVID_AI_VIDEO=hf,comfy`: try each backend in order; a provider-level failure (quota
    exhausted, server unreachable, Space down) moves on to the next one and stays there. Bugs and
    ffmpeg errors are not swallowed. Free-first ordering is the cost-aware default: spend the free
    cloud quota, then use the local GPU, and only a paid API if you listed one."""

    def __init__(self, providers: list):
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self.providers = providers
        self.active = 0

    @property
    def name(self) -> str:
        return self.providers[self.active].name

    @property
    def model(self) -> str:
        return self.providers[self.active].model

    def estimate(self, n_shots: int, seconds_each: float):
        return self.providers[self.active].estimate(n_shots, seconds_each)

    def generate(self, prompt: str, out_mp4: Path, *, seconds: float, seed: int, log) -> ShotResult:
        from .base import AIVideoError

        while True:
            p = self.providers[self.active]
            try:
                return p.generate(prompt, out_mp4, seconds=seconds, seed=seed, log=log)
            except AIVideoError as e:
                if self.active + 1 >= len(self.providers):
                    raise
                nxt = self.providers[self.active + 1]
                log(f"[s4] {p.name} unavailable ({str(e).splitlines()[0][:120]}) -> falling back to {nxt.name}")
                self.active += 1


def _single(kind: str, settings: Settings):
    if kind == "comfy":
        from .comfy import ComfyUIProvider

        return ComfyUIProvider.from_settings(settings)
    if kind == "hf":
        from .hf_space import HFSpaceProvider

        return HFSpaceProvider.from_settings(settings)
    if kind == "pixazo":
        from .pixazo import PixazoProvider

        return PixazoProvider.from_settings(settings)
    if kind == "minimax":
        from .minimax import MiniMaxProvider

        return MiniMaxProvider.from_settings(settings)
    raise ValueError(f"unknown FINVID_AI_VIDEO backend {kind!r} "
                     f"(none | comfy | hf | pixazo | minimax, comma-separated for fallback)")


def make_provider(settings: Settings) -> AIVideoProvider | None:
    kinds = [k.strip().lower() for k in (settings.ai_video or "none").split(",") if k.strip()]
    if not kinds or kinds == ["none"]:
        return None
    providers = [_single(k, settings) for k in kinds if k != "none"]
    return providers[0] if len(providers) == 1 else FallbackProvider(providers)
