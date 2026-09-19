"""Real stock footage ("B-roll") under each script line, from a free stock library.

The AI providers in `..aivideo` generate a handful of 5 s shots per clip; this package instead
fetches one real vertical clip per line, keyed by the `visual` keywords Pass B already writes.
Same composition path (`compose_clip(shots=...)`, one scene window per line), same cost
discipline: $0, every search and download cached under data/_broll/ so re-runs make no request.
`pexels` is the only backend today; anything with a search + download endpoint plugs in the same way.
"""
from __future__ import annotations

from pathlib import Path

from ...config import Settings


class BrollError(RuntimeError):
    """Missing key, rejected key, rate limit: stop the run with a message that says what to do."""


def make_broll(settings: Settings, cache_dir: Path | None = None):
    kind = (settings.broll or "none").strip().lower()
    if kind == "none":
        return None
    if kind == "pexels":
        from .pexels import PexelsBroll

        return PexelsBroll.from_settings(settings, cache_dir=cache_dir)
    raise ValueError(f"unknown FINVID_BROLL backend {kind!r} (none | pexels)")
