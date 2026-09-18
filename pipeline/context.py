"""RunContext + the stage runner that enforces caching, dry-run and the budget guard."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import DATA_DIR, Settings
from .manifest import Manifest, StageResult


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunContext:
    settings: Settings
    video_url: str
    video_id: str
    workdir: Path
    manifest: Manifest
    dry_run: bool = False
    force: bool = False
    max_clips: int = 3
    log: Callable[[str], None] = print
    # accumulated in this run only (cached stages don't count)
    spent_usd: float = 0.0
    stages_run: list[str] = field(default_factory=list)
    stages_cached: list[str] = field(default_factory=list)
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, settings: Settings, url: str, *, dry_run: bool = False, force: bool = False,
               max_clips: int | None = None, log: Callable[[str], None] = print,
               data_dir: Path | None = None) -> "RunContext":
        vid = extract_video_id(url)
        workdir = (data_dir or DATA_DIR) / vid
        workdir.mkdir(parents=True, exist_ok=True)
        return cls(
            settings=settings, video_url=url, video_id=vid, workdir=workdir,
            manifest=Manifest(workdir, vid, url), dry_run=dry_run, force=force,
            max_clips=max_clips if max_clips is not None else settings.max_clips, log=log,
        )

    def path(self, rel: str) -> Path:
        return self.workdir / rel

    def charge(self, usd: float, what: str) -> None:
        """Called by stages BEFORE an API call with the pre-flight estimate; aborts if over budget."""
        if self.spent_usd + usd > self.settings.max_budget_usd:
            raise BudgetExceeded(
                f"{what} would cost ~${usd:.4f}, total ${self.spent_usd + usd:.4f} "
                f"> FINVID_MAX_BUDGET_USD={self.settings.max_budget_usd}. Aborting before the call."
            )

    def settle(self, result: StageResult) -> None:
        self.spent_usd += result.usd


def run_stage(ctx: RunContext, name: str, cfg: dict[str, Any],
              fn: Callable[[RunContext], StageResult]) -> StageResult:
    """Idempotent stage execution.

    - cache hit (same config hash, outputs on disk, not --force) -> skip, $0
    - --dry-run -> stage must return estimates only (CostEntry.estimated=True) and must not call APIs
    - otherwise run, charge, record to manifest
    """
    if not ctx.force:
        hit = ctx.manifest.cached(name, cfg)
        if hit is not None:
            ctx.stages_cached.append(name)
            ctx.stage_results[name] = hit
            ctx.log(f"[{name}] cache hit -> skipped (saved ${hit.usd:.4f})")
            return hit
    ctx.log(f"[{name}] {'DRY-RUN estimate' if ctx.dry_run else 'running'} ...")
    t0 = time.time()
    result = fn(ctx)
    dt = time.time() - t0
    if not ctx.dry_run:
        ctx.manifest.invalidate_from(name)  # downstream depends on this output
        ctx.manifest.record(name, cfg, result)
        ctx.settle(result)
        ctx.stages_run.append(name)
        ctx.log(f"[{name}] done in {dt:.1f}s, cost ${result.usd:.4f}")
    else:
        ctx.log(f"[{name}] estimate ${result.estimated_usd:.4f}")
    ctx.stage_results[name] = result
    return result


_YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    m = _YT_ID.search(url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"cannot extract a YouTube video id from: {url}")
