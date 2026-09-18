"""RunContext + the stage runner that enforces caching, dry-run and the budget guard."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import DATA_DIR, Settings
from .manifest import Manifest, StageResult


class AlreadyRunning(RuntimeError):
    """Another process is working on this video right now (the cross-process duplicate guard)."""


LOCK_FILE = ".running.lock"
LOCK_STALE_SEC = 3 * 3600  # a crashed process (kill -9, power loss) can't clean up; treat old locks as dead


class WorkdirLock:
    """`data/<id>/.running.lock` so two terminals (or the web UI + a terminal) can't process the
    same video at once and bill twice. O_EXCL creation is atomic on every OS; the file holds
    pid + start time so a stale lock from a crashed run can be identified and taken over."""

    def __init__(self, workdir: Path):
        self.path = workdir / LOCK_FILE
        self.fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid={os.getpid()} started={time.time():.0f}".encode())
                os.close(self.fd)
                return
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    holder = self.path.read_text(encoding="utf-8").strip()
                except OSError:
                    continue  # released between our two calls; retry
                if age > LOCK_STALE_SEC:
                    self.path.unlink(missing_ok=True)  # stale: take over
                    continue
                raise AlreadyRunning(
                    f"{self.path.parent.name} is already being processed ({holder}, {age:.0f}s ago). "
                    f"Wait for it, or delete {self.path} if that process is dead.")
        raise AlreadyRunning(f"could not acquire {self.path}")

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


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
        if not dry_run:
            workdir.mkdir(parents=True, exist_ok=True)
        return cls(
            settings=settings, video_url=url, video_id=vid, workdir=workdir,
            manifest=Manifest(workdir, vid, url, persist=not dry_run), dry_run=dry_run, force=force,
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
        ref = sum(c.usd for c in result.costs if c.provider == "reference")
        est = result.estimated_usd - ref
        ctx.log(f"[{name}] estimate ${est:.4f}" + (f" (+ ${ref:.2f} reference only, not charged)" if ref else ""))
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
