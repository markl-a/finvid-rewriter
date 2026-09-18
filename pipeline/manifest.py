"""Per-video manifest: idempotency cache + cost ledger.

data/<video_id>/manifest.json
{
  "video_id": ..., "url": ..., "created": ..., "updated": ...,
  "stages": {
     "s1_download": {"config_hash": "...", "config": {...},
                      "outputs": {"audio": "01_audio.wav", ...},
                      "meta": {...}, "finished_at": "...", "costs": [CostEntry...]}
  },
  "runs": [ {"started": ..., "dry_run": bool, "stages_run": [...], "stages_cached": [...], "usd": ...} ]
}

A stage is "fresh" when its config hash matches AND every output file still exists.
Re-running with the same config costs $0 because the stage is skipped entirely.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import CostEntry

STAGE_ORDER = ["s1_download", "s2_transcribe", "s3_script", "s4_render"]


def config_hash(cfg: dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class StageResult:
    outputs: dict[str, str]  # name -> path relative to workdir
    costs: list[CostEntry] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    cached: bool = False

    @property
    def usd(self) -> float:
        return sum(c.usd for c in self.costs if not c.estimated)

    @property
    def estimated_usd(self) -> float:
        return sum(c.usd for c in self.costs if c.estimated)


class Manifest:
    def __init__(self, workdir: Path, video_id: str, url: str, *, persist: bool = True):
        self.workdir = workdir
        self.path = workdir / "manifest.json"
        self.data: dict[str, Any]
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "video_id": video_id,
                "url": url,
                "created": _now(),
                "updated": _now(),
                "stages": {},
                "runs": [],
            }
            if persist:  # a --dry-run must not leave an empty data/<id>/ behind
                self.save()

    # ---- persistence ----
    def save(self) -> None:
        self.data["updated"] = _now()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    # ---- cache lookup ----
    def cached(self, stage: str, cfg: dict[str, Any]) -> StageResult | None:
        rec = self.data["stages"].get(stage)
        if not rec or rec.get("config_hash") != config_hash(cfg):
            return None
        outs = rec.get("outputs", {})
        if not outs or not all((self.workdir / p).exists() for p in outs.values()):
            return None
        return StageResult(
            outputs=outs,
            costs=[CostEntry(**c) for c in rec.get("costs", [])],
            meta=rec.get("meta", {}),
            cached=True,
        )

    def record(self, stage: str, cfg: dict[str, Any], result: StageResult) -> None:
        self.data["stages"][stage] = {
            "config_hash": config_hash(cfg),
            "config": cfg,
            "outputs": result.outputs,
            "meta": result.meta,
            "costs": [c.model_dump() for c in result.costs],
            "finished_at": _now(),
        }
        self.save()

    def invalidate_from(self, stage: str) -> None:
        """Drop this stage and every later one (their inputs changed)."""
        idx = STAGE_ORDER.index(stage)
        for s in STAGE_ORDER[idx:]:
            self.data["stages"].pop(s, None)
        self.save()

    # ---- cost views ----
    def stage_costs(self, stage: str) -> list[CostEntry]:
        rec = self.data["stages"].get(stage) or {}
        return [CostEntry(**c) for c in rec.get("costs", [])]

    def all_costs(self) -> list[CostEntry]:
        out: list[CostEntry] = []
        for s in STAGE_ORDER:
            out.extend(self.stage_costs(s))
        return out

    def total_usd(self, include_estimated: bool = False) -> float:
        return sum(c.usd for c in self.all_costs() if include_estimated or not c.estimated)

    def add_run(self, run: dict[str, Any]) -> None:
        self.data["runs"].append(run)
        self.save()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
