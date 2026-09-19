"""Local web UI: one HTML page + a small JSON API over data/<video_id>/.

The page exists to make the cost decisions visible: which stages were cached (saved $),
which transcript segments were selected vs skipped and why, and actual spend vs the
"naive" alternative (no filtering + AI video API).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import DATA_DIR as _DEFAULT_DATA_DIR
from ..context import BudgetExceeded, extract_video_id
from ..manifest import STAGE_ORDER

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Tests monkeypatch this; when None the real pipeline.cli.run_pipeline is imported lazily so
# that importing the UI never pulls in typer/rich/stages.
run_pipeline = None

app = FastAPI(title="finvid cost dashboard")

from .settings_api import router as _settings_router  # noqa: E402 - keys/probes panel

app.include_router(_settings_router)


# ---------------------------------------------------------------- helpers
def data_dir() -> Path:
    """DATA_DIR, overridable per call via FINVID_DATA_DIR (used by tests)."""
    override = os.environ.get("FINVID_DATA_DIR")
    return Path(override).resolve() if override else Path(_DEFAULT_DATA_DIR).resolve()


def _read_json(path: Path) -> Any | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return None


_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _workdir(video_id: str) -> Path:
    if not _SAFE_ID.match(video_id):
        raise HTTPException(400, "bad video_id")
    wd = data_dir() / video_id
    if not wd.is_dir():
        raise HTTPException(404, f"no data for {video_id}")
    return wd


def _all_costs(manifest: dict | None) -> list[dict]:
    out: list[dict] = []
    if not manifest:
        return out
    stages = manifest.get("stages") or {}
    for s in STAGE_ORDER:
        for c in (stages.get(s) or {}).get("costs") or []:
            if isinstance(c, dict):
                out.append(c)
    return out


def _is_ref(c: dict) -> bool:
    return c.get("provider") == "reference"


def _usd(c: dict) -> float:
    try:
        return float(c.get("usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _real(c: dict) -> bool:
    return not c.get("estimated") and not _is_ref(c)


def _spent(manifest: dict | None) -> float:
    return round(sum(_usd(c) for c in _all_costs(manifest) if _real(c)), 6)


def _is_pass_b(c: dict, strong_model: str | None) -> bool:
    note = str(c.get("note") or "").lower().replace("_", " ").replace("-", " ")
    if "pass b" in note or "passb" in note:
        return True
    if "pass a" in note or "passa" in note:
        return False
    return bool(strong_model) and c.get("model") == strong_model


def cost_summary(manifest: dict | None, scripts: dict | None) -> dict:
    """All numbers the cost cards need. Every input may be None / partial."""
    costs = _all_costs(manifest)
    stages = (manifest or {}).get("stages") or {}
    runs = (manifest or {}).get("runs") or []
    last_run = runs[-1] if runs else None
    cached_last = set((last_run or {}).get("stages_cached") or [])

    spent = sum(_usd(c) for c in costs if _real(c))
    estimated = sum(_usd(c) for c in costs if c.get("estimated") and not _is_ref(c))
    reference = sum(_usd(c) for c in costs if _is_ref(c))

    per_stage: dict[str, dict] = {}
    for s in STAGE_ORDER:
        sc = [c for c in costs if c.get("stage") == s]
        per_stage[s] = {
            "present": s in stages,
            "usd": round(sum(_usd(c) for c in sc if _real(c)), 6),
            "estimated_usd": round(sum(_usd(c) for c in sc if c.get("estimated") and not _is_ref(c)), 6),
            "cached_last_run": s in cached_last,
            "finished_at": (stages.get(s) or {}).get("finished_at"),
            "meta": (stages.get(s) or {}).get("meta") or {},
        }
    saved_by_cache = sum(per_stage[s]["usd"] for s in cached_last if s in per_stage)

    # naive alternative: Pass B on every candidate segment + AI-video API for the clips
    segs = (scripts or {}).get("segments") or []
    seg_total = len(segs)
    seg_selected = sum(1 for s in segs if isinstance(s, dict) and s.get("selected"))
    strong_model = None
    try:
        from ..config import get_settings

        strong_model = get_settings().llm_strong_model
    except Exception:  # settings may fail outside the project env; the heuristic still works
        strong_model = None
    pass_b_actual = sum(_usd(c) for c in costs
                        if c.get("stage") == "s3_script" and _real(c) and _is_pass_b(c, strong_model))
    if seg_selected > 0 and pass_b_actual > 0:
        pass_b_all = pass_b_actual * seg_total / seg_selected
    else:
        pass_b_all = 0.0
    naive = reference + pass_b_all

    return {
        "spent_usd": round(spent, 6),
        "estimated_usd": round(estimated, 6),
        "reference_ai_video_usd": round(reference, 6),
        "per_stage": per_stage,
        "saved_by_cache_usd": round(saved_by_cache, 6),
        "last_run_usd": round(float((last_run or {}).get("usd") or 0.0), 6),
        "last_run": last_run,
        "runs": len(runs),
        "naive_alternative_usd": round(naive, 6),
        "naive_breakdown": {
            "reference_ai_video_usd": round(reference, 6),
            "pass_b_actual_usd": round(pass_b_actual, 6),
            "pass_b_all_segments_usd": round(pass_b_all, 6),
            "segments_total": seg_total,
            "segments_selected": seg_selected,
        },
        "ledger": costs,
    }


# ---------------------------------------------------------------- run registry
class RunRequest(BaseModel):
    url: str
    dry_run: bool = False
    force: bool = False
    max_clips: int | None = 3
    until: str | None = None


_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()
_RICH_TAG = re.compile(
    r"\[/?(?:(?:bold|dim|italic|underline|red|green|yellow|blue|cyan|magenta|white|black)\s?)*\]"
)


def _clean_log(line: Any) -> str:
    return _RICH_TAG.sub("", str(line))


def _resolve_runner():
    if run_pipeline is not None:
        return run_pipeline
    from ..cli import run_pipeline as real

    return real


def _worker(rec: dict, req: RunRequest) -> None:
    def log(line: Any) -> None:
        rec["logs"].append(_clean_log(line))

    try:
        _resolve_runner()(
            req.url, dry_run=req.dry_run, force=req.force, max_clips=req.max_clips,
            until=req.until or None, log=log, data_dir=data_dir(),
        )
        rec["status"] = "done"
    except BudgetExceeded as e:
        rec["status"] = "budget_exceeded"
        rec["error"] = str(e)
    except Exception as e:  # surfaced to the page, never crashes the server
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["logs"].append(f"ERROR {type(e).__name__}: {e}")
    finally:
        rec["finished"] = time.time()


# ---------------------------------------------------------------- routes
@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/api/videos")
def list_videos() -> list[dict]:
    base = data_dir()
    out: list[dict] = []
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        m = _read_json(d / "manifest.json")
        if not isinstance(m, dict):
            continue
        info = _read_json(d / "01_info.json") or {}
        stages = m.get("stages") or {}
        out.append({
            "video_id": d.name,  # folder name = identifier (data/demo/ is a checked-in copy of a run)
            "source_video_id": m.get("video_id") or d.name,
            "url": m.get("url") or "",
            "title": (info.get("title") if isinstance(info, dict) else None)
            or (stages.get("s1_download") or {}).get("meta", {}).get("title") or "",
            "updated": m.get("updated") or "",
            "spent_usd": _spent(m),
            "stages_done": [s for s in STAGE_ORDER if s in stages],
        })
    # folders with finished stages first (so a stray empty manifest never hides the demo), newest first
    out.sort(key=lambda v: (bool(v["stages_done"]), v["updated"]), reverse=True)
    return out


@app.get("/api/videos/{video_id}")
def video_detail(video_id: str) -> dict:
    wd = _workdir(video_id)
    manifest = _read_json(wd / "manifest.json")
    info = _read_json(wd / "01_info.json")
    transcript = _read_json(wd / "02_transcript.json")
    scripts = _read_json(wd / "03_scripts.json")
    render = _read_json(wd / "04_render.json")

    tsum = None
    if isinstance(transcript, dict):
        segs = transcript.get("segments") or []
        chars = sum(len(str(s.get("text") or "")) for s in segs if isinstance(s, dict))
        dur = transcript.get("duration_sec")
        if dur is None and segs:
            try:
                dur = max(float(s.get("end") or 0) for s in segs if isinstance(s, dict))
            except ValueError:
                dur = None
        tsum = {"segments": len(segs), "chars": chars, "duration_sec": dur,
                "provider": transcript.get("provider"), "model": transcript.get("model")}

    return {
        "video_id": video_id,
        "manifest": manifest,
        "info": info,
        "transcript": tsum,
        "scripts": scripts,
        "render": render,
        "cost_summary": cost_summary(manifest if isinstance(manifest, dict) else None,
                                     scripts if isinstance(scripts, dict) else None),
    }


@app.get("/files/{video_id}/{path:path}")
def serve_file(video_id: str, path: str) -> FileResponse:
    if not _SAFE_ID.match(video_id):
        raise HTTPException(400, "bad video_id")
    wd = (data_dir() / video_id).resolve()
    if ".." in path.replace("\\", "/").split("/"):
        raise HTTPException(400, "path traversal rejected")
    target = (wd / path).resolve()
    if wd != target and wd not in target.parents:
        raise HTTPException(400, "path traversal rejected")
    if not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.post("/api/runs", status_code=202)
def start_run(req: RunRequest) -> dict:
    try:
        video_id = extract_video_id(req.url.strip())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if req.until:
        req.until = req.until.strip().lower()
        if req.until in ("", "all", "none"):
            req.until = None
        elif req.until not in ("s1", "s2", "s3", "s4"):
            raise HTTPException(400, "until must be one of s1|s2|s3|s4")
    with _runs_lock:
        for r in _runs.values():
            if r["video_id"] == video_id and r["status"] == "running":
                raise HTTPException(
                    409, f"{video_id} already running (run {r['run_id']}) - "
                         "that's the duplicate-processing guard")
        run_id = uuid.uuid4().hex[:12]
        rec = {
            "run_id": run_id, "video_id": video_id, "status": "running", "logs": [],
            "started": time.time(), "finished": None, "error": None,
            "request": req.model_dump(),
        }
        _runs[run_id] = rec
    t = threading.Thread(target=_worker, args=(rec, req), name=f"run-{run_id}", daemon=True)
    t.start()
    return {"run_id": run_id, "video_id": video_id}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    with _runs_lock:
        return [{k: v for k, v in r.items() if k != "logs"} | {"lines": len(r["logs"])}
                for r in _runs.values()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    rec = _runs.get(run_id)
    if rec is None:
        raise HTTPException(404, "unknown run")
    return JSONResponse(dict(rec, logs=list(rec["logs"])))
