"""Settings panel backend: read/write the project's .env from the local web UI, and probe the
services behind each key so a first-time user can see green/red before spending anything.

Keys are never sent back to the browser: GET returns `set: true` plus the last 4 characters.
The server binds to 127.0.0.1 (see cli.serve), so this is a local convenience, not an admin API.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import PROJECT_ROOT

router = APIRouter()
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# what the panel edits. Secrets are masked on read; the rest are shown as-is.
SECRETS = ["OPENAI_API_KEY", "HF_TOKEN", "PEXELS_API_KEY", "PIXAZO_API_KEY", "MINIMAX_API_KEY",
           "KLING_ACCESS_KEY", "KLING_SECRET_KEY"]
PLAIN = ["FINVID_AI_VIDEO", "FINVID_BROLL", "FINVID_AI_SHOTS_PER_CLIP", "FINVID_COMFY_URL",
         "FINVID_MAX_BUDGET_USD", "FINVID_MAX_CLIPS", "FINVID_STT_PROVIDER", "FINVID_TTS_PROVIDER"]
EDITABLE = set(SECRETS + PLAIN)
_LINE = re.compile(r"^\s*(?:#\s*)?(?P<key>[A-Z][A-Z0-9_]*)\s*=(?P<val>.*)$")  # also matches "# KEY=" placeholders


def read_env(path: Path | None = None) -> dict[str, str]:
    """KEY=value pairs from .env (comments/blank lines ignored, surrounding quotes stripped)."""
    path = path or ENV_PATH  # resolved at call time so tests can point it elsewhere
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line)
        if m and not line.lstrip().startswith("#"):
            out[m.group("key")] = m.group("val").strip().strip('"').strip("'")
    return out


def write_env(updates: dict[str, str], path: Path | None = None) -> None:
    """Merge updates into .env in place: existing lines are rewritten, comments kept, new keys
    appended. Missing .env is seeded from .env.example so the user's file keeps the documentation."""
    path = path or ENV_PATH
    if not path.exists() and ENV_EXAMPLE.exists():
        path.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    for i, line in enumerate(lines):
        m = _LINE.match(line)
        if not m:
            continue
        key = m.group("key")
        is_comment = line.lstrip().startswith("#")
        if key in updates and not is_comment:
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
        elif key in updates and is_comment and key not in seen:
            # "# HF_TOKEN=hf_...（說明）" placeholder line -> replace it with the real assignment
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
    for key, val in updates.items():
        if key not in seen:
            lines.append(f"{key}={val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key, val in updates.items():  # the running server sees the change immediately
        if val == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _mask(v: str) -> dict:
    v = v or ""
    return {"set": bool(v.strip()), "hint": ("…" + v[-4:]) if len(v) >= 8 else ""}


class SettingsUpdate(BaseModel):
    values: dict[str, str]


@router.get("/api/settings")
def get_settings_view() -> dict:
    env = read_env()
    return {
        "env_path": str(ENV_PATH),
        "secrets": {k: _mask(env.get(k, "")) for k in SECRETS},
        "plain": {k: env.get(k, "") for k in PLAIN},
    }


@router.post("/api/settings")
def update_settings(req: SettingsUpdate) -> dict:
    bad = [k for k in req.values if k not in EDITABLE]
    if bad:
        raise HTTPException(400, f"not editable: {bad}")
    cleaned = {k: v.strip() for k, v in req.values.items()}
    for k, v in cleaned.items():
        if v and ("\n" in v or not v.isascii()):
            raise HTTPException(400, f"{k}: value must be a single ASCII line (looks like a placeholder?)")
    write_env(cleaned)
    return {"saved": sorted(cleaned), "env_path": str(ENV_PATH)}


# ---------------------------------------------------------------- probes
def _ok(**kw) -> dict:
    return {"ok": True, **kw}


def _fail(msg: str, **kw) -> dict:
    return {"ok": False, "message": msg, **kw}


def probe_comfy(url: str) -> dict:
    url = (url or "http://127.0.0.1:8188").rstrip("/")
    try:
        with httpx.Client(base_url=url, timeout=5.0) as c:
            stats = c.get("/system_stats").json()
            dev = (stats.get("devices") or [{}])[0]
            ckpts = c.get("/object_info/CheckpointLoaderSimple").json()["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            encs = c.get("/object_info/CLIPLoader").json()["CLIPLoader"]["input"]["required"]["clip_name"][0]
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        return _fail(f"ComfyUI 沒有回應（{url}）：{e}。請先啟動 ComfyUI，或改用 hf / pixazo。")
    from ..config import get_settings

    s = get_settings()
    missing = [m for m in (s.comfy_checkpoint, s.comfy_text_encoder) if m not in ckpts + encs]
    if missing:
        return _fail(f"ComfyUI 有回應，但缺模型檔：{missing}（放到 ComfyUI/models/checkpoints 與 models/text_encoders）",
                     device=dev.get("name"))
    return _ok(device=dev.get("name"), vram_gb=round((dev.get("vram_total") or 0) / 1e9, 1),
               checkpoint=s.comfy_checkpoint, text_encoder=s.comfy_text_encoder)


def probe_openai(key: str) -> dict:
    if not key:
        return _fail("OPENAI_API_KEY 未填")
    try:
        r = httpx.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except httpx.HTTPError as e:
        return _fail(f"連不到 OpenAI：{e}")
    if r.status_code == 401:
        return _fail("OpenAI 拒絕這把 key（401）")
    if r.status_code != 200:
        return _fail(f"OpenAI 回應 {r.status_code}")
    return _ok(models=len(r.json().get("data", [])))


def probe_hf(token: str) -> dict:
    if not token:
        return _ok(note="未填 token：匿名可用，但 ZeroGPU 一天只夠 1–2 段；建議填免費帳號的 token")
    try:
        r = httpx.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    except httpx.HTTPError as e:
        return _fail(f"連不到 Hugging Face：{e}")
    if r.status_code != 200:
        return _fail(f"Hugging Face 拒絕這個 token（{r.status_code}）")
    return _ok(user=r.json().get("name"))


def probe_pexels(key: str) -> dict:
    if not key:
        return _fail("PEXELS_API_KEY 未填（pexels.com/api 免費申請）")
    try:
        r = httpx.get("https://api.pexels.com/videos/search", params={"query": "city", "per_page": 1},
                      headers={"Authorization": key}, timeout=10)
    except httpx.HTTPError as e:
        return _fail(f"連不到 Pexels：{e}")
    if r.status_code == 401:
        return _fail("Pexels 拒絕這把 key（401）")
    if r.status_code != 200:
        return _fail(f"Pexels 回應 {r.status_code}")
    return _ok(monthly_remaining=r.headers.get("X-Ratelimit-Remaining"))


def probe_pixazo(key: str) -> dict:
    if not key:
        return _fail("PIXAZO_API_KEY 未填（pixazo.ai 免費申請）")
    return _ok(note="key 已填；Pixazo 沒有免費的驗證端點，第一次生成時才會知道是否有效")


def probe_minimax(key: str) -> dict:
    if not key:
        return _fail("MINIMAX_API_KEY 未填（付費，可不填）")
    try:
        r = httpx.get("https://api.minimax.io/v1/query/video_generation", params={"task_id": "0"},
                      headers={"Authorization": f"Bearer {key}"}, timeout=10)
        code = (r.json().get("base_resp") or {}).get("status_code")
    except (httpx.HTTPError, ValueError) as e:
        return _fail(f"連不到 MiniMax：{e}")
    if code in (1004, 2049):
        return _fail(f"MiniMax 拒絕這把 key（{code}）")
    return _ok(note="認證通過（沒有花錢）")


class ProbeRequest(BaseModel):
    what: str


@router.post("/api/settings/probe")
def probe(req: ProbeRequest) -> dict:
    env = read_env()
    t0 = time.time()
    if req.what == "comfy":
        res = probe_comfy(env.get("FINVID_COMFY_URL") or os.environ.get("FINVID_COMFY_URL", ""))
    elif req.what == "openai":
        res = probe_openai(env.get("OPENAI_API_KEY", ""))
    elif req.what == "hf":
        res = probe_hf(env.get("HF_TOKEN", ""))
    elif req.what == "pexels":
        res = probe_pexels(env.get("PEXELS_API_KEY", ""))
    elif req.what == "pixazo":
        res = probe_pixazo(env.get("PIXAZO_API_KEY", ""))
    elif req.what == "minimax":
        res = probe_minimax(env.get("MINIMAX_API_KEY", ""))
    else:
        raise HTTPException(400, f"unknown probe {req.what!r}")
    res["ms"] = int((time.time() - t0) * 1000)
    return res
