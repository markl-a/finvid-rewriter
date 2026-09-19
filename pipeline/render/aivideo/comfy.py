"""ComfyUI backend: drive a running ComfyUI server (local GPU, $0) through its HTTP API.

    POST /prompt            queue a workflow (API-format JSON)      -> prompt_id
    GET  /history/<id>      outputs once finished                   -> {node: {"gifs"|"images"|"videos": [...]}}
    GET  /view?filename=..  download an output file
    GET  /system_stats      liveness + device name (for the ledger)

The workflow is a template with <<placeholders>>; we fill prompt/size/frames/seed and post it.
Cost is $0 but time is not: the ledger records GPU wall seconds so the trade-off against paid
APIs (20 s, $0.1-0.5 per shot) is visible in the same table.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

from ...config import Settings
from .base import NEGATIVE, AIVideoError, CachedShotProvider

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


class ComfyUIError(AIVideoError):
    pass


def _placeholders(obj, values: dict):
    """Replace "<<name>>" strings (whole-string or embedded) anywhere in the workflow JSON."""
    if isinstance(obj, dict):
        return {k: _placeholders(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_placeholders(v, values) for v in obj]
    if isinstance(obj, str):
        if obj.startswith("<<") and obj.endswith(">>") and obj[2:-2] in values:
            return values[obj[2:-2]]  # keeps ints/floats typed
        for k, v in values.items():
            obj = obj.replace(f"<<{k}>>", str(v))
        return obj
    return obj


class ComfyUIProvider(CachedShotProvider):
    name = "comfyui"

    def __init__(self, url: str, workflow: Path, *, checkpoint: str, text_encoder: str,
                 width: int, height: int, fps: int, steps: int, cfg: float,
                 timeout_sec: float = 1800, est_gpu_seconds: float = 90.0,
                 ffmpeg_bin: str = "ffmpeg", client: httpx.Client | None = None):
        super().__init__(width=width, height=height, fps=fps, ffmpeg_bin=ffmpeg_bin, est_seconds=est_gpu_seconds)
        self.url = url.rstrip("/")
        self.workflow_path = workflow
        self.template = json.loads(workflow.read_text(encoding="utf-8"))
        self.template.pop("_comment", None)
        self.checkpoint = self.model = checkpoint
        self.text_encoder = text_encoder
        self.steps, self.cfg = steps, cfg
        self.timeout_sec = timeout_sec
        self.est_gpu_seconds = est_gpu_seconds
        self.client = client or httpx.Client(base_url=self.url, timeout=60.0)

    @classmethod
    def from_settings(cls, s: Settings) -> "ComfyUIProvider":
        wf = Path(s.comfy_workflow) if s.comfy_workflow else WORKFLOWS_DIR / "ltxv_t2v.json"
        return cls(s.comfy_url, wf, checkpoint=s.comfy_checkpoint, text_encoder=s.comfy_text_encoder,
                   width=s.ai_shot_width, height=s.ai_shot_height, fps=s.ai_shot_fps,
                   steps=s.comfy_steps, cfg=s.comfy_cfg, timeout_sec=s.comfy_timeout_sec,
                   est_gpu_seconds=s.comfy_est_gpu_seconds, ffmpeg_bin=s.ffmpeg_bin())

    # ------------------------------------------------------------------ server
    def ping(self) -> dict:
        try:
            r = self.client.get("/system_stats")
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise ComfyUIError(
                f"ComfyUI not reachable at {self.url} ({e}). Start it with "
                f"`python main.py --listen 127.0.0.1 --port 8188` or set FINVID_AI_VIDEO=none.") from e

    def frames_for(self, seconds: float) -> int:
        n = int(round(seconds * self.fps))
        return max(9, ((n - 1 + 7) // 8) * 8 + 1)  # LTX wants 8k+1 frames; round up

    def build_workflow(self, prompt: str, *, seconds: float, seed: int, prefix: str) -> dict:
        values = {
            "checkpoint": self.checkpoint, "text_encoder": self.text_encoder,
            "prompt": prompt, "negative": NEGATIVE,
            "width": self.width, "height": self.height, "frames": self.frames_for(seconds),
            "fps": self.fps, "steps": self.steps, "cfg": self.cfg, "seed": seed, "prefix": prefix,
        }
        return _placeholders(self.template, values)

    def cache_fields(self, prompt: str, *, seconds: float, seed: int) -> dict:
        # the whole filled workflow is the cache key: any node/param change re-renders
        return {"provider": self.name, "workflow": self.build_workflow(prompt, seconds=seconds, seed=seed, prefix="x")}

    def _render(self, prompt: str, raw_out: Path, *, seconds: float, seed: int, log) -> dict:
        stats = self.ping()
        device = (stats.get("devices") or [{}])[0].get("name", "?")
        wf = self.build_workflow(prompt, seconds=seconds, seed=seed, prefix=raw_out.stem)
        r = self.client.post("/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex})
        if r.status_code != 200:
            raise ComfyUIError(f"ComfyUI rejected the workflow ({r.status_code}): {r.text[:800]}")
        prompt_id = r.json()["prompt_id"]
        log(f"[s4] comfyui: queued {raw_out.stem} ({self.frames_for(seconds)} frames "
            f"{self.width}x{self.height}, {self.steps} steps) on {device}")
        outputs = self._wait(prompt_id)
        self._download(outputs, raw_out)
        return {"device": device, "workflow": self.workflow_path.name}

    def _wait(self, prompt_id: str) -> dict:
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            r = self.client.get(f"/history/{prompt_id}")
            r.raise_for_status()
            hist = r.json().get(prompt_id)
            if hist:
                status = hist.get("status") or {}
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message") if msgs else "see ComfyUI console"
                    raise ComfyUIError(f"ComfyUI execution failed: {detail}")
                if hist.get("outputs"):
                    return hist["outputs"]
            time.sleep(2.0)
        raise ComfyUIError(f"ComfyUI did not finish within {self.timeout_sec:.0f}s")

    def _download(self, outputs: dict, dest: Path) -> Path:
        for node_out in outputs.values():
            for kind in ("gifs", "videos", "images"):
                for f in node_out.get(kind, []) or []:
                    if not f.get("filename"):
                        continue
                    r = self.client.get("/view", params={"filename": f["filename"],
                                                         "subfolder": f.get("subfolder", ""),
                                                         "type": f.get("type", "output")})
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    return dest
        raise ComfyUIError("ComfyUI finished but produced no video output (check the save node)")
