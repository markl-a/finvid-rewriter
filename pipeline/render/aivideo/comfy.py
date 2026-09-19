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

import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import httpx

from ...config import Settings
from ...models import CostEntry
from . import ShotResult

WORKFLOWS_DIR = Path(__file__).parent / "workflows"
NEGATIVE = ("low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
            "text, watermark, logo, subtitles, blurry")
STAGE = "s4_render"


class ComfyUIError(RuntimeError):
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


class ComfyUIProvider:
    name = "comfyui"

    def __init__(self, url: str, workflow: Path, *, checkpoint: str, text_encoder: str,
                 width: int, height: int, fps: int, steps: int, cfg: float,
                 timeout_sec: float = 1800, est_gpu_seconds: float = 180.0,
                 ffmpeg_bin: str = "ffmpeg", client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.workflow_path = workflow
        self.template = json.loads(workflow.read_text(encoding="utf-8"))
        self.template.pop("_comment", None)
        self.checkpoint = checkpoint
        self.model = checkpoint
        self.text_encoder = text_encoder
        self.width, self.height, self.fps = width, height, fps
        self.steps, self.cfg = steps, cfg
        self.timeout_sec = timeout_sec
        self.est_gpu_seconds = est_gpu_seconds
        self.ffmpeg_bin = ffmpeg_bin
        self.client = client or httpx.Client(base_url=self.url, timeout=60.0)

    @classmethod
    def from_settings(cls, s: Settings) -> "ComfyUIProvider":
        wf = Path(s.comfy_workflow) if s.comfy_workflow else WORKFLOWS_DIR / "ltxv_t2v.json"
        return cls(s.comfy_url, wf, checkpoint=s.comfy_checkpoint, text_encoder=s.comfy_text_encoder,
                   width=s.ai_shot_width, height=s.ai_shot_height, fps=s.ai_shot_fps,
                   steps=s.comfy_steps, cfg=s.comfy_cfg, timeout_sec=s.comfy_timeout_sec,
                   est_gpu_seconds=s.comfy_est_gpu_seconds, ffmpeg_bin=s.ffmpeg_bin())

    # ------------------------------------------------------------------ ledger
    def _entry(self, gpu_seconds: float, *, estimated: bool, note: str) -> CostEntry:
        return CostEntry(stage=STAGE, provider=self.name, model=self.checkpoint, unit="gpu_second",
                         quantity=round(gpu_seconds, 1), unit_price_usd=0.0, usd=0.0,
                         estimated=estimated, note=note)

    def estimate(self, n_shots: int, seconds_each: float) -> list[CostEntry]:
        return [self._entry(self.est_gpu_seconds * n_shots, estimated=True,
                            note=f"{n_shots} AI shot(s) x ~{seconds_each:.0f}s on local ComfyUI, "
                                 f"~{self.est_gpu_seconds:.0f}s GPU each")]

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

    def cache_key(self, prompt: str, *, seconds: float, seed: int) -> str:
        wf = self.build_workflow(prompt, seconds=seconds, seed=seed, prefix="x")
        raw = json.dumps({"provider": self.name, "workflow": wf}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def generate(self, prompt: str, out_mp4: Path, *, seconds: float, seed: int, log) -> ShotResult:
        key = self.cache_key(prompt, seconds=seconds, seed=seed)
        side = out_mp4.with_suffix(".json")
        if out_mp4.exists() and side.exists():
            try:
                if json.loads(side.read_text(encoding="utf-8")).get("key") == key:
                    return ShotResult(path=out_mp4, cached=True,
                                      costs=[self._entry(0, estimated=False, note=f"{out_mp4.name}: cache hit")])
            except ValueError:
                pass

        stats = self.ping()
        device = (stats.get("devices") or [{}])[0].get("name", "?")
        wf = self.build_workflow(prompt, seconds=seconds, seed=seed, prefix=out_mp4.stem)
        t0 = time.time()
        r = self.client.post("/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex})
        if r.status_code != 200:
            raise ComfyUIError(f"ComfyUI rejected the workflow ({r.status_code}): {r.text[:800]}")
        prompt_id = r.json()["prompt_id"]
        log(f"[s4] comfyui: queued {out_mp4.stem} ({self.frames_for(seconds)} frames "
            f"{self.width}x{self.height}, {self.steps} steps) on {device}")

        outputs = self._wait(prompt_id)
        raw = self._download(outputs, out_mp4.parent / f"{out_mp4.stem}.raw")
        self._to_mp4(raw, out_mp4)
        raw.unlink(missing_ok=True)
        wall = time.time() - t0
        side.write_text(json.dumps({"key": key, "prompt": prompt, "seconds": seconds, "seed": seed,
                                    "wall_seconds": round(wall, 1), "device": device,
                                    "workflow": self.workflow_path.name}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return ShotResult(path=out_mp4, wall_seconds=wall, meta={"device": device},
                          costs=[self._entry(wall, estimated=False,
                                             note=f"{out_mp4.name}: {wall:.0f}s on {device}")])

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

    def _to_mp4(self, src: Path, out_mp4: Path) -> None:
        """Whatever the save node emitted (webm/mp4/webp) -> h264 mp4 at our fps, no audio."""
        subprocess.run([self.ffmpeg_bin, "-y", "-v", "error", "-nostdin", "-i", str(src),
                        "-an", "-r", str(self.fps), "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)],
                       check=True, capture_output=True, text=True)
