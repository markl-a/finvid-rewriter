"""Hugging Face ZeroGPU backend: call a public Gradio Space (default: Lightricks' official
LTX-Video distilled demo) through `gradio_client`. Free, but metered: measured 2026-09-19,
anonymous callers get 120 GPU-seconds a day and the Space RESERVES 120 s per call, so anonymous
use is 1-2 shots a day; a free HF account token (HF_TOKEN) gets a larger daily allowance (a full
3-clip run went through). One 5 s 576x1024 shot is ~18-25 s wall. The quota is the "budget" here,
so a quota refusal is surfaced as a clear error and the shots already rendered stay cached -
re-running later only renders what is missing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ...config import Settings
from .base import AIVideoError, CachedShotProvider

DEFAULT_SPACE = "Lightricks/ltx-video-distilled"
NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, text, watermark, logo, subtitles"


class HFSpaceError(AIVideoError):
    pass


class HFSpaceProvider(CachedShotProvider):
    name = "hf-zerogpu"

    def __init__(self, space: str, token: str | None, *, width: int, height: int, fps: int,
                 est_seconds: float = 30.0, ffmpeg_bin: str = "ffmpeg", client=None):
        super().__init__(width=width, height=height, fps=fps, ffmpeg_bin=ffmpeg_bin, est_seconds=est_seconds)
        self.space = self.model = space
        self.token = token
        self._client = client  # injected in tests; otherwise created lazily (network on construction)

    @classmethod
    def from_settings(cls, s: Settings) -> "HFSpaceProvider":
        return cls(s.hf_space, s.hf_token, width=s.ai_shot_width, height=s.ai_shot_height,
                   fps=s.ai_shot_fps, est_seconds=s.hf_est_seconds, ffmpeg_bin=s.ffmpeg_bin())

    def client(self):
        if self._client is None:
            try:
                from gradio_client import Client
            except ImportError as e:  # pragma: no cover
                raise HFSpaceError("pip install gradio_client (it is in the project dependencies)") from e
            try:
                self._client = Client(self.space, token=self.token, verbose=False)  # gradio_client >= 2
            except Exception as e:  # noqa: BLE001 - gradio_client raises many types
                raise HFSpaceError(f"cannot reach Space {self.space}: {e}") from e
        return self._client

    def _render(self, prompt: str, raw_out: Path, *, seconds: float, seed: int, log) -> dict:
        log(f"[s4] hf-zerogpu: {self.space} text_to_video {self.width}x{self.height} {seconds:.0f}s "
            f"({'with' if self.token else 'without'} HF_TOKEN)")
        try:
            res = self.client().predict(
                prompt=prompt, negative_prompt=NEGATIVE,
                input_image_filepath=None, input_video_filepath=None,
                height_ui=self.height, width_ui=self.width, mode="text-to-video",
                duration_ui=seconds, ui_frames_to_use=9, seed_ui=seed, randomize_seed=False,
                ui_guidance_scale=1, improve_texture_flag=True, api_name="/text_to_video")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "quota" in msg.lower() or "GPU" in msg and "exceeded" in msg:
                raise HFSpaceError(
                    f"ZeroGPU quota exhausted on {self.space}: {msg[:300]}. Shots already rendered are "
                    f"cached; re-run later, add HF_TOKEN for a bigger quota, or switch FINVID_AI_VIDEO.") from e
            raise HFSpaceError(f"Space {self.space} failed: {msg[:500]}") from e
        video = res[0] if isinstance(res, (list, tuple)) else res
        path = video.get("video") if isinstance(video, dict) else video
        if not path or not Path(path).exists():
            raise HFSpaceError(f"Space returned no video file: {res!r}"[:300])
        shutil.copy(path, raw_out)
        return {"space": self.space, "returned_seed": res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else None}
