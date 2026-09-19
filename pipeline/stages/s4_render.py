"""Stage 4: accepted scripts -> 1080x1920 vertical MP4s, programmatically (default ~$0).

Per clip:  TTS per line (edge-tts free / OpenAI optional)  ->  matplotlib chart from ChartSpec
           ->  PIL title/subtitle/attribution overlays  ->  one ffmpeg call.
No AI video API is called. We only *record* what a Runway/Kling-class API would have charged for the
same seconds (provider "reference", estimated=True) so the ledger shows the comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..context import RunContext
from ..manifest import StageResult
from ..models import CostEntry, RenderedClip, RenderOutput, ScriptClip, ScriptsOutput
from ..pricing import ai_video_reference_cost
from ..render import tts
from ..render.chart import render_chart
from ..render.aivideo import make_provider
from ..render.compose import compose_clip

STAGE = "s4_render"
RENDER_VERSION = 3  # bump when layout/encoding changes so cached clips are re-rendered

SCRIPTS_FILE = "03_scripts.json"
DRY_RUN_CHARS_PER_CLIP = 200
DRY_RUN_SECONDS_PER_CLIP = 35
RENDER_FILE = "04_render.json"
CLIPS_DIR = "04_clips"


# ----------------------------------------------------------------------------- config
def stage_config(ctx: RunContext) -> dict[str, Any]:
    s = ctx.settings
    s3 = ctx.manifest.data.get("stages", {}).get("s3_script") or {}
    return {
        "video_id": ctx.video_id,
        "render_version": RENDER_VERSION,
        "tts_provider": s.tts_provider,
        "tts_voice": s.tts_voice if s.tts_provider == "edge" else s.openai_tts_model,
        "width": s.video_width,
        "height": s.video_height,
        "max_clips": ctx.max_clips,
        "ai_video": s.ai_video,
        "ai_shot": {"seconds": s.ai_shot_seconds, "w": s.ai_shot_width, "h": s.ai_shot_height,
                    "model": s.comfy_checkpoint, "steps": s.comfy_steps} if s.ai_video != "none" else None,
        "s3_config_hash": s3.get("config_hash"),
    }


def shot_prompt(clip: ScriptClip) -> str:
    """Pass B writes `ai_shot` (prompt v8+). Older scripts get a neutral finance b-roll prompt so the
    provider still runs; the title is appended as context. Never text/logos/faces (legal + LTX quirks)."""
    base = clip.ai_shot.strip() or (
        "Cinematic vertical b-roll for a finance news short: a modern Taiwanese city skyline with "
        "apartment towers at dusk, slow smooth camera push-in, soft warm window lights, realistic, "
        "high detail, no people, no text")
    return f"{base}. Vertical 9:16 framing, no text, no captions, no logos, no watermarks."


# ----------------------------------------------------------------------------- helpers
def load_scripts(ctx: RunContext) -> ScriptsOutput:
    p = ctx.path(SCRIPTS_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{SCRIPTS_FILE} missing in {ctx.workdir}; run stage 3 first")
    return ScriptsOutput.model_validate_json(p.read_text(encoding="utf-8"))


def accepted_clips(scripts: ScriptsOutput, max_clips: int) -> list[ScriptClip]:
    return [c for c in scripts.clips if c.plagiarism_ok][:max_clips]


def _rel(ctx: RunContext, p: Path) -> str:
    return p.relative_to(ctx.workdir).as_posix()


# ----------------------------------------------------------------------------- execute
def execute(ctx: RunContext) -> StageResult:
    s = ctx.settings
    prov, model = tts.provider_model(s)

    if ctx.dry_run:
        if ctx.path(SCRIPTS_FILE).exists():
            clips = accepted_clips(load_scripts(ctx), ctx.max_clips)
            chars = sum(len(c.full_text) for c in clips)
            secs = sum(c.est_seconds for c in clips)
        else:  # nothing upstream yet: assume max_clips clips of ~35 s / ~200 chars each
            clips = []
            n = ctx.max_clips
            chars, secs = n * DRY_RUN_CHARS_PER_CLIP, n * DRY_RUN_SECONDS_PER_CLIP
            ctx.log(f"[{STAGE}] dry-run without {SCRIPTS_FILE}: assuming {n} clips x {DRY_RUN_SECONDS_PER_CLIP}s")
        costs = [tts.tts_cost_entry(s, chars, estimated=True, note=f"{len(clips) or ctx.max_clips} clips, {chars} chars"),
                 ai_video_reference_cost(secs, note=f"{len(clips) or ctx.max_clips} clips x ~{secs / max(len(clips) or ctx.max_clips, 1):.0f}s")]
        ctx.log(f"[{STAGE}] dry-run: TTS ~${costs[0].usd:.4f} for {chars} chars; "
                f"an AI video API would be ~${costs[1].usd:.2f} for the same {secs:.0f}s - not called")
        provider = make_provider(s)
        if provider is not None:
            n = len(clips) or ctx.max_clips
            est = provider.estimate(n, s.ai_shot_seconds)
            costs.extend(est)
            ctx.log(f"[{STAGE}] dry-run: {n} AI opening shot(s) x {s.ai_shot_seconds:.0f}s via {provider.name}/"
                    f"{provider.model}: ${sum(c.usd for c in est):.4f}, ~{sum(c.quantity for c in est):.0f} {est[0].unit}s")
        return StageResult(outputs={}, costs=costs,
                           meta={"clips_planned": len(clips) or ctx.max_clips, "chars": chars, "est_seconds": secs,
                                 "reference_ai_video_usd": costs[1].usd})

    scripts = load_scripts(ctx)
    clips = accepted_clips(scripts, ctx.max_clips)
    ctx.log(f"[{STAGE}] {len(clips)} clip(s) to render with {prov}/{model}, "
            f"{s.video_width}x{s.video_height}")
    clips_dir = ctx.path(CLIPS_DIR)
    clips_dir.mkdir(parents=True, exist_ok=True)
    provider = make_provider(s)  # None unless FINVID_AI_VIDEO is set
    if provider is not None:
        ctx.log(f"[{STAGE}] AI opening shot per clip via {provider.name}/{provider.model} "
                f"({s.ai_shot_width}x{s.ai_shot_height}, {s.ai_shot_seconds:.0f}s)")
    costs: list[CostEntry] = []
    rendered: list[RenderedClip] = []
    outputs: dict[str, str] = {}
    committed = 0.0  # everything actually spent in this stage (TTS + AI shots)
    tts_total = 0.0
    total_seconds = 0.0
    reference_total = 0.0

    for clip in clips:
        sid = clip.segment_id
        chars = len(clip.full_text)
        cost = tts.tts_cost_entry(s, chars, note=f"clip {sid}: {chars} chars")
        ctx.charge(committed + cost.usd, f"TTS for clip {sid}")  # pre-flight, before any call

        stem = f"clip_{sid:02d}"
        texts = [clip.hook] + [ln.text for ln in clip.lines]
        line_audio = tts.synthesize_lines(s, texts, clips_dir, stem)
        committed += cost.usd
        tts_total += cost.usd
        costs.append(cost)

        chart_png: Path | None = None
        if clip.chart is not None:
            chart_png = render_chart(clip.chart, clips_dir / f"chart_{sid:02d}.png")

        shot_path: Path | None = None
        shot_secs = 0.0
        if provider is not None:
            # same gate as everything else: only selected, plagiarism-clean clips get a shot,
            # one per clip, pre-flight charged (free providers charge $0), cached by prompt hash
            est = provider.estimate(1, s.ai_shot_seconds)
            ctx.charge(committed + sum(c.usd for c in est), f"AI shot for clip {sid}")
            shot = provider.generate(shot_prompt(clip), clips_dir / f"ai_{sid:02d}.mp4",
                                     seconds=s.ai_shot_seconds, seed=sid, log=ctx.log)
            costs.extend(shot.costs)
            committed += sum(c.usd for c in shot.costs)
            shot_path, shot_secs = shot.path, shot.wall_seconds
            ctx.log(f"[{STAGE}] clip {sid}: AI shot {'cache hit' if shot.cached else f'{shot.wall_seconds:.0f}s'} "
                    f"-> {shot.path.name}")

        mp4 = clips_dir / f"{stem}.mp4"
        duration = compose_clip(s, clip, line_audio, chart_png, mp4, scripts.source_name, intro_video=shot_path)
        for _, p, _ in line_audio:  # per-line mp3s were only needed for timing
            p.unlink(missing_ok=True)

        ref = ai_video_reference_cost(round(duration, 1), note=f"clip {sid}")
        costs.append(ref)
        total_seconds += duration
        reference_total += ref.usd
        rendered.append(RenderedClip(segment_id=sid, title=clip.title, video_path=_rel(ctx, mp4),
                                     duration_sec=round(duration, 2),
                                     chart_path=_rel(ctx, chart_png) if chart_png else None,
                                     ai_shot_path=_rel(ctx, shot_path) if shot_path else None,
                                     ai_shot_provider=provider.name if provider else None,
                                     ai_shot_seconds=round(shot_secs, 1)))
        outputs[f"clip_{sid}"] = _rel(ctx, mp4)
        if shot_path:
            outputs[f"ai_{sid}"] = _rel(ctx, shot_path)
        ctx.log(f"[{STAGE}] clip {sid} '{clip.title}': {duration:.1f}s, {len(texts)} lines, "
                f"{'chart' if chart_png else 'no chart'}, TTS ${cost.usd:.4f} "
                f"| running: spent ${committed:.4f} vs AI-video-API ~${reference_total:.2f}")

    out = RenderOutput(video_id=ctx.video_id, clips=rendered)
    ctx.path(RENDER_FILE).write_text(json.dumps(out.model_dump(), ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    outputs = {"render": RENDER_FILE, **outputs}
    ctx.log(f"[{STAGE}] rendered {len(rendered)} clip(s), {total_seconds:.0f}s total, "
            f"actual ${committed:.4f} vs ~${reference_total:.2f} with an AI video API")
    return StageResult(outputs=outputs, costs=costs,
                       meta={"clips_rendered": len(rendered), "total_seconds": round(total_seconds, 1),
                             "tts_usd": round(tts_total, 6),
                             "spent_usd": round(committed, 6),
                             "ai_video": provider.name if provider else "none",
                             "ai_shot_gpu_seconds": round(sum(c.ai_shot_seconds for c in rendered), 1),
                             "reference_ai_video_usd": round(reference_total, 4)})
