"""Stage 4: accepted scripts -> 1080x1920 vertical MP4s, programmatically (default ~$0).

Per clip:  TTS per line (edge-tts free / OpenAI optional)  ->  matplotlib chart from ChartSpec
           ->  PIL title/subtitle/attribution overlays  ->  one ffmpeg call.
No AI video API is called. We only *record* what a Runway/Kling-class API would have charged for the
same seconds (provider "reference", estimated=True) so the ledger shows the comparison.
Optional footage under the whole clip: AI shots (FINVID_AI_VIDEO, a fixed few per clip) and/or real
stock footage (FINVID_BROLL=pexels, one clip per line, $0) - both go through compose_clip(shots=...).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..context import RunContext
from ..manifest import StageResult
from ..models import CostEntry, RenderedClip, RenderOutput, ScriptClip, ScriptsOutput
from ..pricing import ai_video_reference_cost
from ..render import tts
from ..render.chart import render_chart
from ..render.aivideo import make_provider, normalized_chain
from ..render.broll import make_broll
from ..render.compose import compose_clip

STAGE = "s4_render"
RENDER_VERSION = 4  # bump when layout/encoding changes so cached clips are re-rendered

SCRIPTS_FILE = "03_scripts.json"
DRY_RUN_CHARS_PER_CLIP = 200
DRY_RUN_SECONDS_PER_CLIP = 35
DRY_RUN_LINES_PER_CLIP = 6  # body lines per clip, for the b-roll request estimate without 03_scripts.json
RENDER_FILE = "04_render.json"
CLIPS_DIR = "04_clips"


# ----------------------------------------------------------------------------- config
def stage_config(ctx: RunContext) -> dict[str, Any]:
    s = ctx.settings
    s3 = ctx.manifest.data.get("stages", {}).get("s3_script") or {}
    chain = normalized_chain(s.ai_video)
    return {
        "video_id": ctx.video_id,
        "render_version": RENDER_VERSION,
        "tts_provider": s.tts_provider,
        "tts_voice": s.tts_voice if s.tts_provider == "edge" else s.openai_tts_model,
        "width": s.video_width,
        "height": s.video_height,
        "max_clips": ctx.max_clips,
        "ai_video": ",".join(chain),
        # every knob that changes a generated shot (per-shot caches key on these too, but a stage-level
        # hit never reaches them): providers, size, count, and each backend's model/params
        "ai_shot": {"seconds": s.ai_shot_seconds, "w": s.ai_shot_width, "h": s.ai_shot_height,
                    "fps": s.ai_shot_fps, "per_clip": s.ai_shots_per_clip,
                    "hf": {"space": s.hf_space} if "hf" in chain else None,
                    "pixazo": True if "pixazo" in chain else None,
                    "comfy": {"model": s.comfy_checkpoint, "text_encoder": s.comfy_text_encoder,
                              "workflow": s.comfy_workflow, "steps": s.comfy_steps, "cfg": s.comfy_cfg}
                    if "comfy" in chain else None,
                    "minimax": {"model": s.minimax_model, "resolution": s.minimax_resolution} if "minimax" in chain else None,
                    "kling": {"model": s.kling_model, "mode": s.kling_mode} if "kling" in chain else None,
                    } if chain != ["none"] else None,
        "broll": {"source": s.broll, "max_clip_seconds": s.broll_max_clip_seconds} if s.broll != "none" else None,
        "s3_config_hash": s3.get("config_hash"),
    }


_SUFFIX = " Vertical 9:16 framing, no text, no captions, no logos, no watermarks, no readable faces."
_FALLBACK = ("Cinematic vertical b-roll for a finance news short: a modern Taiwanese city skyline with "
             "apartment towers at dusk, slow smooth camera push-in, soft warm window lights, realistic, "
             "high detail, no people, no text")


def shot_prompts(clip: ScriptClip, n: int) -> list[str]:
    """n prompts for n scene windows. Shot 0 is the opening Pass B described in `ai_shot`; the
    others are built from the `visual` keywords of the first line in each scene (deterministic,
    no extra LLM call). Older scripts without those fields get a neutral finance b-roll prompt."""
    first = clip.ai_shot.strip() or _FALLBACK
    prompts = [first + _SUFFIX]
    if n <= 1:
        return prompts
    body = clip.lines
    groups = [body[j * len(body) // (n - 1):(j + 1) * len(body) // (n - 1)] for j in range(n - 1)]
    for g in groups:
        kw = next((ln.visual.strip() for ln in g if ln.visual.strip()), "")
        if kw:
            prompts.append(f"Cinematic vertical b-roll: {kw}. Slow smooth camera movement, soft natural "
                           f"light, realistic, high detail, shallow depth of field." + _SUFFIX)
        else:
            prompts.append(first + _SUFFIX)
    return prompts[:n]


def broll_queries(clip: ScriptClip, with_hook: bool) -> list[str]:
    """One stock-footage query per scene window, in playback order. Body lines use their own
    `visual` keywords (empty -> the backend's generic query). The hook, when it is not an AI shot,
    borrows the first non-empty `visual` (or the `ai_shot` description): Pass B writes no keywords for it."""
    body = [ln.visual.strip() for ln in clip.lines]
    if not with_hook:
        return body
    hook = next((kw for kw in body if kw), "") or clip.ai_shot.strip()
    return [hook] + body


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


def _rel_any(ctx: RunContext, p: Path) -> str:
    """Like _rel, but for the b-roll cache shared next to the workdirs ("../_broll/<id>_8s.mp4")."""
    return Path(os.path.relpath(p, ctx.workdir)).as_posix()


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
        broll = make_broll(s, cache_dir=ctx.workdir.parent / "_broll")
        per_clip = 1 if broll is not None else s.ai_shots_per_clip  # with b-roll the AI shot is the hook only
        if provider is not None:
            n = (len(clips) or ctx.max_clips) * per_clip
            est = provider.estimate(n, s.ai_shot_seconds)
            costs.extend(est)
            ctx.log(f"[{STAGE}] dry-run: {n} AI shot(s) ({per_clip} per clip) x {s.ai_shot_seconds:.0f}s "
                    f"via {provider.name}/{provider.model}: ${sum(c.usd for c in est):.4f}, "
                    f"~{sum(c.quantity for c in est):.0f} {est[0].unit}s")
        if broll is not None:
            lines = sum(len(c.lines) for c in clips) if clips else ctx.max_clips * DRY_RUN_LINES_PER_CLIP
            n = lines + (0 if provider is not None else (len(clips) or ctx.max_clips))  # + one hook query per clip
            est = broll.estimate(n)
            costs.extend(est)
            ctx.log(f"[{STAGE}] dry-run: ~{n} Pexels requests, $0 (one per line"
                    f"{'' if provider is not None else ' + hook'}; cached queries make none)")
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
    broll = make_broll(s, cache_dir=ctx.workdir.parent / "_broll")  # None unless FINVID_BROLL is set
    shots_per_clip = s.ai_shots_per_clip
    if broll is not None:
        broll.require_key()  # fail here, not after clip 1's TTS
        if provider is not None:
            shots_per_clip = 1
            ctx.log(f"[{STAGE}] FINVID_BROLL={broll.name}: the AI shot is the hook only, "
                    f"FINVID_AI_SHOTS_PER_CLIP={s.ai_shots_per_clip} ignored; every body line gets stock footage")
        ctx.log(f"[{STAGE}] stock footage via {broll.name}: one clip per line (query = the line's `visual`), "
                f"trimmed to {s.broll_max_clip_seconds:g}s, cached in {broll.cache_dir}")
    if provider is not None:
        ctx.log(f"[{STAGE}] {shots_per_clip} AI shot(s) per clip via {provider.name}/{provider.model} "
                f"({s.ai_shot_width}x{s.ai_shot_height}, {s.ai_shot_seconds:.0f}s each), footage under the whole clip")
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

        shot_paths: list[Path] = []
        shot_secs = 0.0
        shot_usd = 0.0
        if provider is not None:
            # same gate as everything else: only selected, plagiarism-clean clips get shots,
            # a fixed number per clip, pre-flight charged (free providers charge $0), cached by prompt hash
            for k, prompt in enumerate(shot_prompts(clip, shots_per_clip)):
                est = provider.estimate(1, s.ai_shot_seconds)
                ctx.charge(committed + sum(c.usd for c in est), f"AI shot {k + 1} for clip {sid}")
                name = f"ai_{sid:02d}.mp4" if k == 0 else f"ai_{sid:02d}_{k}.mp4"
                shot = provider.generate(prompt, clips_dir / name, seconds=s.ai_shot_seconds,
                                         seed=sid * 10 + k, log=ctx.log)
                costs.extend(shot.costs)
                committed += sum(c.usd for c in shot.costs)
                shot_usd += sum(c.usd for c in shot.costs)
                shot_paths.append(shot.path)
                shot_secs += shot.wall_seconds
                ctx.log(f"[{STAGE}] clip {sid}: AI shot {k + 1}/{shots_per_clip} "
                        f"{'cache hit' if shot.cached else f'{shot.wall_seconds:.0f}s'} -> {shot.path.name}")

        broll_paths: list[Path] = []
        scene_shots = list(shot_paths)  # playback order: hook first, then one per body line
        if broll is not None:
            # one stock clip per line -> compose gets len(lines)+1 shots = one scene window per line.
            # A search with no result reuses the previous scene's clip, so the footage never has a gap.
            for q in broll_queries(clip, with_hook=not shot_paths):
                p = broll.fetch(q, ctx.log)
                if p is None and not scene_shots:
                    p = broll.fetch("", ctx.log)  # hook with nothing found: the generic query
                if p is None:
                    if not scene_shots:
                        raise RuntimeError(f"{broll.name}: no footage at all for clip {sid}")
                    p = scene_shots[-1]
                scene_shots.append(p)
                broll_paths.append(p)
            ctx.log(f"[{STAGE}] clip {sid}: {len(broll_paths)} stock clip(s) from {broll.name} "
                    f"({broll.requests} request(s) so far) -> {len(scene_shots)} scenes, one per line")

        mp4 = clips_dir / f"{stem}.mp4"
        duration = compose_clip(s, clip, line_audio, chart_png, mp4, scripts.source_name, shots=scene_shots or None)
        for _, p, _ in line_audio:  # per-line mp3s were only needed for timing
            p.unlink(missing_ok=True)

        ref = ai_video_reference_cost(round(duration, 1), note=f"clip {sid}")
        costs.append(ref)
        total_seconds += duration
        reference_total += ref.usd
        rendered.append(RenderedClip(segment_id=sid, title=clip.title, video_path=_rel(ctx, mp4),
                                     duration_sec=round(duration, 2),
                                     chart_path=_rel(ctx, chart_png) if chart_png else None,
                                     ai_shot_path=_rel(ctx, shot_paths[0]) if shot_paths else None,
                                     ai_shot_paths=[_rel(ctx, sp) for sp in shot_paths],
                                     ai_shot_provider=provider.name if provider else None,
                                     ai_shot_seconds=round(shot_secs, 1), ai_shot_usd=round(shot_usd, 4),
                                     broll_paths=[_rel_any(ctx, bp) for bp in broll_paths],
                                     broll_provider=broll.name if broll else None))
        outputs[f"clip_{sid}"] = _rel(ctx, mp4)
        for k, sp in enumerate(shot_paths):
            outputs[f"ai_{sid}_{k}"] = _rel(ctx, sp)
        ctx.log(f"[{STAGE}] clip {sid} '{clip.title}': {duration:.1f}s, {len(texts)} lines, "
                f"{'chart' if chart_png else 'no chart'}, TTS ${cost.usd:.4f} "
                f"| running: spent ${committed:.4f} vs AI-video-API ~${reference_total:.2f}")

    if broll is not None:
        costs.append(broll.cost_entry())
        ctx.log(f"[{STAGE}] {broll.name}: {broll.requests} API request(s), {broll.downloads} download(s), $0")
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
                             "broll": broll.name if broll else "none",
                             "broll_requests": broll.requests if broll else 0,
                             "reference_ai_video_usd": round(reference_total, 4)})
