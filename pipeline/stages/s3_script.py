"""Stage 3: transcript -> a few rewritten short-video scripts.

Cost design (this is the step the brief calls "usually the most expensive"):

  Pass A  (cheap model, 1 call)   whole transcript -> 5-10 topic candidates + hook_score
  Gate    (pure Python, $0)       threshold + de-dup + prefer chart data -> top max_clips
  Pass B  (strong model, K calls) only the selected segments get a full script
  Check   (pure Python, $0)       n-gram / LCS plagiarism gate, 1 retry, else reject

Every LLM call is pre-flighted through ctx.charge() so the budget guard fires before
money is spent, and every clip carries a mandatory source-attribution line that is
injected by code if the model forgets it.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .. import numbers, plagiarism, pricing
from ..context import RunContext
from ..llm import chat_json
from ..manifest import StageResult, config_hash
from ..models import (
    AudioInfo, ChartSpec, CostEntry, DataPoint, ScriptClip, ScriptLine, ScriptsOutput,
    Transcript, TranscriptSegment, TopicSegment,
)

STAGE = "s3_script"
PROMPT_VERSION = 7  # bump whenever a prompt below changes -> invalidates the s3 cache

TRANSCRIPT_FILE = "02_transcript.json"
INFO_FILE = "01_info.json"
OUTPUT_FILE = "03_scripts.json"
PASS_A_CACHE_FILE = "03a_pass_a.json"  # scratch cache of the cheap pass, see execute()

ATTRIBUTION_TEMPLATE = "根據{source}報導指出"

# token budget assumptions used for dry-run and pre-flight estimates
PASS_A_PROMPT_TOKENS = 600     # system prompt + formatting overhead
PASS_A_OUTPUT_TOKENS = 3300    # 8 segments of JSON + reasoning tokens (measured 3271 on the demo video)
PASS_B_INPUT_TOKENS = 1200     # system prompt + segment brief + ~2 min of transcript
PASS_B_OUTPUT_TOKENS = 1600    # one script JSON + reasoning tokens on gpt-5 (measured 1534-1753)
FALLBACK_TRANSCRIPT_CHARS = 3000
ZH_CHARS_PER_SEC = 4.0         # spoken Mandarin, used only when s2 gave a duration but no text

PLAGIARISM_CONTEXT_SEC = 20.0  # widen the transcript window when checking a clip
DUP_JACCARD = 0.5              # char-bigram Jaccard on topic+summary -> "duplicate"
DUP_SHARED_NUMBERS = 2         # >= this many identical (value, unit) data points -> same story
DUP_TIME_OVERLAP = 0.5         # >= this share of the shorter segment's time range overlaps -> same footage

# ---------------------------------------------------------------- prompts

PASS_A_SYSTEM = """你是財經短影音製作人。使用者會給你一段附時間戳的節目逐字稿（繁體中文）{topic_hint}。
任務：把逐字稿切成 5 到 10 個「主題連貫」的段落，每段輸出以下欄位：
- id：整數，從 1 開始遞增
- start、end：該段在逐字稿中的起訖秒數，必須是純數字（例如 785，不要寫 13:05）
- topic：主題，15 字以內
- summary：摘要，60 字以內
- key_points：2 到 4 條重點（字串陣列）
- data_points：段落中提到的每一個具體數字，格式 {"label": 說明, "value": 數值, "unit": 單位}，value 必須是純數字，例如 {"label": "央行重貼現率", "value": 2.0, "unit": "%"}
- has_chart_data：只有在該段有 2 個以上可以互相比較的數據時才為 true
- hook_score：1 到 5 的整數，代表用這個主題做 30 秒直式短影音，能讓人停下滑動的機率。有具體數字、令人意外的說法、可以馬上執行的建議給高分；閒聊、主持人寒暄、廣告、節目宣傳給 1 分。
只回傳 JSON，格式為 {"segments": [...]}，不要輸出任何其他文字。"""

PASS_B_SYSTEM = """你是財經短影音編劇。使用者會給你一個主題的摘要、重點、數據，以及對應的原始逐字稿片段。
請寫一支 30 到 45 秒的直式短影音腳本，嚴格遵守以下規則：
1. 一定要用自己的話改寫。不可以照抄逐字稿裡的任何句子或片語，要重組句子結構、換用不同的措辭與語序；但所有數字必須與原文完全一致，不可以捏造、推算或修改任何數據。
2. hook：第一句，20 字以內，要能讓人停下來。
3. lines：共 5 到 9 句，每句 28 字以內（一行字幕的長度），口語、好唸、一句一個意思。
4. lines 裡必須有一句「完整包含」這個出處句：「{attribution}」，而且要放在第 1 或第 2 句。
5. 如果 has_chart_data 為 true，提供 chart：{{"type": "bar" 或 "line", "title": 圖表標題, "y_label": Y 軸說明, "series": [{{"name": 系列名稱, "points": [{{"label": 標籤, "value": 數字, "unit": 單位}}]}}]}}，只能使用 data_points 或逐字稿裡真的出現過的數字；否則 chart 為 null。
6. title：15 字以內。est_seconds：預估播放秒數（數字）。
7. 台詞裡的金額用台灣口語單位寫：1000萬、2380萬、40萬/坪、1.5萬，不要展開成 10000000 或用科學記號。同一張 chart 的所有 points 必須是同一個單位，不同單位的數字不要放進同一張圖。
只回傳 JSON，格式如下，不要輸出任何其他文字：
{{"segment_id": 整數, "title": "...", "hook": "...", "lines": [{{"text": "...", "emphasis": false}}], "chart": null 或 chart 物件, "attribution": "{attribution}", "est_seconds": 35}}"""

REWRITE_INSTRUCTION = """
【重寫要求】上一版腳本與逐字稿的重疊太高：{n}-gram 重疊率 {overlap:.0%}（上限 {max_overlap:.0%}），最長相同字串 {lcs} 字（上限 {max_lcs} 字），相同的片段是：「{lcs_text}」。
請徹底改寫：換掉這個片段和所有相似句子，避免出現任何與逐字稿相同的連續 {n} 字以上片語，改變句型與用詞，但數字必須保持完全一致。"""


# ---------------------------------------------------------------- config

def stage_config(ctx: RunContext) -> dict[str, Any]:
    s = ctx.settings
    return {
        "video_id": ctx.video_id,
        "prompt_version": PROMPT_VERSION,
        "llm_cheap_model": s.llm_cheap_model,
        "llm_strong_model": s.llm_strong_model,
        "hook_threshold": s.hook_threshold,
        "max_clips": ctx.max_clips,
        "plagiarism_ngram": s.plagiarism_ngram,
        "plagiarism_max_overlap": s.plagiarism_max_overlap,
        "plagiarism_max_lcs": s.plagiarism_max_lcs,
        "s2_config_hash": ctx.manifest.data["stages"].get("s2_transcribe", {}).get("config_hash"),
    }


# ---------------------------------------------------------------- helpers

def source_name_for(info: AudioInfo | None) -> str:
    """The programme the clip cites. The YouTube channel is 健康2.0 / HEALTH 2.0 but the
    broadcaster is TVBS, so cite the broadcaster + programme; otherwise the channel name."""
    if info is None:
        return "原始節目"
    ch = info.channel or ""
    if "HEALTH 2.0" in ch.upper() or "健康2.0" in ch or "健康 2.0" in ch:
        return "TVBS《健康2.0》"
    return ch or "原始節目"


def attribution_for(source: str) -> str:
    return ATTRIBUTION_TEMPLATE.format(source=source)


def _mmss(sec: float) -> str:
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def format_transcript(segments: list[TranscriptSegment], *, max_chars: int = 60,
                      max_span: float = 15.0) -> str:
    """Render as `[mm:ss] text` lines, merging tiny STT segments so Pass A's prompt is a
    few hundred lines instead of thousands."""
    lines: list[str] = []
    buf = ""
    buf_start = 0.0
    buf_end = 0.0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if not buf:
            buf, buf_start, buf_end = text, seg.start, seg.end
            continue
        if len(buf) + len(text) > max_chars or seg.end - buf_start > max_span:
            lines.append(f"[{_mmss(buf_start)}] {buf}")
            buf, buf_start, buf_end = text, seg.start, seg.end
        else:
            buf += text
            buf_end = seg.end
    if buf:
        lines.append(f"[{_mmss(buf_start)}] {buf}")
    return "\n".join(lines)


def transcript_slice(transcript: Transcript, start: float, end: float, pad: float = 0.0) -> str:
    """Raw transcript text overlapping [start-pad, end+pad]."""
    lo, hi = start - pad, end + pad
    return "".join(s.text for s in transcript.segments if s.end > lo and s.start < hi)


def _as_float(v: Any, default: float = 0.0) -> float:
    """Accept numbers, numeric strings, and mm:ss / h:mm:ss timestamps (the model sometimes
    echoes the [mm:ss] format it saw in the transcript)."""
    try:
        if isinstance(v, str):
            v = v.replace(",", "").replace("%", "").strip()
            if ":" in v:
                parts = [float(x) for x in v.split(":")]
                total = 0.0
                for x in parts:
                    total = total * 60 + x
                return total
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是")
    return bool(v)


def _as_str_list(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    return []


def parse_data_points(raw: Any) -> list[DataPoint]:
    out: list[DataPoint] = []
    if not isinstance(raw, list):
        return out
    for dp in raw:
        if not isinstance(dp, dict):
            continue
        val = dp.get("value")
        try:
            if isinstance(val, str):
                val = val.replace(",", "").replace("%", "").strip()
            fval = float(val)
        except (TypeError, ValueError):
            continue  # value must be numeric; drop it rather than fabricate
        out.append(DataPoint(label=str(dp.get("label", "")), value=fval, unit=str(dp.get("unit", "") or "")))
    return out


def parse_pass_a(payload: dict[str, Any], log) -> list[TopicSegment]:
    raw = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        log("[s3] Pass A returned no 'segments' list")
        return []
    out: list[TopicSegment] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            log(f"[s3] Pass A: dropped non-object segment #{idx}")
            continue
        data_points = parse_data_points(item.get("data_points"))
        has_chart = _as_bool(item.get("has_chart_data")) and len(data_points) >= 2
        try:
            seg = TopicSegment(
                id=_as_int(item.get("id"), idx),
                start=_as_float(item.get("start")),
                end=_as_float(item.get("end")),
                topic=str(item.get("topic", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                key_points=_as_str_list(item.get("key_points")),
                data_points=data_points,
                has_chart_data=has_chart,
                hook_score=max(1, min(5, _as_int(item.get("hook_score"), 1))),
            )
        except ValidationError as e:
            log(f"[s3] Pass A: dropped segment #{idx} ({e.errors()[0].get('msg', e)})")
            continue
        if not seg.topic or seg.end <= seg.start:
            log(f"[s3] Pass A: dropped segment #{idx} (empty topic or bad time range)")
            continue
        out.append(seg)
    # make ids unique and stable
    seen: set[int] = set()
    for i, seg in enumerate(out, start=1):
        if seg.id in seen:
            seg.id = max(seen) + 1
        seen.add(seg.id)
    return out


# ---------------------------------------------------------------- gate ($0)

def _bigrams(s: str) -> set[str]:
    s = plagiarism.normalise(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def topic_similarity(a: TopicSegment, b: TopicSegment) -> float:
    x = _bigrams(a.topic + a.summary)
    y = _bigrams(b.topic + b.summary)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def shared_numbers(a: TopicSegment, b: TopicSegment) -> int:
    """Identical (value, unit) pairs. Paraphrased Chinese summaries defeat text similarity
    (a clear duplicate scored Jaccard 0.19 on the demo video), but two segments quoting the
    same figures are telling the same story."""
    x = {(p.value, p.unit) for p in a.data_points}
    y = {(p.value, p.unit) for p in b.data_points}
    return len(x & y)


def time_overlap(a: TopicSegment, b: TopicSegment) -> float:
    """Overlap as a share of the shorter segment (Pass A sometimes returns overlapping ranges)."""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shorter = max(1e-6, min(a.end - a.start, b.end - b.start))
    return inter / shorter


def duplicate_reason(seg: TopicSegment, chosen: TopicSegment) -> str | None:
    if topic_similarity(seg, chosen) >= DUP_JACCARD:
        return f"duplicate of segment {chosen.id} (text similarity)"
    if shared_numbers(seg, chosen) >= DUP_SHARED_NUMBERS:
        return f"duplicate of segment {chosen.id} (same figures)"
    if time_overlap(seg, chosen) >= DUP_TIME_OVERLAP:
        return f"duplicate of segment {chosen.id} (time overlap)"
    return None


def select_segments(segments: list[TopicSegment], *, threshold: int,
                    max_clips: int) -> list[TopicSegment]:
    """Pure-Python selection gate. Mutates selected/skip_reason on each segment and
    returns the selected ones in transcript order."""
    for seg in segments:
        seg.selected = False
        seg.skip_reason = ""
    # chart data first, then hook score, then earlier in the show
    ranked = sorted(segments, key=lambda s: (s.has_chart_data, s.hook_score, -s.start), reverse=True)
    chosen: list[TopicSegment] = []
    for seg in ranked:
        if seg.hook_score < threshold:
            seg.skip_reason = f"hook_score {seg.hook_score} < threshold {threshold}"
            continue
        dup = next((r for r in (duplicate_reason(seg, c) for c in chosen) if r), None)
        if dup is not None:
            seg.skip_reason = dup
            continue
        if len(chosen) >= max_clips:
            seg.skip_reason = "over max_clips budget"
            continue
        seg.selected = True
        chosen.append(seg)
    return sorted(chosen, key=lambda s: s.start)


def _log_gate_table(segments: list[TopicSegment], log) -> None:
    log("[s3] gate: id  score chart  topic                 -> decision")
    for seg in sorted(segments, key=lambda s: s.id):
        decision = "SELECTED" if seg.selected else f"skip: {seg.skip_reason}"
        log(f"[s3]       {seg.id:<3} {seg.hook_score:<5} {'yes' if seg.has_chart_data else 'no ':<5}  "
            f"{seg.topic[:20]:<20}  -> {decision}")


# ---------------------------------------------------------------- Pass B parsing

_SCI = re.compile(r"(\d+(?:\.\d+)?)[eE]\+?(\d+)")
_BIG = re.compile(r"(?<![\d.])(\d{5,})(?![\d.])")


def humanize_numbers(text: str) -> str:
    """Deterministic clean-up of numbers the model expanded: 1e+07 -> 1000萬, 23800000 -> 2380萬,
    400000 -> 40萬. Numbers below 10000 are left alone (15000 is fine to say)."""
    def _sci(m: re.Match) -> str:
        return str(int(float(m.group(1)) * (10 ** int(m.group(2)))))
    text = _SCI.sub(_sci, text)

    def _big(m: re.Match) -> str:
        n = int(m.group(1))
        if n < 100000:
            return m.group(1)
        if n % 100000000 == 0:
            return f"{n // 100000000}億"
        if n >= 100000000:
            return f"{n / 100000000:g}億"
        w = n / 10000
        return f"{int(w)}萬" if w == int(w) else f"{w:g}萬"
    return _BIG.sub(_big, text)


def unify_chart_units(points: list[DataPoint], log=None) -> list[DataPoint]:
    """Keep only points sharing the most common unit; a bar chart mixing 元/坪 and 元 is wrong."""
    if len(points) < 2:
        return points
    units = [p.unit for p in points]
    major = max(set(units), key=units.count)
    kept = [p for p in points if p.unit == major]
    if log and len(kept) != len(points):
        log(f"[s3] chart: dropped {len(points) - len(kept)} point(s) whose unit != '{major}'")
    return kept if len(kept) >= 2 else points[:0]


def parse_pass_b(payload: dict[str, Any], seg: TopicSegment, attribution: str) -> ScriptClip:
    if not isinstance(payload, dict):
        raise ValueError("Pass B reply is not a JSON object")
    lines: list[ScriptLine] = []
    for item in payload.get("lines") or []:
        if isinstance(item, str):
            text = item.strip()
            emphasis = False
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            emphasis = _as_bool(item.get("emphasis"))
        else:
            continue
        if text:
            lines.append(ScriptLine(text=humanize_numbers(text), emphasis=emphasis))
    chart: ChartSpec | None = None
    raw_chart = payload.get("chart")
    if seg.has_chart_data and isinstance(raw_chart, dict):
        series = []
        for s in raw_chart.get("series") or []:
            if not isinstance(s, dict):
                continue
            pts = unify_chart_units(parse_data_points(s.get("points")))
            if pts:
                series.append({"name": str(s.get("name", "")), "points": pts})
        if series:
            ctype = str(raw_chart.get("type", "bar")).lower()
            chart = ChartSpec(type=ctype if ctype in ("bar", "line") else "bar",
                              title=str(raw_chart.get("title") or seg.topic),
                              y_label=str(raw_chart.get("y_label", "") or ""), series=series)
    hook = humanize_numbers(str(payload.get("hook", "")).strip())
    if not hook and lines:
        hook = lines[0].text
    if lines and plagiarism.normalise(lines[0].text) == plagiarism.normalise(hook):
        lines.pop(0)  # the hook is spoken first anyway; don't say it twice
    clip = ScriptClip(
        segment_id=seg.id,
        title=humanize_numbers(str(payload.get("title") or seg.topic).strip()),
        hook=hook,
        lines=lines,
        chart=chart,
        attribution=attribution,
        est_seconds=_as_float(payload.get("est_seconds"), 30.0) or 30.0,
    )
    ensure_attribution(clip, attribution)
    if not clip.lines:
        raise ValueError("Pass B reply has no lines")
    return clip


def ensure_attribution(clip: ScriptClip, attribution: str) -> None:
    """Never rely on the LLM for the legal bit: force the attribution field and make sure
    some line carries the sentence (inject as line 2, i.e. right after the hook line)."""
    clip.attribution = attribution
    if any(attribution in line.text for line in clip.lines):
        return
    clip.lines.insert(min(1, len(clip.lines)), ScriptLine(text=attribution, emphasis=False))


# ---------------------------------------------------------------- estimates

def _load_pass_a_cache(ctx: RunContext, key: str) -> dict[str, Any] | None:
    p = ctx.path(PASS_A_CACHE_FILE)
    if not p.exists():
        return None
    try:
        cached = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if cached.get("key") != key or not isinstance(cached.get("payload"), dict):
        return None
    return cached["payload"]


def _pass_a_estimate(model: str, transcript_chars: int, estimated: bool = True) -> list[CostEntry]:
    tokens_in = pricing.estimate_tokens_zh("x" * transcript_chars) + PASS_A_PROMPT_TOKENS
    return pricing.llm_cost(STAGE, "openai", model, tokens_in, PASS_A_OUTPUT_TOKENS,
                            estimated=estimated, note="pass A (cheap model, whole transcript)")


def _pass_b_estimate(model: str, clips: int, estimated: bool = True, note: str = "") -> list[CostEntry]:
    return pricing.llm_cost(STAGE, "openai", model, PASS_B_INPUT_TOKENS * clips,
                            PASS_B_OUTPUT_TOKENS * clips, estimated=estimated,
                            note=note or f"pass B (strong model) x {clips} clips")


def _pass_b_per_clip_usd(model: str) -> float:
    return sum(c.usd for c in _pass_b_estimate(model, 1))


def _dry_run(ctx: RunContext) -> StageResult:
    s = ctx.settings
    tpath = ctx.path(TRANSCRIPT_FILE)
    chars = FALLBACK_TRANSCRIPT_CHARS
    basis = "assumed"
    if tpath.exists():
        try:
            chars = len(Transcript.model_validate_json(tpath.read_text(encoding="utf-8")).text)
            basis = TRANSCRIPT_FILE
        except (ValidationError, ValueError, OSError):
            pass
    else:
        meta = getattr(ctx.stage_results.get("s2_transcribe"), "meta", None) or {}
        for key in ("transcript_chars", "chars", "text_chars", "estimated_chars"):
            if isinstance(meta.get(key), (int, float)) and meta[key] > 0:
                chars, basis = int(meta[key]), f"s2 meta.{key}"
                break
        else:
            for key in ("seconds_estimated", "seconds_billed", "duration_sec", "processed_duration_sec"):
                if isinstance(meta.get(key), (int, float)) and meta[key] > 0:
                    chars, basis = int(meta[key] * ZH_CHARS_PER_SEC), f"s2 meta.{key} x {ZH_CHARS_PER_SEC} chars/s"
                    break
    costs = _pass_a_estimate(s.llm_cheap_model, chars) + _pass_b_estimate(s.llm_strong_model, ctx.max_clips)
    total = sum(c.usd for c in costs)
    ctx.log(f"[s3] dry-run: transcript ~{chars} chars ({basis}); pass A on {s.llm_cheap_model} + "
            f"pass B x{ctx.max_clips} on {s.llm_strong_model} ~= ${total:.4f}")
    return StageResult(outputs={}, costs=costs, meta={
        "transcript_chars_basis": basis, "transcript_chars": chars,
        "pass_b_clips": ctx.max_clips, "estimated_usd": round(total, 6),
    })


# ---------------------------------------------------------------- execute

def execute(ctx: RunContext) -> StageResult:
    if ctx.dry_run:
        return _dry_run(ctx)

    s = ctx.settings
    transcript = Transcript.model_validate_json(ctx.path(TRANSCRIPT_FILE).read_text(encoding="utf-8"))
    info: AudioInfo | None = None
    ipath = ctx.path(INFO_FILE)
    if ipath.exists():
        try:
            info = AudioInfo.model_validate_json(ipath.read_text(encoding="utf-8"))
        except (ValidationError, ValueError):
            ctx.log(f"[s3] {INFO_FILE} unreadable; attribution falls back to a generic source")
    source = source_name_for(info)
    attribution = attribution_for(source)
    costs: list[CostEntry] = []

    def spent() -> float:
        return sum(c.usd for c in costs)

    # ---- Pass A: cheap model reads the whole transcript once
    transcript_lines = format_transcript(transcript.segments)
    est_a = sum(c.usd for c in _pass_a_estimate(s.llm_cheap_model, len(transcript.text)))
    ctx.charge(est_a, f"s3 Pass A ({s.llm_cheap_model})")
    user_a = (f"節目：{source}\n影片長度：{_mmss(transcript.duration_sec)}\n\n逐字稿：\n{transcript_lines}")
    # Pass A has its own on-disk cache keyed by (model, prompts): if Pass B aborts half-way
    # (budget guard, network) or only the Pass B prompt changes, the cheap pass is not paid again.
    # the topic hint comes from the video title, not a hard-coded theme, so another finance show works unchanged
    pa_system = PASS_A_SYSTEM.replace("{topic_hint}", f"，節目標題：「{info.title}」" if info and info.title else "")
    pass_a_key = config_hash({"model": s.llm_cheap_model, "system": pa_system, "user": user_a})
    payload_a = None if ctx.force else _load_pass_a_cache(ctx, pass_a_key)
    if payload_a is not None:
        ctx.log(f"[s3] Pass A: reusing {PASS_A_CACHE_FILE} (same model + prompt), $0")
    else:
        payload_a, cost_a = chat_json(s, s.llm_cheap_model, pa_system, user_a, stage=STAGE, note="pass A")
        costs.extend(cost_a)
        ctx.path(PASS_A_CACHE_FILE).write_text(
            json.dumps({"key": pass_a_key, "payload": payload_a}, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.log(f"[s3] Pass A: cost ${sum(c.usd for c in cost_a):.4f}")
    segments = parse_pass_a(payload_a, ctx.log)
    ctx.log(f"[s3] Pass A: {len(segments)} candidate segments")

    # ---- Gate: $0
    selected = select_segments(segments, threshold=s.hook_threshold, max_clips=ctx.max_clips)
    _log_gate_table(segments, ctx.log)
    per_clip = _pass_b_per_clip_usd(s.llm_strong_model)
    skipped = len(segments) - len(selected)
    saved = round(skipped * per_clip, 4)
    ctx.log(f"[s3] Pass B skipped for {skipped}/{len(segments)} segments -> saved ~${saved:.4f} "
            f"(~${per_clip:.4f} per clip on {s.llm_strong_model})")

    # ---- Pass B: strong model, only for selected segments
    pb_system = PASS_B_SYSTEM.format(attribution=attribution)
    source_numbers = numbers.numbers_in(transcript.text)  # every figure the show actually states
    clips: list[ScriptClip] = []
    rejected: list[ScriptClip] = []
    unparsable = 0
    for seg in selected:
        user_b = _pass_b_user(seg, transcript, attribution)
        clip = _call_pass_b(ctx, seg, pb_system, user_b, attribution, costs, spent, note=f"pass B seg {seg.id}")
        if clip is None:
            unparsable += 1
            continue
        verify_numbers(clip, seg, source_numbers, ctx.log)
        source_text = transcript_slice(transcript, seg.start, seg.end, pad=PLAGIARISM_CONTEXT_SEC)
        ok, overlap, lcs = _plag(clip, source_text, s)
        if not ok:
            lcs_text = plagiarism.longest_common_substring_text(clip.full_text, source_text)
            ctx.log(f"[s3] seg {seg.id}: too close to transcript (overlap {overlap:.0%} > {s.plagiarism_max_overlap:.0%}"
                    f" or lcs {lcs} > {s.plagiarism_max_lcs}: 「{lcs_text}」) -> asking for a rewrite")
            extra = REWRITE_INSTRUCTION.format(n=s.plagiarism_ngram, overlap=overlap,
                                               max_overlap=s.plagiarism_max_overlap, lcs=lcs,
                                               max_lcs=s.plagiarism_max_lcs, lcs_text=lcs_text)
            clip2 = _call_pass_b(ctx, seg, pb_system, user_b + "\n" + extra, attribution, costs, spent,
                                 note=f"pass B seg {seg.id} rewrite")
            if clip2 is not None:
                clip = clip2
            clip.rewrite_attempts = 1
            ok, overlap, lcs = _plag(clip, source_text, s)
        clip.plagiarism_overlap = round(overlap, 4)
        clip.plagiarism_lcs = lcs
        clip.plagiarism_ok = ok
        if ok:
            clips.append(clip)
            ctx.log(f"[s3] seg {seg.id} '{clip.title}': accepted (overlap {overlap:.0%}, lcs {lcs}, "
                    f"{len(clip.lines)} lines, ~{clip.est_seconds:.0f}s)")
        else:
            rejected.append(clip)
            ctx.log(f"[s3] seg {seg.id} '{clip.title}': rejected: not rewritten enough "
                    f"(overlap {overlap:.0%}, lcs {lcs})")

    out = ScriptsOutput(video_id=ctx.video_id, source_name=source, segments=segments,
                        clips=clips, rejected_clips=rejected)
    ctx.path(OUTPUT_FILE).write_text(json.dumps(out.model_dump(), ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    ctx.log(f"[s3] wrote {OUTPUT_FILE}: {len(clips)} clips, {len(rejected)} rejected, "
            f"total ${spent():.4f}")
    return StageResult(
        outputs={"scripts": OUTPUT_FILE},
        costs=costs,
        meta={
            "source_name": source,
            "title": info.title if info else "",
            "segments_total": len(segments),
            "segments_selected": len(selected),
            "clips_accepted": len(clips),
            "clips_rejected": len(rejected),
            "clips_unparsable": unparsable,
            "chart_points_dropped": sum(1 for c in clips + rejected for t in c.numbers_unverified if t.startswith("chart:")),
            "pass_b_skipped_saved_usd_est": saved,
        },
    )


def _pass_b_user(seg: TopicSegment, transcript: Transcript, attribution: str) -> str:
    dps = "\n".join(f"- {d.label}: {d.value:g}{d.unit}" for d in seg.data_points) or "- （無）"
    kps = "\n".join(f"- {k}" for k in seg.key_points) or "- （無）"
    raw = transcript_slice(transcript, seg.start, seg.end)
    return (
        f"segment_id: {seg.id}\n"
        f"topic: {seg.topic}\n"
        f"summary: {seg.summary}\n"
        f"has_chart_data: {'true' if seg.has_chart_data else 'false'}\n"
        f"出處句（必須完整出現在第 1 或第 2 句）：{attribution}\n\n"
        f"key_points:\n{kps}\n\n"
        f"data_points（數字只能用這些，或逐字稿裡出現的）:\n{dps}\n\n"
        f"原始逐字稿片段（{_mmss(seg.start)}–{_mmss(seg.end)}，只供了解事實，禁止照抄）：\n{raw}"
    )


def _call_pass_b(ctx: RunContext, seg: TopicSegment, system: str, user: str, attribution: str,
                 costs: list[CostEntry], spent, *, note: str) -> ScriptClip | None:
    s = ctx.settings
    tokens_in = pricing.estimate_tokens_zh(system + user)
    est = sum(c.usd for c in pricing.llm_cost(STAGE, "openai", s.llm_strong_model, tokens_in,
                                              PASS_B_OUTPUT_TOKENS, estimated=True))
    ctx.charge(spent() + est, f"s3 {note} ({s.llm_strong_model}, incl. ${spent():.4f} already spent in s3)")
    payload, cost = chat_json(s, s.llm_strong_model, system, user, stage=STAGE, note=note)
    costs.extend(cost)
    try:
        return parse_pass_b(payload, seg, attribution)
    except (ValidationError, ValueError, TypeError) as e:
        ctx.log(f"[s3] seg {seg.id}: Pass B reply unusable ({e}); skipping this clip")
        return None


def verify_numbers(clip: ScriptClip, seg: TopicSegment, source_numbers: set[float], log) -> None:
    """$0 provenance gate. The prompt asks the model to use only figures from the transcript;
    this checks it. Chart values are the legal risk (they get drawn as "data"), so any chart point
    not stated in the transcript is dropped, and a chart left with fewer than two points is dropped
    whole. Only the transcript counts as evidence - Pass A's data_points come from the same model,
    so they cannot vouch for Pass B. Spoken numbers are only flagged (numbers_unverified) -
    a paraphrase like 「差了四千」 is legitimate arithmetic, not fabrication."""
    src = source_numbers
    if clip.chart:
        kept_series = []
        for ser in clip.chart.series:
            pts = []
            for pt in ser.points:
                if numbers.value_stated(pt.value, pt.unit, src):
                    pts.append(pt)
                else:
                    tag = f"chart:{pt.label}={pt.value:g}{pt.unit}"
                    clip.numbers_unverified.append(tag)
                    log(f"[s3] seg {seg.id}: chart point {tag} is not in the transcript -> dropped")
            if len(pts) >= 2:
                ser.points = pts
                kept_series.append(ser)
        clip.chart.series = kept_series
        if not kept_series:
            clip.chart = None
            log(f"[s3] seg {seg.id}: chart dropped (no verifiable points left)")
    spoken = clip.full_text.replace(clip.attribution, "")
    for tok in numbers.unstated_numbers(spoken, src):
        clip.numbers_unverified.append(f"text:{tok}")
    if any(t.startswith("text:") for t in clip.numbers_unverified):
        log(f"[s3] seg {seg.id}: spoken numbers not traceable to the transcript: "
            + ", ".join(t[5:] for t in clip.numbers_unverified if t.startswith("text:")))


def _plag(clip: ScriptClip, source_text: str, s) -> tuple[bool, float, int]:
    return plagiarism.check(clip.full_text, source_text, n=s.plagiarism_ngram,
                            max_overlap=s.plagiarism_max_overlap, max_lcs=s.plagiarism_max_lcs)


_SEG_ID_RE = re.compile(r"segment_id:\s*(\d+)")


def segment_id_from_prompt(user_prompt: str) -> int | None:
    """Helper for tests/fakes: recover which segment a Pass B prompt is about."""
    m = _SEG_ID_RE.search(user_prompt)
    return int(m.group(1)) if m else None
