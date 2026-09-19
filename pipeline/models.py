"""Pydantic models shared by all stages. These are the on-disk JSON contracts."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- cost ledger ----------
class CostEntry(BaseModel):
    stage: str
    provider: str
    model: str
    unit: str  # "minute" | "input_tokens" | "output_tokens" | "characters" | "second" | "call"
    quantity: float
    unit_price_usd: float  # price per single unit
    usd: float
    estimated: bool = False  # True for dry-run / pre-flight estimates
    note: str = ""


# ---------- step 1 ----------
class AudioInfo(BaseModel):
    video_id: str
    title: str
    channel: str
    url: str
    original_duration_sec: float
    processed_duration_sec: float
    audio_path: str
    preprocessing: dict = Field(default_factory=dict)


# ---------- step 2 ----------
class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    video_id: str
    language: str = "zh-TW"
    provider: str
    model: str
    duration_sec: float
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)


# ---------- step 3 ----------
class DataPoint(BaseModel):
    label: str
    value: float
    unit: str = ""


class TopicSegment(BaseModel):
    """Output of Pass A: one candidate topic from the transcript."""
    id: int
    start: float
    end: float
    topic: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    data_points: list[DataPoint] = Field(default_factory=list)
    has_chart_data: bool = False
    hook_score: int = Field(ge=1, le=5)
    # filled by gate
    selected: bool = False
    skip_reason: str = ""


class ChartSeries(BaseModel):
    name: str
    points: list[DataPoint]


class ChartSpec(BaseModel):
    type: str = "bar"  # bar | line
    title: str
    y_label: str = ""
    series: list[ChartSeries]


class ScriptLine(BaseModel):
    text: str
    emphasis: bool = False
    visual: str = ""  # 2-4 English stock-footage keywords for this line (b-roll lookup, optional)


class ScriptClip(BaseModel):
    """Output of Pass B for one selected segment."""
    segment_id: int
    title: str
    hook: str
    lines: list[ScriptLine]
    chart: ChartSpec | None = None
    attribution: str
    est_seconds: float = 30
    # English prompt for the AI-generated opening shot (step 4, optional); no text/faces/logos
    ai_shot: str = ""
    # filled by plagiarism check
    plagiarism_overlap: float = 0.0
    plagiarism_lcs: int = 0
    plagiarism_ok: bool = True
    rewrite_attempts: int = 0
    # filled by the number provenance gate: chart points not stated in the transcript are dropped
    # ("chart:<label>=<value><unit>"); spoken numbers that can't be traced are listed ("text:<token>")
    numbers_unverified: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return self.hook + "".join(l.text for l in self.lines)


class ScriptsOutput(BaseModel):
    video_id: str
    source_name: str
    segments: list[TopicSegment]  # all candidates, with selected/skip_reason
    clips: list[ScriptClip]  # only selected + passed plagiarism gate
    rejected_clips: list[ScriptClip] = Field(default_factory=list)


# ---------- step 4 ----------
class RenderedClip(BaseModel):
    segment_id: int
    title: str
    video_path: str
    duration_sec: float
    chart_path: str | None = None
    ai_shot_path: str | None = None      # first generated shot (opening), if FINVID_AI_VIDEO != none
    ai_shot_paths: list[str] = Field(default_factory=list)  # every shot, in scene order
    ai_shot_provider: str | None = None
    ai_shot_seconds: float = 0.0         # wall/GPU seconds for all shots of this clip (0 on cache hit)
    broll_paths: list[str] = Field(default_factory=list)  # stock clips in scene order, if FINVID_BROLL != none
    broll_provider: str | None = None


class RenderOutput(BaseModel):
    video_id: str
    clips: list[RenderedClip]
