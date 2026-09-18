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


class ScriptClip(BaseModel):
    """Output of Pass B for one selected segment."""
    segment_id: int
    title: str
    hook: str
    lines: list[ScriptLine]
    chart: ChartSpec | None = None
    attribution: str
    est_seconds: float = 30
    # filled by plagiarism check
    plagiarism_overlap: float = 0.0
    plagiarism_lcs: int = 0
    plagiarism_ok: bool = True
    rewrite_attempts: int = 0

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


class RenderOutput(BaseModel):
    video_id: str
    clips: list[RenderedClip]
