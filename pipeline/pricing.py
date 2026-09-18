"""Unit price table (USD). Single source of truth for every cost estimate in the pipeline.

Prices checked against https://developers.openai.com/api/docs/pricing on 2026-09-18.
Change these if the vendor changes prices; nothing else in the code hard-codes a number.
"""
from __future__ import annotations

from .models import CostEntry

# per-minute audio pricing
STT_PER_MINUTE: dict[str, float] = {
    "openai/gpt-4o-mini-transcribe": 0.003,
    "openai/gpt-4o-transcribe": 0.006,
    "openai/gpt-transcribe": 0.0045,
    "openai/whisper-1": 0.006,
    "local/faster-whisper": 0.0,
}

# per 1M tokens (input, output)
LLM_PER_1M: dict[str, tuple[float, float]] = {
    "openai/gpt-5-nano": (0.05, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-5": (1.25, 10.00),
}

# per 1M characters of input text
TTS_PER_1M_CHARS: dict[str, float] = {
    "edge/edge-tts": 0.0,
    "openai/gpt-4o-mini-tts": 0.60,  # text input; audio output billed separately, approximated below
    "openai/tts-1": 15.00,
}
# gpt-4o-mini-tts audio output is ~$12/1M audio tokens. Rough: 1 zh char ≈ 0.25s ≈ 4 audio tokens.
OPENAI_TTS_AUDIO_USD_PER_CHAR = 12.0 / 1_000_000 * 4

# Reference price for "just generate the clip with an AI video API" (Runway/Kling/Veo class).
# Used ONLY for the comparison table; the pipeline never calls such an API.
AI_VIDEO_USD_PER_SECOND = 0.25


def _key(provider: str, model: str) -> str:
    return f"{provider}/{model}"


def stt_cost(stage: str, provider: str, model: str, seconds: float, estimated: bool = False) -> CostEntry:
    minutes = seconds / 60.0
    price = STT_PER_MINUTE.get(_key(provider, model), 0.0)
    return CostEntry(
        stage=stage, provider=provider, model=model, unit="minute",
        quantity=round(minutes, 3), unit_price_usd=price, usd=round(minutes * price, 6),
        estimated=estimated, note=f"{seconds:.0f}s audio",
    )


def llm_cost(stage: str, provider: str, model: str, input_tokens: int, output_tokens: int,
             estimated: bool = False, note: str = "") -> list[CostEntry]:
    pin, pout = LLM_PER_1M.get(_key(provider, model), (0.0, 0.0))
    return [
        CostEntry(stage=stage, provider=provider, model=model, unit="input_tokens",
                  quantity=input_tokens, unit_price_usd=pin / 1e6,
                  usd=round(input_tokens * pin / 1e6, 6), estimated=estimated, note=note),
        CostEntry(stage=stage, provider=provider, model=model, unit="output_tokens",
                  quantity=output_tokens, unit_price_usd=pout / 1e6,
                  usd=round(output_tokens * pout / 1e6, 6), estimated=estimated, note=note),
    ]


def tts_cost(stage: str, provider: str, model: str, characters: int, estimated: bool = False) -> CostEntry:
    price = TTS_PER_1M_CHARS.get(_key(provider, model), 0.0) / 1e6
    if provider == "openai" and model == "gpt-4o-mini-tts":
        price += OPENAI_TTS_AUDIO_USD_PER_CHAR
    return CostEntry(
        stage=stage, provider=provider, model=model, unit="characters",
        quantity=characters, unit_price_usd=price, usd=round(characters * price, 6),
        estimated=estimated,
    )


def ai_video_reference_cost(seconds: float, note: str = "") -> CostEntry:
    return CostEntry(
        stage="s4_render", provider="reference", model="ai-video-api", unit="second",
        quantity=seconds, unit_price_usd=AI_VIDEO_USD_PER_SECOND,
        usd=round(seconds * AI_VIDEO_USD_PER_SECOND, 4), estimated=True,
        note=note or "what a Runway/Kling-class video API would charge for this clip",
    )


def estimate_tokens_zh(text: str) -> int:
    """Rough token estimate for Chinese text (~1 token per 1.3 chars on OpenAI tokenizers)."""
    return int(len(text) / 1.3) + 1
