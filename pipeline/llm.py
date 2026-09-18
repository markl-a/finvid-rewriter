"""Thin LLM wrapper: one function that returns parsed JSON + a CostEntry list.

Only OpenAI for now; the interface is provider-agnostic so Gemini/Anthropic can be added
without touching the stages.
"""
from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .models import CostEntry
from .pricing import llm_cost

_client = None


def _openai(settings: Settings):
    global _client
    if _client is None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def chat_json(settings: Settings, model: str, system: str, user: str, *, stage: str,
              note: str = "", reasoning_effort: str | None = "low",
              max_output_tokens: int = 4000) -> tuple[dict[str, Any], list[CostEntry]]:
    """Call the model, force a JSON object reply, return (parsed, costs)."""
    client = _openai(settings)
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_output_tokens
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["max_tokens"] = max_output_tokens
        kwargs["temperature"] = 0.7
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or "{}"
    usage = resp.usage
    costs = llm_cost(stage, "openai", model,
                     usage.prompt_tokens if usage else 0,
                     usage.completion_tokens if usage else 0, note=note)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # tolerate ```json fences
        s = content.strip().strip("`")
        if s.startswith("json"):
            s = s[4:]
        parsed = json.loads(s)
    return parsed, costs
