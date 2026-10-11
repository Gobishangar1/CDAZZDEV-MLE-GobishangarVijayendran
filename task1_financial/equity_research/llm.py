"""Groq LLM clients and helpers for reading JSON out of LLM replies.

Fallback design (one free provider, one key):
    Groq free-tier limits are per model. If the primary model fails (429, timeout, 5xx after
    the client's own retries), the same request goes to the OTHER GPT-OSS model.
        headlines: gpt-oss-20b  -> gpt-oss-120b
        signal:    gpt-oss-120b -> gpt-oss-20b

LLMs are passed into the graph (build_graph(fast_llm, strong_llm)), so tests use fakes.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Groq ChatGroq factory with rate limiter, timeout,
# retries, JSON mode and a fallback to the other model; helpers to extract JSON and summarise
# Pydantic errors', Date: 2026-10-11

from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from equity_research import tracing
from equity_research.config import (
    FAST_MODEL,
    HEADLINE_REASONING,
    LLM_MAX_RETRIES,
    LLM_REQUESTS_PER_SECOND,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    SIGNAL_REASONING,
    STRONG_MODEL,
    require_groq_key,
)


def make_llm(model: str, reasoning_effort: str, json_mode: bool = True):
    """One Groq chat model. JSON mode makes the reply JSON text; Pydantic still checks the schema."""
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_groq import ChatGroq

    require_groq_key()
    return ChatGroq(
        model=model,
        temperature=LLM_TEMPERATURE,
        reasoning_effort=reasoning_effort,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
        rate_limiter=InMemoryRateLimiter(requests_per_second=LLM_REQUESTS_PER_SECOND),
        model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
    )


def make_fast_llm(json_mode: bool = True):
    """Per-headline sentiment: small model first, big model as fallback."""
    return make_llm(FAST_MODEL, HEADLINE_REASONING, json_mode).with_fallbacks(
        [make_llm(STRONG_MODEL, HEADLINE_REASONING, json_mode)])


def make_strong_llm(json_mode: bool = True):
    """Buy/Hold/Sell signal: big model first, small model as fallback."""
    return make_llm(STRONG_MODEL, SIGNAL_REASONING, json_mode).with_fallbacks(
        [make_llm(FAST_MODEL, SIGNAL_REASONING, json_mode)])


def message_text(message: Any) -> str:
    """The text of an LLM reply (content can be a string or a list of parts)."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def invoke_llm(llm, prompt_value, purpose: str, attempt: int, detail: str = "") -> str:
    """Call the LLM, record one 'llm_call' trace event (model, latency, tokens), return the text.

    The model name comes from the response, so the trace shows when the fallback model answered.
    API errors are recorded and re-raised (the caller decides what to do).
    """
    t0 = time.perf_counter()
    try:
        message = llm.invoke(prompt_value)
    except Exception as exc:
        tracing.record("llm_call", purpose=purpose, attempt=attempt, detail=detail[:80], ok=False,
                       latency_ms=round((time.perf_counter() - t0) * 1000),
                       error=f"{type(exc).__name__}: {str(exc)[:200]}")
        raise
    text = message_text(message)
    usage = getattr(message, "usage_metadata", None) or {}
    meta = getattr(message, "response_metadata", None) or {}
    tracing.record("llm_call", purpose=purpose, attempt=attempt, detail=detail[:80], ok=True,
                   model=meta.get("model_name"), latency_ms=round((time.perf_counter() - t0) * 1000),
                   input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                   output_preview=tracing.preview(text))
    return text


def extract_json(text: str) -> str:
    """Text from the first '{' to the last '}'. Handles replies like 'Sure! {...}'."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return text[start:end + 1]


def short_error(exc: Exception) -> str:
    """Compact error text for logs and for the repair prompt."""
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in exc.errors())
    return str(exc)
