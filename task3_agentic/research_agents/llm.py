"""Groq clients and the structured-output helper.

    make_agent_llm()     tool-calling model (gpt-oss-120b) with gpt-oss-20b as fallback
    make_json_llm()      JSON-mode model for structured outputs (report, brief, clarification)
    make_sentiment_llm() small JSON-mode model for the llm_sentiment tool
    invoke_json(...)     call -> extract JSON -> validate with Pydantic -> repair retry (logged)
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from research_agents.config import (
    AGENT_MODEL,
    AGENT_REASONING,
    FALLBACK_MAX_RETRIES,
    FALLBACK_MODEL,
    JSON_ATTEMPTS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    PRIMARY_MAX_RETRIES,
    SENTIMENT_MODEL,
    SENTIMENT_REASONING,
    require_groq_key,
)

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _chat(model: str, reasoning: str, json_mode: bool, fallback: bool = False):
    """One Groq chat model. Two models with separate free-tier quotas share the load: the primary retries
    briefly on a rate limit, then the fallback takes over and waits longer."""
    from langchain_groq import ChatGroq

    require_groq_key()
    return ChatGroq(model=model, temperature=LLM_TEMPERATURE, reasoning_effort=reasoning, timeout=LLM_TIMEOUT_S,
                    max_retries=FALLBACK_MAX_RETRIES if fallback else PRIMARY_MAX_RETRIES,
                    model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {})


class ToolLLM:
    """A chat model pair for agents: bind_tools() returns primary-with-fallback, both bound to the same tools.
    A malformed tool call on Groq (400 tool_use_failed) or a rate limit on one model moves to the other."""

    def __init__(self, primary: Any, fallback: Any | None = None) -> None:
        self.primary, self.fallback = primary, fallback

    def bind_tools(self, tools: list, **kwargs: Any) -> Any:
        bound = self.primary.bind_tools(tools, **kwargs)
        return bound.with_fallbacks([self.fallback.bind_tools(tools, **kwargs)]) if self.fallback else bound

    def invoke(self, messages: list) -> Any:          # a plain call without tools
        runnable = self.primary.with_fallbacks([self.fallback]) if self.fallback else self.primary
        return runnable.invoke(messages)


def make_agent_llm() -> ToolLLM:
    return ToolLLM(_chat(AGENT_MODEL, AGENT_REASONING, False),
                   _chat(FALLBACK_MODEL, AGENT_REASONING, False, fallback=True))


def make_json_llm():
    return _chat(AGENT_MODEL, AGENT_REASONING, True).with_fallbacks(
        [_chat(FALLBACK_MODEL, AGENT_REASONING, True, fallback=True)])


def make_sentiment_llm():
    return _chat(SENTIMENT_MODEL, SENTIMENT_REASONING, True).with_fallbacks(
        [_chat(AGENT_MODEL, SENTIMENT_REASONING, True, fallback=True)])


# -----------------------------------------------------------------------------
# Structured output
# -----------------------------------------------------------------------------
def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return text[start:end + 1]


def short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in exc.errors())[:600]
    return str(exc)[:600]


class StructuredOutputError(RuntimeError):
    pass


def invoke_json(llm: Any, system: str, human: str, schema: type[T], repair_template: str,
                label: str, tracer: Any = None, attempts: int = JSON_ATTEMPTS, fixed: dict | None = None) -> T:
    """System + human -> JSON -> schema. On failure: log, add the error to the prompt, retry.
    `fixed` fields (e.g. ticker, date) are set by code, so the model cannot get them wrong."""
    repair, error = "", ""
    for attempt in range(1, attempts + 1):
        raw = ""
        try:
            raw = message_text(llm.invoke([SystemMessage(system), HumanMessage(human + repair)]))
            data = json.loads(extract_json(raw))
            if not isinstance(data, dict):
                raise ValueError("the reply is not a JSON object")
            return schema.model_validate({**data, **(fixed or {})})
        except (ValueError, ValidationError) as exc:
            error = short_error(exc)
        except Exception as exc:                       # API error after client retries and fallback
            error = f"LLM call failed: {type(exc).__name__}: {str(exc)[:300]}"
        log.warning("structured_output_invalid label=%s attempt=%d error=%s", label, attempt, error)
        if tracer is not None:
            tracer.note("validation_failed", f"{label} attempt {attempt}: {error[:300]}", label=label,
                        attempt=attempt, raw_output=raw[:300])
        repair = repair_template.format(error=error)
    raise StructuredOutputError(f"{label}: no valid output after {attempts} attempts ({error})")
