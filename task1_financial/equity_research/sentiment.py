"""Per-headline LLM sentiment and the aggregate score (Task 1B).

classify_headline: ONE LLM call per headline (+ one repair try).
    reply -> extract JSON -> validate with HeadlineSentiment
    invalid -> log the error AND the raw text, retry once with the error message
    still invalid -> failure record (the run continues with the other headlines)

aggregate_sentiment: confidence- and recency-weighted mean
    S = sum(w_i * c_i * s_i) / sum(w_i * c_i),   s = +1/0/-1,  c = confidence,
    w_i = 0.5 ** (age_days / half_life)
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Per-headline LLM sentiment with Pydantic
# validation, logged failures and a repair retry; confidence- and recency-weighted aggregation with
# agreement and coverage', Date: 2026-10-11

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import ValidationError

from equity_research.config import (
    HEADLINE_ATTEMPTS,
    LOG_TRUNCATE,
    SENTIMENT_HALF_LIFE_DAYS,
    SENTIMENT_THRESHOLD,
    SENTIMENT_UNKNOWN_AGE_DAYS,
)
from equity_research import tracing
from equity_research.llm import extract_json, invoke_llm, short_error
from equity_research.prompts import HEADLINE_PROMPT, HEADLINE_REPAIR
from equity_research.schemas import HeadlineSentiment, SentimentSummary

log = logging.getLogger(__name__)

SENTIMENT_VALUE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def classify_headline(llm, ticker: str, company: str | None, item: dict
                      ) -> tuple[HeadlineSentiment | None, dict | None]:
    """Returns (result, None) on success or (None, failure_record). Never raises."""
    repair, error, raw = "", "", ""
    for attempt in range(1, HEADLINE_ATTEMPTS + 1):
        try:
            messages = HEADLINE_PROMPT.invoke({
                "ticker": ticker, "company": company or ticker,
                "headline": item["title"], "publisher": item.get("publisher") or "unknown",
                "repair": repair,
            })
            raw = ""
            raw = invoke_llm(llm, messages, "headline", attempt, detail=item["title"])
            result = HeadlineSentiment.model_validate_json(extract_json(raw))
            tracing.record("validation", purpose="headline", attempt=attempt, valid=True, problems=[],
                           detail=item["title"][:80])
            # Keep the exact original title, even if the model paraphrased it.
            return result.model_copy(update={"headline": item["title"]}), None
        except (ValueError, ValidationError) as exc:          # bad JSON or wrong schema
            error = short_error(exc)
        except Exception as exc:                               # API error after retries + fallback model
            error = f"LLM call failed: {type(exc).__name__}: {str(exc)[:LOG_TRUNCATE]}"
        tracing.record("validation", purpose="headline", attempt=attempt, valid=False, problems=[error],
                       detail=item["title"][:80])
        log.warning("headline_invalid attempt=%d headline=%r error=%s raw=%r",
                    attempt, item["title"][:60], error, raw[:LOG_TRUNCATE])
        repair = HEADLINE_REPAIR.format(error=error)
    return None, {"headline": item["title"], "error": error, "raw_output": raw[:LOG_TRUNCATE]}


def aggregate_sentiment(results: list[dict], now: datetime, total: int | None = None) -> SentimentSummary | None:
    """results: dicts with 'sentiment', 'confidence', 'published_at'. None if there are no results.

    Neutral headlines stay in the denominator on purpose: mostly-neutral news should give ~0.
    """
    if not results:
        return None
    num = den = 0.0
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        published = r.get("published_at")
        age_days = (now - published).total_seconds() / 86400 if published else SENTIMENT_UNKNOWN_AGE_DAYS
        weight = 0.5 ** (max(age_days, 0.0) / SENTIMENT_HALF_LIFE_DAYS)
        num += weight * r["confidence"] * SENTIMENT_VALUE[r["sentiment"]]
        den += weight * r["confidence"]
        counts[r["sentiment"]] += 1
    score = num / den if den else 0.0
    label = ("positive" if score >= SENTIMENT_THRESHOLD else
             "negative" if score <= -SENTIMENT_THRESHOLD else "neutral")
    n = len(results)
    return SentimentSummary(score=round(score, 3), label=label, counts=counts,
                            agreement=round(max(counts.values()) / n, 2),
                            coverage=round(n / max(total or n, n), 2), n_valid=n)
