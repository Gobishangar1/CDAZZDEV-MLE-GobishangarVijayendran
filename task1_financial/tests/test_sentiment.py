"""Per-headline JSON (10 marks) and structured-output validation: schema, repair, logged failures,
aggregation logic."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for headline classification with a
# fake LLM (valid, repaired, failed, API error) and for sentiment aggregation', Date: 2026-10-11

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from equity_research.schemas import HeadlineSentiment
from equity_research.sentiment import aggregate_sentiment, classify_headline
from tests.conftest import FakeLLM

ITEM = {"title": "Nvidia beats estimates as data center sales soar", "publisher": "Reuters"}
GOOD = json.dumps({"headline": ITEM["title"], "sentiment": "positive", "confidence": 0.9,
                   "brief_reason": "An earnings beat is directly positive for the share price."})


def test_schema_has_the_four_brief_fields():
    assert set(HeadlineSentiment.model_fields) == {"headline", "sentiment", "confidence", "brief_reason"}
    with pytest.raises(ValidationError):
        HeadlineSentiment(headline="h", sentiment="bullish", confidence=1.4, brief_reason="r")


def test_valid_reply_first_try():
    llm = FakeLLM(lambda h, n: GOOD)
    result, failure = classify_headline(llm, "NVDA", "NVIDIA", ITEM)
    assert failure is None and result.sentiment == "positive" and llm.calls == 1


def test_invalid_reply_is_logged_and_repaired(caplog):
    replies = ['Sure! {"headline": "x", "sentiment": "bullish", "confidence": 1.4, "brief_reason": "r"}', GOOD]
    llm = FakeLLM(lambda h, n: replies[n - 1])
    with caplog.at_level(logging.WARNING):
        result, failure = classify_headline(llm, "NVDA", "NVIDIA", ITEM)
    assert result.sentiment == "positive" and failure is None and llm.calls == 2
    assert "headline_invalid" in caplog.text and "bullish" in caplog.text        # error AND raw text logged
    assert "previous reply was invalid" in llm.prompts[1]                         # repair prompt sent


def test_two_failures_return_a_record_not_an_exception():
    result, failure = classify_headline(FakeLLM(lambda h, n: "I think positive"), "NVDA", None, ITEM)
    assert result is None and failure["error"] == "no JSON object found in the reply"
    assert failure["raw_output"] == "I think positive"


def test_api_error_is_caught():
    def boom(h, n):
        raise ConnectionError("groq down")
    result, failure = classify_headline(FakeLLM(boom), "NVDA", None, ITEM)
    assert result is None and "LLM call failed" in failure["error"]


def test_headline_text_is_kept_exactly():
    reply = json.dumps({"headline": "paraphrased", "sentiment": "neutral", "confidence": 0.5, "brief_reason": "r"})
    result, _ = classify_headline(FakeLLM(lambda h, n: reply), "NVDA", None, ITEM)
    assert result.headline == ITEM["title"]


NOW = datetime(2026, 10, 10, tzinfo=timezone.utc)


def r(sentiment, confidence, days_old):
    return {"sentiment": sentiment, "confidence": confidence,
            "published_at": None if days_old is None else NOW - timedelta(days=days_old)}


def test_aggregate_all_neutral_is_zero():
    s = aggregate_sentiment([r("neutral", 0.8, 1)] * 5, NOW)
    assert (s.score, s.label, s.agreement) == (0.0, "neutral", 1.0)


def test_aggregate_newer_news_weighs_more():
    s = aggregate_sentiment([r("positive", 0.9, 0), r("negative", 0.9, 6)], NOW)
    assert s.score == pytest.approx(0.6, abs=1e-3)    # weights 1 and 0.25 -> (1 - 0.25) / 1.25


def test_aggregate_mixed_with_coverage():
    s = aggregate_sentiment([r("positive", 0.9, 0.5), r("positive", 0.7, 1), r("neutral", 0.6, 2),
                             r("negative", 0.8, 6), r("neutral", 0.5, None)], NOW, total=6)
    assert s.label == "positive" and s.counts == {"positive": 2, "neutral": 2, "negative": 1}
    assert (s.agreement, s.coverage, s.n_valid) == (0.4, 0.83, 5)


def test_aggregate_empty_is_none():
    assert aggregate_sentiment([], NOW) is None
