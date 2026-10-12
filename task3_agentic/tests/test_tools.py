"""The five tools: correct types, correct numbers, and errors turned into suggestions (never raised)."""

import json
import math

import numpy as np
import pytest
from conftest import FakeJSONLLM, failing_search, make_prices, search_results

from research_agents.config import SNIPPET_CHARS, TOOL_OUTPUT_TO_LLM_CHARS
from research_agents.schemas import NewsResult, PriceData, SearchResult, SentimentResult, VolatilityResult
from research_agents.tools import ALL_TOOLS, ToolBox

ALL = set(ALL_TOOLS)


@pytest.fixture
def box(sentiment_llm, tracer):
    return ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)


def test_every_tool_returns_its_schema(box):
    assert isinstance(box.get_price_data("nvda", "1y"), PriceData)
    assert isinstance(box.get_news("NVDA", 5), NewsResult)
    assert isinstance(box.calculate_volatility("NVDA", 30), VolatilityResult)
    assert isinstance(box.llm_sentiment(["NVIDIA beats estimates"]), SentimentResult)
    assert isinstance(box.web_search("NVDA analyst"), SearchResult)


def test_price_data_has_indicators_trend_and_few_rows(box):
    p = box.get_price_data("NVDA", "6mo")
    assert p.ticker == "NVDA" and p.period == "6mo"
    assert set(p.indicators) >= {"sma_50", "sma_200", "rsi_14", "macd", "macd_hist", "bb_upper", "bb_lower"}
    assert all(v is not None for v in p.indicators.values())        # 2 years fetched: SMA-200 is ready
    assert len(p.recent_ohlcv) == 5 and 100 < p.rows < 140           # ~6 months of trading days
    assert "RSI" in p.trend and p.low_52w <= p.last_close <= p.high_52w


def test_volatility_matches_an_independent_calculation(box):
    v = box.calculate_volatility("NVDA", 30)
    close = make_prices()["Close"].to_numpy()
    expected = np.std(np.diff(np.log(close))[-30:], ddof=1) * math.sqrt(252) * 100
    assert v.annualised_vol_pct == pytest.approx(expected, abs=0.01)
    assert 0 <= v.vol_percentile_1y <= 100
    assert v.max_drawdown_1y_pct <= 0 and v.var_95_1d_pct < 0
    assert "30-day annualised volatility" in v.interpretation


def test_sentiment_is_confidence_weighted(tracer):
    def reply(system, human, n):
        return json.dumps({"items": [
            {"headline": "a", "sentiment": "positive", "confidence": 0.9, "brief_reason": "x"},
            {"headline": "b", "sentiment": "negative", "confidence": 0.3, "brief_reason": "y"},
            {"headline": "c", "sentiment": "neutral", "confidence": 0.6, "brief_reason": "z"}]})
    out = ToolBox(sentiment_llm=FakeJSONLLM(reply), tracer=tracer).llm_sentiment(["a", "b", "c"])
    assert out.score == pytest.approx((0.9 - 0.3) / 1.8, abs=1e-3)
    assert out.label == "positive" and out.counts == {"positive": 1, "neutral": 1, "negative": 1}


def test_invalid_sentiment_json_is_logged_and_repaired(tracer):
    replies = iter(["not json", json.dumps({"items": [
        {"headline": "a", "sentiment": "negative", "confidence": 0.7, "brief_reason": "x"}]})])
    llm = FakeJSONLLM(lambda s, h, n: next(replies))
    out = ToolBox(sentiment_llm=llm, tracer=tracer).llm_sentiment(["a"])
    assert out.label == "negative" and llm.calls == 2
    assert "previous JSON was invalid" in llm.prompts[1][1]           # the error went back to the model
    assert [e["kind"] for e in tracer.events] == ["validation_failed"]


def test_web_search_maps_and_shortens_results(box):
    r = box.web_search("NVDA analyst", max_results=3)
    assert len(r.results) == 3 and r.results[0].url.startswith("https://")
    assert all(len(h.snippet) <= SNIPPET_CHARS for h in r.results)


# ----------------------------------------------------------------------------- executor
def test_execute_ok_is_traced(box, tracer):
    text, status, result = box.execute("tester", ALL, "calculate_volatility", {"ticker": "NVDA", "window": 20})
    assert status == "ok" and isinstance(result, VolatilityResult)
    assert json.loads(text)["window"] == 20
    call = tracer.tool_calls()[0]
    assert call["tool"] == "calculate_volatility" and call["inputs"] == {"ticker": "NVDA", "window": 20}


def test_bad_arguments_become_an_error_message(box):
    text, status, result = box.execute("tester", ALL, "calculate_volatility", {"ticker": "NVDA", "window": 1})
    assert status == "error" and result is None
    assert "invalid arguments" in json.loads(text)["error"]


def test_unknown_ticker_error_suggests_what_to_try(box, fake_ticker):
    fake_ticker.fail = RuntimeError("404 Not Found")
    text, status, _ = box.execute("tester", ALL, "get_price_data", {"ticker": "ZZZZ", "period": "1y"})
    payload = json.loads(text)
    assert status == "error" and "No price data" in payload["error"]
    assert "shorter period" in payload["suggestion"]
    assert fake_ticker.history_calls == 3                                # retried, then gave up cleanly


def test_failed_search_suggests_only_tools_the_agent_has(tracer):
    box = ToolBox(tracer=tracer, search_backend=failing_search)
    text, status, _ = box.execute("B", {"web_search", "get_news"}, "web_search", {"query": "NVDA risks"})
    assert status == "error" and "get_news" in json.loads(text)["suggestion"]
    text, _, _ = box.execute("C", {"web_search"}, "web_search", {"query": "NVDA risks"})
    assert "get_news" not in json.loads(text)["suggestion"]


def test_empty_news_is_marked_empty_not_ok(box, fake_ticker):
    fake_ticker.news_items = []                    # yfinance empty; the RSS fallbacks are offline too
    text, status, result = box.execute("B", {"get_news", "web_search"}, "get_news", {"ticker": "NVDA", "n": 5})
    payload = json.loads(text)
    assert status == "empty" and result.is_empty()
    assert "web_search" in payload["suggestion"] and payload["result"]["warnings"]


def test_denied_tool_is_not_executed(box, fake_ticker, tracer):
    text, status, result = box.execute("Agent B", {"web_search", "get_news"}, "get_price_data", {"ticker": "NVDA"})
    assert status == "denied" and result is None
    assert json.loads(text)["your_tools"] == ["get_news", "web_search"]
    assert fake_ticker.history_calls == 0
    assert tracer.tool_calls()[0]["status"] == "denied"


def test_sentiment_without_a_model_is_an_error_not_a_crash(tracer):
    text, status, _ = ToolBox(tracer=tracer).execute("A", ALL, "llm_sentiment", {"headlines": ["x"]})
    assert status == "error" and "not configured" in text


def test_output_to_the_llm_is_capped(tracer):
    def huge(query, n):
        return [{"title": "t" * 300, "href": "https://x", "body": "b" * 300}] * n
    text, _, _ = ToolBox(tracer=tracer, search_backend=huge).execute("A", ALL, "web_search",
                                                                      {"query": "big", "max_results": 8})
    assert len(text) <= TOOL_OUTPUT_TO_LLM_CHARS and text.endswith('"[truncated]"')


def test_langchain_tool_schemas_match_the_brief(box):
    specs = {t.name: t.args for t in box.langchain_tools(list(ALL_TOOLS))}
    assert set(specs["get_price_data"]) == {"ticker", "period"}
    assert set(specs["get_news"]) == {"ticker", "n"}
    assert set(specs["calculate_volatility"]) == {"ticker", "window"}
    assert set(specs["llm_sentiment"]) == {"headlines"}
    assert "query" in specs["web_search"]
