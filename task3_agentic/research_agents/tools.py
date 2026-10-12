"""The five agent tools and the executor that runs them.

    get_price_data(ticker, period)        yfinance OHLCV + computed indicators            -> PriceData
    get_news(ticker, n)                   recent headlines as a structured list            -> NewsResult
    calculate_volatility(ticker, window)  annualised historical volatility (+ drawdown, VaR) -> VolatilityResult
    llm_sentiment(headlines)              one LLM call, validated, aggregated score         -> SentimentResult
    web_search(query, max_results)        duckduckgo search (ddgs) for analyst commentary    -> SearchResult

ToolBox.execute() is the only way agents run tools. It:
    - enforces each agent's allowed tools (3B tool restriction)    -> status "denied"
    - validates the arguments with the tool's Pydantic schema       -> status "error"
    - turns every exception into an error result + a suggestion of an alternative the agent CAN use
    - marks empty results as "empty" (with a suggestion), never as success
    - logs every call to agent_trace.jsonl (inputs, 200-char output, duration, status)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from research_agents import market, news
from research_agents.config import (
    DEFAULT_NEWS,
    DEFAULT_PERIOD,
    DEFAULT_SEARCH_RESULTS,
    DEFAULT_VOL_WINDOW,
    INDICATOR_FETCH_PERIOD,
    MAX_NEWS,
    MAX_SEARCH_RESULTS,
    MAX_SENTIMENT_HEADLINES,
    MAX_VOL_WINDOW,
    MIN_SEARCH_RESULTS,
    MIN_VOL_WINDOW,
    RECENT_ROWS,
    SENTIMENT_THRESHOLD,
    SNIPPET_CHARS,
    TOOL_OUTPUT_TO_LLM_CHARS,
    TRADING_DAYS_PER_YEAR,
    VAR_QUANTILE,
)
from research_agents.llm import invoke_json
from research_agents.market import ToolError
from research_agents.prompts import REPORT_REPAIR, SENTIMENT_HUMAN, SENTIMENT_SYSTEM
from research_agents.schemas import (
    Compact,
    Headline,
    HeadlineLabel,
    NewsResult,
    OHLCVRow,
    PriceData,
    SearchHit,
    SearchResult,
    SentimentLLMOutput,
    SentimentResult,
    VolatilityResult,
)

log = logging.getLogger(__name__)
ALL_TOOLS = ("get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search")
_PERIOD_OFFSET = {"1mo": {"months": 1}, "3mo": {"months": 3}, "6mo": {"months": 6}, "1y": {"years": 1},
                  "2y": {"years": 2}, "5y": {"years": 5}}


# -----------------------------------------------------------------------------
# Argument schemas (what the LLM sees, and what the executor validates)
# -----------------------------------------------------------------------------
class PriceArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y"] = Field(DEFAULT_PERIOD, description="History length")


class NewsArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    n: int = Field(DEFAULT_NEWS, ge=1, le=MAX_NEWS, description="Number of headlines")


class VolatilityArgs(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. NVDA")
    window: int = Field(DEFAULT_VOL_WINDOW, ge=MIN_VOL_WINDOW, le=MAX_VOL_WINDOW,
                        description="Trading days used for the volatility (e.g. 20, 30, 60, 90)")


class SentimentArgs(BaseModel):
    headlines: list[str] = Field(min_length=1, max_length=MAX_SENTIMENT_HEADLINES,
                                 description="Headline texts to score (max 15)")


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, description="Web search query, e.g. 'NVDA analyst downgrade risk'")
    max_results: int = Field(DEFAULT_SEARCH_RESULTS, ge=1, le=MAX_SEARCH_RESULTS)


DESCRIPTIONS = {
    "get_price_data": "Daily OHLCV price history from Yahoo Finance with computed indicators (SMA-50, SMA-200, "
                      "RSI-14, MACD, Bollinger Bands), period return and 52-week range.",
    "get_news": "Recent news headlines about the company (title, publisher, time, source) as a structured list.",
    "calculate_volatility": "Annualised historical volatility of daily log returns over a window of trading days, "
                            "its percentile within the last year, 1-year max drawdown and 1-day 95% VaR.",
    "llm_sentiment": "Score the sentiment of a list of headlines with an LLM. Returns a score from -1 to +1, a "
                     "label and a label per headline.",
    "web_search": "Search the web (DuckDuckGo) for analyst commentary, risks, events and competition.",
}

# What to try when a tool fails or returns nothing (only tools the agent has are suggested).
ALTERNATIVES: dict[str, list[tuple[str, str]]] = {
    "get_price_data": [("get_price_data", "check the ticker or retry with a shorter period such as '6mo'")],
    "calculate_volatility": [("calculate_volatility", "retry with another window such as 20 or 60"),
                             ("get_price_data", "use get_price_data and read the Bollinger and trend values")],
    "get_news": [("web_search", "use web_search with a query like '<TICKER> stock news this week'"),
                 ("get_news", "retry get_news with a larger n")],
    "web_search": [("web_search", "retry ONCE with a shorter, simpler query"),
                   ("get_news", "use get_news to get recent headlines instead")],
    "llm_sentiment": [("llm_sentiment", "retry with fewer headlines (5 to 10)")],
}


def suggestion(tool: str, allowed: set[str]) -> str:
    options = [text for name, text in ALTERNATIVES.get(tool, []) if name in allowed]
    return "; or ".join(options) if options else "continue with the evidence you already have"


def default_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """DuckDuckGo text search via the ddgs package (the renamed duckduckgo-search library)."""
    try:
        from ddgs import DDGS
    except ImportError:                                   # older installs
        from duckduckgo_search import DDGS  # type: ignore[no-redef]
    return DDGS().text(query, region="us-en", safesearch="moderate", max_results=max_results)


# -----------------------------------------------------------------------------
# The toolbox
# -----------------------------------------------------------------------------
class ToolBox:
    def __init__(self, sentiment_llm: Any = None, tracer: Any = None,
                 search_backend: Callable[[str, int], list[dict]] | None = None) -> None:
        self.sentiment_llm = sentiment_llm
        self.tracer = tracer
        self.search_backend = search_backend or default_search     # tests and the failure demo pass their own
        self.specs: dict[str, tuple[Callable[..., Compact], type[BaseModel]]] = {
            "get_price_data": (self.get_price_data, PriceArgs),
            "get_news": (self.get_news, NewsArgs),
            "calculate_volatility": (self.calculate_volatility, VolatilityArgs),
            "llm_sentiment": (self.llm_sentiment, SentimentArgs),
            "web_search": (self.web_search, SearchArgs),
        }

    def langchain_tools(self, names: list[str]) -> list[StructuredTool]:
        """Tool definitions for llm.bind_tools(). Execution always goes through execute()."""
        return [StructuredTool.from_function(func=self.specs[n][0], name=n, description=DESCRIPTIONS[n],
                                             args_schema=self.specs[n][1]) for n in names]

    # ------------------------------------------------------------- 1. prices
    def get_price_data(self, ticker: str, period: str = DEFAULT_PERIOD) -> PriceData:
        ticker = market.normalize_ticker(ticker)
        if period not in _PERIOD_OFFSET:
            raise ToolError(f"period must be one of {list(_PERIOD_OFFSET)}")
        fetch_period = period if period in ("2y", "5y") else INDICATOR_FETCH_PERIOD   # warm-up for SMA-200
        df = market.add_indicators(market.fetch_history(ticker, fetch_period))
        window = df[df.index >= df.index[-1] - pd.DateOffset(**_PERIOD_OFFSET[period])]
        last, year = df.iloc[-1], df.iloc[-TRADING_DAYS_PER_YEAR:]

        def num(v: Any) -> float | None:
            return None if v is None or pd.isna(v) else round(float(v), 4)

        ind = {k: num(last[k]) for k in ("sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
                                         "bb_upper", "bb_middle", "bb_lower", "bb_pct_b")}
        close = float(last["Close"])
        parts = []
        for name in ("sma_50", "sma_200"):
            if ind[name] is not None:
                parts.append(f"close {'above' if close > ind[name] else 'below'} {name.upper().replace('_', '-')}")
        if ind["sma_50"] is not None and ind["sma_200"] is not None:
            parts.append("long-term uptrend (SMA-50 > SMA-200)" if ind["sma_50"] > ind["sma_200"]
                         else "long-term downtrend (SMA-50 < SMA-200)")
        if ind["rsi_14"] is not None:
            zone = "overbought" if ind["rsi_14"] > 70 else "oversold" if ind["rsi_14"] < 30 else "neutral"
            parts.append(f"RSI {ind['rsi_14']:.1f} ({zone})")
        if ind["macd_hist"] is not None:
            parts.append(f"MACD histogram {'positive' if ind['macd_hist'] > 0 else 'negative'}")
        high_52w = float(year["High"].max())
        return PriceData(
            ticker=ticker, period=period, start=str(window.index[0].date()), end=str(window.index[-1].date()),
            rows=len(window), last_close=round(close, 2),
            period_return_pct=round((close / float(window["Close"].iloc[0]) - 1) * 100, 2),
            high_52w=round(high_52w, 2), low_52w=round(float(year["Low"].min()), 2),
            pct_from_52w_high=round((close / high_52w - 1) * 100, 2), indicators=ind, trend="; ".join(parts),
            recent_ohlcv=[OHLCVRow(date=str(i.date()), open=round(r.Open, 2), high=round(r.High, 2),
                                   low=round(r.Low, 2), close=round(r.Close, 2), volume=float(r.Volume))
                          for i, r in window.iloc[-RECENT_ROWS:].iterrows()])

    # --------------------------------------------------------------- 2. news
    def get_news(self, ticker: str, n: int = DEFAULT_NEWS) -> NewsResult:
        ticker = market.normalize_ticker(ticker)
        items, warnings = news.fetch_headlines(ticker, n)
        return NewsResult(ticker=ticker, warnings=warnings, headlines=[
            Headline(title=h["title"], publisher=h.get("publisher"), source=h["source"],
                     published_at=h["published_at"].isoformat() if h.get("published_at") else None,
                     relevant=h.get("relevant", True)) for h in items])

    # --------------------------------------------------------- 3. volatility
    def calculate_volatility(self, ticker: str, window: int = DEFAULT_VOL_WINDOW) -> VolatilityResult:
        ticker = market.normalize_ticker(ticker)
        df = market.fetch_history(ticker, INDICATOR_FETCH_PERIOD)
        log_ret = np.log(df["Close"]).diff().dropna()
        if len(log_ret) < window + 1:
            raise ToolError(f"Only {len(log_ret)} daily returns; need more than {window}")
        annual = math.sqrt(TRADING_DAYS_PER_YEAR) * 100
        current = float(log_ret.iloc[-window:].std(ddof=1)) * annual
        last_year = log_ret.iloc[-TRADING_DAYS_PER_YEAR:]
        rolling = (log_ret.rolling(window).std(ddof=1) * annual).dropna().iloc[-TRADING_DAYS_PER_YEAR:]
        percentile = float((rolling < current).mean() * 100)
        closes = df["Close"].iloc[-TRADING_DAYS_PER_YEAR:]
        drawdown = float((closes / closes.cummax() - 1).min() * 100)
        var_95 = float((np.exp(last_year.quantile(VAR_QUANTILE)) - 1) * 100)
        level = "elevated" if percentile >= 70 else "low" if percentile <= 30 else "normal"
        return VolatilityResult(
            ticker=ticker, window=window, as_of=str(df.index[-1].date()),
            annualised_vol_pct=round(current, 2), one_year_vol_pct=round(float(last_year.std(ddof=1)) * annual, 2),
            vol_percentile_1y=round(percentile, 1), max_drawdown_1y_pct=round(drawdown, 2),
            var_95_1d_pct=round(var_95, 2),
            interpretation=(f"{window}-day annualised volatility is {current:.1f}%, at the {percentile:.0f}th "
                            f"percentile of the past year ({level}). On 5% of days in the past year the stock "
                            f"fell more than {abs(var_95):.1f}%; the worst peak-to-trough fall was "
                            f"{abs(drawdown):.1f}%."))

    # ---------------------------------------------------------- 4. sentiment
    def llm_sentiment(self, headlines: list[str]) -> SentimentResult:
        if self.sentiment_llm is None:
            raise ToolError("sentiment model is not configured")
        clean = [h.strip() for h in headlines if h and h.strip()][:MAX_SENTIMENT_HEADLINES]
        if not clean:
            raise ToolError("no headlines given")
        numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(clean, 1))
        out = invoke_json(self.sentiment_llm, SENTIMENT_SYSTEM.format(),
                          SENTIMENT_HUMAN.format(headlines=numbered, repair=""), SentimentLLMOutput,
                          REPORT_REPAIR, label="llm_sentiment", tracer=self.tracer)
        value = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        weight = sum(i.confidence for i in out.items)
        score = sum(i.confidence * value[i.sentiment] for i in out.items) / weight if weight else 0.0
        counts = {k: sum(i.sentiment == k for i in out.items) for k in value}
        label = ("positive" if score >= SENTIMENT_THRESHOLD else
                 "negative" if score <= -SENTIMENT_THRESHOLD else "neutral")
        return SentimentResult(n=len(out.items), score=round(score, 3), label=label, counts=counts,
                               items=[HeadlineLabel(**i.model_dump()) for i in out.items])

    # --------------------------------------------------------- 5. web search
    def web_search(self, query: str, max_results: int = DEFAULT_SEARCH_RESULTS) -> SearchResult:
        max_results = max(max_results, MIN_SEARCH_RESULTS)
        try:
            raw = self.search_backend(query, max_results) or []
        except Exception as exc:                          # rate limits, timeouts, network
            raise ToolError(f"web search failed: {type(exc).__name__}: {str(exc)[:150]}") from exc
        hits = [SearchHit(title=r.get("title", ""), url=r.get("href") or r.get("url", ""),
                          snippet=" ".join((r.get("body") or r.get("snippet") or "").split())[:SNIPPET_CHARS])
                for r in raw if r.get("title")]
        return SearchResult(query=query, results=hits[:max_results])

    # ------------------------------------------------------------ executor
    def execute(self, agent: str, allowed: set[str], name: str, args: dict[str, Any]
                ) -> tuple[str, str, Compact | None]:
        """Run one tool call for one agent. Returns (text for the LLM, status, result model or None)."""
        t0 = time.perf_counter()
        result: Compact | None = None
        error: str | None = None
        if name not in self.specs:
            status, error = "error", f"Unknown tool '{name}'"
            payload: dict[str, Any] = {"status": "error", "error": error, "your_tools": sorted(allowed)}
        elif name not in allowed:
            status, error = "denied", f"Tool '{name}' is not available to {agent}"
            payload = {"status": "error", "error": error + ".", "your_tools": sorted(allowed),
                       "suggestion": "use only your own tools"}
        else:
            func, schema = self.specs[name]
            try:
                parsed = schema(**(args or {}))
                result = func(**parsed.model_dump())
                status = "empty" if result.is_empty() else "ok"
                payload = json.loads(result.to_llm())
                if status == "empty":
                    payload = {"status": "empty", "result": payload, "suggestion": suggestion(name, allowed)}
            except ValidationError as exc:
                status, error = "error", "invalid arguments: " + "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                payload = {"status": "error", "error": error, "suggestion": "fix the arguments and call again"}
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {str(exc)[:200]}"
                payload = {"status": "error", "error": error, "suggestion": suggestion(name, allowed)}
        text = json.dumps(payload, default=str, separators=(",", ":"))
        if len(text) > TOOL_OUTPUT_TO_LLM_CHARS:
            text = text[:TOOL_OUTPUT_TO_LLM_CHARS - 20] + '..."[truncated]"'
        duration_ms = round((time.perf_counter() - t0) * 1000)
        if self.tracer is not None:
            self.tracer.tool_call(agent, name, args or {}, text, duration_ms, status, error)
        return text, status, result
