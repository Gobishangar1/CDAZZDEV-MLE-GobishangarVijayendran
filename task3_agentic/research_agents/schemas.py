"""Pydantic schemas: every tool result, every agent-to-agent message, and the final report.

    Tool results        PriceData, NewsResult, VolatilityResult, SentimentResult, SearchResult
    Agent handoff (3B)  DataBrief (A -> B), ClarificationRequest (B -> A), ClarificationResponse (A -> B)
    Final output        ResearchReport (Financial Health Summary, Top Three Risks, Hedge Strategy)

Nothing travels between agents as a raw string: each message is one of these models.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "negative", "neutral"]


class Compact(BaseModel):
    """Base model with a compact JSON form for the LLM (no spaces, no nulls)."""

    def to_llm(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=True), separators=(",", ":"), default=str)

    def is_empty(self) -> bool:
        return False


# -----------------------------------------------------------------------------
# Tool results
# -----------------------------------------------------------------------------
class OHLCVRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceData(Compact):
    """get_price_data: OHLCV with computed indicators (latest values + the last rows)."""

    ticker: str
    period: str
    start: str
    end: str
    rows: int
    last_close: float
    period_return_pct: float
    high_52w: float
    low_52w: float
    pct_from_52w_high: float
    indicators: dict[str, float | None]      # sma_50, sma_200, rsi_14, macd, macd_signal, macd_hist, bb_*
    trend: str                                # plain-language reading computed by code
    recent_ohlcv: list[OHLCVRow]


class Headline(BaseModel):
    title: str
    publisher: str | None = None
    published_at: str | None = None
    source: str
    relevant: bool = True


class NewsResult(Compact):
    ticker: str
    headlines: list[Headline]
    warnings: list[str] = []

    def is_empty(self) -> bool:
        return not self.headlines


class VolatilityResult(Compact):
    """calculate_volatility: annualised historical volatility of daily log returns."""

    ticker: str
    window: int
    as_of: str
    annualised_vol_pct: float                 # std(log returns over window) * sqrt(252) * 100
    one_year_vol_pct: float
    vol_percentile_1y: float                  # where today's window-vol sits within the last year (0-100)
    max_drawdown_1y_pct: float
    var_95_1d_pct: float                      # historical 1-day Value-at-Risk at 95% (a loss, negative)
    interpretation: str


class HeadlineLabel(BaseModel):
    headline: str
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=1, max_length=300)


class SentimentLLMOutput(BaseModel):
    """What the sentiment LLM must return (validated before use)."""

    items: list[HeadlineLabel] = Field(min_length=1)


class SentimentResult(Compact):
    n: int
    score: float = Field(ge=-1.0, le=1.0)     # confidence-weighted mean of +1 / 0 / -1
    label: Sentiment
    counts: dict[str, int]
    items: list[HeadlineLabel]


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class SearchResult(Compact):
    query: str
    results: list[SearchHit]

    def is_empty(self) -> bool:
        return not self.results


# -----------------------------------------------------------------------------
# Agent-to-agent messages (3B)
# -----------------------------------------------------------------------------
class DataBrief(Compact):
    """Agent A -> Agent B. Numbers are copied from tool results by code, never retyped by the LLM."""

    ticker: str
    as_of: str
    price: dict[str, Any] | None = None
    volatility: dict[str, Any] | None = None
    sentiment: dict[str, Any] | None = None
    analyst_notes: str = Field(min_length=1)  # Agent A's own words
    data_gaps: list[str] = []
    tools_used: list[str] = []


class ClarificationRequest(Compact):
    """Agent B -> Agent A: ONE specific request for data that only Agent A's tools can produce."""

    question: str = Field(min_length=20)
    requested_data: list[str] = Field(min_length=1)
    headlines: list[str] = Field(default=[], max_length=15)   # B can pass headlines; A cannot fetch news
    reason: str = Field(min_length=10)

    @field_validator("headlines", mode="before")
    @classmethod
    def _at_most_15(cls, value: Any) -> Any:
        """The sentiment tool takes 15 headlines at most: keep the first 15 instead of failing."""
        return value[:15] if isinstance(value, list) else value


class ClarificationResponse(Compact):
    """Agent A -> Agent B: the answer, with the raw tool data behind it."""

    question: str
    answer: str = Field(min_length=1)
    data: dict[str, Any] = {}
    tools_used: list[str] = []


# -----------------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------------
class FinancialHealthSummary(BaseModel):
    summary: str = Field(min_length=40)
    key_metrics: list[str] = Field(min_length=2)
    market_sentiment: str = Field(min_length=10)


class Risk(BaseModel):
    title: str
    description: str = Field(min_length=20)
    evidence: list[str] = Field(min_length=1)   # specific facts: numbers or headlines
    sources: list[str] = Field(min_length=1)    # tool names or URLs the evidence came from
    severity: Literal["high", "medium", "low"]

    @field_validator("severity", mode="before")
    @classmethod
    def _normalise_severity(cls, value: Any) -> Any:
        """Accept 'High', 'Medium-High', 'moderate', 'critical' etc. instead of failing the whole report."""
        if not isinstance(value, str):
            return value
        v = value.strip().lower()
        if v in ("high", "medium", "low"):
            return v
        if "high" in v or "critical" in v or "severe" in v:
            return "high"
        if "medium" in v or "moderate" in v or "mid" in v:
            return "medium"
        return "low" if "low" in v or "minor" in v else value


class HedgeStrategy(BaseModel):
    strategy: str
    instrument: str
    rationale: str = Field(min_length=30)
    data_points: list[str] = Field(min_length=1)
    sizing_or_trigger: str


class ResearchReport(Compact):
    ticker: str
    as_of: str
    financial_health_summary: FinancialHealthSummary
    top_three_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy
    clarification_used: str | None = None       # 3B: how Agent A's answer changed the report
    sources: list[str] = []
    disclaimer: str = ""
