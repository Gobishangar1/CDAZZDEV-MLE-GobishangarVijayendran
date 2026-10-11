"""Pydantic models: the contracts for every structured object in the pipeline.

LLM-facing (validated before use):
    HeadlineSentiment   one per headline (brief: headline, sentiment, confidence, brief_reason)
    TradeSignal         the Buy/Hold/Sell call

Pipeline outputs (built by code, validated so every required field is present):
    MomentumSignal, MarketSummary (the "summary dictionary"), SentimentSummary
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic schemas for headline sentiment
# (headline, sentiment, confidence, brief_reason), trade signal with a 3-5 sentence validator,
# market summary and momentum signal', Date: 2026-10-11

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from equity_research.config import JUSTIFICATION_MAX_SENTENCES, JUSTIFICATION_MIN_SENTENCES

SentimentLabel = Literal["positive", "negative", "neutral"]
MomentumLabel = Literal["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"]

# -----------------------------------------------------------------------------
# Text helpers used by the TradeSignal validator and by trade_signal.py
# -----------------------------------------------------------------------------
# GPT-OSS often writes Unicode dashes ("52‑week" with U+2011). Normalise them before any regex.
_DASHES = re.compile(r"[‐-―−­]")
# A sentence ends at . ! ? followed by space + capital letter, so "2.2%" is not a sentence end.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def clean_text(text: str) -> str:
    return _DASHES.sub("-", text).replace(" ", " ").strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(clean_text(text)) if s.strip()]


# -----------------------------------------------------------------------------
# LLM-facing models
# -----------------------------------------------------------------------------
class HeadlineSentiment(BaseModel):
    """Brief: 'headline, sentiment (positive, negative, or neutral), confidence (0-1), brief_reason'."""

    headline: str = Field(description="The headline text, copied exactly")
    sentiment: SentimentLabel = Field(description="Likely effect on this stock's price")
    confidence: float = Field(ge=0.0, le=1.0, description="0 = guess, 1 = certain")
    brief_reason: str = Field(min_length=1, max_length=300, description="One sentence, max 25 words")


class TradeSignal(BaseModel):
    signal: Literal["BUY", "HOLD", "SELL"]
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = Field(description="3 to 5 sentences explaining how the indicators combine")
    key_drivers: list[str] = Field(min_length=2, description="Indicator combinations that drove the call")
    conflicting_evidence: list[str] = Field(description="Signals against the call, and why they do not override it")

    @field_validator("justification")
    @classmethod
    def three_to_five_sentences(cls, v: str) -> str:
        n = len(split_sentences(v))
        if not JUSTIFICATION_MIN_SENTENCES <= n <= JUSTIFICATION_MAX_SENTENCES:
            raise ValueError(f"justification has {n} sentences; it must have "
                             f"{JUSTIFICATION_MIN_SENTENCES} to {JUSTIFICATION_MAX_SENTENCES}")
        return v


# -----------------------------------------------------------------------------
# Pipeline outputs
# -----------------------------------------------------------------------------
class MomentumSignal(BaseModel):
    label: MomentumLabel
    score: float = Field(ge=-1.0, le=1.0)
    components: dict[str, float]        # each indicator's vote, so the score is explainable
    missing: list[str] = []             # indicators that could not vote (e.g. no SMA-200 yet)


class MarketSummary(BaseModel):
    """The summary dictionary required by Task 1A (use .model_dump() to get the dict)."""

    ticker: str
    company_name: str | None
    currency: str | None
    as_of: str                          # date of the last close used
    current_price: float
    high_52w: float
    low_52w: float
    pct_from_52w_high: float
    pe_ratio: float | None              # None when not meaningful (losses, ETFs, missing)
    pe_source: Literal["trailingPE", "computed_from_eps", "unavailable"]
    ytd_return_pct: float | None
    ytd_base: Literal["prior_year_close", "first_close_this_year"]
    momentum: MomentumSignal


class SentimentSummary(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    label: SentimentLabel
    counts: dict[str, int]
    agreement: float = Field(ge=0.0, le=1.0)   # share of headlines with the majority label
    coverage: float = Field(ge=0.0, le=1.0)    # valid results / all headlines
    n_valid: int
