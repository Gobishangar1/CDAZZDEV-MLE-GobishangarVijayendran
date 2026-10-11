"""Summary dictionary and rule-based momentum signal (Task 1A).

    indicator_snapshot(df)        latest indicator values as plain floats (what the LLM sees)
    momentum_signal(snapshot)     weighted vote of the indicators -> MomentumSignal
    build_summary(...)            the required summary -> MarketSummary (.model_dump() for the dict)
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A summary (price, 52-week range,
# P/E with fallbacks, YTD return) and a weighted rule-based momentum score with re-normalisation
# for missing indicators', Date: 2026-10-11

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.config import (
    MACD_FADE_DAMPING,
    MACD_FADE_STEPS,
    MACD_HIST_KEEP,
    MOMENTUM_LABELS,
    MOMENTUM_WEIGHTS,
    RSI_LOOKBACK_DAYS,
    RSI_NEUTRAL,
    RSI_PERIOD,
    RSI_SCALE,
    SMA_LONG,
    SMA_SHORT,
    TRADING_DAYS_PER_YEAR,
)
from equity_research.indicators import last_ma_cross
from equity_research.schemas import MarketSummary, MomentumSignal

SMA_S, SMA_L, RSI_COL = f"SMA_{SMA_SHORT}", f"SMA_{SMA_LONG}", f"RSI_{RSI_PERIOD}"


def safe_float(value: object) -> float | None:
    """A real, finite number, or None. Handles None, text and 'Infinity' from Yahoo."""
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# -----------------------------------------------------------------------------
# Summary fields
# -----------------------------------------------------------------------------
def pe_ratio(price: float, info: dict) -> tuple[float | None, str]:
    """(P/E, source) in 3 levels: Yahoo trailing P/E -> price / EPS (only if EPS > 0) -> None.

    A negative P/E has no meaning (the company lost money), so it is never computed.
    """
    pe = safe_float(info.get("trailingPE"))
    if pe is not None and pe > 0:
        return round(pe, 2), "trailingPE"
    eps = safe_float(info.get("trailingEps"))
    if eps is not None and eps > 0:
        return round(price / eps, 2), "computed_from_eps"
    return None, "unavailable"


def ytd_return(close: pd.Series) -> tuple[float | None, str]:
    """(YTD return %, base). Base = last close of the previous year (standard definition).

    For a stock that started trading this year, the base is its first close this year.
    """
    last_day = close.index[-1]
    prev_year = close[close.index.year < last_day.year]
    if not prev_year.empty:
        base, source = prev_year.iloc[-1], "prior_year_close"
    else:
        base, source = close[close.index.year == last_day.year].iloc[0], "first_close_this_year"
    value = safe_float(close.iloc[-1] / base - 1)
    return (round(value * 100, 2) if value is not None else None), source


# -----------------------------------------------------------------------------
# Snapshot and momentum
# -----------------------------------------------------------------------------
def indicator_snapshot(df: pd.DataFrame) -> dict:
    """Latest values from the indicator frame. NaN becomes None (JSON-friendly)."""
    last = df.iloc[-1]

    def val(col: str) -> float | None:
        v = last.get(col)
        return None if v is None or pd.isna(v) else round(float(v), 4)

    rsi_series = df[RSI_COL].dropna()
    cross = last_ma_cross(df[SMA_S], df[SMA_L])
    return {
        "close": val("Close"),
        "sma_50": val(SMA_S),
        "sma_200": val(SMA_L),
        "rsi": val(RSI_COL),
        "rsi_10d_ago": (round(float(rsi_series.iloc[-RSI_LOOKBACK_DAYS - 1]), 2)
                        if len(rsi_series) > RSI_LOOKBACK_DAYS else None),
        "macd": val("MACD"),
        "macd_signal": val("MACD_signal"),
        "macd_hist_last5": [round(float(x), 4) for x in df["MACD_hist"].dropna().iloc[-MACD_HIST_KEEP:]],
        "bb_upper": val("BB_upper"),
        "bb_lower": val("BB_lower"),
        "bb_pctb": val("BB_pctB"),
        "cross": {"kind": cross[0], "date": str(cross[1].date()), "days_ago": cross[2]} if cross else None,
    }


def macd_is_fading(hist: list[float]) -> bool:
    """True if |histogram| shrank on each of the last MACD_FADE_STEPS days."""
    recent = hist[-(MACD_FADE_STEPS + 1):]
    return len(recent) == MACD_FADE_STEPS + 1 and all(abs(b) < abs(a) for a, b in zip(recent, recent[1:]))


def _label(score: float) -> str:
    for cutoff, name in MOMENTUM_LABELS:
        if (score >= cutoff) if cutoff > 0 else (score > cutoff):
            return name
    return "strong_bearish"


def momentum_signal(s: dict) -> MomentumSignal:
    """Each indicator votes in [-1, +1]; the weighted average is the score.

    Missing indicators (e.g. no SMA-200 for a young stock) are dropped and the remaining
    weights re-normalised, so a missing vote does not pull the score toward neutral.
    """
    votes: dict[str, float] = {}
    if s.get("sma_50") is not None and s.get("sma_200") is not None:
        votes["long_trend"] = 1.0 if s["sma_50"] > s["sma_200"] else -1.0
    if s.get("sma_50") is not None:
        votes["short_trend"] = 1.0 if s["close"] > s["sma_50"] else -1.0
    hist = s.get("macd_hist_last5") or []
    if hist:
        damping = MACD_FADE_DAMPING if macd_is_fading(hist) else 1.0
        votes["macd"] = float(np.sign(hist[-1])) * damping
    if s.get("rsi") is not None:
        votes["rsi"] = float(np.clip((s["rsi"] - RSI_NEUTRAL) / RSI_SCALE, -1, 1))
    if s.get("bb_pctb") is not None:
        votes["bollinger"] = float(np.clip((s["bb_pctb"] - 0.5) * 2, -1, 1))   # %B 0..1 -> -1..+1

    missing = [k for k in MOMENTUM_WEIGHTS if k not in votes]
    if not votes:
        return MomentumSignal(label="neutral", score=0.0, components={}, missing=missing)
    total_weight = sum(MOMENTUM_WEIGHTS[k] for k in votes)
    score = sum(MOMENTUM_WEIGHTS[k] * v for k, v in votes.items()) / total_weight
    return MomentumSignal(label=_label(score), score=round(score, 3),
                          components={k: round(v, 2) for k, v in votes.items()}, missing=missing)


# -----------------------------------------------------------------------------
# The summary dictionary
# -----------------------------------------------------------------------------
def build_summary(ticker: str, df: pd.DataFrame, info: dict, momentum: MomentumSignal) -> MarketSummary:
    """Current price, 52-week high/low (from High/Low columns), P/E, YTD return, momentum."""
    price = float(df["Close"].iloc[-1])
    last_year = df.iloc[-TRADING_DAYS_PER_YEAR:]
    high_52w = float(last_year["High"].max())
    low_52w = float(last_year["Low"].min())
    pe, pe_source = pe_ratio(price, info)
    ytd, ytd_base = ytd_return(df["Close"])
    return MarketSummary(
        ticker=ticker,
        company_name=info.get("longName"),
        currency=info.get("currency"),
        as_of=str(df.index[-1].date()),
        current_price=round(price, 2),
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        pct_from_52w_high=round((price / high_52w - 1) * 100, 2),
        pe_ratio=pe,
        pe_source=pe_source,
        ytd_return_pct=ytd,
        ytd_base=ytd_base,
        momentum=momentum,
    )
