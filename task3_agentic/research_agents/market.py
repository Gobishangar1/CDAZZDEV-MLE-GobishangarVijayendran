"""Price history from yfinance and technical indicators (pandas only, no TA-Lib).

Adapted from Task 1 (task1_financial/equity_research/market_data.py and indicators.py), kept as a separate copy
so Task 3 runs on its own.
"""
# SOURCE: Adapted from task1_financial/equity_research/market_data.py and indicators.py (this repository)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Compact yfinance history fetch with retries and a
# per-session cache, plus SMA/RSI/MACD/Bollinger for the Task 3 tools', Date: 2026-10-11

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from research_agents.config import (
    BB_DDOF,
    BB_STD,
    BB_WINDOW,
    FETCH_ATTEMPTS,
    FETCH_BACKOFF_S,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
)

log = logging.getLogger(__name__)
TICKER_PATTERN = r"^[A-Z0-9.\-^]{1,10}$"
_history_cache: dict[tuple[str, str], pd.DataFrame] = {}   # (ticker, period) -> frame, for one Python session
_info_cache: dict[str, dict] = {}


class ToolError(RuntimeError):
    """A tool could not produce a result. The message is shown to the agent so it can replan."""


def normalize_ticker(raw: object) -> str:
    if not isinstance(raw, str) or not re.fullmatch(TICKER_PATTERN, raw.strip().upper()):
        raise ToolError(f"'{raw}' is not a valid ticker symbol")
    return raw.strip().upper()


def _get_ticker(ticker: str) -> yf.Ticker:
    """Single factory for yfinance objects (tests replace it)."""
    return yf.Ticker(ticker)


def clear_caches() -> None:
    _history_cache.clear()
    _info_cache.clear()


@retry(stop=stop_after_attempt(FETCH_ATTEMPTS), wait=wait_exponential(multiplier=FETCH_BACKOFF_S),
       retry=retry_if_not_exception_type(ToolError), reraise=True)
def _download(ticker: str, period: str) -> pd.DataFrame:
    raw = _get_ticker(ticker).history(period=period, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no rows for {ticker}")
    return raw


def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    """Clean daily OHLCV (adjusted). Cached per (ticker, period) for this Python session."""
    ticker = normalize_ticker(ticker)
    key = (ticker, period)
    if key not in _history_cache:
        try:
            raw = _download(ticker, period)
        except Exception as exc:
            raise ToolError(f"No price data for {ticker} ({period}): {exc}") from exc
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df[df["Close"].notna()]
        df = df[~df.index.duplicated(keep="last")]
        if df.empty:
            raise ToolError(f"No valid closing prices for {ticker}")
        _history_cache[key] = df
    return _history_cache[key].copy()


def fetch_company_name(ticker: str) -> str | None:
    """longName from Yahoo, or None (never raises). Used to filter relevant headlines."""
    ticker = normalize_ticker(ticker)
    if ticker not in _info_cache:
        try:
            _info_cache[ticker] = _get_ticker(ticker).get_info() or {}
        except Exception as exc:
            log.warning("info_failed ticker=%s error=%s", ticker, exc)
            _info_cache[ticker] = {}
    return _info_cache[ticker].get("longName")


# -----------------------------------------------------------------------------
# Indicators (same formulas as Task 1)
# -----------------------------------------------------------------------------
def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI: seeded with a simple mean, then Wilder smoothing."""
    delta = close.diff().to_numpy(dtype=float)
    gains, losses = np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta, 0.0)
    out = np.full(len(close), np.nan)

    def value(g: float, lo: float) -> float:
        return (50.0 if g == 0 else 100.0) if lo == 0 else 100 - 100 / (1 + g / lo)

    if len(close) > period:
        g, lo = gains[1:period + 1].mean(), losses[1:period + 1].mean()
        out[period] = value(g, lo)
        for i in range(period + 1, len(close)):
            g = (g * (period - 1) + gains[i]) / period
            lo = (lo * (period - 1) + losses[i]) / period
            out[i] = value(g, lo)
    return pd.Series(out, index=close.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]
    out["sma_50"], out["sma_200"], out["rsi_14"] = sma(c, SMA_SHORT), sma(c, SMA_LONG), rsi(c)
    macd_line = ema(c, MACD_FAST) - ema(c, MACD_SLOW)
    signal = ema(macd_line, MACD_SIGNAL)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd_line, signal, macd_line - signal
    mid = c.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = c.rolling(BB_WINDOW, min_periods=BB_WINDOW).std(ddof=BB_DDOF)
    out["bb_upper"], out["bb_middle"], out["bb_lower"] = mid + BB_STD * std, mid, mid - BB_STD * std
    out["bb_pct_b"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    return out
