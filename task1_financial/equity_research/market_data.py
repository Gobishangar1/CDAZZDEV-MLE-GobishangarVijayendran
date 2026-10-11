"""Daily OHLCV prices and company info from yfinance (Task 1A).

Public functions
    normalize_ticker(raw)          -> "NVDA" (or InvalidTickerError)
    probe_ticker(ticker)           -> True if Yahoo has recent prices (never raises)
    fetch_prices(ticker, today)    -> (clean OHLCV DataFrame, warnings)   MarketDataError if no prices
    check_prices(df)               -> warnings for impossible or suspicious rows (never changes data)
    fetch_info(ticker)             -> company facts dict, {} on failure (never raises)

Design rules
    - No hard-coded date strings: the start date is computed from today with DateOffset.
    - Prices are split/dividend adjusted (auto_adjust=True).
    - Network calls retry with exponential backoff; a malformed ticker is never retried.
    - Only missing prices are fatal. Missing company info is not.
    - `_get_ticker` is the only place that creates a yfinance object, so tests can fake it.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yfinance OHLCV fetch with a computed start date
# plus warm-up buffer, tenacity retries, cleaning of null closes and duplicate dates, data checks
# and a fail-soft company info fetch', Date: 2026-10-11

from __future__ import annotations

import logging
import re

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from equity_research.config import (
    FETCH_ATTEMPTS,
    FETCH_BACKOFF_S,
    HISTORY_YEARS,
    MAX_DAILY_MOVE,
    PROBE_PERIOD,
    TICKER_PATTERN,
    TRADING_DAYS_PER_YEAR,
    WARMUP_CALENDAR_DAYS,
)

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
INFO_KEYS = ["longName", "sector", "industry", "currency", "marketCap", "trailingPE", "trailingEps"]


class InvalidTickerError(ValueError):
    """The ticker is malformed, or Yahoo has no prices for it."""


class MarketDataError(RuntimeError):
    """Prices could not be fetched. The only fatal data error."""


class _EmptyHistory(RuntimeError):
    """Internal: Yahoo returned no rows. Retried, because it is often temporary."""


def _get_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


_network_retry = retry(
    stop=stop_after_attempt(FETCH_ATTEMPTS),
    wait=wait_exponential(multiplier=FETCH_BACKOFF_S),
    retry=retry_if_not_exception_type(InvalidTickerError),   # retrying a bad ticker never helps
    reraise=True,
)


# -----------------------------------------------------------------------------
# Ticker
# -----------------------------------------------------------------------------
def normalize_ticker(raw: object) -> str:
    """Uppercase and strip, then check the format."""
    if not isinstance(raw, str):
        raise InvalidTickerError(f"Ticker must be text, got {type(raw).__name__}")
    ticker = raw.strip().upper()
    if not re.fullmatch(TICKER_PATTERN, ticker):
        raise InvalidTickerError(f"'{raw}' is not a valid ticker format")
    return ticker


def probe_ticker(ticker: str) -> bool:
    """Cheap 5-day fetch so a bad ticker stops the run BEFORE any LLM call."""
    try:
        history = _get_ticker(ticker).history(period=PROBE_PERIOD)
    except Exception as exc:
        log.warning("ticker_probe_failed ticker=%s error=%s", ticker, exc)
        return False
    return history is not None and not history.empty


# -----------------------------------------------------------------------------
# Prices
# -----------------------------------------------------------------------------
def compute_start_dates(today: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(fetch_start, display_start). display_start = today - 2 years; fetch_start adds the warm-up."""
    today = pd.Timestamp(today).normalize()
    display_start = today - pd.DateOffset(years=HISTORY_YEARS)
    fetch_start = display_start - pd.DateOffset(days=WARMUP_CALENDAR_DAYS)   # DateOffset: no numpy "generic unit" warning
    return fetch_start, display_start


@_network_retry
def _download_history(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    raw = _get_ticker(ticker).history(start=start.strftime("%Y-%m-%d"), auto_adjust=True)
    if raw is None or raw.empty:
        raise _EmptyHistory(f"Yahoo returned no rows for {ticker}")
    return raw


def clean_ohlcv(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep OHLCV, remove the timezone, sort, drop null closes and duplicate dates."""
    missing = [c for c in OHLCV_COLUMNS if c not in raw.columns]
    if missing:
        raise MarketDataError(f"Price data is missing columns: {missing}")

    df = raw[OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)   # plain dates avoid timezone comparison bugs
    df.index = df.index.normalize()
    df.index.name = "Date"
    df = df.sort_index()

    null_close = int(df["Close"].isna().sum())
    df = df[df["Close"].notna()]                 # every indicator depends on Close
    duplicated = df.index.duplicated(keep="last")
    df = df[~duplicated]
    return df, {"null_close_dropped": null_close, "duplicate_dates_dropped": int(duplicated.sum())}


def check_prices(df: pd.DataFrame) -> list[str]:
    """Flag impossible or suspicious rows. Never changes the data: in equities a big move is usually news."""
    warnings = []
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        warnings.append("Some prices are zero or negative")
    if (df["High"] < df["Low"]).any():
        warnings.append("Some rows have High below Low")
    outside = (df["Close"] > df["High"]) | (df["Close"] < df["Low"])
    if outside.any():
        warnings.append(f"{int(outside.sum())} rows have Close outside the High-Low range")
    big = df["Close"].pct_change().abs() > MAX_DAILY_MOVE
    if big.any():
        days = ", ".join(str(d.date()) for d in df.index[big][:3])
        warnings.append(f"{int(big.sum())} daily moves above {MAX_DAILY_MOVE:.0%} (check: {days})")
    return warnings


def fetch_prices(ticker: str, today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    """2 years + warm-up of clean, adjusted daily OHLCV, and a list of data-quality warnings.

    The frame includes the warm-up rows (indicators need them). Reports show only recent rows.
    """
    ticker = normalize_ticker(ticker)
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    fetch_start, display_start = compute_start_dates(today)

    try:
        raw = _download_history(ticker, fetch_start)
    except Exception as exc:
        raise MarketDataError(f"Could not fetch prices for {ticker} after {FETCH_ATTEMPTS} attempts: {exc}") from exc

    df, removed = clean_ohlcv(raw)
    if df.empty:
        raise MarketDataError(f"No valid closing prices for {ticker} after cleaning")

    warnings = []
    if removed["null_close_dropped"]:
        warnings.append(f"Dropped {removed['null_close_dropped']} rows with a missing close price")
    if removed["duplicate_dates_dropped"]:
        warnings.append(f"Dropped {removed['duplicate_dates_dropped']} duplicate dates")
    if df.index[0] > display_start:
        warnings.append(f"History starts {df.index[0].date()}, after the 2-year window start "
                        f"{display_start.date()} (young stock); long indicators may be missing")
    rows_in_window = int((df.index >= display_start).sum())
    if rows_in_window < HISTORY_YEARS * TRADING_DAYS_PER_YEAR * 0.95:   # allow a few holidays
        warnings.append(f"Only {rows_in_window} trading days in the 2-year window")
    warnings += check_prices(df)

    for w in warnings:
        log.warning("price_quality ticker=%s %s", ticker, w)
    log.info("prices_ok ticker=%s rows=%d first=%s last=%s", ticker, len(df), df.index[0].date(), df.index[-1].date())
    return df, warnings


# -----------------------------------------------------------------------------
# Company info
# -----------------------------------------------------------------------------
@_network_retry
def _download_info(ticker: str) -> dict:
    info = _get_ticker(ticker).get_info()
    if not isinstance(info, dict):
        raise RuntimeError("get_info() did not return a dict")
    return info


def fetch_info(ticker: str) -> dict:
    """Company name, sector, currency, market cap, trailing P/E and EPS. {} on any failure."""
    try:
        info = _download_info(normalize_ticker(ticker))
    except Exception as exc:
        log.warning("info_failed ticker=%s error=%s", ticker, exc)
        return {}
    return {k: info.get(k) for k in INFO_KEYS}
