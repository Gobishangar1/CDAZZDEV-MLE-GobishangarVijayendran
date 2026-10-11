"""Technical indicators from first principles: pandas/numpy only, no TA-Lib (Task 1A, 25 marks).

Every function takes a pd.Series of adjusted closes and returns a Series/DataFrame on the same index.
Values are NaN until enough history exists (no partial windows).

    sma(close, n)        simple moving average
    ema(close, n)        exponential moving average, alpha = 2/(n+1), recursive form
    rsi(close, 14)       Wilder's RSI (seeded with a simple mean, then Wilder smoothing)
    macd(close)          MACD line, signal line, histogram (12, 26, 9)
    bollinger(close)     bands (20, 2 sigma, population std) + %B + bandwidth
    add_indicators(df)   all of the above as columns
    last_ma_cross(a, b)  latest golden/death cross
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement SMA, EMA, Wilder RSI, MACD (12,26,9) and
# Bollinger Bands (20, 2 std, ddof=0) from first principles with pandas, no TA-Lib', Date: 2026-10-11

from __future__ import annotations

import numpy as np
import pandas as pd

from equity_research.config import (
    BB_DDOF,
    BB_STD,
    BB_WINDOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_NEUTRAL,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
)


def sma(close: pd.Series, window: int) -> pd.Series:
    """SMA_n(t) = mean of the last n closes. min_periods=n: no value until the window is full."""
    return close.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """EMA_t = a*x_t + (1-a)*EMA_{t-1}, a = 2/(span+1).

    adjust=False gives this recursive (trader's) definition, seeded with the first value.
    The seed effect fades within a few multiples of `span`; the warm-up buffer hides it.
    """
    return series.ewm(span=span, adjust=False).mean()


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    """RSI = 100 - 100/(1 + avg_gain/avg_loss), with the divide-by-zero cases handled."""
    if avg_loss == 0:
        return RSI_NEUTRAL if avg_gain == 0 else 100.0   # flat price -> 50; only gains -> 100
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI.

    1. change = close - previous close; gain = max(change, 0), loss = max(-change, 0)
    2. first average = simple mean of the first `period` gains (and losses)
    3. then Wilder smoothing: avg = ((period - 1) * avg + today) / period

    Note: ewm(alpha=1/period) alone seeds with ONE value, not the `period`-day mean, so its
    early values differ (by ~10 RSI points at the start). We implement Wilder's seed exactly.
    """
    delta = close.diff().to_numpy(dtype=float)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    out = np.full(len(close), np.nan)
    if len(close) > period:
        avg_gain = gains[1:period + 1].mean()     # index 0 is the NaN change
        avg_loss = losses[1:period + 1].mean()
        out[period] = _rsi_from_averages(avg_gain, avg_loss)
        for i in range(period + 1, len(close)):   # each value depends on the previous one
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            out[i] = _rsi_from_averages(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name=f"RSI_{period}")


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """MACD = EMA_fast - EMA_slow; Signal = EMA_signal(MACD); Histogram = MACD - Signal."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": signal_line,
                         "MACD_hist": macd_line - signal_line}, index=close.index)


def bollinger(close: pd.Series, window: int = BB_WINDOW, n_std: float = BB_STD,
              ddof: int = BB_DDOF) -> pd.DataFrame:
    """Middle = SMA_20; Upper/Lower = Middle +/- 2 sigma_20.

    ddof=0 (population std) is Bollinger's definition. pandas defaults to ddof=1, which makes
    the bands sqrt(20/19) = 2.6% wider. %B = position inside the band (0 = lower, 1 = upper).
    Bandwidth = (Upper - Lower) / Middle, so it compares across price levels.
    """
    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=ddof)
    upper = middle + n_std * std
    lower = middle - n_std * std
    band = (upper - lower).replace(0, np.nan)    # flat price: width 0 -> NaN, not a divide-by-zero
    return pd.DataFrame({"BB_middle": middle, "BB_upper": upper, "BB_lower": lower,
                         "BB_pctB": (close - lower) / band, "BB_width": band / middle}, index=close.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the OHLCV frame with all indicator columns added."""
    out = df.copy()
    close = out["Close"]
    out[f"SMA_{SMA_SHORT}"] = sma(close, SMA_SHORT)
    out[f"SMA_{SMA_LONG}"] = sma(close, SMA_LONG)
    out[f"RSI_{RSI_PERIOD}"] = rsi(close)
    return out.join(macd(close)).join(bollinger(close))


def last_ma_cross(sma_short: pd.Series, sma_long: pd.Series) -> tuple[str, pd.Timestamp, int] | None:
    """Latest golden (short crosses above long) or death cross: (kind, date, trading_days_ago)."""
    both = pd.concat([sma_short, sma_long], axis=1).dropna()   # only days where both exist
    if both.empty:
        return None
    above = (both.iloc[:, 0] > both.iloc[:, 1]).astype(int)
    change = above.diff()                                       # +1 golden, -1 death, 0 none
    events = change[change != 0].dropna()
    if events.empty:
        return None
    date = events.index[-1]
    kind = "golden" if events.iloc[-1] > 0 else "death"
    return kind, date, len(both.loc[date:]) - 1
