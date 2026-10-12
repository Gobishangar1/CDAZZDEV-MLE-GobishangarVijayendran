"""Indicator accuracy (25 marks): hand-computed values, an independent re-implementation, and
mathematical properties. All expected values below were worked out by hand in the lessons."""

import numpy as np
import pandas as pd
import pytest

from equity_research.indicators import add_indicators, bollinger, ema, last_ma_cross, macd, rsi, sma
from tests.conftest import make_prices

HAND = pd.Series([10, 11, 12, 11, 13, 12], dtype=float)


# ------------------------------------------------------------------ SMA
def test_sma_hand_values():
    assert sma(HAND, 3).round(4).tolist()[2:] == [11.0, 11.3333, 12.0, 12.0]
    assert sma(HAND, 3).iloc[:2].isna().all()          # no partial windows


# ------------------------------------------------------------------ EMA
def test_ema_hand_values():
    # EMA-3: alpha = 0.5, seeded with the first close
    assert ema(HAND, 3).round(4).tolist() == [10.0, 10.5, 11.25, 11.125, 12.0625, 12.0312]


def test_ema_matches_recursive_loop():
    close = make_prices(300)["Close"].reset_index(drop=True)
    alpha, expected = 2 / (12 + 1), [close.iloc[0]]
    for x in close.iloc[1:]:
        expected.append(alpha * x + (1 - alpha) * expected[-1])
    np.testing.assert_allclose(ema(close, 12).to_numpy(), expected, rtol=1e-12)


# ------------------------------------------------------------------ RSI
def test_rsi_hand_values():
    # Period 3: seed = mean of first 3 gains/losses, then (2*avg + today)/3
    out = rsi(HAND, 3).round(2).tolist()
    assert np.isnan(out[:3]).all()
    assert out[3:] == [66.67, 83.33, 60.61]


def test_rsi_flat_day_keeps_value():
    # A flat day shrinks both averages by the same factor, so RSI stays the same
    assert rsi(pd.Series([10, 11, 12, 11, 13, 12, 12], dtype=float), 3).round(2).iloc[-1] == 60.61


def test_rsi_edge_cases():
    assert rsi(pd.Series([5.0] * 20)).iloc[-1] == 50.0                 # flat price
    assert rsi(pd.Series(np.arange(20.0))).iloc[-1] == 100.0           # only gains
    assert rsi(pd.Series(np.arange(20.0, 0, -1))).iloc[-1] == 0.0      # only losses
    assert rsi(pd.Series([1.0, 2.0, 3.0])).isna().all()                # too short: no crash


def test_rsi_bounds_and_first_value_position():
    out = rsi(make_prices(400)["Close"])
    assert out.iloc[:14].isna().all() and not np.isnan(out.iloc[14])  # 15 prices -> first RSI
    assert out.dropna().between(0, 100).all()


def test_rsi_seed_differs_from_plain_ewm_then_converges():
    """Wilder seeds with a 14-day mean; plain ewm seeds with one value. They converge later,
    which is why the 300-day warm-up makes the displayed window independent of the seed."""
    close = make_prices(400)["Close"].reset_index(drop=True)
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plain = 100 - 100 / (1 + gain / loss)
    diff = (rsi(close) - plain).abs()
    assert diff.iloc[14] > 0.5                  # different at the start
    assert diff.iloc[250:].max() < 1e-6         # identical after convergence


# ------------------------------------------------------------------ MACD
def test_macd_definition():
    close = make_prices(300)["Close"]
    m = macd(close)
    np.testing.assert_allclose(m["MACD"], ema(close, 12) - ema(close, 26))
    np.testing.assert_allclose(m["MACD_signal"], ema(m["MACD"], 9))
    np.testing.assert_allclose(m["MACD_hist"], m["MACD"] - m["MACD_signal"])


def test_macd_sign_follows_trend():
    up = macd(pd.Series(np.linspace(100, 200, 120)))
    down = macd(pd.Series(np.linspace(200, 100, 120)))
    assert up["MACD"].iloc[-1] > 0 and down["MACD"].iloc[-1] < 0


# ------------------------------------------------------------------ Bollinger
def test_bollinger_matches_numpy_population_std():
    close = make_prices(100)["Close"]
    bb = bollinger(close)
    window = close.iloc[-20:].to_numpy()
    mid, sd = window.mean(), window.std(ddof=0)          # numpy default ddof=0 = population std
    assert bb["BB_middle"].iloc[-1] == pytest.approx(mid)
    assert bb["BB_upper"].iloc[-1] == pytest.approx(mid + 2 * sd)
    assert bb["BB_lower"].iloc[-1] == pytest.approx(mid - 2 * sd)
    assert bb["BB_pctB"].iloc[-1] == pytest.approx((close.iloc[-1] - (mid - 2 * sd)) / (4 * sd))
    assert bb["BB_width"].iloc[-1] == pytest.approx(4 * sd / mid)


def test_bollinger_ddof_difference_is_sqrt_20_over_19():
    close = make_prices(100)["Close"]
    sample = bollinger(close, ddof=1).iloc[-1]
    pop = bollinger(close, ddof=0).iloc[-1]
    ratio = (sample["BB_upper"] - sample["BB_middle"]) / (pop["BB_upper"] - pop["BB_middle"])
    assert ratio == pytest.approx((20 / 19) ** 0.5)


def test_bollinger_order_and_flat_price():
    bb = bollinger(make_prices(200)["Close"]).dropna()
    assert (bb["BB_upper"] >= bb["BB_middle"]).all() and (bb["BB_middle"] >= bb["BB_lower"]).all()
    flat = bollinger(pd.Series([7.0] * 30))
    assert flat["BB_width"].isna().iloc[-1] and flat["BB_pctB"].isna().iloc[-1]   # NaN, not a crash


# ------------------------------------------------------------------ helpers
def test_add_indicators_columns_and_warmup():
    df = add_indicators(make_prices(720))
    for col in ["SMA_50", "SMA_200", "RSI_14", "MACD", "MACD_signal", "MACD_hist",
                "BB_upper", "BB_middle", "BB_lower", "BB_pctB", "BB_width"]:
        assert col in df.columns
    assert df["SMA_200"].iloc[:199].isna().all() and df["SMA_200"].iloc[199:].notna().all()


def test_last_ma_cross():
    idx = pd.bdate_range("2026-01-01", periods=6)
    short = pd.Series([1, 1, 3, 3, 3, 3.0], index=idx)
    long = pd.Series([2, 2, 2, 2, 2, 2.0], index=idx)
    kind, date, days_ago = last_ma_cross(short, long)
    assert (kind, date, days_ago) == ("golden", idx[2], 3)
    assert last_ma_cross(pd.Series([np.nan] * 6, index=idx), long) is None
