"""Summary dictionary (10 marks) and the rule-based momentum signal."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for P/E fallbacks, YTD return,
# momentum scenarios and the summary fields', Date: 2026-10-11

import pandas as pd
import pytest

from equity_research.analysis import build_summary, indicator_snapshot, momentum_signal, pe_ratio, ytd_return
from equity_research.indicators import add_indicators
from equity_research.schemas import MarketSummary
from tests.conftest import make_prices


@pytest.mark.parametrize("info,expected", [
    ({"trailingPE": 35.2}, (35.2, "trailingPE")),
    ({"trailingPE": None, "trailingEps": 3.0}, (50.0, "computed_from_eps")),
    ({"trailingPE": "Infinity", "trailingEps": -2.0}, (None, "unavailable")),   # loss-making
    ({"trailingPE": -12.0}, (None, "unavailable")),
    ({}, (None, "unavailable")),
])
def test_pe_ratio_levels(info, expected):
    assert pe_ratio(150.0, info) == expected


def test_ytd_uses_prior_year_close():
    idx = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-06-01"])
    close = pd.Series([90.0, 100.0, 105.0, 120.0], index=idx)
    assert ytd_return(close) == (20.0, "prior_year_close")      # 120 / 100 - 1, not 120 / 105 - 1


def test_ytd_for_stock_listed_this_year():
    idx = pd.to_datetime(["2026-03-02", "2026-06-01"])
    assert ytd_return(pd.Series([50.0, 60.0], index=idx)) == (20.0, "first_close_this_year")


BASE = dict(close=190, sma_50=180, sma_200=150, rsi=64, bb_pctb=0.8, macd_hist_last5=[0.5, 0.7, 0.9, 1.1, 1.3])


@pytest.mark.parametrize("snap,label", [
    (BASE, "strong_bullish"),
    ({**BASE, "close": 176, "rsi": 48, "bb_pctb": 0.35, "macd_hist_last5": [1.3, 1.1, 0.8, 0.5, 0.2]}, "neutral"),
    (dict(close=120, sma_50=130, sma_200=150, rsi=33, bb_pctb=0.1,
          macd_hist_last5=[-0.5, -0.8, -1.0, -1.2, -1.4]), "strong_bearish"),
])
def test_momentum_scenarios(snap, label):
    assert momentum_signal(snap).label == label


def test_momentum_fading_macd_is_halved():
    fading = momentum_signal({**BASE, "macd_hist_last5": [1.3, 1.1, 0.8, 0.5, 0.2]})
    assert fading.components["macd"] == 0.5


def test_momentum_missing_indicator_is_renormalised():
    m = momentum_signal({**BASE, "sma_200": None})
    assert "long_trend" in m.missing and -1 <= m.score <= 1 and m.label == "strong_bullish"
    assert momentum_signal({"close": 10}).label == "neutral"                 # nothing to vote


def test_summary_has_all_required_fields():
    df = add_indicators(make_prices(720, tz=None)[["Open", "High", "Low", "Close", "Volume"]])
    snap = indicator_snapshot(df)
    s = build_summary("DEMO", df, {"longName": "Demo Corp", "currency": "USD", "trailingPE": 31.5},
                      momentum_signal(snap))
    assert isinstance(s, MarketSummary)
    d = s.model_dump()
    for key in ["current_price", "high_52w", "low_52w", "pe_ratio", "ytd_return_pct", "momentum"]:
        assert d[key] is not None
    last_year = df.iloc[-252:]
    assert d["high_52w"] == round(last_year["High"].max(), 2)      # from High, not Close
    assert d["low_52w"] == round(last_year["Low"].min(), 2)
    assert d["low_52w"] <= d["current_price"] <= d["high_52w"]
    assert snap["cross"] is None or snap["cross"]["kind"] in {"golden", "death"}
