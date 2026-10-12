"""OHLCV fetch (10 marks) and robustness: computed dates, cleaning, retries, fail-soft info."""

import numpy as np
import pandas as pd
import pytest

from equity_research import market_data as md
from equity_research.config import HISTORY_YEARS, WARMUP_CALENDAR_DAYS
from tests.conftest import FakeTicker, make_prices

TODAY = pd.Timestamp("2026-10-09")    # fixed date in TESTS only, so results repeat


@pytest.mark.parametrize("raw,expected", [(" nvda ", "NVDA"), ("brk-b", "BRK-B"), ("^gspc", "^GSPC")])
def test_normalize_ticker_ok(raw, expected):
    assert md.normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["", "NV DA", "TOOLONGTICKER", "NVDA;DROP", 123, None])
def test_normalize_ticker_rejects(raw):
    with pytest.raises(md.InvalidTickerError):
        md.normalize_ticker(raw)


def test_probe_never_raises(use_ticker):
    use_ticker(FakeTicker(valid=True))
    assert md.probe_ticker("NVDA") is True
    use_ticker(FakeTicker(valid=False))
    assert md.probe_ticker("XXXXZZ") is False


def test_start_dates_are_computed_not_hardcoded():
    fetch_start, display_start = md.compute_start_dates(TODAY)
    assert display_start == TODAY - pd.DateOffset(years=HISTORY_YEARS)
    assert (display_start - fetch_start).days == WARMUP_CALENDAR_DAYS


def test_fetch_prices_cleans_data(use_ticker):
    raw = make_prices(720)
    raw.iloc[10, raw.columns.get_loc("Close")] = np.nan            # a null close
    raw = pd.concat([raw, raw.iloc[[20]]]).sample(frac=1, random_state=1)   # duplicate date, shuffled
    use_ticker(FakeTicker(history=raw))

    df, warnings = md.fetch_prices("nvda", today=TODAY)

    assert list(df.columns) == md.OHLCV_COLUMNS          # Dividends / Stock Splits removed
    assert df.index.tz is None and df.index.is_monotonic_increasing
    assert not df.index.duplicated().any() and df["Close"].notna().all()
    assert len(df) == 720 - 1                            # 1 null dropped, duplicate collapsed
    assert any("missing close" in w for w in warnings) and any("duplicate" in w for w in warnings)


def test_fetch_retries_then_succeeds(use_ticker):
    fake = use_ticker(FakeTicker(history=[ConnectionError("blip"), pd.DataFrame(), make_prices(720)]))
    df, _ = md.fetch_prices("NVDA", today=TODAY)
    assert fake.history_calls == 3 and not df.empty


def test_fetch_gives_up_with_clear_error(use_ticker):
    use_ticker(FakeTicker(history=[ConnectionError("down")] * 3))
    with pytest.raises(md.MarketDataError, match="after 3 attempts"):
        md.fetch_prices("NVDA", today=TODAY)


def test_malformed_ticker_is_not_fetched(use_ticker):
    fake = use_ticker(FakeTicker(history=make_prices(720)))
    with pytest.raises(md.InvalidTickerError):
        md.fetch_prices("bad ticker!", today=TODAY)
    assert fake.history_calls == 0


def test_young_stock_warns(use_ticker):
    use_ticker(FakeTicker(history=make_prices(300)))
    _, warnings = md.fetch_prices("NEWCO", today=TODAY)
    assert any("young stock" in w for w in warnings)


def test_check_prices_flags_but_never_changes():
    df = make_prices(50, tz=None)[md.OHLCV_COLUMNS]
    df.iloc[30, df.columns.get_loc("Close")] *= 3        # +200% day
    df.iloc[5, df.columns.get_loc("Low")] = df.iloc[5]["High"] + 5
    before = df.copy()
    warnings = md.check_prices(df)
    assert any("daily moves above" in w for w in warnings)
    assert any("High below Low" in w for w in warnings)
    pd.testing.assert_frame_equal(df, before)


def test_fetch_info_keeps_needed_keys(use_ticker):
    use_ticker(FakeTicker(info={"longName": "NVIDIA Corporation", "trailingPE": 35.0, "junk": 1}))
    info = md.fetch_info("NVDA")
    assert info["longName"] == "NVIDIA Corporation" and info["trailingPE"] == 35.0 and "junk" not in info


def test_fetch_info_fails_soft(use_ticker):
    use_ticker(FakeTicker(info=ConnectionError("down")))
    assert md.fetch_info("NVDA") == {}
