"""Shared test fakes. Every test runs offline: no Yahoo, no news sites, no Groq."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest fixtures: synthetic OHLCV, a fake yfinance
# Ticker and a scripted fake LLM', Date: 2026-10-11

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from tenacity import wait_none

from equity_research import market_data, news

END_DATE = "2026-10-09"


def make_prices(periods: int = 720, end: str = END_DATE, drift: float = 0.15, seed: int = 7,
                tz: str | None = "America/New_York") -> pd.DataFrame:
    """Synthetic daily OHLCV shaped like yfinance output (tz-aware index, extra columns)."""
    idx = pd.bdate_range(end=end, periods=periods, tz=tz)
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(drift, 1.5, periods))
    close = np.maximum(close, 5.0)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close,
                         "Volume": 1_000_000, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)


def make_news(n: int = 12) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [{"content": {"title": f"Demo Corp headline number {i}",
                         "pubDate": (now - timedelta(hours=10 * i)).isoformat(),
                         "provider": {"displayName": "Wire"},
                         "canonicalUrl": {"url": f"https://example.com/{i}"}}} for i in range(n)]


class FakeTicker:
    """Mimics the yfinance methods we use. `history` may be a DataFrame, an Exception, or a list
    of them (consumed one per call). `valid` controls the 5-day probe."""

    def __init__(self, history=None, info=None, news=None, valid=True):
        self._history = history if isinstance(history, list) else [history]
        self._info, self._news, self.valid = info, news if news is not None else [], valid
        self.history_calls = 0

    def history(self, period=None, **kwargs):
        if period is not None:                         # the probe
            return make_prices(5) if self.valid else pd.DataFrame()
        self.history_calls += 1
        item = self._history.pop(0) if len(self._history) > 1 else self._history[0]
        if isinstance(item, Exception):
            raise item
        return item.copy()

    def get_info(self):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info or {}

    def get_news(self, count=10, tab="news"):
        if isinstance(self._news, dict):           # {"news": [...], "all": [...]} to fake each tab
            return self._news.get(tab, [])[:count]
        return self._news[:count]


class FakeLLM:
    """Returns scripted replies. `reply` is a function (human_message_text, call_number) -> str."""

    def __init__(self, reply):
        self.reply, self.calls, self.prompts = reply, 0, []

    def invoke(self, prompt_value):
        self.calls += 1
        human = prompt_value.to_messages()[-1].content
        self.prompts.append(human)
        return AIMessage(content=self.reply(human, self.calls))


@pytest.fixture(autouse=True)
def no_retry_wait():
    """Keep the retry logic, skip the 2 s / 4 s backoff waits."""
    for fn in (market_data._download_history, market_data._download_info):
        fn.retry.wait = wait_none()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests are offline: any real HTTP call for RSS news fails immediately."""
    def blocked(url):
        raise ConnectionError(f"network disabled in tests: {url}")
    monkeypatch.setattr(news, "_http_get", blocked)


@pytest.fixture
def use_ticker(monkeypatch):
    """use_ticker(FakeTicker(...)) or use_ticker({"NVDA": FakeTicker(...)})."""
    def _use(fake):
        monkeypatch.setattr(market_data, "_get_ticker",
                            (lambda t: fake[t]) if isinstance(fake, dict) else (lambda t: fake))
        return fake
    return _use
