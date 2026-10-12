"""News retrieval (10 marks): both yfinance formats, RSS parsing, de-duplication, provider chain."""

from datetime import datetime, timezone

import pytest

from equity_research import news
from tests.conftest import FakeTicker, make_news

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Nvidia Beats Estimates! - Reuters</title><link>https://g1</link><pubDate>Thu, 09 Oct 2026 14:00:00 GMT</pubDate></item>
<item><title>Q3 results - beat - CNBC</title><link>https://g2</link><pubDate>Thu, 09 Oct 2026 12:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_yfinance_new_and_old_formats():
    new = {"id": "1", "content": {"title": "Nvidia beats estimates", "pubDate": "2026-10-09T13:00:00Z",
                                  "provider": {"displayName": "Reuters"}, "canonicalUrl": {"url": "https://x"}}}
    old = {"title": "Nvidia shares slip", "publisher": "Bloomberg", "providerPublishTime": 1760000000,
           "link": "https://y"}
    a, b = news.from_yfinance_item(new), news.from_yfinance_item(old)
    assert (a["title"], a["publisher"], a["url"]) == ("Nvidia beats estimates", "Reuters", "https://x")
    assert a["published_at"] == datetime(2026, 10, 9, 13, tzinfo=timezone.utc)
    assert (b["publisher"], b["url"]) == ("Bloomberg", "https://y") and b["published_at"].tzinfo is not None
    assert news.from_yfinance_item({"content": {"title": ""}}) is None


def test_google_rss_splits_publisher_at_last_dash():
    items = news.parse_rss(RSS, "google_news")
    assert [(i["title"], i["publisher"]) for i in items] == [
        ("Nvidia Beats Estimates!", "Reuters"), ("Q3 results - beat", "CNBC")]


def test_dedupe_ignores_case_punctuation_and_publisher_suffix():
    items = [{"title": "Nvidia beats estimates"}, {"title": "Nvidia Beats Estimates! - Reuters"},
             {"title": "Chip stocks rally"}]
    assert [i["title"] for i in news.dedupe(items)] == ["Nvidia beats estimates", "Chip stocks rally"]


def test_chain_stops_when_enough(monkeypatch):
    calls = []

    def provider(name, n):
        def get():
            calls.append(name)
            return [{"title": f"NVDA {name} story {i}", "publisher": "P", "published_at": None,
                     "url": None, "source": name} for i in range(n)]
        return name, get

    monkeypatch.setattr(news, "news_providers", lambda t, c=None: [provider("a", 6), provider("b", 6), provider("c", 6)])
    items, warnings = news.fetch_news("NVDA")
    assert calls == ["a", "b"] and len(items) == 12 and warnings == []


def test_chain_survives_failing_source_and_warns(monkeypatch):
    def broken():
        raise TimeoutError("slow site")

    monkeypatch.setattr(news, "news_providers", lambda t, c=None: [
        ("yfinance", broken),
        ("google_news", lambda: news.parse_rss(RSS, "google_news"))])
    items, warnings = news.fetch_news("NVDA")
    assert len(items) == 2
    assert any("yfinance failed" in w for w in warnings) and any("Only 2" in w for w in warnings)


def test_yfinance_provider_newest_first_and_capped(use_ticker):
    use_ticker(FakeTicker(news=make_news(20)))
    items, warnings = news.fetch_news("DEMO", "Demo Corp")
    assert len(items) == 15 and warnings == []
    times = [i["published_at"] for i in items]
    assert times == sorted(times, reverse=True)


def test_company_aliases():
    assert news.company_aliases("nvda", "NVIDIA Corporation") == ("NVDA", ["NVIDIA"])
    assert news.company_aliases("META", "Meta Platforms, Inc.") == ("META", ["Meta", "Meta Platforms"])
    assert news.company_aliases("AMD", "Advanced Micro Devices, Inc.") == ("AMD", ["Advanced Micro Devices"])


@pytest.mark.parametrize("title,expected", [
    ("Nvidia Weighs Buying Reflection AI", True),
    ("Cerebras Is About as Big as Nvidia's Data Center Business", True),
    ("NVDA slips 2% after hours", True),
    ("Is It Too Late to Buy Palantir Stock?", False),          # off-topic headlines from the real run
    ("Got $2,000? 2 Natural Gas Pipeline Stocks to Buy Before Winter.", False),
    ("nvda in lower case is not the ticker", False),           # tickers match in exact case only
])
def test_is_relevant(title, expected):
    assert news.is_relevant(title, "NVDA", ["NVIDIA"]) is expected


def test_relevant_first_general_only_to_fill(monkeypatch):
    def items(titles):
        return [{"title": t, "publisher": None, "published_at": None, "url": f"https://www.fool.com/{i}",
                 "source": "yahoo_rss"} for i, t in enumerate(titles)]
    relevant = [f"Nvidia story {i}" for i in range(7)]
    general = [f"Bitcoin story {i}" for i in range(9)]
    monkeypatch.setattr(news, "news_providers", lambda t, c=None: [("yahoo_rss", lambda: items(relevant + general))])
    chosen, warnings = news.fetch_news("NVDA", "NVIDIA Corporation")
    assert len(chosen) == 10 and sum(h["relevant"] for h in chosen) == 7       # 7 relevant + 3 fill
    assert any("added 3 general market headlines" in w for w in warnings)
    assert chosen[0]["publisher"] == "fool.com"                                 # domain used as publisher


def test_yfinance_empty_tab_falls_back_to_all_tab(use_ticker):
    use_ticker(FakeTicker(news={"news": [], "all": make_news(12)}))
    items, warnings = news.fetch_news("DEMO", "Demo Corp")
    assert len(items) == 12 and warnings == []


def test_empty_source_is_reported_not_silent(use_ticker):
    use_ticker(FakeTicker(news={"news": [], "all": []}))
    items, warnings = news.fetch_news("NVDA", "NVIDIA Corporation")
    assert "News source yfinance returned no items" in warnings
    assert any("yahoo_rss failed" in w for w in warnings)        # network is blocked in tests
