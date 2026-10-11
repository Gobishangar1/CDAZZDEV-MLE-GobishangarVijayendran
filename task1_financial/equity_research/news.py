"""Recent news headlines from free sources (Task 1A).

Provider chain: stop as soon as NEWS_MIN unique headlines are collected.
    1. yfinance Ticker.news     (reads both the old flat format and the 2024+ nested "content" format)
    2. Yahoo Finance RSS feed
    3. Google News RSS search

Every headline becomes the same dict:
    {"title", "publisher", "published_at" (UTC datetime or None), "url", "source"}
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'News provider chain (yfinance news in old and new
# formats, Yahoo RSS, Google News RSS) with timeouts, UTC timestamps and title de-duplication',
# Date: 2026-10-11

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import quote_plus, urlparse

import feedparser
import pandas as pd
import requests

from equity_research import market_data
from equity_research.config import NEWS_MAX, NEWS_MIN, NEWS_TIMEOUT_S, YF_NEWS_COUNT, YF_NEWS_TABS

log = logging.getLogger(__name__)

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_OLDEST = datetime.min.replace(tzinfo=timezone.utc)


def parse_time(value: object) -> datetime | None:
    """Unix seconds or a date string -> timezone-aware UTC datetime. None if unreadable."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        t = pd.Timestamp(value)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.to_pydatetime()
    except Exception:
        return None


def from_yfinance_item(item: dict) -> dict | None:
    """One yfinance news item -> our dict. Works for the old and the new (nested 'content') format."""
    c = item.get("content") or item
    title = (c.get("title") or "").strip()
    if not title:
        return None
    return {
        "title": title,
        "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
        "published_at": parse_time(c.get("pubDate") or c.get("providerPublishTime")),
        "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link"),
        "source": "yfinance",
    }


def _http_get(url: str) -> bytes:
    """Download with a timeout (feedparser.parse(url) has none) and a browser-like User-Agent."""
    resp = requests.get(url, timeout=NEWS_TIMEOUT_S, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


def parse_rss(content: bytes | str, source: str) -> list[dict]:
    items = []
    for e in feedparser.parse(content).entries:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        publisher = (e.get("source") or {}).get("title")
        if source == "google_news" and publisher is None and " - " in title:
            title, publisher = title.rsplit(" - ", 1)       # "Headline - Reuters"
        items.append({"title": title.strip(), "publisher": publisher,
                      "published_at": parse_time(e.get("published")),
                      "url": e.get("link"), "source": source})
    return items


def normalize_title(title: str) -> str:
    """Lowercase, drop a trailing ' - Publisher', drop punctuation. Used to find duplicates."""
    title = re.sub(r"\s+-\s+[^-]+$", "", title)
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = normalize_title(it["title"])
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def news_providers(ticker: str, company_name: str | None = None) -> list[tuple[str, Callable[[], list[dict]]]]:
    """(name, function) pairs, called in order only when still needed."""
    query = quote_plus(f"{company_name or ticker} stock")

    def yfinance_news() -> list[dict]:
        """yfinance's get_news returns [] (no exception) when Yahoo sends a bad response, so we ask
        for more items and try a second tab before giving up on this source."""
        t = market_data._get_ticker(ticker)
        raw: list = []
        for tab in YF_NEWS_TABS:
            try:
                raw = t.get_news(count=YF_NEWS_COUNT, tab=tab) or []
            except Exception as exc:
                log.warning("yfinance_news_failed tab=%s error=%s", tab, exc)
                raw = []
            if raw:
                break
            log.warning("yfinance_news_empty ticker=%s tab=%s", ticker, tab)
        return [h for h in map(from_yfinance_item, raw) if h]

    return [
        ("yfinance", yfinance_news),
        ("yahoo_rss", lambda: parse_rss(_http_get(YAHOO_RSS.format(ticker=ticker)), "yahoo_rss")),
        ("google_news", lambda: parse_rss(_http_get(GOOGLE_RSS.format(query=query)), "google_news")),
    ]


_COMPANY_SUFFIX = re.compile(r"\b(corporation|corp|incorporated|inc|limited|ltd|plc|holdings?|group|company|co|"
                             r"class [a-z])\b\.?", re.IGNORECASE)


# First words too generic to identify a company on their own ("Advanced Micro Devices" -> not "Advanced").
_GENERIC_FIRST_WORDS = {"advanced", "american", "applied", "general", "global", "first", "international",
                        "national", "united", "the", "new", "bank", "digital", "energy"}


def company_aliases(ticker: str, company_name: str | None) -> tuple[str, list[str]]:
    """(ticker, names) used to decide if a headline is about this company.

    "NVIDIA Corporation" -> ["NVIDIA"];  "Meta Platforms, Inc." -> ["Meta", "Meta Platforms"]
    """
    names: set[str] = set()
    if company_name:
        short = " ".join(_COMPANY_SUFFIX.sub(" ", company_name).replace(",", " ").split())
        if len(short) >= 3:
            names.add(short)
            first = short.split()[0]
            if len(first) >= 4 and first.lower() not in _GENERIC_FIRST_WORDS:
                names.add(first)        # headlines usually say "Meta", not "Meta Platforms"
    return ticker.upper(), sorted(names)


def is_relevant(title: str, ticker: str, names: list[str]) -> bool:
    """True if the title names the ticker (exact case, as a whole word) or the company (any case)."""
    if re.search(rf"(?<![A-Za-z0-9$]){re.escape(ticker)}(?![A-Za-z0-9])", title):
        return True
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", title, re.IGNORECASE) for n in names)


def _domain(url: str | None) -> str | None:
    host = urlparse(url).netloc if url else ""
    return host[4:] if host.startswith("www.") else (host or None)


def _newest_first(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda h: h["published_at"] or _OLDEST, reverse=True)


def fetch_news(ticker: str, company_name: str | None = None) -> tuple[list[dict], list[str]]:
    """Up to NEWS_MAX unique headlines ABOUT this company, newest first, plus warnings. Never raises.

    Free feeds mix in general market stories (Bitcoin, other stocks). Headlines that name the
    ticker or the company come first; general headlines are added only to reach NEWS_MIN, and
    that is reported as a warning. Each headline carries "relevant": True/False.
    """
    ticker, names = company_aliases(ticker, company_name)
    collected: list[dict] = []
    warnings: list[str] = []
    for name, get_items in news_providers(ticker, company_name):
        try:
            new_items = get_items()
            if not new_items:                       # an empty source is reported, never silent
                warnings.append(f"News source {name} returned no items")
            collected = dedupe(collected + new_items)
            n_relevant = sum(is_relevant(h["title"], ticker, names) for h in collected)
            log.info("news_provider name=%s items=%d unique_total=%d relevant_total=%d",
                     name, len(new_items), len(collected), n_relevant)
        except Exception as exc:                    # one bad source must not stop the run
            warnings.append(f"News source {name} failed: {type(exc).__name__}")
            log.warning("news_provider_failed name=%s error=%s", name, exc)
            continue
        if n_relevant >= NEWS_MIN:
            break

    for h in collected:
        h["relevant"] = is_relevant(h["title"], ticker, names)
        h["publisher"] = h.get("publisher") or _domain(h.get("url"))   # RSS items often have no source
    relevant = _newest_first([h for h in collected if h["relevant"]])[:NEWS_MAX]
    general = _newest_first([h for h in collected if not h["relevant"]])
    fill = general[:max(NEWS_MIN - len(relevant), 0)]
    if fill:
        warnings.append(f"Only {len(relevant)} headlines name {ticker}/{', '.join(names) or ticker}; "
                        f"added {len(fill)} general market headlines to reach {NEWS_MIN}")
    chosen = _newest_first(relevant + fill)
    if len(chosen) < NEWS_MIN:
        warnings.append(f"Only {len(chosen)} unique headlines found (target {NEWS_MIN})")
    return chosen, warnings
