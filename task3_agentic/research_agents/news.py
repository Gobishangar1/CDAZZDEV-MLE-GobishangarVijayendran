"""Recent headlines: yfinance -> Yahoo RSS -> Google News RSS, de-duplicated, relevant headlines first.

Adapted from Task 1 (task1_financial/equity_research/news.py), kept as a separate copy.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import feedparser
import pandas as pd
import requests

from research_agents import market
from research_agents.config import NEWS_TIMEOUT_S, YF_NEWS_COUNT

log = logging.getLogger(__name__)
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_OLDEST = datetime.min.replace(tzinfo=timezone.utc)
_SUFFIX = re.compile(
    r"\b(corporation|corp|incorporated|inc|limited|ltd|plc|holdings?|group|company|co|class [a-z])\b\.?",
    re.IGNORECASE)
_GENERIC = {"advanced", "american", "applied", "general", "global", "first", "international", "national",
            "united", "the", "new", "bank", "digital", "energy"}


def _parse_time(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        t = pd.Timestamp(value)
        return (t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).to_pydatetime()
    except Exception:
        return None


def _from_yfinance(item: dict) -> dict | None:
    c = item.get("content") or item
    title = (c.get("title") or "").strip()
    if not title:
        return None
    return {"title": title, "publisher": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
            "published_at": _parse_time(c.get("pubDate") or c.get("providerPublishTime")),
            "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link"), "source": "yfinance"}


def _http_get(url: str) -> bytes:
    resp = requests.get(url, timeout=NEWS_TIMEOUT_S, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


def _parse_rss(content: bytes | str, source: str) -> list[dict]:
    items = []
    for e in feedparser.parse(content).entries:
        title = (e.get("title") or "").strip()
        if not title:
            continue
        publisher = (e.get("source") or {}).get("title")
        if source == "google_news" and publisher is None and " - " in title:
            title, publisher = title.rsplit(" - ", 1)
        items.append({"title": title.strip(), "publisher": publisher, "published_at": _parse_time(e.get("published")),
                      "url": e.get("link"), "source": source})
    return items


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        key = re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+-\s+[^-]+$", "", it["title"]).lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def aliases(ticker: str, company_name: str | None) -> list[str]:
    names: set[str] = set()
    if company_name:
        short = " ".join(_SUFFIX.sub(" ", company_name).replace(",", " ").split())
        if len(short) >= 3:
            names.add(short)
            first = short.split()[0]
            if len(first) >= 4 and first.lower() not in _GENERIC:
                names.add(first)
    return sorted(names)


def is_relevant(title: str, ticker: str, names: list[str]) -> bool:
    if re.search(rf"(?<![A-Za-z0-9$]){re.escape(ticker)}(?![A-Za-z0-9])", title):
        return True
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", title, re.IGNORECASE) for n in names)


def providers(ticker: str, company_name: str | None):
    query = quote_plus(f"{company_name or ticker} stock")

    def yfinance_news() -> list[dict]:
        t, raw = market._get_ticker(ticker), []
        for tab in ("news", "all"):
            try:
                raw = t.get_news(count=YF_NEWS_COUNT, tab=tab) or []
            except Exception as exc:
                log.warning("yfinance_news_failed tab=%s error=%s", tab, exc)
            if raw:
                break
        return [h for h in map(_from_yfinance, raw) if h]

    return [("yfinance", yfinance_news),
            ("yahoo_rss", lambda: _parse_rss(_http_get(YAHOO_RSS.format(ticker=ticker)), "yahoo_rss")),
            ("google_news", lambda: _parse_rss(_http_get(GOOGLE_RSS.format(query=query)), "google_news"))]


def fetch_headlines(ticker: str, n: int) -> tuple[list[dict], list[str]]:
    """Up to n headlines about the company (newest first) + warnings. Never raises."""
    ticker = market.normalize_ticker(ticker)
    company = market.fetch_company_name(ticker)
    names = aliases(ticker, company)
    collected: list[dict] = []
    warnings: list[str] = []
    for name, get in providers(ticker, company):
        try:
            got = get()
            if not got:
                warnings.append(f"News source {name} returned no items")
            collected = _dedupe(collected + got)
        except Exception as exc:
            warnings.append(f"News source {name} failed: {type(exc).__name__}")
            continue
        if sum(is_relevant(h["title"], ticker, names) for h in collected) >= n:
            break
    for h in collected:
        h["relevant"] = is_relevant(h["title"], ticker, names)
        if not h.get("publisher") and h.get("url"):
            host = urlparse(h["url"]).netloc
            h["publisher"] = host[4:] if host.startswith("www.") else host
    newest = lambda xs: sorted(xs, key=lambda h: h["published_at"] or _OLDEST, reverse=True)  # noqa: E731
    relevant = newest([h for h in collected if h["relevant"]])[:n]
    fill = newest([h for h in collected if not h["relevant"]])[:max(n - len(relevant), 0)]
    if fill:
        warnings.append(f"Only {len(relevant)} headlines name {ticker}; added {len(fill)} general headlines")
    return newest(relevant + fill), warnings
