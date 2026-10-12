"""Shared offline fakes for the Task 3 tests: no Yahoo, no news sites, no DuckDuckGo, no Groq.

    FakeTicker     the yfinance methods the tools use (history, get_info, get_news)
    PolicyLLM      a scripted "agent model": it reads the conversation and picks the next tool from what it
                   has observed (so the loop, replanning and memory are tested without a real LLM)
    FakeJSONLLM    answers the JSON prompts (sentiment, clarification request, final report)
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tenacity import wait_none

from research_agents import config, market, news
from research_agents import tools as tools_module
from research_agents.trace import Tracer

TICKER = "NVDA"
COMPANY = "NVIDIA Corporation"
HEADLINES = [
    "NVIDIA beats estimates as data-center revenue hits a record",
    "NVIDIA faces new export ban on AI chips to China",
    "NVIDIA shares fall after antitrust probe report",
    "NVIDIA unveils next-generation GPU platform",
    "Analysts debate NVIDIA valuation after strong rally",
    "NVIDIA supplier warns of tight packaging capacity",
    "NVIDIA expands partnership with major cloud provider",
    "NVIDIA lawsuit over crypto disclosures moves forward",
    "NVIDIA stock rises as AI spending outlook improves",
    "NVIDIA CEO comments on demand at developer conference",
    "NVIDIA options traders price large post-earnings move",
    "NVIDIA market value tops peers in chip sector",
]


# -----------------------------------------------------------------------------
# Market data fakes
# -----------------------------------------------------------------------------
def make_prices(periods: int = 520, end: str = "2026-10-09", seed: int = 7) -> pd.DataFrame:
    """Synthetic daily OHLCV shaped like yfinance output (tz-aware index, extra columns)."""
    idx = pd.bdate_range(end=end, periods=periods, tz="America/New_York")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0008, 0.02, periods)))
    return pd.DataFrame({"Open": close * 0.995, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": 2_000_000.0, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)


def make_news(titles: list[str] = HEADLINES) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [{"content": {"title": t, "pubDate": (now - timedelta(hours=6 * i)).isoformat(),
                         "provider": {"displayName": "Wire"},
                         "canonicalUrl": {"url": f"https://example.com/{i}"}}} for i, t in enumerate(titles)]


class FakeTicker:
    def __init__(self, prices: pd.DataFrame | None = None, info: dict | None = None,
                 news_items: list[dict] | None = None, fail: Exception | None = None) -> None:
        self.prices = make_prices() if prices is None else prices
        self.info = {"longName": COMPANY} if info is None else info
        self.news_items = make_news() if news_items is None else news_items
        self.fail = fail
        self.history_calls = 0

    def history(self, period: str = "1y", **kwargs: Any) -> pd.DataFrame:
        self.history_calls += 1
        if self.fail is not None:
            raise self.fail
        return self.prices.copy()

    def get_info(self) -> dict:
        return self.info

    def get_news(self, count: int = 10, tab: str = "news") -> list[dict]:
        return self.news_items[:count]


def search_results(query: str, max_results: int) -> list[dict]:
    """A fake DuckDuckGo backend (same keys as ddgs: title, href, body)."""
    return [{"title": f"Analyst note {i}: {query}", "href": f"https://news.example.com/{i}",
             "body": "Analysts flag export controls and valuation as the main risks. " * 6}
            for i in range(max_results)]


def failing_search(query: str, max_results: int) -> list[dict]:
    raise RuntimeError("202 Ratelimit (simulated DuckDuckGo failure)")


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path) -> FakeTicker:
    """Every test: fake Yahoo, no HTTP, no real search, trace and memory files in a temp folder."""
    fake = FakeTicker()

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("network disabled in tests")

    monkeypatch.setattr(market, "_get_ticker", lambda t: fake)
    monkeypatch.setattr(news, "_http_get", blocked)
    monkeypatch.setattr(tools_module, "default_search", blocked)
    monkeypatch.setattr(config, "TRACE_FILE", tmp_path / "logs" / "agent_trace.jsonl")
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "memory")
    market._download.retry.wait = wait_none()            # keep the retries, skip the backoff waits
    market.clear_caches()
    yield fake
    market.clear_caches()


@pytest.fixture
def fake_ticker(offline: FakeTicker) -> FakeTicker:
    return offline


@pytest.fixture
def tracer(tmp_path) -> Tracer:
    return Tracer(path=tmp_path / "logs" / "agent_trace.jsonl", run_id="test", verbose=False)


# -----------------------------------------------------------------------------
# A scripted tool-calling model
# -----------------------------------------------------------------------------
class Observations:
    """What the model can 'see': the system prompt, the latest task and all tool results so far."""

    def __init__(self, messages: list) -> None:
        self.messages = messages
        self.system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        self.task_index = humans[-1] if humans else 0
        self.task = str(messages[self.task_index].content) if humans else ""
        self.results: list[tuple[str, str, dict]] = []          # (tool, status, payload) in order
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    payload = json.loads(m.content)
                except ValueError:
                    payload = {}
                status = (m.artifact or {}).get("status", "ok")
                self.results.append((m.name, status, payload))

    def called(self, tool: str) -> bool:
        return any(t == tool for t, _, _ in self.results)

    def ok(self, tool: str) -> dict | None:
        found = [p for t, s, p in self.results if t == tool and s == "ok"]
        return found[-1] if found else None

    def failed(self, tool: str) -> bool:
        return any(t == tool and s in ("error", "empty") for t, s, _ in self.results)


class _Bound:
    def __init__(self, parent: PolicyLLM, tools: list[str]) -> None:
        self.parent, self.tools = parent, tools

    def invoke(self, messages: list) -> AIMessage:
        return self.parent.respond(messages, self.tools)


class PolicyLLM:
    """policy(observations, tool_names or None) -> str (final answer) or (reasoning, tool, args) or AIMessage.
    tool_names is None when tool calls are disabled (tool_choice="none": the step budget is used)."""

    def __init__(self, policy: Callable[[Observations, list[str] | None], Any]) -> None:
        self.policy = policy
        self.calls = 0
        self.seen: list[list] = []
        self._ids = itertools.count(1)

    def bind_tools(self, tools: list, tool_choice: str | None = None) -> _Bound:
        return _Bound(self, None if tool_choice == "none" else [t.name for t in tools])

    def invoke(self, messages: list) -> AIMessage:
        return self.respond(messages, None)

    def respond(self, messages: list, tools: list[str] | None) -> AIMessage:
        self.calls += 1
        self.seen.append(list(messages))
        out = self.policy(Observations(messages), tools)
        if isinstance(out, AIMessage):
            return out
        if isinstance(out, str):
            return AIMessage(content=out)
        reasoning, tool, args = out
        return AIMessage(content=reasoning, tool_calls=[{"name": tool, "args": args, "id": f"call_{next(self._ids)}",
                                                         "type": "tool_call"}])


def research_policy(obs: Observations, tools: list[str] | None) -> Any:
    """A sensible analyst: every next step depends on what the previous results showed."""
    if tools is None:
        return "Final answer from the evidence gathered so far."
    vol = obs.ok("calculate_volatility")
    if "follow-up" in obs.task.lower() or obs.task.lower().startswith("what"):
        if vol:                                               # answer from memory, no tool call
            return (f"Earlier I computed {vol['window']}-day annualised volatility of "
                    f"{vol['annualised_vol_pct']}%, so a protective put should be sized for that.")
        if "calculate_volatility" in tools:
            return ("Reasoning: I have no volatility yet.", "calculate_volatility", {"ticker": TICKER, "window": 30})

    provided = re.findall(r"^- (.+)$", obs.task, flags=re.M)     # headlines sent by Agent B
    if provided and "llm_sentiment" in tools and not obs.called("llm_sentiment"):
        return ("Reasoning: Agent B sent headlines; I score them.", "llm_sentiment", {"headlines": provided[:15]})
    if "window=60" in obs.task and "calculate_volatility" in tools and not any(
            t == "calculate_volatility" and p.get("window") == 60 for t, _, p in obs.results):
        return ("Reasoning: the writer asked for the 60-day window.", "calculate_volatility",
                {"ticker": TICKER, "window": 60})
    if provided or "window=60" in obs.task:
        return "Sentiment and the 60-day volatility are in the tool results above."

    if "get_price_data" in tools and not obs.called("get_price_data"):
        return ("Reasoning: start with the price trend.", "get_price_data", {"ticker": TICKER, "period": "1y"})
    price = obs.ok("get_price_data")
    if "calculate_volatility" in tools and not obs.called("calculate_volatility"):
        rsi = (price or {}).get("indicators", {}).get("rsi_14") or 50
        window = 20 if rsi > 70 else 30                      # the argument depends on the observation
        return (f"Reasoning: RSI is {rsi:.0f}; I measure {window}-day volatility next.", "calculate_volatility",
                {"ticker": TICKER, "window": window})
    if "get_news" in tools and not obs.called("get_news"):
        return ("Reasoning: now the news flow.", "get_news", {"ticker": TICKER, "n": 10})
    if obs.failed("get_news") and "web_search" in tools and not obs.called("web_search"):
        return ("Reasoning: get_news returned nothing; following the suggestion I use web_search.", "web_search",
                {"query": f"{TICKER} stock news this week"})
    news_result = obs.ok("get_news")
    if news_result and "llm_sentiment" in tools and not obs.called("llm_sentiment"):
        titles = [h["title"] for h in news_result["headlines"]][:15]
        return ("Reasoning: I have headlines; I score their sentiment.", "llm_sentiment", {"headlines": titles})
    if "web_search" in tools and not obs.called("web_search"):
        return ("Reasoning: I still need outside analyst views.", "web_search",
                {"query": f"{TICKER} analyst commentary risks"})
    if obs.failed("web_search") and "get_news" in tools and obs.results[-1][0] == "web_search":
        return ("Reasoning: web_search failed; the suggestion says use get_news instead.", "get_news",
                {"ticker": TICKER, "n": 15})
    return "Final answer: trend, volatility, sentiment and analyst risks are covered by the tool results."


@pytest.fixture
def policy_llm() -> PolicyLLM:
    return PolicyLLM(research_policy)


# -----------------------------------------------------------------------------
# A JSON model for the structured outputs
# -----------------------------------------------------------------------------
REPORT = {
    "financial_health_summary": {
        "summary": "The stock is in a long-term uptrend with positive momentum, but volatility is elevated.",
        "key_metrics": ["Last close: 120.5", "30-day annualised volatility: 32.8%"],
        "market_sentiment": "News sentiment is mixed: strong demand against export-control headlines."},
    "top_three_risks": [
        {"title": "Export controls", "description": "New limits on AI chip sales to China could cut revenue.",
         "evidence": ["NVIDIA faces new export ban on AI chips to China"], "sources": ["get_news"],
         "severity": "high"},
        {"title": "Antitrust probe", "description": "A regulatory probe can weigh on the multiple for months.",
         "evidence": ["NVIDIA shares fall after antitrust probe report"], "sources": ["web_search"],
         "severity": "medium"},
        {"title": "Volatility shock", "description": "High realised volatility means large moves around events.",
         "evidence": ["1-day 95% VaR -3.1%"], "sources": ["calculate_volatility"], "severity": "medium"}],
    "hedge_strategy": {
        "strategy": "Protective put", "instrument": "3-month put option, 10% out of the money",
        "rationale": "Elevated volatility and event risk justify paying for downside protection now.",
        "data_points": ["30-day annualised volatility 32.8%"], "sizing_or_trigger": "Hedge 50% of the position"},
    "clarification_used": "Agent A scored the headlines as mixed and gave 60-day volatility, used for the put.",
    "sources": ["get_price_data", "calculate_volatility", "get_news", "web_search"],
}

CLARIFICATION = {"question": "Please score the sentiment of these headlines and give the 60-day volatility.",
                 "requested_data": ["llm_sentiment of the headlines", "calculate_volatility window=60"],
                 "headlines": [],
                 "reason": "The brief has no sentiment and the hedge horizon is about 90 days."}

POSITIVE = ("beats", "record", "rises", "unveils", "expands", "tops")
NEGATIVE = ("ban", "fall", "probe", "lawsuit", "warns")


def label_headlines(human: str) -> dict:
    titles = re.findall(r"^\d+\. (.+)$", human, flags=re.M)
    items = []
    for t in titles:
        low = t.lower()
        sentiment = ("positive" if any(w in low for w in POSITIVE) else
                     "negative" if any(w in low for w in NEGATIVE) else "neutral")
        items.append({"headline": t, "sentiment": sentiment, "confidence": 0.8, "brief_reason": "Test label."})
    return {"items": items}


def json_reply(system: str, human: str) -> str:
    if "label each news headline" in system:
        return json.dumps(label_headlines(human))
    if "review Agent A's data" in system:
        return json.dumps(CLARIFICATION)
    if "final equity research report" in system:
        return "Here is the report:\n" + json.dumps(REPORT)
    raise AssertionError(f"unexpected prompt: {system[:80]}")


class FakeJSONLLM:
    """reply(system, human, call_number) -> str. Records every prompt."""

    def __init__(self, reply: Callable[[str, str, int], str] | None = None) -> None:
        self.reply = reply or (lambda system, human, n: json_reply(system, human))
        self.calls = 0
        self.prompts: list[tuple[str, str]] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        system, human = str(messages[0].content), str(messages[-1].content)
        self.prompts.append((system, human))
        return AIMessage(content=self.reply(system, human, self.calls))


@pytest.fixture
def json_llm() -> FakeJSONLLM:
    return FakeJSONLLM()


@pytest.fixture
def sentiment_llm() -> FakeJSONLLM:
    return FakeJSONLLM()
