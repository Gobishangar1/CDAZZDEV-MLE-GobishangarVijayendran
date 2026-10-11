"""The whole LangGraph workflow, offline: fake Yahoo, fake news, fake LLMs.

Scenarios: happy path with a signal retry, LLM always broken (fallback), bad tickers,
no news at all, report files written."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'End-to-end pytest for the LangGraph pipeline with
# fakes: retry loop, fallback, bad ticker, no news, report output', Date: 2026-10-11

import json
import re

import pytest

from equity_research import news
from equity_research.graph import build_graph, run_pipeline
from tests.conftest import FakeLLM, FakeTicker, make_news, make_prices


def headline_reply(human, n):
    title = human.split("Headline: ")[1].split("\n")[0]
    if title.endswith("number 3") and "previous reply was invalid" not in human:
        return "not json"                                   # repaired on the 2nd try
    if title.endswith("number 7"):
        return '{"sentiment": "bullish"}'                    # fails twice -> failure record
    label = ["positive", "neutral", "negative"][int(title.split()[-1]) % 3]
    return json.dumps({"headline": title, "sentiment": label, "confidence": 0.7, "brief_reason": "test reason"})


def good_signal(human):
    rsi = re.search(r"RSI-14 is ([\d.]+)", human).group(1)
    has_conflicts = "CONFLICTS:\n- none" not in human
    return json.dumps({
        "signal": "HOLD", "confidence": 0.5,
        "justification": (f"The SMA trend and the MACD histogram are read together, because trend alone is not "
                          f"enough. RSI-14 at {rsi} confirms the momentum picture while the Bollinger bands show "
                          f"where price sits. News sentiment is weighed as well, but it does not override the "
                          f"technical picture."),
        "key_drivers": ["SMA trend with MACD", "RSI with Bollinger"],
        "conflicting_evidence": ["each conflict was weighed"] if has_conflicts else []})


def signal_bad_then_good(human, n):
    if n == 1:
        return json.dumps({"signal": "BUY", "confidence": 0.95, "justification": "RSI is 99. MACD is 5. Price is 1.",
                           "key_drivers": ["a", "b"], "conflicting_evidence": []})
    return good_signal(human)


@pytest.fixture
def market(use_ticker):
    return use_ticker({"DEMO": FakeTicker(history=make_prices(720), news=make_news(12),
                                          info={"longName": "Demo Corp", "currency": "USD", "trailingPE": 31.5,
                                                "sector": "Technology", "marketCap": 2.5e11}),
                       "XXXXZZ": FakeTicker(valid=False)})


def test_happy_path_with_one_signal_retry(market, tmp_path):
    fast, strong = FakeLLM(headline_reply), FakeLLM(signal_bad_then_good)
    r = run_pipeline("demo", fast, strong, output_dir=str(tmp_path), verbose=False)

    assert r["status"] == "ok" and r["summary"]["ticker"] == "DEMO"
    assert len(r["headlines"]) == 12 and len(r["headline_results"]) == 11 and len(r["headline_failures"]) == 1
    assert fast.calls == 14                                   # 12 + 1 repair + 1 extra for the failed one
    assert r["sentiment"]["n_valid"] == 11 and r["sentiment"]["coverage"] == 0.92
    assert [len(h["problems"]) > 0 for h in r["signal_history"]] == [True, False]
    assert r["signal"]["signal"] == "HOLD" and r["signal_source"] == "llm"
    assert "Previous answer failed" not in strong.prompts[0] and "failed these checks" in strong.prompts[1]
    for key in ("chart", "markdown", "html", "run_json"):
        assert key in r["report_paths"]
    html = open(r["report_paths"]["html"], encoding="utf-8").read()
    assert "data:image/png;base64," in html and "Risk disclaimer" in html and "Demo Corp" in html
    saved = json.load(open(r["report_paths"]["run_json"], encoding="utf-8"))
    assert saved["signal"]["signal"] == "HOLD"

    # Trace: one event per LLM call, failed validations counted, file written next to run.json
    t = r["trace_summary"]
    assert t["llm_calls"] == fast.calls + strong.calls == 16
    assert t["validation_failures"] == 2 + 1 + 1         # headline 7 twice, headline 3 once, signal attempt 1
    events = [json.loads(line) for line in open(r["report_paths"]["trace"], encoding="utf-8")]
    kinds = {e["kind"] for e in events}
    assert kinds == {"node", "llm_call", "validation"}
    assert sum(e["kind"] == "node" and e["node"] == "classify_headline" for e in events) == 12
    assert all(len(e.get("output_preview", "")) <= 200 for e in events)


def test_llm_always_broken_uses_fallback(market):
    r = run_pipeline("DEMO", FakeLLM(headline_reply), FakeLLM(lambda h, n: "still not json"),
                     output_dir=None, verbose=False)
    assert r["status"] == "ok" and r["signal_source"] == "rule_based_fallback"
    assert len(r["signal_history"]) == 3 and r["signal"]["signal"] in {"BUY", "HOLD", "SELL"}


@pytest.mark.parametrize("ticker,message", [("XXXXZZ", "unknown or delisted"), ("bad ticker!", "not a valid ticker")])
def test_bad_ticker_clean_error_and_no_llm_calls(market, ticker, message):
    fast, strong = FakeLLM(headline_reply), FakeLLM(good_signal)
    r = run_pipeline(ticker, fast, strong, output_dir=None, verbose=False)
    assert r["status"] == "error" and message in r["message"]
    assert fast.calls == 0 and strong.calls == 0


def test_no_news_still_produces_a_signal(market, monkeypatch):
    monkeypatch.setattr(news, "news_providers", lambda t, c=None: [("yfinance", lambda: [])])
    r = run_pipeline("DEMO", FakeLLM(headline_reply), FakeLLM(lambda h, n: good_signal(h)),
                     output_dir=None, verbose=False)
    assert r["status"] == "ok" and r["sentiment"] is None
    assert any("Only 0 unique headlines" in w for w in r["warnings"])
    assert "News sentiment is unavailable." in r["facts"]


def test_price_fetch_failure_stops_cleanly(use_ticker):
    use_ticker(FakeTicker(history=[ConnectionError("down")] * 3, news=make_news(12)))
    fast = FakeLLM(headline_reply)
    r = run_pipeline("DEMO", fast, FakeLLM(good_signal), output_dir=None, verbose=False)
    assert r["status"] == "error" and "Could not fetch prices" in r["message"]
    assert fast.calls == 0                                     # no headline LLM calls once the run is stopping


def test_graph_compiles_and_draws():
    mermaid = build_graph(None, None).get_graph().draw_mermaid()
    for node in ("validate_input", "classify_headline", "build_context", "generate_signal", "rule_fallback"):
        assert node in mermaid
