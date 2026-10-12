"""Bonus report: required sections, embedded chart, ranking of top headlines, safe table cells."""

from equity_research.indicators import add_indicators
from equity_research.report import _money, top_headlines, write_report
from tests.conftest import make_prices


def final_result(signal_source="llm"):
    return {
        "summary": {"ticker": "DEMO", "company_name": "Demo | Corp", "currency": "USD", "as_of": "2026-10-09",
                    "current_price": 98.11, "high_52w": 102.46, "low_52w": 57.62, "pct_from_52w_high": -4.2,
                    "pe_ratio": None, "pe_source": "unavailable", "ytd_return_pct": 26.4, "ytd_base": "prior_year_close",
                    "momentum": {"label": "bullish", "score": 0.31, "components": {"rsi": 0.47}, "missing": []}},
        "info": {"sector": "Technology", "marketCap": 2.5e11},
        "headline_results": [
            {"headline": "Small news", "publisher": "A", "sentiment": "positive", "confidence": 0.3, "brief_reason": "r"},
            {"headline": "Big lawsuit", "publisher": "B", "sentiment": "negative", "confidence": 0.9, "brief_reason": "r"},
            {"headline": "Market flat", "publisher": "C", "sentiment": "neutral", "confidence": 0.9, "brief_reason": "r"},
            {"headline": "Record | revenue", "publisher": "D", "sentiment": "positive", "confidence": 0.8, "brief_reason": "r"},
        ],
        "headline_failures": [],
        "sentiment": {"score": 0.1, "label": "neutral", "counts": {"positive": 2, "neutral": 1, "negative": 1},
                      "agreement": 0.5, "coverage": 1.0, "n_valid": 4},
        "facts": ["Close 98.11 is 8.9% above SMA-50 (90.06)."], "conflicts": [],
        "signal": {"signal": "HOLD", "confidence": 0.5, "justification": "One. Two. Three.",
                   "key_drivers": ["a", "b"], "conflicting_evidence": []},
        "signal_source": signal_source, "warnings": ["Dropped 1 duplicate dates"], "errors": [],
    }


def test_top_headlines_are_the_strongest():
    ranked = [h["headline"] for h in top_headlines(final_result()["headline_results"])]
    assert ranked == ["Big lawsuit", "Record | revenue", "Small news"]     # neutral scores 0, so it is last


def test_money_format():
    assert (_money(2.5e11, "USD"), _money(4.4e12, "USD"), _money(98.1, "EUR"), _money(None, "USD")) == \
           ("$250.00B", "$4.40T", "EUR 98.10", "n/a")


def test_report_has_every_required_section(tmp_path):
    df = add_indicators(make_prices(400, tz=None)[["Open", "High", "Low", "Close", "Volume"]])
    paths = write_report(final_result("rule_based_fallback"), df, tmp_path)
    md = open(paths["markdown"], encoding="utf-8").read()
    html = open(paths["html"], encoding="utf-8").read()
    for section in ("Company snapshot", "Technical outlook", "News sentiment", "Recommendation", "Risk disclaimer"):
        assert section in md and section in html
    assert "![" in md and "chart.png" in md                                  # Markdown links the PNG
    assert "data:image/png;base64," in html                                  # HTML is self-contained
    assert "Record / revenue" in md and "Demo / Corp" in md                  # '|' cannot break a table
    assert "P/E (trailing) | n/a (not reported)" in md
    assert "recommendation is rule-based" in md and "Dropped 1 duplicate dates" in md
