"""Signal reasoning (15 marks) and validation (10 marks): facts, conflicts, the semantic validator,
one LLM attempt, and the rule-based fallback."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for facts/conflicts, the signal
# validator (invented numbers, echo, confidence, unicode dashes) and the fallback', Date: 2026-10-11

import json

import pytest
from pydantic import ValidationError

from equity_research.schemas import TradeSignal, split_sentences
from equity_research.signal_context import build_facts, find_conflicts
from equity_research.trade_signal import (
    attempt_signal,
    numbers_in,
    rule_based_fallback,
    signal_prompt_values,
    validate_signal,
)
from tests.conftest import FakeLLM

SNAP = dict(close=176.0, sma_50=180.0, sma_200=150.0, rsi=48.0, rsi_10d_ago=61.0, macd=1.2, macd_signal=1.0,
            bb_pctb=0.35, macd_hist_last5=[1.3, 1.1, 0.8, 0.5, 0.2],
            cross={"kind": "golden", "date": "2026-06-01", "days_ago": 84})
SUMMARY = {"pct_from_52w_high": -9.3, "high_52w": 194.0, "low_52w": 120.0, "ytd_return_pct": 21.4, "pe_ratio": 45.2}
SENT = {"score": 0.31, "label": "positive", "n_valid": 12, "agreement": 0.58, "coverage": 0.92}
MOM = {"label": "neutral", "score": 0.125}
FACTS = build_facts(SNAP, SUMMARY, SENT)
CONFLICTS = find_conflicts(SNAP, MOM, SENT)

GOOD = {
    "signal": "HOLD", "confidence": 0.55,
    "justification": (
        "The long-term uptrend is intact, since SMA-50 remains above SMA-200 and the golden cross is 84 "
        "trading days old. However, price has slipped 2.2% below SMA-50 while the MACD histogram is "
        "shrinking, which signals fading momentum. RSI falling from 61.0 to 48.0 confirms that buying "
        "strength is draining, even though it is not oversold. Positive news sentiment of +0.31 argues "
        "against selling, but it does not override the weakening momentum, so HOLD is the balanced call."),
    "key_drivers": ["Fading MACD with price below SMA-50", "Intact SMA-50/SMA-200 uptrend with positive news"],
    "conflicting_evidence": ["Long-term uptrend argues against SELL, but short-term momentum is fading"],
}
ECHO = {"signal": "HOLD", "confidence": 0.95,
        "justification": "The RSI is 48. The MACD histogram is 0.2. The price is 176 and SMA-50 is 185.",
        "key_drivers": ["RSI", "MACD"], "conflicting_evidence": []}


def test_facts_state_relationships_not_raw_values():
    text = " ".join(FACTS)
    assert "2.2% below SMA-50" in text and "long-term trend is up" in text
    assert "golden cross 84 trading days ago" in text and "momentum fading" in text
    assert "it was 61.0 ten trading days ago" in text and "(positive)" in text


def test_conflicts_found():
    assert CONFLICTS == ["Long-term trend is up, but price has fallen below SMA-50 (short-term weakness).",
                         "Trend is up, but MACD momentum is fading or negative."]
    assert find_conflicts({**SNAP, "rsi": 75, "close": 190, "macd_hist_last5": [1, 2, 3, 4, 5]},
                          {"score": 0.6}, None) == ["Momentum is bullish, but RSI is overbought (risk of a pullback)."]


def test_sentence_count_ignores_decimals():
    assert len(split_sentences("Price fell 2.2% today. RSI is 48.0 now. Done.")) == 3


def test_numbers_ignore_dates_and_handle_unicode_and_commas():
    assert numbers_in("near its 52‑week high on 2026-10-09 at 1,234.50") == [52.0, 1234.5]


def test_good_answer_passes():
    assert validate_signal(TradeSignal(**GOOD), FACTS, CONFLICTS) == []


def test_echo_answer_fails_every_semantic_check():
    problems = " | ".join(validate_signal(TradeSignal(**ECHO), FACTS, CONFLICTS))
    assert "185 does not appear" in problems                 # invented number (fact was 180)
    assert "conflicting_evidence is empty" in problems
    assert "Confidence 0.95 is too high" in problems
    assert "link signals together" in problems


def test_single_indicator_answer_fails():
    one = {**GOOD, "justification": "RSI is 48.0 so buyers are weak. RSI is not oversold yet, so wait. "
                                    "RSI fell from 61.0, therefore HOLD."}
    assert any("indicator group" in p for p in validate_signal(TradeSignal(**one), FACTS, CONFLICTS))


def test_schema_rejects_two_sentences_and_bad_enum():
    with pytest.raises(ValidationError):
        TradeSignal(**{**GOOD, "justification": "Too short. Only two."})
    with pytest.raises(ValidationError):
        TradeSignal(**{**GOOD, "signal": "STRONG BUY"})


def test_attempt_signal_returns_problems_and_feedback_is_sent():
    llm = FakeLLM(lambda h, n: json.dumps(ECHO))
    values = signal_prompt_values("NVDA", "2026-10-09", FACTS, CONFLICTS, MOM, SENT)
    sig, problems, raw = attempt_signal(llm, values, FACTS, CONFLICTS, ["previous problem X"], MOM["score"])
    assert sig is None and problems and raw
    assert "previous problem X" in llm.prompts[0]           # feedback from the last attempt is in the prompt
    sig, problems, _ = attempt_signal(FakeLLM(lambda h, n: json.dumps(GOOD)), values, FACTS, CONFLICTS, [], 0.125)
    assert sig.signal == "HOLD" and problems == []
    sig, problems, _ = attempt_signal(FakeLLM(lambda h, n: "not json"), values, FACTS, CONFLICTS, [], 0.125)
    assert sig is None and "Output format error" in problems[0]


@pytest.mark.parametrize("mom,sent,expected", [
    ({"label": "strong_bullish", "score": 0.8}, {"score": 0.5}, "BUY"),
    ({"label": "neutral", "score": 0.125}, {"score": 0.31}, "HOLD"),
    ({"label": "bearish", "score": -0.6}, None, "SELL"),
])
def test_fallback_is_deterministic_and_valid(mom, sent, expected):
    sig = rule_based_fallback(CONFLICTS, mom, sent)
    assert sig.signal == expected and sig.confidence <= 0.6
    assert "rule-based" in sig.justification
