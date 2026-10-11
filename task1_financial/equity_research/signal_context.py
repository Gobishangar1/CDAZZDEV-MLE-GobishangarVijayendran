"""Turn numbers into FACTS and CONFLICTS for the signal LLM (Task 1B, 'reasoning' criterion).

Why: given raw numbers, LLMs tend to "echo" them ("RSI is 48. MACD is 0.2. HOLD.") and they are
weak at arithmetic. So code does the maths and states relationships as facts, and a deterministic
detector lists signals that disagree. The prompt then requires the LLM to resolve each conflict.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Convert an indicator snapshot into short factual
# sentences and detect conflicting signal pairs with deterministic rules', Date: 2026-10-11

from __future__ import annotations

from equity_research.config import (
    BB_NEAR_LOWER,
    BB_NEAR_UPPER,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SENTIMENT_THRESHOLD,
)
from equity_research.analysis import macd_is_fading


def _pct(a: float, b: float) -> float:
    return (a / b - 1) * 100


def _uptrend(s: dict) -> bool:
    return s.get("sma_50") is not None and s.get("sma_200") is not None and s["sma_50"] > s["sma_200"]


def _downtrend(s: dict) -> bool:
    return s.get("sma_50") is not None and s.get("sma_200") is not None and s["sma_50"] < s["sma_200"]


def build_facts(s: dict, summary: dict, sentiment: dict | None) -> list[str]:
    """Short sentences that already state direction, distance, zone and change over time."""
    facts: list[str] = []
    close = s["close"]

    if s.get("sma_50") is not None:
        side = "above" if close > s["sma_50"] else "below"
        facts.append(f"Close {close:.2f} is {abs(_pct(close, s['sma_50'])):.1f}% {side} SMA-50 ({s['sma_50']:.2f}).")
    if s.get("sma_200") is not None:
        side = "above" if close > s["sma_200"] else "below"
        facts.append(f"Close is {abs(_pct(close, s['sma_200'])):.1f}% {side} SMA-200 ({s['sma_200']:.2f}).")
    if _uptrend(s) or _downtrend(s):
        facts.append(f"SMA-50 is {'above' if _uptrend(s) else 'below'} SMA-200: long-term trend is "
                     f"{'up' if _uptrend(s) else 'down'}.")
    else:
        facts.append("SMA-200 is not available yet (short price history); long-term trend unknown.")
    if s.get("cross"):
        facts.append(f"Last moving-average cross: {s['cross']['kind']} cross {s['cross']['days_ago']} trading days ago.")

    hist = s.get("macd_hist_last5") or []
    if hist:
        direction = "positive" if hist[-1] > 0 else "negative"
        side = "bullish" if hist[-1] > 0 else "bearish"          # a negative histogram = bearish momentum
        if len(hist) >= 4:
            trend = (f"shrinking toward zero ({side} momentum fading)" if macd_is_fading(hist) else
                     f"growing away from zero ({side} momentum building)" if abs(hist[-1]) > abs(hist[-4]) else
                     "mixed over the last 3 days")
            facts.append(f"MACD histogram is {direction} ({hist[-1]:+.2f}) and {trend}.")
        else:
            facts.append(f"MACD histogram is {direction} ({hist[-1]:+.2f}, {side} momentum).")
    if s.get("macd") is not None and s.get("macd_signal") is not None:
        facts.append(f"MACD line is {'above' if s['macd'] > s['macd_signal'] else 'below'} its signal line "
                     f"and {'above' if s['macd'] > 0 else 'below'} zero.")

    if s.get("rsi") is not None:
        r = s["rsi"]
        zone = "overbought" if r > RSI_OVERBOUGHT else "oversold" if r < RSI_OVERSOLD else "neutral zone"
        prev = f"; it was {s['rsi_10d_ago']:.1f} ten trading days ago" if s.get("rsi_10d_ago") is not None else ""
        facts.append(f"RSI-14 is {r:.1f} ({zone}){prev}.")

    if s.get("bb_pctb") is not None:
        b = s["bb_pctb"]
        position = ("above the upper band" if b > 1 else "below the lower band" if b < 0 else
                    "near the upper band" if b > BB_NEAR_UPPER else "near the lower band" if b < BB_NEAR_LOWER
                    else "inside the bands")
        facts.append(f"Bollinger %B is {b:.2f} (price {position}).")

    facts.append(f"Price is {abs(summary['pct_from_52w_high']):.1f}% below its 52-week high "
                 f"({summary['high_52w']:.2f}); 52-week low is {summary['low_52w']:.2f}.")
    if summary.get("ytd_return_pct") is not None:
        facts.append(f"Year-to-date return is {summary['ytd_return_pct']:+.1f}%.")
    if summary.get("pe_ratio") is not None:
        facts.append(f"Trailing P/E is {summary['pe_ratio']:.1f}.")

    if sentiment:
        facts.append(f"News sentiment is {sentiment['score']:+.2f} ({sentiment['label']}) from "
                     f"{sentiment['n_valid']} headlines; agreement {sentiment['agreement']:.0%}, "
                     f"coverage {sentiment['coverage']:.0%}.")
    else:
        facts.append("News sentiment is unavailable.")
    return facts


def find_conflicts(s: dict, momentum: dict, sentiment: dict | None) -> list[str]:
    """Deterministic rules that pair up signals which disagree. The LLM must resolve each one."""
    out: list[str] = []
    hist = s.get("macd_hist_last5") or []
    rsi, score = s.get("rsi"), momentum["score"]

    if _uptrend(s) and s.get("sma_50") is not None and s["close"] < s["sma_50"]:
        out.append("Long-term trend is up, but price has fallen below SMA-50 (short-term weakness).")
    if _downtrend(s) and s.get("sma_50") is not None and s["close"] > s["sma_50"]:
        out.append("Long-term trend is down, but price is above SMA-50 (short-term bounce).")
    if _uptrend(s) and hist and (hist[-1] < 0 or macd_is_fading(hist)):
        out.append("Trend is up, but MACD momentum is fading or negative.")
    if _downtrend(s) and hist and hist[-1] > 0:
        out.append("Trend is down, but MACD momentum has turned positive.")
    if rsi is not None and rsi > RSI_OVERBOUGHT and score > 0:
        out.append("Momentum is bullish, but RSI is overbought (risk of a pullback).")
    if rsi is not None and rsi < RSI_OVERSOLD and score < 0:
        out.append("Momentum is bearish, but RSI is oversold (a bounce is possible).")
    if sentiment and sentiment["label"] == "positive" and score < -SENTIMENT_THRESHOLD:
        out.append("News sentiment is positive, but price momentum is bearish.")
    if sentiment and sentiment["label"] == "negative" and score > SENTIMENT_THRESHOLD:
        out.append("News sentiment is negative, but price momentum is bullish.")
    return out
