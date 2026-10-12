"""Buy/Hold/Sell signal: one LLM attempt, a semantic validator, and a rule-based fallback (Task 1B).

Two validation levels:
    1. format  : TradeSignal (Pydantic): enum, ranges, 3-5 sentences, >= 2 key drivers
    2. meaning : validate_signal(): the reasoning itself
        (a) every number comes from FACTS (no invented numbers)
        (b) at least 2 different indicator groups are named (combination)
        (c) conflicts given -> conflicting_evidence is not empty
        (d) conflicts given -> confidence <= 0.75
        (e) anti-echo: at least half of the sentences link signals (but, while, because...)

The retry loop itself is the graph edge generate_signal -> generate_signal (see graph.py).
"""

from __future__ import annotations

import logging
import math
import re

from pydantic import ValidationError

from equity_research.config import (
    BB_STD,
    BB_WINDOW,
    FALLBACK_BUY,
    FALLBACK_CONFIDENCE_SCALE,
    FALLBACK_SELL,
    FALLBACK_SENTIMENT_WEIGHT,
    LOG_TRUNCATE,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_CONFIDENCE_WITH_CONFLICTS,
    MIN_INDICATOR_GROUPS,
    NUMBER_TOLERANCE,
    RSI_NEUTRAL,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
)
from equity_research import tracing
from equity_research.llm import extract_json, invoke_llm, short_error
from equity_research.prompts import SIGNAL_FEEDBACK, SIGNAL_PROMPT
from equity_research.schemas import TradeSignal, clean_text, split_sentences

log = logging.getLogger(__name__)

# Numbers that may appear without being "facts": indicator parameters, RSI levels, 52-week, the 1-3 month horizon.
WELL_KNOWN_NUMBERS = {float(x) for x in (SMA_SHORT, SMA_LONG, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                                         BB_WINDOW, BB_STD, RSI_OVERBOUGHT, RSI_OVERSOLD, RSI_NEUTRAL, 52, 1, 3)}

_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")

INDICATOR_GROUPS = {
    "trend": r"\bSMA|moving average|golden cross|death cross|\btrend\b",
    "rsi": r"\bRSI\b|overbought|oversold",
    "macd": r"\bMACD\b|histogram",
    "bollinger": r"Bollinger|%B|\bbands?\b",
}
LINK_WORDS = re.compile(r"\b(but|while|although|though|despite|because|however|yet|whereas|since|"
                        r"confirms?|supports?|offsets?|outweighs?|so|therefore|combined|together|alongside)\b",
                        re.IGNORECASE)


def numbers_in(text: str) -> list[float]:
    """Absolute values of all numbers in the text, ignoring dates and thousands separators."""
    text = _THOUSANDS.sub("", _ISO_DATE.sub(" ", clean_text(text)))
    return [abs(float(x)) for x in _NUMBER.findall(text)]


def _same_number(x: float, allowed: float) -> bool:
    """Equal within NUMBER_TOLERANCE, or a normal rounding of it (176.00 -> 176, 61.04 -> 61.0)."""
    return (abs(x - allowed) <= NUMBER_TOLERANCE * max(abs(allowed), 1.0)
            or x == round(allowed) or x == round(allowed, 1))


def validate_signal(sig: TradeSignal, facts: list[str], conflicts: list[str],
                    extra_numbers: tuple[float, ...] = ()) -> list[str]:
    """Semantic checks on a schema-valid TradeSignal. Returns a list of problems ([] = pass)."""
    problems: list[str] = []
    text = clean_text(sig.justification)

    allowed = set(numbers_in(" ".join(facts))) | WELL_KNOWN_NUMBERS | {abs(x) for x in extra_numbers}
    for x in sorted(set(numbers_in(text))):
        if not any(_same_number(x, a) for a in allowed):
            problems.append(f"The number {x:g} does not appear in FACTS. Use only numbers from FACTS.")

    named = [g for g, pattern in INDICATOR_GROUPS.items() if re.search(pattern, text, re.IGNORECASE)]
    if len(named) < MIN_INDICATOR_GROUPS:
        problems.append(f"The justification names {len(named)} indicator group(s) {named}; combine at least "
                        f"{MIN_INDICATOR_GROUPS} of: trend (SMA), RSI, MACD, Bollinger.")

    if conflicts and not sig.conflicting_evidence:
        problems.append("CONFLICTS were given but conflicting_evidence is empty; address each conflict.")
    if conflicts and sig.confidence > MAX_CONFIDENCE_WITH_CONFLICTS:
        problems.append(f"Confidence {sig.confidence} is too high: there are conflicts, so it must be at most "
                        f"{MAX_CONFIDENCE_WITH_CONFLICTS}.")

    sentences = split_sentences(text)
    linked = sum(bool(LINK_WORDS.search(s)) for s in sentences)
    if linked < math.ceil(len(sentences) / 2):
        problems.append(f"Only {linked} of {len(sentences)} sentences link signals together; explain how the "
                        f"signals combine (while, but, because, despite, confirms).")
    return problems


def signal_prompt_values(ticker: str, as_of: str, facts: list[str], conflicts: list[str],
                         momentum: dict, sentiment: dict | None) -> dict:
    return {
        "ticker": ticker, "as_of": as_of,
        "facts": "\n".join(f"- {f}" for f in facts),
        "conflicts": "\n".join(f"- {c}" for c in conflicts) or "- none",
        "momentum_label": momentum["label"], "momentum_score": momentum["score"],
        "sentiment_line": f"{sentiment['score']:+.2f} ({sentiment['label']})" if sentiment else "unavailable",
    }


def attempt_signal(llm, values: dict, facts: list[str], conflicts: list[str], previous_problems: list[str],
                   momentum_score: float, attempt_no: int | None = None) -> tuple[TradeSignal | None, list[str], str]:
    """ONE LLM attempt + both validation levels. Returns (signal or None, problems, raw reply)."""
    feedback = (SIGNAL_FEEDBACK.format(problems="\n".join(f"- {p}" for p in previous_problems))
                if previous_problems else "")
    raw = ""
    attempt = attempt_no or (2 if previous_problems else 1)
    try:
        raw = invoke_llm(llm, SIGNAL_PROMPT.invoke({**values, "feedback": feedback}), "signal", attempt)
        sig = TradeSignal.model_validate_json(extract_json(raw))                      # level 1: format
        problems = validate_signal(sig, facts, conflicts, extra_numbers=(momentum_score,))  # level 2: meaning
    except (ValueError, ValidationError) as exc:
        sig, problems = None, [f"Output format error: {short_error(exc)}"]
    except Exception as exc:                       # API failure after client retries and the fallback model
        sig, problems = None, [f"LLM call failed: {type(exc).__name__}: {str(exc)[:LOG_TRUNCATE]}"]
    tracing.record("validation", purpose="signal", attempt=attempt, valid=not problems, problems=problems)
    return (sig if not problems else None), problems, raw


def rule_based_fallback(conflicts: list[str], momentum: dict, sentiment: dict | None) -> TradeSignal:
    """Deterministic signal: 70% momentum + 30% sentiment. Used after 3 failed LLM attempts. Cannot fail."""
    sent = sentiment["score"] if sentiment else 0.0
    score = (1 - FALLBACK_SENTIMENT_WEIGHT) * momentum["score"] + FALLBACK_SENTIMENT_WEIGHT * sent
    signal = "BUY" if score >= FALLBACK_BUY else "SELL" if score <= FALLBACK_SELL else "HOLD"
    justification = (
        f"This is a rule-based signal because the LLM answer failed validation on every attempt. "
        f"It combines the momentum score of {momentum['score']:+.2f} ({momentum['label']}) with news "
        f"sentiment of {sent:+.2f}, weighted {1 - FALLBACK_SENTIMENT_WEIGHT:.0%} to {FALLBACK_SENTIMENT_WEIGHT:.0%}. "
        f"The combined score of {score:+.2f} maps to {signal}, because BUY needs at least {FALLBACK_BUY:+.2f} "
        f"and SELL needs at most {FALLBACK_SELL:+.2f}."
    )
    return TradeSignal(
        signal=signal,
        confidence=round(min(abs(score), 1.0) * FALLBACK_CONFIDENCE_SCALE, 2),
        justification=justification,
        key_drivers=[f"momentum score {momentum['score']:+.2f} ({momentum['label']})",
                     f"news sentiment {sent:+.2f}"],
        conflicting_evidence=conflicts,
    )
