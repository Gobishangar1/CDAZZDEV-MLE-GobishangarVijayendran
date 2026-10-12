"""Every constant used by the pipeline, plus API-key and logging setup.

Rule: no other module contains a "magic number". If a value is a choice
(a window, a threshold, a weight), it lives here with a one-line reason.
"""

from __future__ import annotations

import logging
import os
from typing import Final

# -----------------------------------------------------------------------------
# Market data (Task 1A)
# -----------------------------------------------------------------------------
HISTORY_YEARS: Final = 2               # brief: at least two years of daily OHLCV
WARMUP_CALENDAR_DAYS: Final = 300      # ~200 trading days, so SMA-200 has a value on day 1 of the 2-year window
TRADING_DAYS_PER_YEAR: Final = 252     # US market convention; 52-week window
TICKER_PATTERN: Final = r"^[A-Z0-9.\-^]{1,10}$"   # letters, digits, '.', '-', '^' (BRK-B, ^GSPC)
PROBE_PERIOD: Final = "5d"             # tiny fetch to check a ticker exists before any LLM call
FETCH_ATTEMPTS: Final = 3              # yfinance is unofficial and sometimes flaky
FETCH_BACKOFF_S: Final = 2             # exponential backoff base: 2 s, 4 s
MAX_DAILY_MOVE: Final = 0.40           # flag (never delete) daily moves above 40%: real news or a data error

# -----------------------------------------------------------------------------
# Technical indicators (values fixed by the brief)
# -----------------------------------------------------------------------------
SMA_SHORT: Final = 50
SMA_LONG: Final = 200
RSI_PERIOD: Final = 14
MACD_FAST: Final = 12
MACD_SLOW: Final = 26
MACD_SIGNAL: Final = 9
BB_WINDOW: Final = 20
BB_STD: Final = 2.0
BB_DDOF: Final = 0                     # population std = Bollinger's definition (pandas default is 1)

RSI_OVERBOUGHT: Final = 70.0           # Wilder's classic levels
RSI_OVERSOLD: Final = 30.0
RSI_NEUTRAL: Final = 50.0
RSI_LOOKBACK_DAYS: Final = 10          # fact: "RSI was X ten trading days ago"
MACD_HIST_KEEP: Final = 5              # last N histogram values kept in the snapshot
MACD_FADE_STEPS: Final = 3             # histogram shrinking 3 days in a row = momentum fading
BB_NEAR_UPPER: Final = 0.8             # %B above this = "near the upper band"
BB_NEAR_LOWER: Final = 0.2             # %B below this = "near the lower band"

# -----------------------------------------------------------------------------
# Momentum score (rule-based). Starting values, NOT tuned on price history (avoids overfitting).
# -----------------------------------------------------------------------------
MOMENTUM_WEIGHTS: Final[dict[str, float]] = {
    "long_trend": 0.25,    # SMA-50 vs SMA-200: slowest, most reliable trend read
    "short_trend": 0.20,   # close vs SMA-50
    "macd": 0.25,          # momentum direction, halved if fading
    "rsi": 0.20,           # strength, scaled around 50
    "bollinger": 0.10,     # mean-reverting signal, so the smallest say
}
MACD_FADE_DAMPING: Final = 0.5         # halve the MACD vote when the histogram is shrinking
RSI_SCALE: Final = 20.0                # (RSI - 50) / 20: RSI 70 -> +1, RSI 30 -> -1
MOMENTUM_LABELS: Final[list[tuple[float, str]]] = [   # checked top-down
    (0.50, "strong_bullish"),          # score >= 0.50
    (0.15, "bullish"),                 # score >= 0.15
    (-0.15, "neutral"),                # score >  -0.15
    (-0.50, "bearish"),                # score >  -0.50, otherwise strong_bearish
]

# -----------------------------------------------------------------------------
# News (Task 1A)
# -----------------------------------------------------------------------------
NEWS_MIN: Final = 10                   # brief: at least ten recent headlines
NEWS_MAX: Final = 15                   # each headline = one LLM call; keeps us inside free-tier limits
NEWS_TIMEOUT_S: Final = 10             # one slow RSS feed must not block the run
YF_NEWS_COUNT: Final = 30              # ask yfinance for more than its default 10: more chances of relevant items
YF_NEWS_TABS: Final = ("news", "all")  # second Yahoo tab tried when the first comes back empty

# -----------------------------------------------------------------------------
# Sentiment aggregation (Task 1B)
# -----------------------------------------------------------------------------
SENTIMENT_HALF_LIFE_DAYS: Final = 3.0  # a 3-day-old headline counts half as much as today's
SENTIMENT_UNKNOWN_AGE_DAYS: Final = 3.0  # headline without a timestamp is treated as 3 days old
SENTIMENT_THRESHOLD: Final = 0.15      # |score| below this = overall "neutral"

# -----------------------------------------------------------------------------
# Trade signal (Task 1B)
# -----------------------------------------------------------------------------
HEADLINE_ATTEMPTS: Final = 2           # 1 normal try + 1 repair try per headline
MAX_SIGNAL_ATTEMPTS: Final = 3         # LLM tries before the rule-based fallback
JUSTIFICATION_MIN_SENTENCES: Final = 3 # brief: three to five sentences
JUSTIFICATION_MAX_SENTENCES: Final = 5
MIN_INDICATOR_GROUPS: Final = 2        # justification must combine at least 2 indicators
NUMBER_TOLERANCE: Final = 0.01         # 1%; 5% let an invented "SMA-50 is 185" pass when the fact was 180
MAX_CONFIDENCE_WITH_CONFLICTS: Final = 0.75   # prompt rule 6, also enforced in code
FALLBACK_SENTIMENT_WEIGHT: Final = 0.30       # fallback score = 0.7 * momentum + 0.3 * sentiment
FALLBACK_BUY: Final = 0.30             # fallback score >= 0.30 -> BUY
FALLBACK_SELL: Final = -0.30           # fallback score <= -0.30 -> SELL, otherwise HOLD
FALLBACK_CONFIDENCE_SCALE: Final = 0.6 # a rule-based signal never claims high confidence

# -----------------------------------------------------------------------------
# LLM (Groq free tier)
# -----------------------------------------------------------------------------
# The brief suggests Llama-3 on Groq; Groq now serves GPT-OSS as its free production models.
# Model names are config, so switching is a one-line change. Check console.groq.com/docs/models.
FAST_MODEL: Final = "openai/gpt-oss-20b"     # per-headline sentiment (simple task)
STRONG_MODEL: Final = "openai/gpt-oss-120b"  # Buy/Hold/Sell reasoning (harder task)
LLM_TEMPERATURE: Final = 0.0           # reproducible output
LLM_TIMEOUT_S: Final = 30              # a hung call must not hang the graph
LLM_MAX_RETRIES: Final = 2             # client retries on 429 / 5xx
LLM_REQUESTS_PER_SECOND: Final = 0.2   # 12 RPM per model: under the 30 RPM cap and ~8K tokens/min
HEADLINE_REASONING: Final = "low"      # GPT-OSS thinks first; thinking tokens count against limits
SIGNAL_REASONING: Final = "medium"     # the signal needs real reasoning
MAX_CONCURRENCY: Final = 3             # parallel headline branches at once
LOG_TRUNCATE: Final = 200              # characters of raw LLM text kept in logs

# -----------------------------------------------------------------------------
# Report (bonus)
# -----------------------------------------------------------------------------
CHART_DAYS: Final = 252                # chart shows the last 12 months
CHART_DPI: Final = 150
TOP_HEADLINES: Final = 3               # brief: "top three headlines"

DISCLAIMER: Final = (
    "This report is generated automatically for educational purposes only. It is not investment "
    "advice and must not be used to make investment decisions. Data may be delayed, incomplete or "
    "wrong, LLM output can be mistaken, and past performance does not predict future results."
)

assert abs(sum(MOMENTUM_WEIGHTS.values()) - 1.0) < 1e-9, "MOMENTUM_WEIGHTS must sum to 1"


# -----------------------------------------------------------------------------
# API key (never hard-coded, never printed)
# -----------------------------------------------------------------------------
def load_colab_secret(name: str = "GROQ_API_KEY") -> bool:
    """In Colab, copy a secret from the Secrets panel into os.environ.

    Returns True if it was loaded. Never returns or prints the value.
    Outside Colab this does nothing and returns False.
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        value = userdata.get(name)
    except Exception:  # secret missing, or notebook access not granted
        return False
    if value:
        os.environ[name] = value
        return True
    return False


def load_dotenv_file(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win. No extra dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_groq_key() -> None:
    """Fail early with a clear message (not a traceback deep inside the LLM client)."""
    load_dotenv_file()
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise RuntimeError(
            "GROQ_API_KEY is not set. In Colab: add it in the Secrets panel (key icon). "
            "Locally: export GROQ_API_KEY=... or put it in a .env file (git-ignored)."
        )


def setup_logging(level: int = logging.INFO) -> None:
    """Our logs at `level`, noisy third-party libraries at WARNING."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)
    logging.getLogger("equity_research").setLevel(level)


def constants_table() -> list[tuple[str, object]]:
    """All UPPER_CASE constants, for the notebook's 'no magic numbers' cell."""
    return [(k, v) for k, v in globals().items() if k.isupper() and k != "DISCLAIMER"]
