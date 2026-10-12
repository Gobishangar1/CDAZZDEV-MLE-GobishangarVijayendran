"""Every constant for Task 3 (no magic numbers elsewhere), plus API-key and logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

PACKAGE_ROOT: Final = Path(__file__).resolve().parent.parent      # task3_agentic/

# -----------------------------------------------------------------------------
# The research question (from the brief)
# -----------------------------------------------------------------------------
DEFAULT_TICKER: Final = "NVDA"
RESEARCH_QUERY: Final = ("Analyse the current financial health and market sentiment of {ticker}. Identify the top "
                         "three risks to its share price over the next 90 days and suggest one data-driven hedge "
                         "strategy.")

# -----------------------------------------------------------------------------
# Tool: get_price_data / calculate_volatility
# -----------------------------------------------------------------------------
ALLOWED_PERIODS: Final = ("1mo", "3mo", "6mo", "1y", "2y", "5y")
DEFAULT_PERIOD: Final = "1y"
INDICATOR_FETCH_PERIOD: Final = "2y"   # SMA-200 needs 200 trading days of history before the window starts
TRADING_DAYS_PER_YEAR: Final = 252     # annualisation factor and 52-week window
RECENT_ROWS: Final = 5                 # OHLCV rows returned to the LLM (the rest stays in Python: saves tokens)
DEFAULT_VOL_WINDOW: Final = 30         # ~6 weeks of trading days
MIN_VOL_WINDOW: Final = 5
MAX_VOL_WINDOW: Final = 252
VAR_QUANTILE: Final = 0.05             # 1-day historical Value-at-Risk at 95%
FETCH_ATTEMPTS: Final = 3
FETCH_BACKOFF_S: Final = 2

SMA_SHORT, SMA_LONG = 50, 200
RSI_PERIOD: Final = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD, BB_DDOF = 20, 2.0, 0   # population std: Bollinger's definition

# -----------------------------------------------------------------------------
# Tool: get_news / web_search / llm_sentiment
# -----------------------------------------------------------------------------
DEFAULT_NEWS: Final = 10
MAX_NEWS: Final = 20
NEWS_TIMEOUT_S: Final = 10
YF_NEWS_COUNT: Final = 30
DEFAULT_SEARCH_RESULTS: Final = 5
MIN_SEARCH_RESULTS: Final = 3          # one result per search wastes an LLM turn (seen in a real run)
MAX_SEARCH_RESULTS: Final = 8
SNIPPET_CHARS: Final = 220             # long snippets waste tokens; the title + 220 chars carry the point
MAX_SENTIMENT_HEADLINES: Final = 15    # one LLM call for all headlines: keep the prompt small
SENTIMENT_THRESHOLD: Final = 0.15      # |score| below this = neutral

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
AGENT_MAX_STEPS: Final = 8             # LLM turns per agent run; stops endless tool loops
TOOL_OUTPUT_TO_LLM_CHARS: Final = 2500 # cap per tool result in the conversation (token budget)
MODEL_ERROR_RETRIES: Final = 2         # extra tries when the model returns a malformed tool call
JSON_ATTEMPTS: Final = 3               # structured outputs: 1 try + 2 repairs
CLARIFICATION_ROUNDS: Final = 1        # brief: Agent B may send ONE clarification request

# -----------------------------------------------------------------------------
# LLM (Groq free tier). Limits are per model, so each model is the other's fallback.
# -----------------------------------------------------------------------------
AGENT_MODEL: Final = "openai/gpt-oss-120b"     # tool calling and reasoning
FALLBACK_MODEL: Final = "openai/gpt-oss-20b"
SENTIMENT_MODEL: Final = "openai/gpt-oss-20b"  # simple labelling task
LLM_TEMPERATURE: Final = 0.0
LLM_TIMEOUT_S: Final = 45
PRIMARY_MAX_RETRIES: Final = 1         # 120b: give up quickly on 429 and move to the fallback model
FALLBACK_MAX_RETRIES: Final = 3        # 20b: wait and retry (free tier: ~8K tokens/min per model)
AGENT_REASONING: Final = "medium"
SENTIMENT_REASONING: Final = "low"

# -----------------------------------------------------------------------------
# Observability and memory
# -----------------------------------------------------------------------------
LOGS_DIR: Final = PACKAGE_ROOT / "logs"
TRACE_FILE: Final = LOGS_DIR / "agent_trace.jsonl"
TRACE_OUTPUT_CHARS: Final = 200        # brief: output truncated to 200 characters
CONSOLE_CHARS: Final = 400             # how much of each message the notebook trace prints
HANDOFF_FIELD_CHARS: Final = 1000      # each field of a handoff message, as printed in the notebook
MEMORY_DIR: Final = PACKAGE_ROOT / "memory"

DISCLAIMER: Final = ("Generated automatically for educational purposes. Not investment advice. Data may be delayed "
                     "or incomplete, and LLM output can be wrong.")


def load_colab_secret(name: str = "GROQ_API_KEY") -> bool:
    """Copy a Colab secret into os.environ. Never returns or prints the value."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        value = userdata.get(name)
    except Exception:
        return False
    if value:
        os.environ[name] = value
        return True
    return False


def load_dotenv_file(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE). Existing environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_groq_key() -> None:
    for env_file in (Path(".env"), PACKAGE_ROOT / ".env", PACKAGE_ROOT.parent / ".env"):   # cwd, task3, repo root
        load_dotenv_file(str(env_file))
    if not os.environ.get("GROQ_API_KEY", "").strip():
        raise RuntimeError("GROQ_API_KEY is not set. Colab: add it in the Secrets panel. "
                           "Locally: export GROQ_API_KEY=... or put it in a .env file (git-ignored).")


def setup_logging(level: int = logging.WARNING) -> None:
    """Quiet library logs; the readable trace is printed by research_agents.trace."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)
    logging.getLogger("research_agents").setLevel(level)
