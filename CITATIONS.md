# Citations

This file follows Section 2.2 of the assessment brief. Every AI-assisted file also carries an inline
comment in the required format, for example:

```python
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Implement SMA, EMA, Wilder RSI, MACD (12,26,9) and
# Bollinger Bands (20, 2 std, ddof=0) from first principles with pandas, no TA-Lib', Date: 2026-10-11
```

## How AI was used

I used Claude (model `claude-opus-5-5`) as a coding assistant and tutor. The architecture (a fixed
LangGraph workflow, "Python does the maths, the LLM only judges", facts and conflicts for the signal,
two-level validation) was worked out in an AI-assisted design document, which I reviewed and changed.
I then went through each component one function at a time, checked the indicator maths against
hand-computed examples, and ran the pipeline on real data in Colab. Some changes came directly from
my own runs: the Unicode-dash fix in the number check (GPT-OSS wrote "52‑week" with U+2011), and the
confidence cap when signals conflict, and the news relevance filter (my first real run returned 13 off-topic
headlines out of 15).

## AI-assisted code (Task 1)

| File | Prompt (short) | Date |
|---|---|---|
| `task1_financial/equity_research/config.py` | Every constant with a one-line reason; Groq key loading for Colab and local; logging setup | 2026-10-11 |
| `task1_financial/equity_research/schemas.py` | Pydantic schemas for headline sentiment (headline, sentiment, confidence, brief_reason), trade signal with a 3-5 sentence validator, market summary | 2026-10-11 |
| `task1_financial/equity_research/prompts.py` | System and human prompt templates for per-headline sentiment and a Buy/Hold/Sell signal that reasons over combinations | 2026-10-11 |
| `task1_financial/equity_research/market_data.py` | yfinance OHLCV fetch with computed start date + warm-up, tenacity retries, cleaning, data checks, fail-soft info | 2026-10-11 |
| `task1_financial/equity_research/indicators.py` | SMA, EMA, Wilder RSI, MACD (12,26,9), Bollinger (20, 2 std, ddof=0) from first principles, no TA-Lib | 2026-10-11 |
| `task1_financial/equity_research/analysis.py` | Summary (price, 52-week range, P/E fallbacks, YTD) and weighted rule-based momentum score | 2026-10-11 |
| `task1_financial/equity_research/news.py` | News provider chain (yfinance old/new formats, Yahoo RSS, Google News RSS), timeouts, dedupe | 2026-10-11 |
| `task1_financial/equity_research/llm.py` | Groq ChatGroq factory with rate limiter, timeout, retries, JSON mode, model fallback; JSON helpers | 2026-10-11 |
| `task1_financial/equity_research/sentiment.py` | Per-headline LLM sentiment with validation, logged failures, repair retry; weighted aggregation | 2026-10-11 |
| `task1_financial/equity_research/signal_context.py` | Indicator snapshot to factual sentences; deterministic conflict detection | 2026-10-11 |
| `task1_financial/equity_research/trade_signal.py` | Semantic validator (grounded numbers, >= 2 indicators, conflicts, anti-echo) and deterministic fallback | 2026-10-11 |
| `task1_financial/equity_research/report.py` + `templates/brief.md.j2` | One-page brief: Jinja2 Markdown, 3-panel matplotlib chart as base64 in styled HTML, disclaimer | 2026-10-11 |
| `task1_financial/equity_research/graph.py` | LangGraph StateGraph: parallel fetches, Send fan-out with reducers, joins, validation retry loop, fallback, node_guard | 2026-10-11 |
| `task1_financial/equity_research/tracing.py` | Thread-safe run tracer: node durations and LLM calls (model, latency, tokens, validation) to JSONL | 2026-10-11 |
| `task1_financial/equity_research/__main__.py` | Small argparse CLI | 2026-10-11 |
| `task1_financial/tests/*.py` | Offline pytest suite with a fake yfinance Ticker and a scripted fake LLM | 2026-10-11 |
| `task1_financial/notebooks/task1_equity_research.ipynb` | Notebook cells in rubric order, including a fault-injection demo | 2026-10-11 |
| `README.md`, `task1_financial/README.md`, `REFLECTION.md` (first draft) | Documentation drafts, edited by me | 2026-10-11 |

## Adapted open-source code

None. No code was copied from other repositories. Indicator formulas follow their standard published
definitions (Wilder 1978 for RSI; Appel for MACD; Bollinger for Bollinger Bands); no TA-Lib or other
indicator library is used, not even in the tests.

## Teacher-model prompts

Not applicable to Task 1 (no synthetic training data). All runtime prompts are in
`task1_financial/equity_research/prompts.py` and are printed in the notebook appendix.
