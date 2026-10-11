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

## AI-assisted code (Task 3)

For Task 3 I used Claude in the same way: it drafted the code from a design I agreed (LangGraph ReAct loop,
one executor for all tool calls, typed handoffs, one critique round), and I reviewed it, ran the offline tests and
ran the notebook on real data in Colab.

| File | Prompt (short) | Date |
|---|---|---|
| `task3_agentic/research_agents/config.py` | Constants for models, limits, windows and trace/memory paths; Groq key loading | 2026-10-11 |
| `task3_agentic/research_agents/schemas.py` | Pydantic schemas for five tool results, a typed handoff (data brief, clarification request/response) and a three-section report | 2026-10-11 |
| `task3_agentic/research_agents/prompts.py` | System prompts for a ReAct research agent, a data analyst, a research writer with a critique step, a report writer and a sentiment tool | 2026-10-11 |
| `task3_agentic/research_agents/tools.py` | Five tools with Pydantic arguments, and an executor that enforces per-agent tool access, turns errors into suggestions and logs every call | 2026-10-11 |
| `task3_agentic/research_agents/llm.py` | Groq model factories with fallback; JSON + Pydantic structured output with repair retries | 2026-10-11 |
| `task3_agentic/research_agents/agent.py` | Reusable LangGraph ReAct agent: tool restriction, malformed-call retries, step budget, checkpointer memory, trace | 2026-10-11 |
| `task3_agentic/research_agents/single_agent.py` | 3A session with all five tools, validated report, follow-up questions from session memory | 2026-10-11 |
| `task3_agentic/research_agents/multi_agent.py` | Two-agent pipeline with restricted tools, Pydantic handoffs, one clarification round and a JSON brief cache | 2026-10-11 |
| `task3_agentic/research_agents/memory.py` | JSON brief store keyed by ticker and date | 2026-10-11 |
| `task3_agentic/research_agents/report.py` | Validated three-section report from agent evidence; Markdown rendering | 2026-10-11 |
| `task3_agentic/research_agents/trace.py`, `trace_view.py` | Thread-safe JSONL tracer (200-character outputs) with a readable console trace; trace-to-table helpers | 2026-10-11 |
| `task3_agentic/research_agents/market.py`, `news.py` | Compact versions of the Task 1 data and news code for the tools (see below) | 2026-10-11 |
| `task3_agentic/research_agents/__main__.py` | Small argparse CLI | 2026-10-11 |
| `task3_agentic/dashboard/app.py` | Streamlit dashboard that reads `agent_trace.jsonl` (bonus) | 2026-10-11 |
| `task3_agentic/tests/*.py` | Offline pytest suite: fake yfinance, fake search, an observation-driven scripted tool-calling model, a JSON model | 2026-10-11 |
| `task3_agentic/notebooks/task3_agentic_research.ipynb` | Notebook cells in rubric order, with a tool-failure demo | 2026-10-11 |
| `task3_agentic/README.md`, root `README.md` and `REFLECTION.md` updates | Documentation drafts, edited by me | 2026-10-11 |

## Adapted open-source code

No code was copied from other repositories. Indicator formulas follow their standard published
definitions (Wilder 1978 for RSI; Appel for MACD; Bollinger for Bollinger Bands); no TA-Lib or other
indicator library is used, not even in the tests.

Code reused inside this repository (Task 3 is self-contained, so it has its own copies):

```python
# SOURCE: Adapted from task1_financial/equity_research/market_data.py and indicators.py (this repository)
# SOURCE: Adapted from task1_financial/equity_research/news.py (this repository)
```

The Task 3 agent loop follows the standard LangGraph pattern (an agent node, a tools node and a conditional
edge between them) described in the LangGraph documentation; it is written from scratch, not copied.
`web_search` calls the `ddgs` library (the `duckduckgo-search` package under its new name) as a dependency.

## Teacher-model prompts

Not applicable (no synthetic training data in Task 1 or Task 3). All runtime prompts are constants in
`task1_financial/equity_research/prompts.py` and `task3_agentic/research_agents/prompts.py`, and each
notebook prints them in its appendix.
