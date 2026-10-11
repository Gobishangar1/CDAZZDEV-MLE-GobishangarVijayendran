# CDAZZDEV Senior ML Engineer Assessment: Gobishangar Vijayendran

| Task | Folder | Notebook |
|---|---|---|
| Task 1: Financial AI (LLM-powered equity research pipeline) | [`task1_financial/`](task1_financial/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gobishangar1/CDAZZDEV-MLE-GobishangarVijayendran/blob/main/task1_financial/notebooks/task1_equity_research.ipynb) |
| Task 3: Agentic Workflows (multi-agent financial research system) | [`task3_agentic/`](task3_agentic/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gobishangar1/CDAZZDEV-MLE-GobishangarVijayendran/blob/main/task3_agentic/notebooks/task3_agentic_research.ipynb) |

## Task 1 in one paragraph

A LangGraph workflow fetches two years of daily OHLCV data for a ticker (yfinance), computes SMA-50,
SMA-200, RSI-14, MACD (12, 26, 9) and Bollinger Bands (20, 2σ) from first principles, collects 10+
recent headlines from free sources, and builds the required summary dictionary. An LLM on Groq then
classifies each headline into validated JSON, and a second LLM call produces a Buy / Hold / Sell signal
that must reason over combinations of indicators. Every LLM output is validated (Pydantic + semantic
checks); failures are logged, repaired, retried, and finally replaced by a rule-based fallback. The run
ends with a one-page HTML research brief with an embedded chart.

Details, the workflow diagram and the rubric map are in [`task1_financial/README.md`](task1_financial/README.md).

## Task 3 in one paragraph

A LangGraph system answers the brief's research query on its own. In 3A, one ReAct agent has five tools
(price data with indicators, news, annualised volatility, LLM sentiment, DuckDuckGo search); the code fixes no
tool order, so the model chooses each tool and argument from what it has observed, and failed or empty tools come
back as data with a suggested alternative. In 3B, a Data Analyst (price, volatility, sentiment) and a Research
Writer (search, news) have enforced, separate tools, pass Pydantic messages, and run one critique round in which
the writer sends its headlines to the analyst for scoring. In 3C, a checkpointer gives session memory, the final
brief is cached as `memory/<TICKER>_<date>.json`, and every tool call is logged to
[`task3_agentic/logs/agent_trace.jsonl`](task3_agentic/logs/agent_trace.jsonl). A Streamlit dashboard reads the trace.

Details, diagrams and the rubric map are in [`task3_agentic/README.md`](task3_agentic/README.md).

## Repository files

| File | Purpose |
|---|---|
| [`CITATIONS.md`](CITATIONS.md) | all AI-assisted code and documents |
| [`REFLECTION.md`](REFLECTION.md) | design decisions, improvements, limitations for both tasks (≤ 600 words) |
| `.env.example` | the one variable needed (`GROQ_API_KEY`); real keys are never committed |

## Quick start

```bash
git clone https://github.com/Gobishangar1/CDAZZDEV-MLE-GobishangarVijayendran.git
cd CDAZZDEV-MLE-GobishangarVijayendran/task1_financial
pip install -r requirements.txt
python -m pytest                      # offline tests
export GROQ_API_KEY=...               # free key from console.groq.com
python -m equity_research NVDA        # writes outputs/NVDA_<date>/brief.html

cd ../task3_agentic
pip install -r requirements.txt
python -m pytest                      # offline tests
python -m research_agents NVDA        # two-agent pipeline; trace in logs/agent_trace.jsonl
```
