# CDAZZDEV Senior ML Engineer Assessment: Gobishangar Vijayendran

| Task | Folder | Notebook |
|---|---|---|
| Task 1: Financial AI (LLM-powered equity research pipeline) | [`task1_financial/`](task1_financial/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gobishangar1/CDAZZDEV-MLE-GobishangarVijayendran/blob/main/task1_financial/notebooks/task1_equity_research.ipynb) |

## Task 1 in one paragraph

A LangGraph workflow fetches two years of daily OHLCV data for a ticker (yfinance), computes SMA-50,
SMA-200, RSI-14, MACD (12, 26, 9) and Bollinger Bands (20, 2σ) from first principles, collects 10+
recent headlines from free sources, and builds the required summary dictionary. An LLM on Groq then
classifies each headline into validated JSON, and a second LLM call produces a Buy / Hold / Sell signal
that must reason over combinations of indicators. Every LLM output is validated (Pydantic + semantic
checks); failures are logged, repaired, retried, and finally replaced by a rule-based fallback. The run
ends with a one-page HTML research brief with an embedded chart.

Details, the workflow diagram and the rubric map are in [`task1_financial/README.md`](task1_financial/README.md).

## Repository files

| File | Purpose |
|---|---|
| [`CITATIONS.md`](CITATIONS.md) | all AI-assisted code and documents |
| [`REFLECTION.md`](REFLECTION.md) | design decisions, improvements, limitations (≤ 600 words) |
| `.env.example` | the one variable needed (`GROQ_API_KEY`); real keys are never committed |

## Quick start

```bash
git clone https://github.com/Gobishangar1/CDAZZDEV-MLE-GobishangarVijayendran.git
cd CDAZZDEV-MLE-GobishangarVijayendran/task1_financial
pip install -r requirements.txt
python -m pytest                      # offline tests
export GROQ_API_KEY=...               # free key from console.groq.com
python -m equity_research NVDA        # writes outputs/NVDA_<date>/brief.html
```
