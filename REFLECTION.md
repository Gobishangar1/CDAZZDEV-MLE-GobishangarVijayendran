# Reflection

Tasks attempted: **Task 1 (Financial AI)** and **Task 3 (Agentic Workflows)**.

## Architectural decisions

**Task 1: a fixed workflow, not an agent.** The steps are known, so a LangGraph workflow is cheaper, testable
node by node, and the same every run. Fetches run in parallel, `Send()` gives each headline its own LLM branch
(one bad reply loses one headline), and the signal has a retry loop with feedback.

**Python does the maths; the LLM only judges.** Indicators, returns, volatility, drawdown and VaR are
deterministic code in both tasks. In Task 1, code turns numbers into facts and lists conflicts ("trend up, but
MACD fading") that the signal must resolve. This is my main lever against answers that only restate values.

**Never trust LLM output.** Pydantic checks the format; a second validator checks meaning (every number must
appear in the facts, conflicts must be addressed). Failures are logged, fed back, retried, and finally replaced
by a rule-based result, so a run always finishes. My first real run showed why: GPT-OSS wrote "52‑week" with a
Unicode hyphen and the validator rejected it, so I normalised the text.

**Task 3: the graph fixes no order; the model decides.** The ReAct graph only says "if the model asked for a
tool, run it and return the result". All tool calls go through one executor that enforces each agent's tools,
turns exceptions and empty results into data with a suggested alternative (only tools that agent has), and writes
the trace. So restriction, error handling and observability live in one place.

**Coordination by design.** Agent A can score sentiment but has no news; Agent B has news but no quantitative
tools. The critique round therefore carries B's headlines to A, and the report schema forces B to say how A's
answer was used. Handoff numbers are copied from tool results by code, so the LLM cannot change them in transit.

**Free-tier engineering.** Groq limits are per model, so GPT-OSS-120B and 20B are each other's fallback. Tool
outputs sent to the model are capped (2,500 characters, five OHLCV rows) to fit about 8K tokens per minute.

## What I would improve with more time

1. Backtest the Task 1 momentum rules with walk-forward splits and calibrate LLM confidence against realised
   returns.
2. Evaluate the agents on a fixed set of tickers and questions (tool choice, grounded numbers, cost) instead of
   judging single runs.
3. Base the hedge on option prices and implied volatility; historical volatility is only a proxy.
4. Persist session memory (SQLite checkpointer) and summarise long conversations to save tokens.
5. Add LangSmith tracing next to the JSONL file, and an LLM-as-judge check for reasoning quality.

## Limitations encountered

- **Free news is noisy.** yfinance returned no news on my first run and Yahoo RSS gave 13 off-topic headlines
  out of 15. A relevance filter and a three-source chain help, but miss articles that do not name the company.
- **DuckDuckGo rate-limits without warning.** The agents treat this as a tool failure and switch to news.
- **Agents are not deterministic.** Two runs can call tools in a different order. The tests use scripted
  models, so they check the mechanics, not the model's judgement.
- **Rate limits** make a full Task 3 notebook run take several minutes.
- **The brief suggests Llama-3 on Groq**, which is no longer on the free tier, so I used GPT-OSS. Model names
  are configuration.
- **Nothing is backtested.** The signal and the hedge are illustrations, not investment advice.
