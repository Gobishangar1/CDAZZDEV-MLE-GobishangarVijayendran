# Reflection

Tasks attempted: **Task 1 (Financial AI)**.

## Architectural decisions

**A fixed workflow, not an agent.** Task 1 has a known order of steps, so I built a LangGraph workflow
instead of a tool-calling agent: cheaper, testable node by node, and the same path every run. LangGraph still earns its place: the three fetches run in parallel, `Send()` creates
one LLM branch per headline (so one bad reply loses one headline, not all of them), joins wait for all
inputs, and the signal has a retry loop with feedback.

**Python does the maths; the LLM only judges.** All five indicators, returns and the momentum score
are deterministic code. The LLM only labels headlines and weighs signals that disagree. Before the
signal call, code turns numbers into facts ("Close is 2.2% below SMA-50", "RSI 48, down from 61 ten
days ago") and lists conflicts ("trend up, but MACD fading"). The prompt requires the model to resolve
each conflict. This is my main lever against "echo" answers that only restate values.

**Validate twice, never trust the LLM.** Pydantic checks the format. A second validator checks the
meaning: every number must appear in the facts, at least two indicators must be combined, conflicts must
be addressed, and confidence is capped when signals conflict. Failures are logged with the raw reply,
fed back into the next attempt, and after three attempts a rule-based signal is used, so the run always
finishes. My first real run showed why this matters: the validator rejected "52" because GPT-OSS writes
"52‑week" with a Unicode hyphen, so I added text normalisation.

**Indicator details I chose deliberately.** RSI uses Wilder's seed (a simple 14-day mean), not plain
`ewm`, which differs by up to about 10 points at the start. Bollinger uses population std (`ddof=0`),
as Bollinger defined it. A 300-day warm-up makes the displayed window independent of the seeds.
Anomalies are flagged, not removed: in equities a big move is usually news, unlike sensor data.

**One free provider, two models.** Groq's free-tier limits are per model, so GPT-OSS-20B (headlines)
and GPT-OSS-120B (signal) are each other's fallback with one key.

## What I would improve with more time

1. **Backtest the momentum rules** with walk-forward splits. The weights are transparent starting
   values; tuning them on the same history would overfit.
2. **Calibrate the LLM confidence** against realised returns, so 0.7 means something measurable.
3. **Read full articles, not only headlines**, and add a second sentiment opinion (FinBERT) to measure
   agreement with the LLM.
4. **An LLM-as-judge check** for reasoning quality. The anti-echo check uses linking words, which is
   cheap but can be gamed.
5. **Earnings dates** as a feature, since earnings drive many 1-3 month moves.

## Limitations encountered

- **Free news is noisy.** yfinance returned no news on my first real run, and Yahoo's RSS feed gave 13
  off-topic headlines out of 15. I added a relevance filter (ticker or company name) and a three-source
  chain; it still misses headlines that discuss the company without naming it.
- **Single provider risk.** If Groq is down, both models fail; the rule-based fallback still produces a
  signal, but without LLM reasoning.
- **Free-tier rate limits** (about 30 requests and 8K tokens per minute per model) make the headline
  step slow, so calls are rate-limited and concurrency is capped at three.
- **The brief suggests Llama-3 on Groq**, but those models are no longer on Groq's free tier, so I used
  GPT-OSS. Model names are configuration.
- **No proof of skill.** Indicators use only past data, but the signal is not backtested.
