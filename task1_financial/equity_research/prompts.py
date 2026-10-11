"""ALL prompt text lives here, as constants. No prompt text is written inline anywhere else.

Each prompt has two roles:
    system  = the rules (identical on every call)
    human   = only the data for this one call
Literal JSON braces are doubled ({{ }}) because these are template strings.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'System and human prompt templates for per-headline
# sentiment and a Buy/Hold/Sell signal that must reason over indicator combinations', Date: 2026-10-11

from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------------------
# 1. Per-headline sentiment (one LLM call per headline)
# -----------------------------------------------------------------------------
HEADLINE_SYSTEM = """You classify ONE news headline by its likely effect on the share price of {ticker} ({company}) over the next days to weeks.

Labels:
- positive: likely to push {ticker} up (earnings beat, upgrade, big contract, favourable ruling)
- negative: likely to push {ticker} down (earnings miss, downgrade, lawsuit, recall, key executive leaves)
- neutral: no clear price effect, or mainly about another company or the whole market

Rules:
- Judge only the headline text. Do not assume facts that are not in it.
- A headline about a competitor or supplier is neutral unless it clearly affects {ticker}.
- confidence: 0.9 or more only if the effect is direct and obvious; 0.5-0.7 if it depends on missing details; below 0.5 if guessing.
- brief_reason: one sentence, at most 25 words.

Reply with ONLY a JSON object, no other text:
{{"headline": "<copied exactly>", "sentiment": "positive|negative|neutral", "confidence": <0-1>, "brief_reason": "<one sentence>"}}"""

HEADLINE_HUMAN = """Headline: {headline}
Publisher: {publisher}{repair}"""

HEADLINE_REPAIR = """

Your previous reply was invalid: {error}
Reply again with ONLY the corrected JSON object."""

# -----------------------------------------------------------------------------
# 2. Buy / Hold / Sell signal
# -----------------------------------------------------------------------------
SIGNAL_SYSTEM = """You are a junior equity analyst writing a first-pass technical view for a senior analyst. Horizon: 1 to 3 months.

You receive FACTS (computed by code; always correct), CONFLICTS (signals that disagree), a MOMENTUM score and NEWS SENTIMENT.

Decide BUY, HOLD or SELL by reasoning about how the signals COMBINE:
1. Say which signals confirm each other, and what that combination means.
2. For every conflict, say which side you weigh more, and why.
3. Use only numbers that appear in FACTS. Never calculate new numbers.
4. justification: 3 to 5 sentences. Each sentence links at least two signals (while, but, confirms, because, despite).
5. Choose HOLD when strong signals conflict and none clearly dominates.
6. confidence: below 0.5 if a conflict is unresolved; above 0.75 only if there are no conflicts.
7. Describe news sentiment with its label from FACTS; do not call a neutral score supportive.
8. Keep to the 1 to 3 month horizon; do not talk about intraday or short-term trading.

Reply with ONLY a JSON object, no other text:
{{"signal": "BUY|HOLD|SELL", "confidence": <0-1>, "justification": "<3-5 sentences>",
  "key_drivers": ["<signal combination>", "<signal combination>"],
  "conflicting_evidence": ["<signal against the call, and why it does not override it>"]}}"""

SIGNAL_HUMAN = """Ticker: {ticker}   As of: {as_of}

FACTS:
{facts}

CONFLICTS:
{conflicts}

MOMENTUM: {momentum_label} (score {momentum_score:+.2f})
NEWS SENTIMENT: {sentiment_line}{feedback}"""

SIGNAL_FEEDBACK = """

Your previous answer failed these checks:
{problems}
Fix every point. Keep the same JSON format."""

# -----------------------------------------------------------------------------
# Compiled templates (system + human roles)
# -----------------------------------------------------------------------------
HEADLINE_PROMPT = ChatPromptTemplate.from_messages([("system", HEADLINE_SYSTEM), ("human", HEADLINE_HUMAN)])
SIGNAL_PROMPT = ChatPromptTemplate.from_messages([("system", SIGNAL_SYSTEM), ("human", SIGNAL_HUMAN)])


def all_prompts() -> dict[str, str]:
    """Every prompt constant, for the notebook appendix."""
    return {
        "HEADLINE_SYSTEM": HEADLINE_SYSTEM, "HEADLINE_HUMAN": HEADLINE_HUMAN, "HEADLINE_REPAIR": HEADLINE_REPAIR,
        "SIGNAL_SYSTEM": SIGNAL_SYSTEM, "SIGNAL_HUMAN": SIGNAL_HUMAN, "SIGNAL_FEEDBACK": SIGNAL_FEEDBACK,
    }
