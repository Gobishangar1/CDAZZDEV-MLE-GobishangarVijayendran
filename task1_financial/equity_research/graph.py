"""The LangGraph workflow that connects every step (see the Mermaid diagram in the README).

Why LangGraph for a fixed pipeline?
    - parallel branches: prices, company info and news are fetched at the same time
    - Send() fan-out: one classify_headline branch per headline (count known only at runtime)
    - joins: build_summary and build_context wait for ALL their inputs
    - a retry loop: generate_signal -> generate_signal with the validator's feedback
    - node_guard: an exception inside a node becomes an error record, never a crash

Rule used everywhere: nodes WRITE state; routers only READ state and return the next node.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph StateGraph for the Task 1 pipeline:
# parallel fetches, Send fan-out per headline with reducers, two joins, a validation retry loop with
# feedback, rule-based fallback, report node, error report, node_guard', Date: 2026-10-11

from __future__ import annotations

import functools
import json
import logging
import operator
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from equity_research import tracing
from equity_research.analysis import build_summary, indicator_snapshot, momentum_signal
from equity_research.config import MAX_CONCURRENCY, MAX_SIGNAL_ATTEMPTS
from equity_research.indicators import add_indicators
from equity_research.market_data import fetch_info, fetch_prices, normalize_ticker, probe_ticker
from equity_research.news import fetch_news
from equity_research.report import write_report
from equity_research.sentiment import aggregate_sentiment, classify_headline
from equity_research.signal_context import build_facts, find_conflicts
from equity_research.trade_signal import attempt_signal, rule_based_fallback, signal_prompt_values

log = logging.getLogger(__name__)


class EquityState(TypedDict, total=False):
    # input
    ticker: str
    output_dir: str | None
    # data from the three parallel branches
    df: pd.DataFrame                 # in memory only (no checkpointer is used)
    info: dict
    headlines: list[dict]
    # derived
    snapshot: dict
    summary: dict
    sentiment: dict | None
    facts: list[str]
    conflicts: list[str]
    # signal loop
    signal: dict | None
    signal_attempts: int
    signal_problems: list[str]
    signal_history: list[dict]
    signal_source: str
    # outputs
    report_paths: dict[str, str]
    final: dict
    # Reducers: many branches write these in the SAME step, so their lists are concatenated.
    headline_results: Annotated[list[dict], operator.add]
    headline_failures: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[dict], operator.add]


def node_guard(name: str, critical: bool = False):
    """Wrap a node: log its duration; turn any exception into an error record (no traceback)."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(state):
            t0 = time.perf_counter()
            try:
                out = fn(state)
                log.debug("node_ok node=%s ms=%d", name, (time.perf_counter() - t0) * 1000)
                return out
            except Exception as exc:
                log.error("node_failed node=%s critical=%s error=%s: %s", name, critical, type(exc).__name__, exc)
                log.debug("node_failed traceback", exc_info=True)
                return {"errors": [{"node": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                    "critical": critical}]}
        return inner
    return wrap


def traced(name: str, fn):
    """Record one 'node' trace event (duration, ok/error) for every node run."""
    @functools.wraps(fn)
    def inner(state):
        t0 = time.perf_counter()
        out = fn(state)
        status = "error" if isinstance(out, dict) and out.get("errors") else "ok"
        tracing.record("node", node=name, duration_ms=round((time.perf_counter() - t0) * 1000), status=status)
        return out
    return inner


def run_dir_for(output_dir: str | None, ticker: str, as_of: str) -> Path | None:
    return Path(output_dir) / f"{ticker}_{as_of}" if output_dir else None


def _has_critical(state: dict) -> bool:
    return any(e.get("critical") for e in state.get("errors") or [])


def _jsonable_headline(h: dict) -> dict:
    return {**h, "published_at": h["published_at"].isoformat() if h.get("published_at") else None}


def build_graph(fast_llm: Any, strong_llm: Any):
    """Compile the workflow. LLMs are passed in, so tests and the notebook demo can use fakes."""

    # ------------------------------------------------------------------ nodes
    @node_guard("validate_input", critical=True)
    def validate_input(state):
        ticker = normalize_ticker(state["ticker"])
        if not probe_ticker(ticker):
            raise ValueError(f"No recent prices for '{ticker}' (unknown or delisted ticker)")
        return {"ticker": ticker}

    @node_guard("fetch_prices", critical=True)
    def fetch_prices_node(state):
        df, warnings = fetch_prices(state["ticker"])
        return {"df": df, "warnings": warnings}

    @node_guard("fetch_info")
    def fetch_info_node(state):
        info = fetch_info(state["ticker"])
        return {"info": info, "warnings": [] if info else ["Company info unavailable (P/E, sector, name)"]}

    @node_guard("fetch_news")
    def fetch_news_node(state):                        # runs after fetch_info: needs the company name
        headlines, warnings = fetch_news(state["ticker"], (state.get("info") or {}).get("longName"))
        return {"headlines": headlines, "warnings": warnings}

    @node_guard("compute_indicators", critical=True)
    def compute_indicators(state):
        return {"df": add_indicators(state["df"])}

    @node_guard("build_summary", critical=True)
    def build_summary_node(state):
        snapshot = indicator_snapshot(state["df"])
        summary = build_summary(state["ticker"], state["df"], state.get("info") or {}, momentum_signal(snapshot))
        return {"snapshot": snapshot, "summary": summary.model_dump()}

    def dispatch_headlines(state):
        """Join point: waits for news, prices AND info. A router only sees its own branch's writes,
        so without this join a price failure in a sibling branch would not stop the headline LLM calls."""
        return {}

    @node_guard("classify_headline")
    def classify_headline_node(payload):          # receives the small Send payload, not the full state
        item = payload["item"]
        result, failure = classify_headline(fast_llm, payload["ticker"], payload.get("company"), item)
        if result is None:
            return {"headline_failures": [failure]}
        return {"headline_results": [{**result.model_dump(), "publisher": item.get("publisher"),
                                      "published_at": item.get("published_at"), "url": item.get("url"),
                                      "source": item.get("source")}]}

    @node_guard("aggregate_sentiment")
    def aggregate_node(state):
        total = len(state.get("headlines") or [])
        sentiment = aggregate_sentiment(state.get("headline_results") or [], datetime.now(timezone.utc), total)
        return {"sentiment": sentiment.model_dump() if sentiment else None}

    @node_guard("build_context", critical=True)
    def build_context(state):
        snapshot, summary, sentiment = state["snapshot"], state["summary"], state.get("sentiment")
        return {"facts": build_facts(snapshot, summary, sentiment),
                "conflicts": find_conflicts(snapshot, summary["momentum"], sentiment),
                "signal_attempts": 0, "signal_problems": [], "signal_history": []}

    def generate_signal(state):                   # one attempt + validation; never raises
        attempt = state["signal_attempts"] + 1
        summary = state["summary"]
        values = signal_prompt_values(state["ticker"], summary["as_of"], state["facts"], state["conflicts"],
                                      summary["momentum"], state.get("sentiment"))
        sig, problems, raw = attempt_signal(strong_llm, values, state["facts"], state["conflicts"],
                                            state["signal_problems"], summary["momentum"]["score"], attempt)
        if problems:
            log.warning("signal_invalid attempt=%d problems=%s", attempt, problems)
        else:
            log.info("signal_ok attempt=%d signal=%s", attempt, sig.signal)
        return {"signal": sig.model_dump() if sig else None, "signal_source": "llm",
                "signal_attempts": attempt, "signal_problems": problems,
                "signal_history": state["signal_history"] + [{"attempt": attempt, "problems": problems,
                                                              "raw_output": raw[:300]}]}

    def rule_fallback(state):
        sig = rule_based_fallback(state["conflicts"], state["summary"]["momentum"], state.get("sentiment"))
        log.warning("signal_fallback signal=%s after %d attempts", sig.signal, state["signal_attempts"])
        return {"signal": sig.model_dump(), "signal_source": "rule_based_fallback"}

    def _final(state) -> dict:
        return {
            "status": "ok",
            "summary": state["summary"],
            "info": state.get("info") or {},
            "headlines": [_jsonable_headline(h) for h in state.get("headlines") or []],
            "headline_results": [_jsonable_headline(h) for h in state.get("headline_results") or []],
            "headline_failures": state.get("headline_failures") or [],
            "sentiment": state.get("sentiment"),
            "facts": state["facts"],
            "conflicts": state["conflicts"],
            "signal": state["signal"],
            "signal_source": state.get("signal_source", "llm"),
            "signal_history": state.get("signal_history") or [],
            "warnings": state.get("warnings") or [],
            "errors": state.get("errors") or [],
            "report_paths": state.get("report_paths") or {},
        }

    @node_guard("render_report")
    def render_report(state):
        run_dir = run_dir_for(state.get("output_dir"), state["ticker"], state["summary"]["as_of"])
        if run_dir is None:
            return {}
        return {"report_paths": write_report(_final(state), state["df"], run_dir)}

    def finalize(state):                               # run.json and trace.jsonl are written by run_pipeline
        return {"final": _final(state)}

    def error_report(state):
        errors = state.get("errors") or []
        message = "; ".join(e["error"] for e in errors if e.get("critical")) or "Unknown critical error"
        log.error("run_stopped ticker=%s reason=%s", state.get("ticker"), message)
        return {"final": {"status": "error", "ticker": state.get("ticker"), "message": message, "errors": errors}}

    # ---------------------------------------------------------------- routers
    def route_after_validate(state):
        return "error_report" if _has_critical(state) else ["fetch_prices", "fetch_info"]

    def route_after_prices(state):
        return "error_report" if _has_critical(state) or "df" not in state else "compute_indicators"

    def route_headlines(state):
        headlines = state.get("headlines") or []
        if not headlines or _has_critical(state):      # no news, or the run is already stopping
            return "aggregate_sentiment"
        company = (state.get("info") or {}).get("longName")
        return [Send("classify_headline", {"item": h, "ticker": state["ticker"], "company": company})
                for h in headlines]

    def route_after_context(state):
        return "error_report" if _has_critical(state) else "generate_signal"

    def route_after_signal(state):
        if not state["signal_problems"]:
            return "render_report"
        if state["signal_attempts"] >= MAX_SIGNAL_ATTEMPTS:
            return "rule_fallback"
        return "generate_signal"                        # retry WITH the problem list as feedback

    # ----------------------------------------------------------------- wiring
    b = StateGraph(EquityState)
    for name, fn in [
        ("validate_input", validate_input), ("fetch_prices", fetch_prices_node), ("fetch_info", fetch_info_node),
        ("fetch_news", fetch_news_node), ("dispatch_headlines", dispatch_headlines),
        ("compute_indicators", compute_indicators),
        ("build_summary", build_summary_node), ("classify_headline", classify_headline_node),
        ("aggregate_sentiment", aggregate_node), ("build_context", build_context),
        ("generate_signal", generate_signal), ("rule_fallback", rule_fallback),
        ("render_report", render_report), ("finalize", finalize), ("error_report", error_report),
    ]:
        b.add_node(name, traced(name, fn))

    b.add_edge(START, "validate_input")
    b.add_conditional_edges("validate_input", route_after_validate,
                            ["fetch_prices", "fetch_info", "error_report"])                 # list -> parallel
    b.add_edge("fetch_info", "fetch_news")         # news needs the company name to filter relevant headlines
    b.add_conditional_edges("fetch_prices", route_after_prices, ["compute_indicators", "error_report"])
    b.add_edge(["compute_indicators", "fetch_info"], "build_summary")                        # join
    b.add_edge(["fetch_news", "fetch_prices", "fetch_info"], "dispatch_headlines")           # join
    b.add_conditional_edges("dispatch_headlines", route_headlines, ["classify_headline", "aggregate_sentiment"])
    b.add_edge("classify_headline", "aggregate_sentiment")
    b.add_edge(["build_summary", "aggregate_sentiment"], "build_context")                    # join
    b.add_conditional_edges("build_context", route_after_context, ["generate_signal", "error_report"])
    b.add_conditional_edges("generate_signal", route_after_signal,
                            ["render_report", "generate_signal", "rule_fallback"])           # retry loop
    b.add_edge("rule_fallback", "render_report")
    b.add_edge("render_report", "finalize")
    b.add_edge("finalize", END)
    b.add_edge("error_report", END)
    return b.compile()


def run_pipeline(ticker: str, fast_llm: Any = None, strong_llm: Any = None,
                 output_dir: str | None = "outputs", verbose: bool = True) -> dict:
    """Run the whole workflow for one ticker and return the final result dict.

    verbose=True prints each node as it finishes (elapsed time) and a trace summary at the end.
    With an output_dir, writes <output_dir>/<TICKER>_<as_of>/run.json and trace.jsonl
    (the brief, chart and Markdown are written by the render_report node).
    """
    if fast_llm is None or strong_llm is None:
        from equity_research.llm import make_fast_llm, make_strong_llm
        fast_llm = fast_llm or make_fast_llm()
        strong_llm = strong_llm or make_strong_llm()

    graph = build_graph(fast_llm, strong_llm)
    trace = tracing.start()
    t0, final = time.perf_counter(), None
    try:
        for update in graph.stream({"ticker": ticker, "output_dir": output_dir},
                                   config={"max_concurrency": MAX_CONCURRENCY}, stream_mode="updates"):
            for node, out in update.items():
                if verbose:
                    print(f"[{time.perf_counter() - t0:6.1f}s] {node}")
                if out and "final" in out:
                    final = out["final"]
    finally:
        tracing.stop()

    final = final or {"status": "error", "ticker": ticker, "message": "Graph ended without a result"}
    final["trace_summary"] = trace.summary()
    if output_dir and final["status"] == "ok":
        run_dir = run_dir_for(output_dir, final["summary"]["ticker"], final["summary"]["as_of"])
        final["report_paths"]["trace"] = str(trace.write_jsonl(run_dir / "trace.jsonl"))
        final["report_paths"]["run_json"] = str(run_dir / "run.json")
        (run_dir / "run.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    if verbose:
        t = final["trace_summary"]
        print(f"\nTrace: {t['wall_time_s']}s | {t['llm_calls']} LLM calls ({t['llm_api_errors']} API errors, "
              f"{t['validation_failures']} failed validations) | tokens in {t['input_tokens']:,} / out "
              f"{t['output_tokens']:,} | models {t['models'] or '-'}")
    return final
