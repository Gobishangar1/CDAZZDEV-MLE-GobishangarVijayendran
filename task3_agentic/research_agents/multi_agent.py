"""Task 3B: two agents with restricted tools, a typed handoff and one critique round. 3C: persistent cache.

    analyst            Agent A (get_price_data, calculate_volatility, llm_sentiment) -> DataBrief
    writer_research    Agent B (web_search, get_news) researches with the DataBrief
    writer_critique    Agent B -> Agent A: ONE ClarificationRequest (e.g. "score these headlines")
    analyst_clarify    Agent A answers with its own tools -> ClarificationResponse
    writer_report      Agent B writes the final ResearchReport and must say how it used the answer

Why they need each other: Agent A can score sentiment but cannot fetch news; Agent B has the news but no
sentiment, price or volatility tools. The final report needs both.

run_research() wraps the graph: if memory/<TICKER>_<date>.json exists it is loaded (no tools run);
otherwise the pipeline runs end to end with no manual step, and the result is saved.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field

from research_agents.agent import AgentRun, ReActAgent, evidence_from
from research_agents.config import AGENT_MAX_STEPS, RESEARCH_QUERY
from research_agents.llm import invoke_json
from research_agents.memory import BriefStore
from research_agents.prompts import (
    CRITIQUE_HUMAN,
    CRITIQUE_SYSTEM,
    DATA_ANALYST_CLARIFY,
    DATA_ANALYST_SYSTEM,
    REPORT_REPAIR,
    RESEARCH_WRITER_SYSTEM,
    RESEARCH_WRITER_TASK,
)
from research_agents.report import write_report
from research_agents.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    ResearchReport,
)
from research_agents.tools import ToolBox

AGENT_A, AGENT_B = "Agent A (Data Analyst)", "Agent B (Research Writer)"
TOOLS_A = ["get_price_data", "calculate_volatility", "llm_sentiment"]
TOOLS_B = ["web_search", "get_news"]


class MultiAgentReport(ResearchReport):
    """In 3B the report MUST state how Agent A's clarification was used."""

    clarification_used: str = Field(min_length=20)


class PipelineState(TypedDict, total=False):
    ticker: str
    query: str
    as_of: str
    brief: dict[str, Any]
    research_notes: str
    research_evidence: str
    headlines: list[str]
    request: dict[str, Any]
    response: dict[str, Any]
    report: dict[str, Any] | None
    error: str | None


# -----------------------------------------------------------------------------
# Building the typed messages from agent runs (numbers copied by code, not retyped by the LLM)
# -----------------------------------------------------------------------------
def build_data_brief(ticker: str, as_of: str, run: AgentRun) -> DataBrief:
    price, vol, sent = (run.ok_results("get_price_data"), run.ok_results("calculate_volatility"),
                        run.ok_results("llm_sentiment"))
    gaps = [f"{r.tool} failed: {r.text[:120]}" for r in run.records if r.status in ("error", "denied")]
    if not price:
        gaps.append("no price data")
    if not vol:
        gaps.append("no volatility")
    if not sent:
        gaps.append("news sentiment not available (Agent A has no news access)")
    p = price[-1] if price else None
    return DataBrief(
        ticker=ticker, as_of=as_of,
        price=None if p is None else {k: p.get(k) for k in ("period", "last_close", "period_return_pct", "high_52w",
                                                           "low_52w", "pct_from_52w_high", "trend", "indicators")},
        volatility=vol[-1] if vol else None,
        sentiment=None if not sent else {k: sent[-1][k] for k in ("score", "label", "n", "counts")},
        analyst_notes=run.final_text.strip() or "(no notes)", data_gaps=gaps,
        tools_used=sorted({r.tool for r in run.records}))


def headlines_from(run: AgentRun) -> list[str]:
    titles: list[str] = []
    for result in run.ok_results("get_news"):
        titles += [h["title"] for h in result.get("headlines", [])]
    for result in run.ok_results("web_search"):
        titles += [h["title"] for h in result.get("results", [])]
    return list(dict.fromkeys(titles))[:15]


def compact_result(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    """The part of a tool result that Agent B needs (sentiment: totals + the headlines behind each side)."""
    if tool == "llm_sentiment":
        by_label = {k: [i["headline"] for i in result.get("items", []) if i["sentiment"] == k]
                    for k in ("negative", "positive")}
        return {"score": result["score"], "label": result["label"], "n": result["n"], "counts": result["counts"],
                "negative_headlines": by_label["negative"], "positive_headlines": by_label["positive"]}
    if tool == "get_price_data":
        return {k: result.get(k) for k in ("period", "last_close", "period_return_pct", "pct_from_52w_high", "trend")}
    return result


def fallback_request(state: PipelineState) -> ClarificationRequest:
    """Used only if Agent B's critique output fails validation 3 times: the loop still runs once."""
    return ClarificationRequest(
        question="Please score the sentiment of the attached headlines and give the 60-day annualised volatility.",
        requested_data=["llm_sentiment of the headlines", "calculate_volatility window=60"],
        headlines=state.get("headlines", [])[:15],
        reason="The data brief has no news sentiment, and the hedge needs volatility over a horizon near 90 days.")


# -----------------------------------------------------------------------------
# The pipeline graph
# -----------------------------------------------------------------------------
def build_pipeline(llm: Any, json_llm: Any, toolbox: ToolBox, tracer: Any, max_steps: int = AGENT_MAX_STEPS):
    agent_a = ReActAgent(AGENT_A, TOOLS_A, DATA_ANALYST_SYSTEM.format(tools=", ".join(TOOLS_A)), llm, toolbox,
                         tracer, max_steps)
    agent_b = ReActAgent(AGENT_B, TOOLS_B, RESEARCH_WRITER_SYSTEM.format(tools=", ".join(TOOLS_B)), llm, toolbox,
                         tracer, max_steps)

    def analyst(state: PipelineState) -> dict:
        tracer.banner(f"{AGENT_A}: quantitative analysis")
        run = agent_a.run(f"Build a quantitative data brief for {state['ticker']} for a 90-day risk review: price "
                          f"trend and indicators, volatility, drawdown and value-at-risk.")
        brief = build_data_brief(state["ticker"], state["as_of"], run)
        tracer.handoff(AGENT_A, AGENT_B, "DataBrief", brief.model_dump(mode="json"))
        return {"brief": brief.model_dump(mode="json")}

    def writer_research(state: PipelineState) -> dict:
        tracer.banner(f"{AGENT_B}: qualitative research")
        run = agent_b.run(RESEARCH_WRITER_TASK.format(query=state["query"], brief=json.dumps(state["brief"])))
        return {"research_notes": run.final_text, "research_evidence": evidence_from(run.records),
                "headlines": headlines_from(run)}

    def writer_critique(state: PipelineState) -> dict:
        tracer.banner(f"{AGENT_B}: critique of the data brief")
        try:
            request = invoke_json(json_llm, CRITIQUE_SYSTEM.format(),
                                  CRITIQUE_HUMAN.format(brief=json.dumps(state["brief"]),
                                                        notes=state.get("research_notes", ""),
                                                        headlines="\n".join(state.get("headlines", [])) or "(none)"),
                                  ClarificationRequest, REPORT_REPAIR, label="clarification_request", tracer=tracer)
        except Exception as exc:
            tracer.note("critique_fallback", f"critique output invalid ({type(exc).__name__}); using a default request")
            request = fallback_request(state)
        if not request.headlines and any("sentiment" in d.lower() for d in request.requested_data):
            request = request.model_copy(update={"headlines": state.get("headlines", [])[:15]})
        tracer.handoff(AGENT_B, AGENT_A, "ClarificationRequest", request.model_dump(mode="json"))
        return {"request": request.model_dump(mode="json")}

    def analyst_clarify(state: PipelineState) -> dict:
        tracer.banner(f"{AGENT_A}: answering the clarification")
        req = ClarificationRequest(**state["request"])
        task = DATA_ANALYST_CLARIFY.format(
            question=req.question, requested_data="; ".join(req.requested_data), reason=req.reason,
            headlines="\n".join(f"- {h}" for h in req.headlines) or "(none)")
        run = agent_a.run(f"{task}\n\nYour earlier brief for context: {json.dumps(state['brief'])[:1500]}")
        data: dict[str, Any] = {}
        for r in run.records:
            if r.status == "ok" and r.result:
                data[r.tool if r.tool not in data else f"{r.tool}_{len(data)}"] = compact_result(r.tool, r.result)
        response = ClarificationResponse(question=req.question, answer=run.final_text.strip() or "(no answer)",
                                         data=data, tools_used=sorted({r.tool for r in run.records}))
        tracer.handoff(AGENT_A, AGENT_B, "ClarificationResponse", response.model_dump(mode="json"))
        return {"response": response.model_dump(mode="json")}

    def writer_report(state: PipelineState) -> dict:
        tracer.banner(f"{AGENT_B}: final report")
        evidence = (f"DATA BRIEF FROM AGENT A:\n{json.dumps(state['brief'])}\n\n"
                    f"AGENT B TOOL RESULTS:\n{state.get('research_evidence', '')}\n\n"
                    f"AGENT B NOTES:\n{state.get('research_notes', '')}\n\n"
                    f"CLARIFICATION (B asked A): {json.dumps(state['request'])}\n"
                    f"AGENT A ANSWER: {json.dumps(state['response'])[:3000]}")
        try:
            report = write_report(json_llm, state["ticker"], state["as_of"], state["query"], evidence, tracer,
                                  schema=MultiAgentReport,
                                  clarification_hint='"<how Agent A\'s answer to the clarification changed or '
                                                     'confirmed this report, with the numbers>"')
            return {"report": report.model_dump(mode="json"), "error": None}
        except Exception as exc:
            tracer.note("report_failed", f"{type(exc).__name__}: {str(exc)[:300]}")
            return {"report": None, "error": str(exc)}

    g = StateGraph(PipelineState)
    for name, fn in [("analyst", analyst), ("writer_research", writer_research), ("writer_critique", writer_critique),
                     ("analyst_clarify", analyst_clarify), ("writer_report", writer_report)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyst")
    g.add_edge("analyst", "writer_research")
    g.add_edge("writer_research", "writer_critique")
    g.add_edge("writer_critique", "analyst_clarify")     # B -> A (the critique loop, exactly one round)
    g.add_edge("analyst_clarify", "writer_report")        # A -> B
    g.add_edge("writer_report", END)
    return g.compile()


# -----------------------------------------------------------------------------
# Entry point with persistent memory
# -----------------------------------------------------------------------------
def run_research(ticker: str, llm: Any = None, json_llm: Any = None, sentiment_llm: Any = None,
                 tracer: Any = None, store: BriefStore | None = None, use_cache: bool = True,
                 search_backend: Any = None, today: date | None = None) -> dict[str, Any]:
    """Full multi-agent research for one ticker. Returns the saved record (with "from_cache")."""
    from research_agents.trace import Tracer

    ticker = ticker.strip().upper()
    tracer = tracer or Tracer()
    store = store or BriefStore()
    day = today or date.today()
    if use_cache:
        cached = store.load(ticker, day)
        if cached is not None:
            tracer.note("cache_hit", f"Loaded {store.path(ticker, day).name} from memory; no agents or tools run",
                        ticker=ticker, path=str(store.path(ticker, day)))
            return {**cached, "from_cache": True, "path": str(store.path(ticker, day))}

    if llm is None or json_llm is None or sentiment_llm is None:
        from research_agents.llm import (
            make_agent_llm,
            make_json_llm,
            make_sentiment_llm,
        )
        llm, json_llm, sentiment_llm = (llm or make_agent_llm(), json_llm or make_json_llm(),
                                        sentiment_llm or make_sentiment_llm())
    toolbox = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_backend)
    pipeline = build_pipeline(llm, json_llm, toolbox, tracer)
    state = pipeline.invoke({"ticker": ticker, "query": RESEARCH_QUERY.format(ticker=ticker), "as_of": day.isoformat()})

    record = {"ticker": ticker, "date": day.isoformat(), "query": state["query"], "run_id": tracer.run_id,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "report": state.get("report"), "data_brief": state.get("brief"),
              "clarification": {"request": state.get("request"), "response": state.get("response")},
              "trace_summary": tracer.summary(), "error": state.get("error")}
    path = None
    if record["report"]:
        path = store.save(ticker, day, record)
        tracer.note("cache_saved", f"Saved brief to {path.name}", path=str(path))
    return {**record, "from_cache": False, "path": str(path) if path else None}
