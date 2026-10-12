"""Final report: evidence -> validated ResearchReport (JSON + Pydantic + repair), and Markdown rendering."""

from __future__ import annotations

from typing import Any

from research_agents.config import DISCLAIMER
from research_agents.llm import invoke_json
from research_agents.prompts import REPORT_HUMAN, REPORT_REPAIR, REPORT_SYSTEM
from research_agents.schemas import ResearchReport


def write_report(json_llm: Any, ticker: str, as_of: str, query: str, evidence: str, tracer: Any,
                 schema: type[ResearchReport] = ResearchReport, clarification_hint: str = "null") -> ResearchReport:
    report = invoke_json(json_llm, REPORT_SYSTEM.format(ticker=ticker, as_of=as_of,
                                                        clarification_hint=clarification_hint),
                         REPORT_HUMAN.format(query=query, evidence=evidence), schema, REPORT_REPAIR,
                         label="research_report", tracer=tracer,
                         fixed={"ticker": ticker, "as_of": as_of, "disclaimer": DISCLAIMER})
    tracer.note("report", f"Report ready: {len(report.top_three_risks)} risks, "
                          f"hedge = {report.hedge_strategy.strategy}")
    return report


def report_to_markdown(report: ResearchReport | dict) -> str:
    r = report if isinstance(report, ResearchReport) else ResearchReport(**report)
    fh = r.financial_health_summary
    lines = [f"# {r.ticker} research brief ({r.as_of})", "", "## 1. Financial Health Summary", "", fh.summary, "",
             *[f"- {m}" for m in fh.key_metrics], "", f"**Market sentiment:** {fh.market_sentiment}", "",
             "## 2. Top Three Risks (next 90 days)", ""]
    for i, risk in enumerate(r.top_three_risks, 1):
        lines += [f"### {i}. {risk.title} ({risk.severity})", "", risk.description, "", "Evidence:",
                  *[f"- {e}" for e in risk.evidence], f"Sources: {', '.join(risk.sources)}", ""]
    h = r.hedge_strategy
    lines += ["## 3. Hedge Strategy Recommendation", "", f"**{h.strategy}** using {h.instrument}.", "", h.rationale,
              "", "Data points:", *[f"- {d}" for d in h.data_points], f"Sizing / trigger: {h.sizing_or_trigger}", ""]
    if r.clarification_used:
        lines += [f"**How the analyst's clarification was used:** {r.clarification_used}", ""]
    if r.sources:
        lines += ["Sources: " + "; ".join(r.sources), ""]
    lines += [f"*{r.disclaimer}*"]
    return "\n".join(lines)
