"""Task 3A (one research agent, all five tools) and 3C short-term memory (follow-up questions).

    session = ResearchSession("NVDA", ...)
    result = session.research()            # the agent decides which tools to call, then a validated report
    follow = session.ask("What volatility did you compute?")
    follow.tool_calls == 0                 # answered from the conversation memory, no tool re-called
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Single research agent session with all five tools,
# validated three-section report and checkpointer-based short-term memory for follow-up questions',
# Date: 2026-10-11

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from research_agents.agent import AgentRun, ReActAgent, evidence_from
from research_agents.config import RESEARCH_QUERY
from research_agents.prompts import RESEARCH_AGENT_SYSTEM
from research_agents.report import write_report
from research_agents.schemas import ResearchReport
from research_agents.tools import ALL_TOOLS, ToolBox


@dataclass
class SingleAgentResult:
    report: ResearchReport | None
    run: AgentRun
    error: str | None = None


@dataclass
class FollowUp:
    question: str
    answer: str
    tool_calls: int              # tools called while answering (0 = answered from memory)


class ResearchSession:
    def __init__(self, ticker: str, llm: Any, json_llm: Any, toolbox: ToolBox, tracer: Any,
                 thread_id: str | None = None) -> None:
        self.ticker = ticker.upper()
        self.json_llm, self.tracer = json_llm, tracer
        self.thread_id = thread_id or f"session-{tracer.run_id}"
        self.agent = ReActAgent("Research Agent", list(ALL_TOOLS),
                                RESEARCH_AGENT_SYSTEM.format(tools=", ".join(ALL_TOOLS)), llm, toolbox, tracer,
                                checkpointer=InMemorySaver())          # same thread = same conversation

    def research(self) -> SingleAgentResult:
        query = RESEARCH_QUERY.format(ticker=self.ticker)
        self.tracer.banner(f"3A Research Agent: {query}")
        run = self.agent.run(query, self.thread_id)
        evidence = evidence_from(run.records) + f"\n\nAGENT FINAL NOTES:\n{run.final_text}"
        try:
            report = write_report(self.json_llm, self.ticker, date.today().isoformat(), query, evidence, self.tracer)
            return SingleAgentResult(report=report, run=run)
        except Exception as exc:
            self.tracer.note("report_failed", f"{type(exc).__name__}: {str(exc)[:300]}")
            return SingleAgentResult(report=None, run=run, error=str(exc))

    def ask(self, question: str) -> FollowUp:
        """Follow-up in the SAME conversation (short-term memory)."""
        self.tracer.banner(f"Follow-up (same session): {question}")
        run = self.agent.run(question, self.thread_id)
        calls = len(run.tool_calls)
        self.tracer.note("follow_up", f"answered with {calls} new tool calls", tool_calls=calls)
        return FollowUp(question=question, answer=run.final_text, tool_calls=len(run.tool_calls))
