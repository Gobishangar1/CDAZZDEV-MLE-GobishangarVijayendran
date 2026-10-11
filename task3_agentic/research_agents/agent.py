"""A generic tool-using agent (ReAct loop) built with LangGraph.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> (no tool calls) -> END

    agent node : the LLM sees the conversation and DECIDES the next step (call a tool, or answer).
    tools node : runs the calls through ToolBox.execute() (tool restriction, errors -> suggestions, trace).
    The loop continues until the LLM answers without tool calls, or the step budget is used
    (then the LLM must answer from the evidence it has).

Short-term memory: with a checkpointer, every run() on the same thread_id continues the same conversation,
so a follow-up question can be answered from earlier tool results without calling tools again.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Reusable LangGraph ReAct agent with per-agent tool
# restriction, malformed-tool-call retries, a step budget, optional checkpointer memory and trace output',
# Date: 2026-10-11

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from research_agents.config import AGENT_MAX_STEPS, MODEL_ERROR_RETRIES
from research_agents.llm import message_text
from research_agents.tools import ToolBox

FORCE_FINAL = ("You have used your tool budget. Do not call tools. Write your final answer now from the evidence "
               "already in this conversation.")
MALFORMED = ("Your last reply could not be processed (invalid tool call). Call ONE tool with valid JSON "
             "arguments, or write your final answer without tools.")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int


@dataclass
class ToolRecord:
    tool: str
    args: dict[str, Any]
    status: str
    text: str
    result: dict[str, Any] | None


@dataclass
class AgentRun:
    agent: str
    final_text: str
    messages: list[AnyMessage]
    records: list[ToolRecord] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)   # trace events of this run

    def ok_results(self, tool: str) -> list[dict[str, Any]]:
        return [r.result for r in self.records if r.tool == tool and r.status == "ok" and r.result]


class ReActAgent:
    def __init__(self, name: str, tools: list[str], system: str, llm: Any, toolbox: ToolBox, tracer: Any,
                 max_steps: int = AGENT_MAX_STEPS, checkpointer: Any = None) -> None:
        self.name, self.tools = name, list(tools)
        self.system = f"Today's date: {date.today().isoformat()}.\n\n{system}"   # date grounding for news/events
        self.allowed = set(tools)
        self.llm = llm
        schemas = toolbox.langchain_tools(self.tools)
        self.bound = llm.bind_tools(schemas)                           # the model sees ONLY this agent's tools
        self.final = llm.bind_tools(schemas, tool_choice="none")       # budget used: answer in text, no calls
        self.toolbox, self.tracer, self.max_steps = toolbox, tracer, max_steps
        self.checkpointer = checkpointer
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _invoke_model(self, runnable: Any, messages: list) -> AIMessage:
        """Call the LLM; retry with a nudge when the call fails or the tool call is malformed."""
        error = ""
        for _ in range(MODEL_ERROR_RETRIES + 1):
            try:
                reply = runnable.invoke(messages)
                if getattr(reply, "invalid_tool_calls", None):
                    if not reply.tool_calls:
                        raise ValueError(f"malformed tool call: {str(reply.invalid_tool_calls)[:200]}")
                    # keep the valid calls; an unanswered invalid call would break the next API request
                    reply = reply.model_copy(update={"invalid_tool_calls": []})
                return reply
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.tracer.note("llm_error", f"{self.name}: {error}", agent=self.name)
                messages = messages + [HumanMessage(MALFORMED)]
        return AIMessage(content=f"[{self.name}] stopped early: the model failed {MODEL_ERROR_RETRIES + 1} "
                                 f"times ({error}). Answer based on the evidence gathered so far.")

    def _agent_node(self, state: AgentState) -> dict:
        steps = state.get("steps", 0)
        messages = [SystemMessage(self.system), *state["messages"]]
        if steps >= self.max_steps:
            reply = self._invoke_model(self.final, messages + [HumanMessage(FORCE_FINAL)])
            reply = AIMessage(content=message_text(reply))          # drop any tool calls: budget is used
        else:
            reply = self._invoke_model(self.bound, messages)
        thinking = (getattr(reply, "additional_kwargs", None) or {}).get("reasoning_content", "")
        self.tracer.agent_message(self.name, message_text(reply), list(getattr(reply, "tool_calls", []) or []),
                                  reasoning=str(thinking or ""))
        return {"messages": [reply], "steps": steps + 1}

    def _tools_node(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        out = []
        for call in last.tool_calls:
            text, status, result = self.toolbox.execute(self.name, self.allowed, call["name"], call.get("args", {}))
            out.append(ToolMessage(content=text, tool_call_id=call["id"], name=call["name"],
                                   artifact={"status": status,
                                             "result": result.model_dump(mode="json") if result else None}))
        return {"messages": out}

    @staticmethod
    def _route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route, ["tools", END])
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------------------------- run
    def run(self, task: str, thread_id: str | None = None) -> AgentRun:
        config: dict[str, Any] = {"recursion_limit": 2 * self.max_steps + 6}
        before = 0
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": thread_id or "default"}
            snapshot = self.graph.get_state(config)
            before = len(snapshot.values.get("messages", [])) if snapshot and snapshot.values else 0
        mark = self.tracer.mark()
        self.tracer.note("task", f"{self.name} <- {task[:300]}", agent=self.name)
        state = self.graph.invoke({"messages": [HumanMessage(task)], "steps": 0}, config=config)
        new = state["messages"][before:]
        return AgentRun(agent=self.name, final_text=message_text(new[-1]) if new else "", messages=new,
                        records=_records(new), tool_calls=self.tracer.tool_calls_since(mark))


def _records(messages: list[AnyMessage]) -> list[ToolRecord]:
    calls = {c["id"]: c for m in messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])}
    records = []
    for m in messages:
        if isinstance(m, ToolMessage):
            art = m.artifact or {}
            call = calls.get(m.tool_call_id, {})
            records.append(ToolRecord(tool=m.name or call.get("name", "?"), args=call.get("args", {}),
                                      status=art.get("status", "ok"), text=str(m.content),
                                      result=art.get("result")))
    return records


def evidence_from(records: list[ToolRecord], limit: int = 1500) -> str:
    """Tool results as compact text blocks for the report writer (failures included, marked)."""
    lines = []
    for r in records:
        head = f"[{r.tool}({json.dumps(r.args, separators=(',', ':'))})]"
        if r.status in ("ok", "empty") and r.result is not None:
            lines.append(f"{head} {json.dumps(r.result, separators=(',', ':'), default=str)[:limit]}")
        else:
            lines.append(f"{head} {r.status.upper()}: {r.text[:300]}")
    return "\n".join(lines)
