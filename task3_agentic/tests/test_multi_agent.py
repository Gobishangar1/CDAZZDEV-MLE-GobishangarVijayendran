"""3B: restricted tools, typed handoffs, one critique round, end-to-end without manual steps."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for a two-agent LangGraph pipeline with
# restricted tools, Pydantic handoffs and one clarification round', Date: 2026-10-11

import json

import pytest
from conftest import TICKER, FakeJSONLLM, json_reply, search_results
from pydantic import ValidationError

from research_agents.agent import AgentRun, ToolRecord
from research_agents.multi_agent import (
    AGENT_A,
    AGENT_B,
    TOOLS_A,
    TOOLS_B,
    MultiAgentReport,
    build_data_brief,
    build_pipeline,
)
from research_agents.schemas import ClarificationRequest, ClarificationResponse, DataBrief
from research_agents.tools import ToolBox


@pytest.fixture
def state(policy_llm, json_llm, tracer, sentiment_llm):
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)
    pipeline = build_pipeline(policy_llm, json_llm, box, tracer)
    return pipeline.invoke({"ticker": TICKER, "query": f"Analyse {TICKER}", "as_of": "2026-10-11"})


def handoffs(tracer):
    return [e for e in tracer.events if e["kind"] == "handoff"]


def test_pipeline_runs_end_to_end_to_a_valid_report(state):
    report = MultiAgentReport(**state["report"])
    assert len(report.top_three_risks) == 3 and report.clarification_used
    assert state["error"] is None


def test_each_agent_used_only_its_own_tools(state, tracer):
    for call in tracer.tool_calls():
        allowed = TOOLS_A if call["agent"] == AGENT_A else TOOLS_B
        assert call["tool"] in allowed, call
    used_a = {c["tool"] for c in tracer.tool_calls() if c["agent"] == AGENT_A}
    used_b = {c["tool"] for c in tracer.tool_calls() if c["agent"] == AGENT_B}
    assert used_a == set(TOOLS_A) and used_b == set(TOOLS_B)


def test_handoffs_are_typed_and_in_order(state, tracer):
    h = handoffs(tracer)
    assert [(e["sender"], e["receiver"], e["schema"]) for e in h] == [
        (AGENT_A, AGENT_B, "DataBrief"), (AGENT_B, AGENT_A, "ClarificationRequest"),
        (AGENT_A, AGENT_B, "ClarificationResponse")]
    DataBrief(**h[0]["payload"])
    ClarificationRequest(**h[1]["payload"])
    ClarificationResponse(**h[2]["payload"])


def test_brief_numbers_are_copied_from_tool_results(state, tracer):
    vol_call = next(c for c in tracer.events if c["kind"] == "tool_call" and c["tool"] == "calculate_volatility")
    brief = state["brief"]
    assert brief["volatility"]["window"] == vol_call["inputs"]["window"]
    assert brief["sentiment"] is None                       # Agent A has no news, so no sentiment yet
    assert any("no news access" in g for g in brief["data_gaps"])


def test_critique_loop_runs_once_and_is_incorporated(state, tracer, json_llm):
    request = state["request"]
    assert len(request["headlines"]) > 0                     # B's headlines were attached for A to score
    response = state["response"]
    assert {"llm_sentiment", "calculate_volatility"} <= set(response["tools_used"])
    assert response["data"]["llm_sentiment"]["n"] == len(request["headlines"])
    assert any(v.get("window") == 60 for v in response["data"].values() if isinstance(v, dict))
    # the final report prompt contains A's answer
    report_prompt = next(h for s, h in json_llm.prompts if "final equity research report" in s)
    assert "AGENT A ANSWER" in report_prompt and '"llm_sentiment"' in report_prompt
    assert sum(e["schema"] == "ClarificationRequest" for e in handoffs(tracer)) == 1


def test_invalid_critique_falls_back_to_a_default_request(policy_llm, tracer, sentiment_llm):
    def reply(system, human, n):
        return "no json here" if "review Agent A's data" in system else json_reply(system, human)
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)
    state = build_pipeline(policy_llm, FakeJSONLLM(reply), box, tracer).invoke(
        {"ticker": TICKER, "query": "q", "as_of": "2026-10-11"})
    assert any(e["kind"] == "critique_fallback" for e in tracer.events)
    assert state["request"]["headlines"] and state["report"] is not None


def test_report_failure_is_returned_not_raised(policy_llm, tracer, sentiment_llm):
    def reply(system, human, n):
        return '{"bad": 1}' if "final equity research report" in system else json_reply(system, human)
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)
    state = build_pipeline(policy_llm, FakeJSONLLM(reply), box, tracer).invoke(
        {"ticker": TICKER, "query": "q", "as_of": "2026-10-11"})
    assert state["report"] is None and "research_report" in state["error"]


def test_report_without_clarification_used_is_rejected():
    data = json.loads(json_reply("final equity research report", "").split("\n", 1)[1])
    data.update(ticker=TICKER, as_of="2026-10-11", clarification_used=None)
    with pytest.raises(ValidationError):
        MultiAgentReport(**data)


def test_build_data_brief_lists_gaps_for_failed_tools():
    run = AgentRun(agent=AGENT_A, final_text="notes", messages=[], records=[
        ToolRecord("get_price_data", {"ticker": TICKER}, "error", '{"error":"No price data"}', None)])
    brief = build_data_brief(TICKER, "2026-10-11", run)
    assert brief.price is None and brief.volatility is None
    assert any(g.startswith("get_price_data failed") for g in brief.data_gaps) and "no price data" in brief.data_gaps
