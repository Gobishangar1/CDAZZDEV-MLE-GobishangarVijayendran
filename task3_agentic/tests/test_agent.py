"""3A agent loop (observe -> decide -> act), error replanning, step budget, and 3C short-term memory."""

from conftest import TICKER, PolicyLLM, failing_search, research_policy, search_results
from langchain_core.messages import AIMessage, ToolMessage

from research_agents.agent import ReActAgent, evidence_from
from research_agents.schemas import ResearchReport
from research_agents.single_agent import ResearchSession
from research_agents.tools import ALL_TOOLS, ToolBox


def make_agent(llm, tracer, sentiment_llm, search=search_results, tools=ALL_TOOLS, max_steps=8):
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search)
    return ReActAgent("Research Agent", list(tools), "system rules", llm, box, tracer, max_steps=max_steps)


def test_agent_runs_the_loop_and_chooses_arguments_from_observations(policy_llm, tracer, sentiment_llm):
    run = make_agent(policy_llm, tracer, sentiment_llm).run(f"Analyse {TICKER}")
    order = [r.tool for r in run.records]
    assert order == ["get_price_data", "calculate_volatility", "get_news", "llm_sentiment", "web_search"]
    assert all(r.status == "ok" for r in run.records)
    # the sentiment call used the headlines that get_news returned (observe -> act)
    news_titles = [h["title"] for h in run.ok_results("get_news")[0]["headlines"]]
    assert run.records[3].args["headlines"] == news_titles[:15]
    assert run.final_text.startswith("Final answer")
    assert len(run.tool_calls) == 5 and policy_llm.calls == 6


def test_failed_tool_makes_the_agent_replan(policy_llm, tracer, sentiment_llm):
    run = make_agent(policy_llm, tracer, sentiment_llm, search=failing_search).run(f"Analyse {TICKER}")
    statuses = [(r.tool, r.status) for r in run.records]
    assert ("web_search", "error") in statuses
    assert statuses[-1] == ("get_news", "ok")                     # followed the suggestion, did not stop
    assert run.records[-1].args == {"ticker": TICKER, "n": 15}
    assert run.final_text                                          # still finished with an answer


def test_empty_news_leads_to_web_search(policy_llm, tracer, sentiment_llm, fake_ticker):
    fake_ticker.news_items = []
    run = make_agent(policy_llm, tracer, sentiment_llm).run(f"Analyse {TICKER}")
    order = [(r.tool, r.status) for r in run.records]
    i = order.index(("get_news", "empty"))
    assert order[i + 1] == ("web_search", "ok")


def test_tool_restriction_inside_the_agent(tracer, sentiment_llm):
    def sneaky(obs, tools):
        if not obs.results:
            return ("Reasoning: try a price tool.", "get_price_data", {"ticker": TICKER})
        return "I am not allowed to use price tools."
    agent = make_agent(PolicyLLM(sneaky), tracer, sentiment_llm, tools=["web_search", "get_news"])
    run = agent.run("anything")
    assert run.records[0].status == "denied" and "not available" in run.records[0].text
    assert "get_price_data" not in [t.name for t in agent.toolbox.langchain_tools(agent.tools)]


def test_malformed_tool_call_is_retried(tracer, sentiment_llm):
    state = {"n": 0}

    def flaky(obs, tools):
        state["n"] += 1
        if state["n"] == 1:
            return AIMessage(content="", invalid_tool_calls=[{"name": "get_news", "args": "{bad", "id": "x",
                                                              "error": "bad json", "type": "invalid_tool_call"}])
        return research_policy(obs, tools)
    run = make_agent(PolicyLLM(flaky), tracer, sentiment_llm).run(f"Analyse {TICKER}")
    assert any(e["kind"] == "llm_error" for e in tracer.events)
    assert len(run.records) == 5


def test_model_errors_do_not_crash_the_agent(tracer, sentiment_llm):
    def broken(obs, tools):
        raise RuntimeError("503 service unavailable")
    run = make_agent(PolicyLLM(broken), tracer, sentiment_llm).run("anything")
    assert "stopped early" in run.final_text and run.records == []


def test_step_budget_forces_a_final_answer(tracer, sentiment_llm):
    def greedy(obs, tools):
        if tools is None:
            return "Final answer after the budget."
        return ("Reasoning: one more.", "calculate_volatility", {"ticker": TICKER, "window": 20 + len(obs.results)})
    run = make_agent(PolicyLLM(greedy), tracer, sentiment_llm, max_steps=3).run("loop forever")
    assert len(run.records) == 3 and run.final_text == "Final answer after the budget."


def test_evidence_text_marks_failures(policy_llm, tracer, sentiment_llm):
    run = make_agent(policy_llm, tracer, sentiment_llm, search=failing_search).run(f"Analyse {TICKER}")
    text = evidence_from(run.records)
    assert "[web_search(" in text and "ERROR:" in text and '"annualised_vol_pct"' in text


# ----------------------------------------------------------------------------- 3A session + 3C memory
def test_single_agent_report_and_follow_up_from_memory(policy_llm, json_llm, tracer, sentiment_llm):
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)
    session = ResearchSession(TICKER, policy_llm, json_llm, box, tracer)
    result = session.research()
    assert isinstance(result.report, ResearchReport) and len(result.report.top_three_risks) == 3
    assert result.report.ticker == TICKER and "not investment advice" in result.report.disclaimer.lower()

    calls_before = len(tracer.tool_calls())
    follow = session.ask("What annualised volatility did you calculate?")
    assert follow.tool_calls == 0 and len(tracer.tool_calls()) == calls_before
    vol = result.run.ok_results("calculate_volatility")[0]
    assert str(vol["annualised_vol_pct"]) in follow.answer
    # the model saw the earlier tool result in its context (same thread)
    last_context = policy_llm.seen[-1]
    assert any(isinstance(m, ToolMessage) and m.name == "calculate_volatility" for m in last_context)
    assert any(e["kind"] == "follow_up" and e["tool_calls"] == 0 for e in tracer.events)


def test_new_thread_has_no_memory(policy_llm, json_llm, tracer, sentiment_llm):
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)
    session = ResearchSession(TICKER, policy_llm, json_llm, box, tracer, thread_id="t1")
    session.research()
    other = ResearchSession(TICKER, policy_llm, json_llm, box, tracer, thread_id="t2")
    assert other.ask("What annualised volatility did you calculate?").tool_calls == 1


def test_report_failure_is_reported_not_raised(policy_llm, tracer, sentiment_llm):
    from conftest import FakeJSONLLM
    bad = FakeJSONLLM(lambda s, h, n: '{"ticker": "NVDA"}')
    box = ToolBox(sentiment_llm=sentiment_llm, tracer=tracer, search_backend=search_results)
    result = ResearchSession(TICKER, policy_llm, bad, box, tracer).research()
    assert result.report is None and "research_report" in result.error and bad.calls == 3


def test_agent_system_prompt_starts_with_today(tracer, sentiment_llm):
    from datetime import date
    agent = make_agent(PolicyLLM(research_policy), tracer, sentiment_llm)
    assert agent.system.startswith(f"Today's date: {date.today().isoformat()}.")
