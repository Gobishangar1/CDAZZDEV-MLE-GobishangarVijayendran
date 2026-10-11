"""3C persistent memory: the brief is saved as JSON keyed by ticker and date; the next run loads it."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for a ticker+date JSON cache in front of a
# multi-agent pipeline', Date: 2026-10-11

from datetime import date

from conftest import TICKER, FakeJSONLLM, PolicyLLM, json_reply, research_policy, search_results

from research_agents import config
from research_agents.memory import BriefStore
from research_agents.multi_agent import run_research
from research_agents.report import report_to_markdown

DAY = date(2026, 10, 11)


def run(tracer, store, llm, json_llm, sentiment_llm, **kw):
    return run_research(TICKER.lower(), llm=llm, json_llm=json_llm, sentiment_llm=sentiment_llm, tracer=tracer,
                        store=store, search_backend=search_results, today=DAY, **kw)


def test_first_run_saves_second_run_loads(tmp_path, tracer, policy_llm, json_llm, sentiment_llm):
    store = BriefStore(tmp_path / "memory")
    first = run(tracer, store, policy_llm, json_llm, sentiment_llm)
    assert first["from_cache"] is False and first["path"].endswith("NVDA_2026-10-11.json")
    assert (tmp_path / "memory" / "NVDA_2026-10-11.json").exists()
    assert first["clarification"]["request"] and first["clarification"]["response"]
    assert first["trace_summary"]["tool_calls"] > 0

    llm_calls, json_calls, tool_calls = policy_llm.calls, json_llm.calls, len(tracer.tool_calls())
    second = run(tracer, store, policy_llm, json_llm, sentiment_llm)
    assert second["from_cache"] is True and second["report"] == first["report"]
    assert (policy_llm.calls, json_llm.calls, len(tracer.tool_calls())) == (llm_calls, json_calls, tool_calls)
    assert any(e["kind"] == "cache_hit" for e in tracer.events)
    assert "## 3. Hedge Strategy Recommendation" in report_to_markdown(second["report"])


def test_no_cache_flag_runs_again(tmp_path, tracer, policy_llm, json_llm, sentiment_llm):
    store = BriefStore(tmp_path / "memory")
    run(tracer, store, policy_llm, json_llm, sentiment_llm)
    again = run(tracer, store, policy_llm, json_llm, sentiment_llm, use_cache=False)
    assert again["from_cache"] is False


def test_other_day_or_ticker_is_a_miss(tmp_path):
    store = BriefStore(tmp_path)
    store.save("NVDA", DAY, {"report": {"x": 1}})
    assert store.load("nvda", DAY) == {"report": {"x": 1}}
    assert store.load("NVDA", date(2026, 10, 12)) is None and store.load("AMD", DAY) is None


def test_broken_cache_file_means_run_again(tmp_path, tracer, policy_llm, json_llm, sentiment_llm):
    store = BriefStore(tmp_path / "memory")
    store.path(TICKER, DAY).parent.mkdir(parents=True)
    store.path(TICKER, DAY).write_text("{broken", encoding="utf-8")
    assert run(tracer, store, policy_llm, json_llm, sentiment_llm)["from_cache"] is False


def test_failed_report_is_not_cached(tmp_path, tracer, sentiment_llm):
    def reply(system, human, n):
        return "{}" if "final equity research report" in system else json_reply(system, human)
    store = BriefStore(tmp_path / "memory")
    record = run(tracer, store, PolicyLLM(research_policy), FakeJSONLLM(reply), sentiment_llm)
    assert record["report"] is None and record["error"] and record["path"] is None
    assert not store.path(TICKER, DAY).exists()


def test_default_folder_is_task3_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "m")
    assert BriefStore().path("nvda", DAY) == tmp_path / "m" / "NVDA_2026-10-11.json"
