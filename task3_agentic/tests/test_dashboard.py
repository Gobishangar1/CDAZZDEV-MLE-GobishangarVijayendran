"""Trace tables and the Streamlit dashboard (bonus) render from a real trace file without errors."""

from pathlib import Path

import pytest
from conftest import TICKER, search_results

from research_agents import config
from research_agents.memory import BriefStore
from research_agents.multi_agent import run_research
from research_agents.trace import Tracer
from research_agents.trace_view import load_events, of_kind, runs, tool_calls

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


@pytest.fixture
def trace_file(tmp_path, policy_llm, json_llm, sentiment_llm):
    path = tmp_path / "agent_trace.jsonl"
    run_research(TICKER, llm=policy_llm, json_llm=json_llm, sentiment_llm=sentiment_llm,
                 tracer=Tracer(path=path, run_id="r1", verbose=False), store=BriefStore(tmp_path / "m"),
                 search_backend=search_results)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")                                      # a broken line is skipped
    return path


def test_trace_tables(trace_file):
    events = load_events(trace_file)
    assert list(runs(events)["run_id"]) == ["r1"]
    calls = tool_calls(events, "r1")
    assert len(calls) == 6 and (calls["start"] <= calls["ts"]).all()
    assert list(of_kind(events, "handoff", "r1")["schema"]) == ["DataBrief", "ClarificationRequest",
                                                                 "ClarificationResponse"]


def test_missing_file_gives_empty_tables(tmp_path):
    events = load_events(tmp_path / "none.jsonl")
    assert events.empty and runs(events).empty and tool_calls(events).empty


def test_dashboard_renders(trace_file, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(config, "TRACE_FILE", trace_file)
    app = testing.AppTest.from_file(str(APP), default_timeout=30).run()     # default path = TRACE_FILE
    assert not app.exception
    assert [m.value for m in app.metric][0] == "6"
    assert any("Handoffs" in t.label for t in app.tabs)
