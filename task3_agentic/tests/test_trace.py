"""3C observability: agent_trace.jsonl has tool name, inputs, output (max 200 chars) and duration per call."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for a JSONL agent tracer', Date: 2026-10-11

import json

from research_agents import config
from research_agents.config import TRACE_OUTPUT_CHARS
from research_agents.trace import Tracer


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tool_call_line_has_every_required_field(tracer):
    tracer.tool_call("Agent A", "get_price_data", {"ticker": "NVDA", "period": "1y"}, "x" * 1000, 42, "ok")
    (line,) = read_lines(tracer.path)
    assert {"ts", "run_id", "kind", "agent", "tool", "inputs", "output", "duration_ms", "status"} <= set(line)
    assert line["tool"] == "get_price_data" and line["inputs"] == {"ticker": "NVDA", "period": "1y"}
    assert len(line["output"]) == TRACE_OUTPUT_CHARS and line["output"].endswith("...")
    assert line["duration_ms"] == 42 and line["status"] == "ok"


def test_errors_are_kept_and_summarised(tracer):
    tracer.tool_call("B", "web_search", {"query": "q"}, '{"status":"error"}', 5, "error", "RuntimeError: 202")
    tracer.tool_call("B", "get_news", {"ticker": "NVDA"}, "{}", 7, "empty")
    tracer.tool_call("B", "get_news", {"ticker": "NVDA"}, "{}", 8, "ok")
    assert read_lines(tracer.path)[0]["error"] == "RuntimeError: 202"
    s = tracer.summary()
    assert s["tool_calls"] == 3 and s["errors"] == 1 and s["empty"] == 1
    assert s["by_tool"] == {"web_search": 1, "get_news": 2} and s["total_tool_ms"] == 20


def test_runs_append_to_the_same_file(tmp_path):
    path = tmp_path / "agent_trace.jsonl"
    for run in ("r1", "r2"):
        Tracer(path=path, run_id=run, verbose=False).tool_call("A", "t", {}, "o", 1, "ok")
    assert [line["run_id"] for line in read_lines(path)] == ["r1", "r2"]


def test_default_file_is_task3_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRACE_FILE", tmp_path / "logs" / "agent_trace.jsonl")
    assert Tracer(verbose=False).path == tmp_path / "logs" / "agent_trace.jsonl"


def test_console_trace_shows_messages_reasoning_and_handoffs(tmp_path):
    printed = []
    t = Tracer(path=tmp_path / "t.jsonl", printer=printed.append)
    t.agent_message("Agent A", "Reasoning: need volatility", [{"name": "calculate_volatility",
                                                              "args": {"window": 30}}], reasoning="RSI is high")
    t.handoff("Agent A", "Agent B", "DataBrief", {"ticker": "NVDA"})
    text = "\n".join(printed)
    assert "[Agent A] thinks: RSI is high" in text and "-> calls calculate_volatility" in text
    assert "HANDOFF Agent A -> Agent B (DataBrief)" in text


def test_mark_counts_calls_of_one_step(tracer):
    tracer.tool_call("A", "t", {}, "o", 1, "ok")
    mark = tracer.mark()
    tracer.note("task", "follow-up")
    assert tracer.tool_calls_since(mark) == []
