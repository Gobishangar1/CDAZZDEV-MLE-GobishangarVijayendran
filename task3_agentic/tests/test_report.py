"""Final report: exactly three risks with evidence, three Markdown sections, prompts kept as constants."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'pytest tests for the research report schema, markdown
# rendering and prompt constants', Date: 2026-10-11

import json

import pytest
from conftest import REPORT, FakeJSONLLM
from pydantic import ValidationError

from research_agents import prompts
from research_agents.report import report_to_markdown, write_report
from research_agents.schemas import ResearchReport


def full(**changes):
    return {"ticker": "NVDA", "as_of": "2026-10-11", **REPORT, **changes}


def test_exactly_three_risks_are_required():
    with pytest.raises(ValidationError):
        ResearchReport(**full(top_three_risks=REPORT["top_three_risks"][:2]))
    with pytest.raises(ValidationError):
        ResearchReport(**full(top_three_risks=REPORT["top_three_risks"] * 2))


def test_each_risk_needs_evidence_and_a_source():
    risk = {**REPORT["top_three_risks"][0], "evidence": []}
    with pytest.raises(ValidationError):
        ResearchReport(**full(top_three_risks=[risk, *REPORT["top_three_risks"][1:]]))


def test_markdown_has_the_three_sections():
    md = report_to_markdown(ResearchReport(**full()))
    for heading in ("## 1. Financial Health Summary", "## 2. Top Three Risks", "## 3. Hedge Strategy Recommendation"):
        assert heading in md
    assert md.count("Evidence:") == 3


def test_write_report_sets_the_fixed_fields(tracer):
    llm = FakeJSONLLM(lambda s, h, n: json.dumps({**REPORT, "ticker": "WRONG", "as_of": "1999-01-01"}))
    report = write_report(llm, "NVDA", "2026-10-11", "query", "evidence", tracer)
    assert (report.ticker, report.as_of) == ("NVDA", "2026-10-11") and report.disclaimer
    assert any(e["kind"] == "report" for e in tracer.events)


def test_prompts_are_constants_and_format_cleanly():
    found = prompts.all_prompts()
    assert {"RESEARCH_AGENT_SYSTEM", "DATA_ANALYST_SYSTEM", "RESEARCH_WRITER_SYSTEM", "CRITIQUE_SYSTEM",
            "REPORT_SYSTEM", "SENTIMENT_SYSTEM"} <= set(found)
    assert '{"question"' in prompts.CRITIQUE_SYSTEM.format()             # braces escaped correctly
    assert '{"items"' in prompts.SENTIMENT_SYSTEM.format()
    system = prompts.REPORT_SYSTEM.format(ticker="NVDA", as_of="2026-10-11", clarification_hint="null")
    assert '"ticker": "NVDA"' in system and '"clarification_used": null' in system
