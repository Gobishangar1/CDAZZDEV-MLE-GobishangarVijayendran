"""Command line.

    python -m research_agents NVDA              # multi-agent pipeline (uses the memory cache)
    python -m research_agents NVDA --no-cache   # ignore today's cached brief
    python -m research_agents NVDA --single     # 3A single agent + one follow-up question
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'argparse CLI for the Task 3 research agents', Date: 2026-10-11

import argparse
import json
import sys

from research_agents.config import setup_logging
from research_agents.report import report_to_markdown
from research_agents.trace import Tracer


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-agent financial research")
    p.add_argument("ticker")
    p.add_argument("--single", action="store_true", help="run the 3A single agent instead")
    p.add_argument("--no-cache", action="store_true", help="ignore today's cached brief")
    args = p.parse_args()
    setup_logging()
    tracer = Tracer()

    if args.single:
        from research_agents.llm import make_agent_llm, make_json_llm, make_sentiment_llm
        from research_agents.single_agent import ResearchSession
        from research_agents.tools import ToolBox
        session = ResearchSession(args.ticker, make_agent_llm(), make_json_llm(),
                                  ToolBox(sentiment_llm=make_sentiment_llm(), tracer=tracer), tracer)
        result = session.research()
        print("\n" + (report_to_markdown(result.report) if result.report else f"No report: {result.error}"))
        follow = session.ask("What annualised volatility did you calculate, over which window, and what does it "
                             "mean for the hedge?")
        print(f"\nFollow-up answer ({follow.tool_calls} tool calls):\n{follow.answer}")
    else:
        from research_agents.multi_agent import run_research
        record = run_research(args.ticker, tracer=tracer, use_cache=not args.no_cache)
        report = record.get("report")
        print("\n" + (report_to_markdown(report) if report else f"No report: {record.get('error')}"))
        print(f"\nfrom_cache={record['from_cache']} path={record.get('path')}")
    print("\nTrace summary:", json.dumps(tracer.summary()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
