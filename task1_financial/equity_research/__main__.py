"""Command line: python -m equity_research NVDA [--out outputs] [--no-json-mode]"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Small argparse CLI that runs the pipeline and
# prints the signal and report paths', Date: 2026-10-11

import argparse
import json
import sys

from equity_research.config import setup_logging
from equity_research.graph import run_pipeline
from equity_research.llm import make_fast_llm, make_strong_llm


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-powered equity research brief for one ticker")
    parser.add_argument("ticker", help="e.g. NVDA")
    parser.add_argument("--out", default="outputs", help="folder for run.json and the brief (default: outputs)")
    parser.add_argument("--no-json-mode", action="store_true", help="disable Groq JSON mode if the API rejects it")
    args = parser.parse_args()

    setup_logging()
    json_mode = not args.no_json_mode
    result = run_pipeline(args.ticker, make_fast_llm(json_mode), make_strong_llm(json_mode), output_dir=args.out)
    if result["status"] != "ok":
        print(f"\nERROR: {result['message']}")
        return 1
    print("\nSignal:", json.dumps(result["signal"], indent=2))
    print("Source:", result["signal_source"])
    print("Files:", json.dumps(result["report_paths"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
