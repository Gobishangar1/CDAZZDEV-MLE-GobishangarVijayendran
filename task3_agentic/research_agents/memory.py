"""3C persistent memory: the final research brief saved as JSON, keyed by ticker and date.

    memory/NVDA_2026-10-11.json

On a later run with the same ticker (same day), run_research() finds this file and loads it instead of
running the agents and tools again.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'JSON brief store keyed by ticker and date for cached
# agent runs', Date: 2026-10-11

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from research_agents import config


class BriefStore:
    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory) if directory else config.MEMORY_DIR    # default: task3_agentic/memory

    def path(self, ticker: str, day: date) -> Path:
        return self.directory / f"{ticker.upper()}_{day.isoformat()}.json"

    def load(self, ticker: str, day: date) -> dict[str, Any] | None:
        p = self.path(ticker, day)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None                       # a broken cache file means: run again

    def save(self, ticker: str, day: date, record: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        p = self.path(ticker, day)
        p.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return p
