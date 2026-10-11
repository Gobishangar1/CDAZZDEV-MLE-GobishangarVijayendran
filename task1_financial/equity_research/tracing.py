"""Lightweight run tracing: one JSON line per node and per LLM call -> <run_dir>/trace.jsonl

Event kinds
    node         node name, duration_ms, status (ok / error)
    llm_call     purpose (headline / signal), attempt, model that answered, latency_ms,
                 input/output tokens, ok (API success), output preview (200 chars) or error
    validation   purpose, attempt, valid (True/False), problems

Why: terminal logs disappear. The trace shows what was slow, what it cost in tokens, which model
answered (the 20B model or its 120B fallback), and every validation failure, for every run.
Optional: LangSmith also works with zero code (see the README), but this file needs no account.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'A small thread-safe run tracer that records node
# durations and LLM calls (model, latency, tokens, validation result) to JSONL with a summary',
# Date: 2026-10-11

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

PREVIEW_CHARS = 200


class RunTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()          # headline branches run in parallel threads
        self._t0 = time.perf_counter()

    def add(self, kind: str, **fields: Any) -> None:
        event = {"t_s": round(time.perf_counter() - self._t0, 3), "kind": kind, **fields}
        with self._lock:
            self.events.append(event)

    def summary(self) -> dict[str, Any]:
        nodes: dict[str, dict[str, float]] = {}
        for e in self.events:
            if e["kind"] == "node":
                n = nodes.setdefault(e["node"], {"calls": 0, "total_ms": 0})
                n["calls"] += 1
                n["total_ms"] += e["duration_ms"]
        llm = [e for e in self.events if e["kind"] == "llm_call"]
        checks = [e for e in self.events if e["kind"] == "validation"]

        def tokens(key: str) -> int:
            return sum(e.get(key) or 0 for e in llm)

        return {
            "wall_time_s": round(time.perf_counter() - self._t0, 2),
            "llm_calls": len(llm),
            "llm_api_errors": sum(not e["ok"] for e in llm),
            "validation_failures": sum(not e["valid"] for e in checks),
            "input_tokens": tokens("input_tokens"),
            "output_tokens": tokens("output_tokens"),
            "models": sorted({e["model"] for e in llm if e.get("model")}),
            "llm_latency_s": round(sum(e["latency_ms"] for e in llm) / 1000, 2),
            "nodes_ms": {k: round(v["total_ms"]) for k, v in sorted(nodes.items(), key=lambda kv: -kv[1]["total_ms"])},
        }

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, default=str) + "\n")
        return path


_current: RunTrace | None = None


def start() -> RunTrace:
    global _current
    _current = RunTrace()
    return _current


def stop() -> None:
    global _current
    _current = None


def record(kind: str, **fields: Any) -> None:
    """Add an event to the active trace. Does nothing when no run is being traced (e.g. unit tests)."""
    if _current is not None:
        _current.add(kind, **fields)


def preview(text: str) -> str:
    return " ".join(str(text).split())[:PREVIEW_CHARS]
