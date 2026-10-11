"""Observability: every tool call (and every agent message and handoff) -> logs/agent_trace.jsonl

Each tool_call line has: ts, run_id, kind="tool_call", agent, tool, inputs, output (truncated to 200 chars),
duration_ms, status (ok / empty / error / denied). Other kinds: agent_message, handoff, clarification_request,
clarification_response, report, cache_hit, cache_saved, llm_error.

With verbose=True the same events are printed as a readable message trace (what each agent said, which tools
it called, what was handed over), which is what the notebook shows.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Thread-safe JSONL tracer for agent tool calls with
# 200-char output truncation, plus a readable console message trace', Date: 2026-10-11

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_agents import config
from research_agents.config import CONSOLE_CHARS, HANDOFF_FIELD_CHARS, TRACE_OUTPUT_CHARS


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, separators=(",", ":"))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


class Tracer:
    def __init__(self, path: Path | str | None = None, run_id: str | None = None, verbose: bool = True,
                 printer: Callable[[str], None] = print) -> None:
        self.path = Path(path) if path else config.TRACE_FILE     # default: task3_agentic/logs/agent_trace.jsonl
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.verbose = verbose
        self.printer = printer
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ core
    def _record(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "run_id": self.run_id,
                 "kind": kind, **fields}
        with self._lock:
            self.events.append(event)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        return event

    def _print(self, text: str) -> None:
        if self.verbose:
            self.printer(text)

    # ------------------------------------------------------------- event types
    def tool_call(self, agent: str, tool: str, inputs: dict, output: str, duration_ms: int, status: str,
                  error: str | None = None) -> None:
        self._record("tool_call", agent=agent, tool=tool, inputs=inputs,
                     output=_short(output, TRACE_OUTPUT_CHARS), duration_ms=duration_ms, status=status,
                     **({"error": error} if error else {}))
        mark = {"ok": "OK", "empty": "EMPTY", "error": "ERROR", "denied": "DENIED"}.get(status, status)
        self._print(f"      <- {tool} [{mark}, {duration_ms} ms]: {_short(output, CONSOLE_CHARS)}")

    def agent_message(self, agent: str, content: str, tool_calls: list[dict], reasoning: str = "") -> None:
        """content = what the agent said; reasoning = the model's own thinking (gpt-oss returns it separately)."""
        self._record("agent_message", agent=agent, content=_short(content or "", 1000),
                     reasoning=_short(reasoning or "", 600),
                     tool_calls=[{"tool": c["name"], "inputs": c.get("args", {})} for c in tool_calls])
        if reasoning and reasoning.strip():
            self._print(f"  [{agent}] thinks: {_short(reasoning, CONSOLE_CHARS)}")
        if content and content.strip():
            self._print(f"  [{agent}] says: {_short(content, CONSOLE_CHARS)}")
        for c in tool_calls:
            self._print(f"  [{agent}] -> calls {c['name']}({_short(c.get('args', {}), 160)})")

    def handoff(self, sender: str, receiver: str, schema: str, payload: dict) -> None:
        self._record("handoff", sender=sender, receiver=receiver, schema=schema, payload=payload)
        fields = "\n".join(f"  {key}: {_short(value, HANDOFF_FIELD_CHARS)}" for key, value in payload.items())
        self._print(f"\n=== HANDOFF {sender} -> {receiver} ({schema}) ===\n{fields}\n")

    def note(self, kind: str, text: str, **fields: Any) -> None:
        self._record(kind, text=text, **fields)
        self._print(f"  [{kind}] {text}")

    def banner(self, text: str) -> None:
        self._print(f"\n{'-' * 8} {text} {'-' * 8}")

    # ------------------------------------------------------------- queries
    def tool_calls(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == "tool_call"]

    def mark(self) -> int:
        """Position in the event list; use with tool_calls_since() to count calls in one step."""
        return len(self.events)

    def tool_calls_since(self, mark: int) -> list[dict[str, Any]]:
        return [e for e in self.events[mark:] if e["kind"] == "tool_call"]

    def summary(self) -> dict[str, Any]:
        calls = self.tool_calls()
        by_tool: dict[str, int] = {}
        for c in calls:
            by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
        return {"run_id": self.run_id, "tool_calls": len(calls), "by_tool": by_tool,
                "errors": sum(c["status"] in ("error", "denied") for c in calls),
                "empty": sum(c["status"] == "empty" for c in calls),
                "total_tool_ms": sum(c["duration_ms"] for c in calls)}
