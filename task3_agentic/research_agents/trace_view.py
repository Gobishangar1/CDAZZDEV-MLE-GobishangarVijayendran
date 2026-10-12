"""Read agent_trace.jsonl into tables (used by the notebook and the Streamlit dashboard)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TOOL_COLUMNS = ["ts", "start", "run_id", "agent", "tool", "inputs", "status", "duration_ms", "output", "error"]


def load_events(path: Path | str) -> pd.DataFrame:
    """One row per JSONL line. Broken lines are skipped (a crash mid-write must not break the viewer)."""
    rows = []
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df


def runs(events: pd.DataFrame) -> pd.DataFrame:
    """One row per run_id: start, end, number of tool calls and errors, newest first."""
    if events.empty:
        return pd.DataFrame(columns=["run_id", "start", "end", "tool_calls", "errors"])
    calls = events["kind"].eq("tool_call")
    bad = calls & events.get("status", pd.Series(index=events.index, dtype=object)).isin(["error", "denied"])
    out = (events.assign(call=calls, bad=bad).groupby("run_id")
           .agg(start=("ts", "min"), end=("ts", "max"), tool_calls=("call", "sum"), errors=("bad", "sum"))
           .reset_index().sort_values("start", ascending=False))
    return out.reset_index(drop=True)


def tool_calls(events: pd.DataFrame, run_id: str | None = None) -> pd.DataFrame:
    """Tool calls with a start time (ts is written when the call ends)."""
    if events.empty or "tool" not in events:
        return pd.DataFrame(columns=TOOL_COLUMNS)
    df = events[events["kind"] == "tool_call"].copy()
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    df["duration_ms"] = df["duration_ms"].astype(float)
    df["start"] = df["ts"] - pd.to_timedelta(df["duration_ms"], unit="ms")
    df["inputs"] = df["inputs"].apply(lambda v: json.dumps(v, separators=(",", ":")))
    if "error" not in df:
        df["error"] = None
    return df[TOOL_COLUMNS].reset_index(drop=True)


def of_kind(events: pd.DataFrame, kind: str, run_id: str | None = None) -> pd.DataFrame:
    if events.empty:
        return events
    df = events[events["kind"] == kind]
    return (df if run_id is None else df[df["run_id"] == run_id]).dropna(axis=1, how="all").reset_index(drop=True)
