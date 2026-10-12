"""Streamlit dashboard for logs/agent_trace.jsonl (Task 3 bonus: observability).

    cd task3_agentic
    streamlit run dashboard/app.py

Shows, for one run: totals, a timeline of tool calls per agent (coloured by status), time per tool, the full
tool-call table (inputs, 200-char output, duration), what each agent said, and the agent-to-agent handoffs.
"""
import json
import sys
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # task3_agentic/ for research_agents

from research_agents.config import TRACE_FILE  # noqa: E402
from research_agents.trace_view import load_events, of_kind, runs, tool_calls  # noqa: E402

STATUS_COLORS = alt.Scale(domain=["ok", "empty", "error", "denied"],
                          range=["#2e7d32", "#f9a825", "#c62828", "#6a1b9a"])

st.set_page_config(page_title="Agent trace", layout="wide")
st.title("Agent trace viewer")
st.caption("Reads agent_trace.jsonl: every tool call with its inputs, output (200 chars), duration and status.")

path = st.sidebar.text_input("Trace file", str(TRACE_FILE))
events = load_events(path)
if events.empty:
    st.warning(f"No events in {path}. Run the notebook or `python -m research_agents NVDA` first.")
    st.stop()

run_table = runs(events)
labels = {r.run_id: f"{r.run_id} | {r.start:%Y-%m-%d %H:%M} UTC | {r.tool_calls} tool calls"
          for r in run_table.itertuples()}
run_id = st.sidebar.selectbox("Run", list(run_table["run_id"]), format_func=labels.get)
calls = tool_calls(events, run_id)
agents = sorted(calls["agent"].unique())
chosen = st.sidebar.multiselect("Agents", agents, default=agents)
calls = calls[calls["agent"].isin(chosen)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tool calls", len(calls))
c2.metric("Errors / denied", int(calls["status"].isin(["error", "denied"]).sum()))
c3.metric("Empty results", int((calls["status"] == "empty").sum()))
c4.metric("Tool time", f"{calls['duration_ms'].sum() / 1000:.1f} s")

st.subheader("Timeline")
tooltip = ["tool", "status", "duration_ms", "inputs", "output"]
bars = alt.Chart(calls).mark_bar(height=18).encode(
    x=alt.X("start:T", title="time (UTC)"), x2="ts:T", y=alt.Y("agent:N", title=None),
    color=alt.Color("status:N", scale=STATUS_COLORS), tooltip=tooltip)
dots = alt.Chart(calls).mark_circle(size=90).encode(
    x="start:T", y="agent:N", color=alt.Color("status:N", scale=STATUS_COLORS), tooltip=tooltip)
st.altair_chart((bars + dots).properties(height=60 + 50 * max(len(chosen), 1)), width="stretch")

left, right = st.columns(2)
per_tool = calls.groupby("tool").agg(calls=("tool", "size"), total_ms=("duration_ms", "sum")).reset_index()
left.subheader("Time per tool")
left.altair_chart(alt.Chart(per_tool).mark_bar().encode(
    x=alt.X("total_ms:Q", title="total ms"), y=alt.Y("tool:N", sort="-x", title=None),
    tooltip=["tool", "calls", "total_ms"]), width="stretch")
right.subheader("Calls per agent and status")
right.altair_chart(alt.Chart(calls).mark_bar().encode(
    x=alt.X("count():Q", title="calls"), y=alt.Y("agent:N", title=None),
    color=alt.Color("status:N", scale=STATUS_COLORS)), width="stretch")

st.subheader("Tool calls")
st.dataframe(calls[["start", "agent", "tool", "inputs", "status", "duration_ms", "output", "error"]],
             width="stretch", hide_index=True)

messages, handoffs, other = st.tabs(["Agent messages", "Handoffs", "Other events"])
with messages:
    for m in of_kind(events, "agent_message", run_id).itertuples():
        if getattr(m, "agent", None) not in chosen:
            continue
        calls_made = ", ".join(c["tool"] for c in (m.tool_calls or []))
        st.markdown(f"**{m.agent}** · {m.ts:%H:%M:%S}" + (f" → calls `{calls_made}`" if calls_made else ""))
        if isinstance(getattr(m, "reasoning", None), str) and m.reasoning:
            st.caption(f"thinks: {m.reasoning}")
        if m.content:
            st.text(m.content)
with handoffs:
    found = of_kind(events, "handoff", run_id)
    if found.empty:
        st.info("No handoffs in this run (the 3A single agent has none).")
    for h in found.itertuples():
        with st.expander(f"{h.sender} → {h.receiver}  ({h.schema})", expanded=False):
            st.json(h.payload)
with other:
    rest = events[(events["run_id"] == run_id) & ~events["kind"].isin(["tool_call", "agent_message", "handoff"])]
    cols = [c for c in ("ts", "kind", "text") if c in rest]
    st.dataframe(rest[cols], width="stretch", hide_index=True)

with st.sidebar.expander("Raw JSON of the last tool call"):
    if not calls.empty:
        st.code(json.dumps(calls.iloc[-1].astype(str).to_dict(), indent=2), language="json")
