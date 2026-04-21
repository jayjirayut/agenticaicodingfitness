# %% [markdown]
# # Ex 3 Playground: touch the checkpoint
#
# This notebook is for the "ok but what does checkpointing actually DO" moment.
# Four experiments, each shows a different angle of the same SqliteSaver + interrupt() pair.
#
# Run the cells in order the first time. Then re-run experiments independently.
# The `checkpoints.db` file on disk is the whole persistence story.

# %%
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# %% [markdown]
# ## Build the graph once

# %%
class TicketState(TypedDict):
    ticket_body: str
    draft: str | None
    approved: bool | None
    sent: str | None


def draft_node(state: TicketState) -> TicketState:
    reply = llm.invoke(f"2-sentence polite reply to: {state['ticket_body']}").content
    return {"draft": reply}


def human_gate(state: TicketState) -> TicketState:
    decision = interrupt({"draft": state["draft"], "ask": "approve / edit / reject"})
    if isinstance(decision, dict):
        if decision.get("action") == "approve":
            return {"approved": True}
        if decision.get("action") == "edit":
            return {"draft": decision["text"], "approved": True}
    return {"approved": False}


def send_node(state: TicketState) -> TicketState:
    if not state.get("approved"):
        return {"sent": "[REJECTED, not sent]"}
    return {"sent": f"SENT: {state['draft']}"}


DB_PATH = "playground_checkpoints.db"
# Wipe the db at the top of the file so each live demo starts clean.
if Path(DB_PATH).exists():
    Path(DB_PATH).unlink()

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

g = StateGraph(TicketState)
g.add_node("draft", draft_node)
g.add_node("human_gate", human_gate)
g.add_node("send", send_node)
g.add_edge(START, "draft")
g.add_edge("draft", "human_gate")
g.add_edge("human_gate", "send")
g.add_edge("send", END)
app = g.compile(checkpointer=memory)

print("Graph compiled. Checkpoint db:", DB_PATH)


# %% [markdown]
# ## Experiment 1: see the pause
#
# `invoke()` returns as soon as the graph hits `interrupt()`. The return value
# is the state AT that pause point. The graph is literally frozen on disk.

# %%
config_1 = {"configurable": {"thread_id": "ticket-A"}}

state = app.invoke(
    {"ticket_body": "Dashboard won't load.", "draft": None, "approved": None, "sent": None},
    config=config_1,
)
print("After invoke() (graph paused at human_gate):")
print("  draft:     ", state.get("draft"))
print("  approved:  ", state.get("approved"))
print("  sent:      ", state.get("sent"), "  <- still None because send hasn't run")


# %% [markdown]
# ## Experiment 2: resume with APPROVE
#
# `Command(resume=...)` unpauses the graph. The value passed is what the
# `interrupt()` call returns. No re-run of earlier nodes, so no extra LLM cost.

# %%
state = app.invoke(Command(resume={"action": "approve"}), config=config_1)
# DEMO SWAP: comment out the line above and uncomment exactly one below.
# state = app.invoke(Command(resume={"action": "reject"}), config=config_1)
# state = app.invoke(
#     Command(resume={"action": "edit", "text": "We're on it. ETA 1 hour."}),
#     config=config_1,
# )
print("After resume APPROVE:")
print("  approved: ", state.get("approved"))
print("  sent:     ", state.get("sent"))


# %% [markdown]
# ## Experiment 3: same thread, another round, REJECT this time
#
# Start a new ticket on a new thread_id so the first thread's state stays put.

# %%
config_2 = {"configurable": {"thread_id": "ticket-B"}}
# DEMO SWAP: run the same ticket on "ticket-A" (the approved one) to see that
# LangGraph treats it as a new run on the SAME thread, generating a fresh pause.
# config_2 = {"configurable": {"thread_id": "ticket-A"}}

state = app.invoke(
    {"ticket_body": "Cancel my account NOW.", "draft": None, "approved": None, "sent": None},
    config=config_2,
)
print("ticket-B paused, draft:", state.get("draft")[:120], "...")

state = app.invoke(Command(resume={"action": "reject"}), config=config_2)
print("After resume REJECT:")
print("  approved: ", state.get("approved"))
print("  sent:     ", state.get("sent"))


# %% [markdown]
# ## Experiment 4: state lives on disk. Prove it.
#
# Open the SQLite file directly. You will see rows for every checkpoint.
# This is why the graph survives a restart. Kill Python, reopen, resume.

# %%
with sqlite3.connect(DB_PATH) as read_conn:
    rows = read_conn.execute(
        "SELECT thread_id, checkpoint_id, type FROM checkpoints ORDER BY thread_id, checkpoint_id"
    ).fetchall()

print(f"\nTotal checkpoints persisted: {len(rows)}")
for thread_id, checkpoint_id, typ in rows[:10]:
    print(f"  thread={thread_id}  checkpoint={checkpoint_id[:20]}...  type={typ}")


# %% [markdown]
# ## What clicked
#
# - `interrupt()` inside a node pauses the whole graph. `Command(resume=...)`
#   unpauses it. Those two primitives are the whole HITL story.
# - `SqliteSaver` writes a row per node transition, keyed by `thread_id`.
#   Different `thread_id` = different conversation, isolated state.
# - After a crash, `app.invoke(Command(resume=...), config=config)` picks up
#   from the last saved checkpoint. No re-running of `draft_node`, no re-billing
#   the LLM.
#
# ## Your turn during class
#
# - Re-run Experiment 1 with the same `thread_id`. Notice the draft is regenerated
#   (because the pause was already consumed). Then try a fresh thread_id.
# - Pass `{"action": "edit", "text": "We're on it. 1hr ETA."}` instead of approve.
# - Delete `playground_checkpoints.db` between runs to reset the demo.
