# %% [markdown]
# # Ex 1 Playground: feel a StateGraph
#
# This notebook is for live demo, not exercise. Three knobs to turn.
# Run each `# %%` cell one at a time and watch what changes.
#
# **Goal:** after this playground, the 3 primitives (state, node, edge) stop
# feeling like jargon and start feeling like tiny obvious pieces.

# %%
from __future__ import annotations

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    # DEMO SWAP: uncomment the line below and restart the kernel to see drift.
    # temperature=0.7,
)


# %% [markdown]
# ## Knob 1: state is just a TypedDict
#
# The whole "state machine" is this struct. Nothing magic. Each node returns
# a partial dict and LangGraph merges it in.

# %%
class SupportState(TypedDict):
    ticket_body: str
    category: Literal["TECHNICAL", "BILLING", "GENERAL"] | None
    # DEMO SWAP: add a PRICING branch -> comment out the line above and
    # uncomment the line below. Also uncomment CLASSIFY_PROMPT_PRICING later.
    # category: Literal["TECHNICAL", "BILLING", "GENERAL", "PRICING"] | None
    response: str | None


# Print an empty state to show what the graph carries around.
empty: SupportState = {"ticket_body": "", "category": None, "response": None}
print("State shape: ", empty)
print("State keys:  ", list(empty.keys()))


# %% [markdown]
# ## Knob 2: a node is just a function that returns a partial state
#
# Change the prompt below and rerun. Watch `category` flip. Try removing the
# "Reply with ONLY the category word" line to see what happens to the raw output.

# %%
CLASSIFY_PROMPT = (
    "Classify this support ticket into ONE of: TECHNICAL, BILLING, GENERAL.\n"
    "Reply with ONLY the category word.\n\n"
    "Ticket: {ticket}"
)

# DEMO SWAP A: weak prompt, no category list -> classifier drifts into essays.
# Copy the whole block below over CLASSIFY_PROMPT.
# CLASSIFY_PROMPT = "Classify this ticket: {ticket}"

# DEMO SWAP B: add PRICING branch (pairs with the Literal swap above).
# CLASSIFY_PROMPT = (
#     "Classify this support ticket into ONE of: TECHNICAL, BILLING, GENERAL, PRICING.\n"
#     "PRICING = questions about cost, quotes, enterprise plans, discounts.\n"
#     "Reply with ONLY the category word.\n\n"
#     "Ticket: {ticket}"
# )


def classify(state: SupportState) -> SupportState:
    raw = llm.invoke(CLASSIFY_PROMPT.format(ticket=state["ticket_body"])).content.strip().upper()
    # Tiny guard rail: fall back to GENERAL on any unexpected answer.
    valid = {"TECHNICAL", "BILLING", "GENERAL"}
    # valid = {"TECHNICAL", "BILLING", "GENERAL", "PRICING"}  # DEMO SWAP B
    category = raw if raw in valid else "GENERAL"
    print(f"  classify node input  ticket_body = {state['ticket_body']!r}")
    print(f"  classify node raw    llm output  = {raw!r}")
    print(f"  classify node output category    = {category!r}")
    return {"category": category}


def respond(state: SupportState) -> SupportState:
    template = f"[{state['category']} team] We received your ticket. Ref T-12345."
    print(f"  respond  node output response   = {template!r}")
    return {"response": template}


# %% [markdown]
# ## Knob 3: edges = the plumbing
#
# Compile the graph and print its shape. Comment out one `add_edge` and rerun
# to see LangGraph complain about an orphan node.

# %%
graph = StateGraph(SupportState)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.add_edge(START, "classify")
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)
# DEMO SWAP: comment out the "respond" edges above and uncomment below to see
# LangGraph reject a graph that has an orphan node. Good "why END matters" moment.
# graph.add_edge(START, "classify")
# graph.add_edge("classify", END)  # skip respond entirely

app = graph.compile()
print(app.get_graph().draw_ascii())


# %% [markdown]
# ## Run it, then change one thing
#
# First run produces a baseline. Then try these edits and rerun just this cell:
#
# 1. Change the ticket to something ambiguous (e.g. `"hi"`) and see what the
#    guard rail does.
# 2. Set `temperature=0.7` in the `llm = ChatOpenAI(...)` cell at the top of
#    this file, restart the kernel, and see classification drift.
# 3. Add a `"PRICING"` branch to the `SupportState` Literal and update
#    `CLASSIFY_PROMPT` to teach the model about it.

# %%
result = app.invoke({
    "ticket_body": "My API is returning 500 errors since 7am.",
    # DEMO SWAP: uncomment exactly one of these instead to see different routes.
    # "ticket_body": "Hi",                                              # ambiguous, hits guard rail
    # "ticket_body": "Refund for last month please.",                   # routes BILLING
    # "ticket_body": "Do you have SOC 2?",                              # routes GENERAL
    # "ticket_body": "How much is the Enterprise plan?",                # ambiguous, try with PRICING branch
    "category": None,
    "response": None,
})
print("\nFinal state:")
for k, v in result.items():
    print(f"  {k}: {v}")


# %% [markdown]
# ## What clicked
#
# - State is a plain dict. You can print it. You can inspect it. No wizardry.
# - A node reads state, returns a partial dict, LangGraph merges it.
# - Edges are the plumbing. START and END are sentinel node names.
#
# That is the whole surface area for 90% of production graphs. Everything in
# Ex 2, 3, 4, 5 is just this same primitive with more nodes and sharper edges.
