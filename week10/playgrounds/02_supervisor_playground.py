# %% [markdown]
# # Ex 2 Playground: why the supervisor is "just an LLM that calls tools"
#
# This notebook demos the tool-calling supervisor pattern LangChain recommends
# as of April 2026. Three knobs:
#
# 1. The supervisor prompt (what routing behavior we ask for)
# 2. The tool docstrings (how the LLM decides which tool to pick)
# 3. The ticket content (does routing actually work for edge cases?)
#
# **Goal:** after this, "multi-agent system" stops feeling mystical. It is one
# LLM with three functions exposed as tools.

# %%
from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# %% [markdown]
# ## Knob 1: tool docstrings ARE the routing logic
#
# The LLM reads the docstring of every `@tool` function and picks the best
# match. Change a docstring, change routing behavior. No if/else statements.

# %%
@tool
def technical_specialist(ticket: str) -> str:
    """Handle TECHNICAL support (APIs, devices, dashboards, data sync, error codes, 500s, traces)."""
    print(f"  [routed to] technical_specialist")
    return llm.invoke(f"Senior tech support reply in 2 sentences:\n\n{ticket}").content


@tool
def billing_specialist(ticket: str) -> str:
    """Handle BILLING (charges, invoices, refunds, plan upgrades, tax receipts)."""
    print(f"  [routed to] billing_specialist")
    return llm.invoke(f"Billing specialist reply in 2 sentences:\n\n{ticket}").content


@tool
def general_specialist(ticket: str) -> str:
    """Handle GENERAL pre-sales questions (demos, references, SLA, integrations, trials)."""
    print(f"  [routed to] general_specialist")
    return llm.invoke(f"Friendly pre-sales reply in 2 sentences:\n\n{ticket}").content


SPECIALISTS = [technical_specialist, billing_specialist, general_specialist]

# DEMO SWAP: add a 4th specialist and re-include SPECIALISTS below. Uncomment
# the whole block (escalation_specialist + new SPECIALISTS line) to activate.
#
# @tool
# def escalation_specialist(ticket: str) -> str:
#     """Handle ANGRY customers, refund demands, account cancellations, or profanity."""
#     print(f"  [routed to] escalation_specialist")
#     return llm.invoke(
#         f"You are a senior escalation agent. De-escalate in 2 sentences:\n\n{ticket}"
#     ).content
#
# SPECIALISTS = [technical_specialist, billing_specialist, general_specialist, escalation_specialist]


# %% [markdown]
# ## Knob 2: the supervisor prompt shapes behavior
#
# `create_react_agent` gives you a tool-calling ReAct loop. The prompt below
# is the whole "supervisor" logic. Try swapping in the alternates after you
# run the baseline.

# %%
SUPERVISOR_PROMPT_STRICT = (
    "You are the support supervisor. Read the ticket, pick exactly ONE specialist tool, "
    "call it with the original ticket text, then stop. Do not answer the ticket yourself."
)

# Alternates to try:
SUPERVISOR_PROMPT_CHATTY = (
    "You are a support supervisor. Pick a specialist. Before calling it, write one "
    "sentence explaining why you picked that specialist. Then call the tool."
)

SUPERVISOR_PROMPT_GREEDY = (
    "You are the support supervisor. Call EVERY specialist tool you think is relevant, "
    "then summarise the best answer yourself. Do not stop after one tool."
)

supervisor = create_react_agent(
    model=llm,
    tools=SPECIALISTS,
    prompt=SUPERVISOR_PROMPT_STRICT,
    # DEMO SWAP: comment the line above and uncomment exactly one below.
    # prompt=SUPERVISOR_PROMPT_CHATTY,   # LLM narrates WHY before routing
    # prompt=SUPERVISOR_PROMPT_GREEDY,   # LLM calls multiple tools and summarises
)


# %% [markdown]
# ## Knob 3: watch routing happen
#
# Each tool prints `[routed to] ...` when invoked, so you can see the path
# the supervisor chose. Try ambiguous tickets to see the tie-breaker behavior.

# %%
TICKETS = [
    ("easy technical", "My API is returning 500 errors since 7am."),
    ("easy billing",   "I was charged twice for last month's plan."),
    ("easy general",   "Do you have a reference customer in retail?"),
    # Edge cases:
    ("ambiguous",      "How much does your Enterprise plan cost and does it include SLA?"),
    ("angry",          "This is the third time I'm asking for a refund!"),
]


def run_supervisor(state):
    result = supervisor.invoke({"messages": [("user", state["ticket_body"])]})
    return {"final_response": result["messages"][-1].content}


class _State(TypedDict):
    ticket_body: str
    final_response: str | None
    messages: Annotated[list, operator.add]


g = StateGraph(_State)
g.add_node("supervisor", run_supervisor)
g.add_edge(START, "supervisor")
g.add_edge("supervisor", END)
app = g.compile()


for label, ticket in TICKETS:
    print("\n" + "=" * 60)
    print(f"CASE: {label}")
    print(f"Ticket: {ticket}")
    print("-" * 60)
    out = app.invoke({"ticket_body": ticket, "final_response": None, "messages": []})
    print(f"Answer: {out['final_response'][:200]}")


# %% [markdown]
# ## What clicked
#
# - The supervisor is NOT a special class. It is `create_react_agent(llm, tools, prompt)`.
# - Routing happens via tool docstrings, not imperative code. Change a docstring,
#   change the routing.
# - The "multi-agent" metaphor is a helpful story, but under the hood it is one
#   LLM calling functions. No extra services, no message bus.
#
# ## Your turn during class
#
# - Swap `SUPERVISOR_PROMPT_CHATTY` in and see the reasoning trace in stdout.
# - Swap `SUPERVISOR_PROMPT_GREEDY` in and see the LLM chain multiple tools.
# - Add a 4th specialist, e.g. `escalation_specialist` for refund-demand tickets,
#   and watch the "angry" case route differently.
