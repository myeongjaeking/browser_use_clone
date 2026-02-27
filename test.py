"""
SPLX-detectable: Agents SDK + LangGraph (single file)
- Agents: OpenAI Agents SDK `agents.Agent`
- Tools: `@function_tool`
- MCP Servers: `MCPServerStdio` at module scope
- LangGraph: connects agent-level nodes

Business logic intentionally omitted (TODO placeholders).
"""

import os
import asyncio
from typing import TypedDict, Literal, Dict, Any, Optional

from langgraph.graph import StateGraph, END

# OpenAI Agents SDK (these are the key symbols SPLX is likely counting)
from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams


# =========================================================
# 1) Tools (Agents SDK style)
# =========================================================

@function_tool
def people_lookup(name: str) -> str:
    """Lookup internal employee info (deterministic)."""
    return "<TODO: people_lookup result>"


@function_tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location info (deterministic)."""
    return "<TODO: asset_lookup result>"


@function_tool
def faq_search(query: str) -> str:
    """FAQ search (vector/keyword)."""
    return "<TODO: faq_search result>"


@function_tool
def rag_retrieve(query: str) -> str:
    """RAG retrieve from vector index."""
    return "<TODO: rag_retrieve result>"


# =========================================================
# 2) MCP Server (module-scope, SPLX-friendly)
# =========================================================

ATLASSIAN_MCP_PARAMS: MCPServerStdioParams = {
    "command": "uvx",
    "args": ["mcp-atlassian"],
    "env": {
        "JIRA_URL": os.getenv("JIRA_URL", ""),
        "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN", ""),
        "CONFLUENCE_URL": os.getenv("CONFLUENCE_URL", ""),
        "CONFLUENCE_API_TOKEN": os.getenv("CONFLUENCE_API_TOKEN", ""),
    },
}

# Create the MCP server object at import time (static analyzers like this)
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)


# =========================================================
# 3) Agents (module-scope, SPLX-friendly)
# =========================================================

# Read agents
people_agent = Agent(
    name="People Agent",
    instructions="You answer questions about employees/teams using the people_lookup tool.",
    tools=[people_lookup],
)

asset_agent = Agent(
    name="Asset Agent",
    instructions="You answer questions about office assets/locations using the asset_lookup tool.",
    tools=[asset_lookup],
)

faq_agent = Agent(
    name="FAQ Agent",
    instructions="You answer questions using the faq_search tool.",
    tools=[faq_search],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions="You answer questions grounded in documents using rag_retrieve.",
    tools=[rag_retrieve],
)

# Action agent (draft only)
action_agent = Agent(
    name="Action Agent",
    instructions="You create structured drafts for Jira/Confluence actions. Do not execute.",
)

# Guardrail agent (review)
guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="You review drafts for policy/permission/risk and decide if execution is allowed.",
)

# MCP execution agent (the one that can call MCP-provided tools)
# Keep it explicit: mcp_servers set at definition time
mcp_executor_agent = Agent(
    name="MCP Executor Agent",
    instructions="You execute approved drafts by calling MCP tools on the Atlassian MCP server.",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)


# Triage agent (routes)
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Classify the user's request into one of: people, asset, faq, rag, action. "
        "Return ONLY the route label."
    ),
)


# =========================================================
# 4) LangGraph State
# =========================================================

class State(TypedDict, total=False):
    user_input: str
    route: Literal["people", "asset", "faq", "rag", "action"]
    draft: Dict[str, Any]
    approved: bool
    result: str


# =========================================================
# 5) LangGraph nodes (each node calls an Agents SDK Agent)
# =========================================================

async def triage_node(state: State) -> State:
    # TODO: in production, parse the triage output strictly
    res = await Runner.run(triage_agent, state["user_input"])
    route = (res.final_output or "").strip()

    # Safe fallback (keep deterministic for skeleton)
    if route not in {"people", "asset", "faq", "rag", "action"}:
        route = "rag"

    state["route"] = route  # type: ignore
    return state


async def people_node(state: State) -> State:
    res = await Runner.run(people_agent, state["user_input"])
    state["result"] = res.final_output
    return state


async def asset_node(state: State) -> State:
    res = await Runner.run(asset_agent, state["user_input"])
    state["result"] = res.final_output
    return state


async def faq_node(state: State) -> State:
    res = await Runner.run(faq_agent, state["user_input"])
    state["result"] = res.final_output
    return state


async def rag_node(state: State) -> State:
    res = await Runner.run(rag_agent, state["user_input"])
    state["result"] = res.final_output
    return state


async def action_node(state: State) -> State:
    # Draft generation only
    _ = await Runner.run(action_agent, state["user_input"])
    # TODO: structured draft output (pydantic output_type, JSON schema, etc.)
    state["draft"] = {"type": "jira", "payload": {"summary": "<TODO>", "description": "<TODO>"}}
    state["approved"] = False
    state["result"] = "Draft created. Needs guardrail/approval before MCP execution."
    return state


async def guardrail_node(state: State) -> State:
    # TODO: have guardrail_agent produce an explicit allow/deny decision
    _ = await Runner.run(guardrail_agent, "Review draft: <TODO>")
    # Skeleton: auto-approve (replace with real human approval)
    state["approved"] = True
    return state


async def mcp_execute_node(state: State) -> State:
    if not state.get("approved", False):
        state["result"] = "Approval required. Execution skipped."
        return state

    # MCP execution happens via MCP-enabled agent
    # TODO: craft a message instructing which MCP tool to call + args from state["draft"]
    res = await Runner.run(mcp_executor_agent, "Execute draft via MCP: <TODO>")
    state["result"] = res.final_output
    return state


def route_after_triage(state: State) -> str:
    return state.get("route", "rag")


# =========================================================
# 6) Build LangGraph (Agent-to-Agent flow)
# =========================================================

def build_graph():
    g = StateGraph(State)

    g.add_node("triage", triage_node)
    g.add_node("people", people_node)
    g.add_node("asset", asset_node)
    g.add_node("faq", faq_node)
    g.add_node("rag", rag_node)
    g.add_node("action", action_node)
    g.add_node("guardrail", guardrail_node)
    g.add_node("mcp_execute", mcp_execute_node)

    g.set_entry_point("triage")

    g.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "people": "people",
            "asset": "asset",
            "faq": "faq",
            "rag": "rag",
            "action": "action",
        },
    )

    g.add_edge("people", END)
    g.add_edge("asset", END)
    g.add_edge("faq", END)
    g.add_edge("rag", END)

    g.add_edge("action", "guardrail")
    g.add_edge("guardrail", "mcp_execute")
    g.add_edge("mcp_execute", END)

    return g.compile()


# =========================================================
# 7) Runner
# =========================================================

async def main():
    app = build_graph()
    out = await app.ainvoke({"user_input": "회식비 처리 절차 알려줘"})
    print(out.get("result", ""))


if __name__ == "__main__":
    asyncio.run(main())
