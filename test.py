"""
SPLX-recognizable skeleton (single file)

Targets:
- Tools detected via: langchain_core.tools.tool
- Agents detected via: langgraph.prebuilt.create_react_agent
- MCP Servers detected via: langchain_mcp_adapters.client.MultiServerMCPClient (module-scope)
- LangGraph connects agent-level nodes
"""

import os
import asyncio
from typing import TypedDict, Literal, Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# LLM
# =========================================================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# =========================================================
# Local Tools (deterministic placeholders)
# =========================================================

@tool
def people_lookup(name: str) -> str:
    """Lookup internal employee info from directory/HR."""
    return "<TODO: people_lookup result>"


@tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location info from asset DB."""
    return "<TODO: asset_lookup result>"


@tool
def faq_search(query: str) -> str:
    """Search FAQ (vector/keyword)."""
    return "<TODO: faq_search result>"


@tool
def rag_retrieve(query: str) -> str:
    """Retrieve documents from vector DB (Confluence/Jira index)."""
    return "<TODO: rag_retrieve result>"


# =========================================================
# MCP (declare config + client at module scope)
# =========================================================

ATLASSIAN_MCP_SERVER_CONFIG: Dict[str, Any] = {
    "atlassian": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-atlassian"],
        "env": {
            "JIRA_URL": os.getenv("JIRA_URL", ""),
            "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN", ""),
            "CONFLUENCE_URL": os.getenv("CONFLUENCE_URL", ""),
            "CONFLUENCE_API_TOKEN": os.getenv("CONFLUENCE_API_TOKEN", ""),
        },
    }
}

mcp_client = MultiServerMCPClient(ATLASSIAN_MCP_SERVER_CONFIG)
mcp_tools: List[Any] = []

mcp_exec_agent: Optional[Any] = None


async def init_mcp() -> None:
    """Load MCP tools at runtime (client stays module-scope for scanners)."""
    global mcp_tools, mcp_exec_agent
    mcp_tools = await mcp_client.get_tools()
    mcp_exec_agent = create_react_agent(llm, tools=mcp_tools)


# =========================================================
# Explicit Agents (module-scope)
# =========================================================

people_agent = create_react_agent(llm, tools=[people_lookup])
asset_agent = create_react_agent(llm, tools=[asset_lookup])
faq_agent = create_react_agent(llm, tools=[faq_search])
rag_agent = create_react_agent(llm, tools=[rag_retrieve])

action_agent = create_react_agent(llm, tools=[])
guardrail_agent = create_react_agent(llm, tools=[])


# =========================================================
# LangGraph State
# =========================================================

class State(TypedDict, total=False):
    user_input: str
    route: Literal["people", "asset", "faq", "rag", "action"]
    approved: bool
    draft: Dict[str, Any]
    result: str


# =========================================================
# Agent-level nodes (each node calls an Agent)
# =========================================================

async def triage_node(state: State) -> State:
    # TODO: real routing logic (LLM classify or rule)
    state["route"] = "rag"
    return state


async def people_node(state: State) -> State:
    resp = await people_agent.ainvoke(
        {"messages": [{"role": "user", "content": state["user_input"]}]}
    )
    state["result"] = resp["messages"][-1].content
    return state


async def asset_node(state: State) -> State:
    resp = await asset_agent.ainvoke(
        {"messages": [{"role": "user", "content": state["user_input"]}]}
    )
    state["result"] = resp["messages"][-1].content
    return state


async def faq_node(state: State) -> State:
    resp = await faq_agent.ainvoke(
        {"messages": [{"role": "user", "content": state["user_input"]}]}
    )
    state["result"] = resp["messages"][-1].content
    return state


async def rag_node(state: State) -> State:
    resp = await rag_agent.ainvoke(
        {"messages": [{"role": "user", "content": state["user_input"]}]}
    )
    state["result"] = resp["messages"][-1].content
    return state


async def action_node(state: State) -> State:
    _ = await action_agent.ainvoke(
        {"messages": [{"role": "user", "content": state["user_input"]}]}
    )
    state["draft"] = {"type": "jira", "payload": {"summary": "<TODO>", "description": "<TODO>"}}
    state["approved"] = False
    state["result"] = "Draft created. Approval required before MCP execution."
    return state


async def guardrail_node(state: State) -> State:
    _ = await guardrail_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Guardrail check for draft (TODO)."}]}
    )
    # TODO: replace with real approval gate
    state["approved"] = True
    return state


async def mcp_execute_node(state: State) -> State:
    if not state.get("approved", False):
        state["result"] = "Approval required. Execution skipped."
        return state

    if mcp_exec_agent is None:
        state["result"] = "MCP not initialized. Execution skipped."
        return state

    resp = await mcp_exec_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Execute draft via MCP (TODO)."}]}
    )
    state["result"] = resp["messages"][-1].content
    return state


def route_after_triage(state: State) -> str:
    return state.get("route", "rag")


# =========================================================
# Build Graph
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
# Runner
# =========================================================

async def main():
    await init_mcp()
    app = build_graph()

    out = await app.ainvoke({"user_input": "회식비 처리 절차 알려줘"})
    print(out.get("result", ""))


if __name__ == "__main__":
    asyncio.run(main())
