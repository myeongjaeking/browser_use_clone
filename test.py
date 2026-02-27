"""
SPLX-recognizable skeleton (single file)

Goal:
- SPLX "Agents" count: >0 (via create_react_agent)
- SPLX "Tools" count: >0 (via langchain_core.tools @tool)
- SPLX "MCP Servers" count: >0 (via MultiServerMCPClient declared at module scope)
- LangGraph workflow connects agent-level nodes

No internal business logic implemented (TODO only).
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
# 0) LLM
# =========================================================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# =========================================================
# 1) Local Tools (SPLX-friendly: langchain_core.tools.tool)
# =========================================================

@tool
def people_lookup(name: str) -> str:
    """Lookup internal employee info from directory/HR."""
    # TODO: implement
    return "<TODO: people_lookup result>"


@tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location info from asset DB."""
    # TODO: implement
    return "<TODO: asset_lookup result>"


@tool
def faq_search(query: str) -> str:
    """Search FAQ (vector/keyword)."""
    # TODO: implement
    return "<TODO: faq_search result>"


@tool
def rag_retrieve(query: str) -> str:
    """Retrieve documents from vector DB (Confluence/Jira index)."""
    # TODO: implement
    return "<TODO: rag_retrieve result>"


# =========================================================
# 2) MCP (SPLX-friendly: declare MCP client config at module scope)
# =========================================================

# Keep MCP server config explicit and top-level for static analyzers.
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

# Create the client at module scope (some scanners fail if it's only inside an async fn).
mcp_client = MultiServerMCPClient(ATLASSIAN_MCP_SERVER_CONFIG)

# Tools loaded from MCP will be assigned here at runtime.
mcp_tools: List[Any] = []


async def init_mcp_tools() -> None:
    """Load MCP tools at runtime (kept separate, but client/config remain top-level)."""
    global mcp_tools
    # TODO: handle errors, auth, etc.
    mcp_tools = await mcp_client.get_tools()


# =========================================================
# 3) Explicit Agents (SPLX-friendly: create_react_agent at module scope)
# =========================================================

# Read-only / domain agents
people_agent = create_react_agent(llm, tools=[people_lookup])
asset_agent = create_react_agent(llm, tools=[asset_lookup])
faq_agent = create_react_agent(llm, tools=[faq_search])
rag_agent = create_react_agent(llm, tools=[rag_retrieve])

# Action agent: draft only (do not execute writes here)
action_agent = create_react_agent(llm, tools=[])

# Guardrail agent: can be LLM-based or rules (skeleton only)
guardrail_agent = create_react_agent(llm, tools=[])

# MCP execution agent: will get MCP tools at runtime, but the agent object can still be created
# once tools are loaded. We keep a placeholder variable here.
mcp_exec_agent: Optional[Any] = None


def build_mcp_exec_agent() -> None:
    """Create an agent that can call MCP tools (write/read)."""
    global mcp_exec_agent
    # Important: create_react_agent expects the tools list.
    # MCP tools are loaded dynamically into `mcp_tools`.
    mcp_exec_agent = create_react_agent(llm, tools=mcp_tools)


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
# 5) LangGraph Nodes (Agent-level nodes: each node invokes an Agent)
# =========================================================

async def triage_node(state: State) -> State:
    """
    Decide which agent to call next.
    Keep this deterministic or LLM-based.
    """
    # TODO: implement routing (LLM classify / rules).
    # For skeleton, default to rag.
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
    """
    Create a draft action (Jira/Confluence) but do not execute.
    """
    # TODO: generate structured draft with LLM
    _ = await action_agent.ainvoke(
        {"messages": [{"role": "user", "content": state["user_input"]}]}
    )
    state["draft"] = {"type": "jira", "payload": {"summary": "<TODO>", "description": "<TODO>"}}
    state["approved"] = False  # approval gate
    return state


async def guardrail_node(state: State) -> State:
    """
    Validate safety/policy before write.
    """
    # TODO: implement checks (rules + optional LLM review)
    _ = await guardrail_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Check guardrails for draft: <TODO>"}]}
    )
    state["approved"] = True  # for skeleton; real system should require explicit approval
    return state


async def mcp_execute_node(state: State) -> State:
    """
    Execute via MCP tools (Confluence/Jira CRUD).
    Must be gated by approval.
    """
    if not state.get("approved", False):
        state["result"] = "Approval required. Execution skipped."
        return state

    if mcp_exec_agent is None:
        state["result"] = "MCP tools not initialized. Execution skipped."
        return state

    # TODO: choose correct MCP tool based on draft["type"]
    # Example: call the MCP tool via agent tool-calling
    resp = await mcp_exec_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Execute draft via MCP: <TODO>"}]}
    )
    state["result"] = resp["messages"][-1].content
    return state


def route_after_triage(state: State) -> str:
    return state.get("route", "rag")


# =========================================================
# 6) Build Graph
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

    # Action path: action -> guardrail -> mcp_execute -> END
    g.add_edge("action", "guardrail")
    g.add_edge("guardrail", "mcp_execute")
    g.add_edge("mcp_execute", END)

    # Read paths: end immediately
    g.add_edge("people", END)
    g.add_edge("asset", END)
    g.add_edge("faq", END)
    g.add_edge("rag", END)

    return g.compile()


# =========================================================
# 7) Runner
# =========================================================

async def main():
    # Initialize MCP tools at runtime
    await init_mcp_tools()
    build_mcp_exec_agent()

    app = build_graph()

    state: State = {"user_input": "회식비 처리 절차 알려줘", "approved": False}
    out = await app.ainvoke(state)
    print(out.get("result", ""))


if __name__ == "__main__":
    asyncio.run(main())
