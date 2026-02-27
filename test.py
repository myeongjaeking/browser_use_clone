"""
Single-file skeleton:
- LangChain @tool
- LangChain Agent (create_agent)
- LangGraph orchestration
- MCP client (langchain-mcp-adapters)

No business logic implemented.
"""

import os
from typing import TypedDict, List, Dict, Any, Literal, Optional

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END


# =========================================================
# 1️⃣ LLM (Planner / Agent)
# =========================================================

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


# =========================================================
# 2️⃣ LangChain Tools (Local deterministic tools)
# =========================================================

@tool
def people_lookup(name: str) -> str:
    """Lookup employee information from internal directory."""
    # TODO: implement DB/LDAP lookup
    return "<TODO: employee info>"


@tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location info."""
    # TODO: implement asset DB lookup
    return "<TODO: asset info>"


@tool
def faq_search(query: str) -> str:
    """Search FAQ database (vector or keyword)."""
    # TODO: implement FAQ retrieval
    return "<TODO: FAQ answer>"


# =========================================================
# 3️⃣ MCP Setup (Atlassian)
# =========================================================

async def build_mcp_tools():
    """
    Connect to Atlassian MCP server.
    Replace env vars & args as needed.
    """
    client = MultiServerMCPClient(
        {
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
    )

    tools = await client.get_tools()
    return tools, client


# =========================================================
# 4️⃣ RAG Tool Placeholder (vector retrieval)
# =========================================================

@tool
def rag_retrieve(query: str) -> str:
    """Retrieve documents from vector DB (Confluence/Jira index)."""
    # TODO: integrate vector DB retrieval
    return "<TODO: RAG retrieved docs>"


# =========================================================
# 5️⃣ Agent State for LangGraph
# =========================================================

class AgentState(TypedDict, total=False):
    user_input: str
    plan: List[Dict[str, Any]]
    step_idx: int
    route: Literal["people", "asset", "faq", "rag", "action", "final"]
    result: str
    draft: Dict[str, Any]
    needs_approval: bool
    approved: bool


# =========================================================
# 6️⃣ LangGraph Nodes (Flow Only)
# =========================================================

def intake_node(state: AgentState) -> AgentState:
    state.setdefault("step_idx", 0)
    return state


def plan_node(state: AgentState) -> AgentState:
    """
    LLM-based planning (stub).
    """
    # TODO: generate structured plan with LLM
    state["plan"] = [
        {"kind": "rag", "tool": "rag_retrieve"},
    ]
    return state


def route_node(state: AgentState) -> AgentState:
    idx = state.get("step_idx", 0)
    plan = state.get("plan", [])

    if idx >= len(plan):
        state["route"] = "final"
    else:
        state["route"] = plan[idx]["kind"]
    return state


def run_people_node(state: AgentState) -> AgentState:
    # TODO: call people_lookup tool
    state["result"] = "<TODO: people result>"
    state["step_idx"] += 1
    return state


def run_asset_node(state: AgentState) -> AgentState:
    # TODO: call asset_lookup tool
    state["result"] = "<TODO: asset result>"
    state["step_idx"] += 1
    return state


def run_faq_node(state: AgentState) -> AgentState:
    # TODO: call faq_search tool
    state["result"] = "<TODO: FAQ result>"
    state["step_idx"] += 1
    return state


def run_rag_node(state: AgentState) -> AgentState:
    # TODO: call rag_retrieve tool
    state["result"] = "<TODO: RAG result>"
    state["step_idx"] += 1
    return state


def finalize_node(state: AgentState) -> AgentState:
    # TODO: generate final answer using LLM
    state["result"] = state.get("result", "<TODO: final answer>")
    return state


def route_decision(state: AgentState) -> str:
    return state.get("route", "final")


# =========================================================
# 7️⃣ Build Graph
# =========================================================

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("intake", intake_node)
    g.add_node("plan", plan_node)
    g.add_node("route", route_node)

    g.add_node("people", run_people_node)
    g.add_node("asset", run_asset_node)
    g.add_node("faq", run_faq_node)
    g.add_node("rag", run_rag_node)
    g.add_node("final", finalize_node)

    g.set_entry_point("intake")

    g.add_edge("intake", "plan")
    g.add_edge("plan", "route")

    g.add_conditional_edges("route", route_decision, {
        "people": "people",
        "asset": "asset",
        "faq": "faq",
        "rag": "rag",
        "final": "final",
    })

    g.add_edge("people", "route")
    g.add_edge("asset", "route")
    g.add_edge("faq", "route")
    g.add_edge("rag", "route")

    g.add_edge("final", END)

    return g.compile()


# =========================================================
# 8️⃣ Create LangChain Agent (LLM tool-calling)
# =========================================================

def build_langchain_agent(extra_tools: List[Any]):
    """
    Create an LLM agent that can call:
      - local tools (@tool)
      - MCP tools
    """
    tools = [people_lookup, asset_lookup, faq_search, rag_retrieve] + extra_tools
    agent = create_agent(f"openai:{LLM_MODEL}", tools)
    return agent


# =========================================================
# 9️⃣ Example Runner
# =========================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # MCP tools
        mcp_tools, _ = await build_mcp_tools()

        # LangChain agent
        agent = build_langchain_agent(mcp_tools)

        # LangGraph
        app = build_graph()

        state: AgentState = {
            "user_input": "회식비 처리 절차 알려줘",
            "approved": False,
        }

        result = app.invoke(state)
        print("FINAL RESULT:", result.get("result"))

    asyncio.run(main())
