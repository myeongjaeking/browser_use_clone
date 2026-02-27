import os
import asyncio
from typing import TypedDict, Literal, List, Dict, Any

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END


# =========================================================
# LLM
# =========================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model=MODEL, temperature=0)


# =========================================================
# Local Tools
# =========================================================

@tool
def people_lookup(name: str) -> str:
    """Lookup internal employee info."""
    return "<TODO: employee info>"


@tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location info."""
    return "<TODO: asset info>"


@tool
def faq_search(query: str) -> str:
    """Search FAQ DB."""
    return "<TODO: FAQ answer>"


@tool
def rag_retrieve(query: str) -> str:
    """Retrieve docs from vector DB."""
    return "<TODO: RAG retrieved content>"


# =========================================================
# MCP Setup
# =========================================================

async def build_mcp_tools():
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
# Agent State
# =========================================================

class AgentState(TypedDict, total=False):
    user_input: str
    route: Literal["people", "asset", "faq", "rag", "action", "final"]
    draft: Dict[str, Any]
    result: str
    approved: bool


# =========================================================
# Agent Wrappers
# =========================================================

class LLMToolAgent:
    def __init__(self, name: str, tools: List[Any]):
        self.name = name
        self.agent = create_agent(f"openai:{MODEL}", tools)

    async def run(self, message: str) -> str:
        res = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]}
        )
        return str(res["messages"][-1].content)


# =========================================================
# Individual Agents
# =========================================================

# Triage Agent (LLM)
triage_agent = LLMToolAgent("triage", [])

# RAG Agent (LLM + rag tool)
rag_agent = LLMToolAgent("rag", [rag_retrieve])

# FAQ Agent (LLM 없이 tool 호출만)
class FAQAgent:
    def run(self, query: str) -> str:
        return faq_search.invoke({"query": query})

faq_agent = FAQAgent()

# Guardrail Agent
class GuardrailAgent:
    def check(self, draft: Dict[str, Any]) -> bool:
        return True  # TODO

guard_agent = GuardrailAgent()


# =========================================================
# LangGraph Nodes (Agent-level)
# =========================================================

async def triage_node(state: AgentState) -> AgentState:
    # TODO: LLM classification
    state["route"] = "rag"
    return state


async def people_node(state: AgentState) -> AgentState:
    state["result"] = people_lookup.invoke({"name": state["user_input"]})
    return state


async def asset_node(state: AgentState) -> AgentState:
    state["result"] = asset_lookup.invoke({"item": state["user_input"]})
    return state


async def faq_node(state: AgentState) -> AgentState:
    state["result"] = faq_agent.run(state["user_input"])
    return state


async def rag_node(state: AgentState) -> AgentState:
    state["result"] = await rag_agent.run(state["user_input"])
    return state


async def action_node(state: AgentState) -> AgentState:
    # TODO: create draft via LLM
    state["draft"] = {"type": "jira", "payload": {}}
    return state


async def guard_node(state: AgentState) -> AgentState:
    if guard_agent.check(state["draft"]):
        state["approved"] = True
    return state


async def execute_node(state: AgentState) -> AgentState:
    # TODO: call MCP write tool
    state["result"] = "<TODO: executed via MCP>"
    return state


def route_decision(state: AgentState) -> str:
    return state.get("route", "final")


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("people", people_node)
    graph.add_node("asset", asset_node)
    graph.add_node("faq", faq_node)
    graph.add_node("rag", rag_node)
    graph.add_node("action", action_node)
    graph.add_node("guard", guard_node)
    graph.add_node("execute", execute_node)

    graph.set_entry_point("triage")

    graph.add_conditional_edges("triage", route_decision, {
        "people": "people",
        "asset": "asset",
        "faq": "faq",
        "rag": "rag",
        "action": "action",
    })

    graph.add_edge("action", "guard")
    graph.add_edge("guard", "execute")

    graph.add_edge("people", END)
    graph.add_edge("asset", END)
    graph.add_edge("faq", END)
    graph.add_edge("rag", END)
    graph.add_edge("execute", END)

    return graph.compile()


# =========================================================
# Runner
# =========================================================

async def main():
    app = build_graph()

    state: AgentState = {
        "user_input": "회식비 처리 절차 알려줘"
    }

    result = await app.ainvoke(state)
    print(result.get("result"))


if __name__ == "__main__":
    asyncio.run(main())
