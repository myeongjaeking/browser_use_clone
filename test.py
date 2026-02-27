"""
Multi-Agent Interaction Skeleton
- Agents interact with each other (no LangGraph nodes)
- LangChain @tool used
- MCP integrated
- No internal logic implemented
"""

import os
import asyncio
from typing import Any, Dict, List

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# LLM
# =========================================================

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


# =========================================================
# Local Deterministic Tools
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
    """Retrieve documents from vector DB."""
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
# Agent Classes
# =========================================================

class BaseAgent:
    def __init__(self, name: str, tools: List[Any]):
        self.name = name
        self.agent = create_agent(f"openai:{LLM_MODEL}", tools)

    async def run(self, message: str) -> str:
        response = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]}
        )
        return str(response["messages"][-1].content)


class TriageAgent(BaseAgent):
    """
    Decides which agent to delegate to.
    """

    async def decide(self, user_input: str) -> str:
        # TODO: Use LLM classification
        return "rag"  # "faq", "people", "asset", "action"


class FAQAgent:
    """
    Deterministic (LLM X)
    """

    def run(self, query: str) -> str:
        return faq_search.invoke({"query": query})


class RAGAgent(BaseAgent):
    """
    LLM O + RAG tool
    """

    pass


class ActionAgent(BaseAgent):
    """
    Generates draft for Jira/Confluence change.
    """

    async def create_draft(self, user_input: str) -> Dict[str, Any]:
        # TODO: generate structured draft
        return {"type": "jira", "payload": {"summary": "<TODO>"}}


class GuardrailAgent:
    """
    Checks risk/permissions.
    """

    def check(self, draft: Dict[str, Any]) -> bool:
        # TODO: implement rule checks
        return True


class MCPExecutorAgent:
    """
    Executes write via MCP tools.
    """

    def __init__(self, mcp_tools):
        self.mcp_tools = mcp_tools

    async def execute(self, draft: Dict[str, Any]) -> str:
        # TODO: call appropriate MCP tool
        return "<TODO: executed via MCP>"


# =========================================================
# Multi-Agent Interaction Flow
# =========================================================

class CorporateAIAssistant:
    def __init__(self, mcp_tools):
        self.triage = TriageAgent("triage", [])
        self.faq = FAQAgent()
        self.rag = RAGAgent("rag", [rag_retrieve])
        self.action = ActionAgent("action", [])
        self.guard = GuardrailAgent()
        self.executor = MCPExecutorAgent(mcp_tools)

    async def handle(self, user_input: str) -> str:
        route = await self.triage.decide(user_input)

        if route == "people":
            return people_lookup.invoke({"name": user_input})

        elif route == "asset":
            return asset_lookup.invoke({"item": user_input})

        elif route == "faq":
            return self.faq.run(user_input)

        elif route == "rag":
            return await self.rag.run(user_input)

        elif route == "action":
            draft = await self.action.create_draft(user_input)
            if self.guard.check(draft):
                return await self.executor.execute(draft)
            else:
                return "Action blocked by guardrail."

        return "Unable to determine route."


# =========================================================
# Runner
# =========================================================

async def main():
    mcp_tools, _ = await build_mcp_tools()
    assistant = CorporateAIAssistant(mcp_tools)

    result = await assistant.handle("회식비 처리 절차 알려줘")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
