"""
SPLX-detectable: Orchestrator-Driven Multi-Agent Flow (LangGraph)
Orchestrator Agent -> (Knowledge Agent | RAG Agent | MCP Executor Agent) -> END

Goal:
- SPLX shows Agents > 0, Tools > 0, MCP Servers > 0
- Graph shows Orchestrator selecting other Agents via explicit tool calls

Notes:
- Business logic intentionally minimal (demo responses).
- MCP is wired for execution agent; actual Jira/Confluence tool calls are TODO.
"""

import os
import asyncio
from typing import TypedDict, Annotated, Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams


# =========================================================
# 1) Tools (Knowledge / RAG local demo tools)
# =========================================================

@function_tool
def people_lookup(name: str) -> str:
    """Lookup employee info (demo)."""
    return f"{name}: 개발팀, 010-XXXX"


@function_tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location (demo)."""
    return f"{item}: A동 301호"


@function_tool
def faq_search(query: str) -> str:
    """FAQ search (demo)."""
    return f"{query}: 총무 신청 → Slack #expense"


@function_tool
def rag_retrieve(query: str) -> str:
    """RAG retrieve from vector index (demo)."""
    return f"{query}: Q1 보고서 발췌 내용"


# =========================================================
# 2) MCP Server (Atlassian)
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

# Module-scope object for static analyzers (SPLX)
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)


# =========================================================
# 3) Sub Agents (Knowledge / RAG / MCP Execute)
# =========================================================

knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions=(
        "사원/조직/자산/FAQ 성격의 질문에 답한다. "
        "필요 시 tools(people_lookup, asset_lookup, faq_search) 중 적절한 것을 호출한다."
    ),
    tools=[people_lookup, asset_lookup, faq_search],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions=(
        "문서 기반 질의에 답한다. "
        "rag_retrieve로 근거를 가져오고, 그 근거로 답변한다."
    ),
    tools=[rag_retrieve],
)

mcp_executor_agent = Agent(
    name="MCP Executor Agent",
    instructions=(
        "승인된 작업을 Atlassian MCP를 통해 실행한다. "
        "Jira/Confluence 생성/수정/조회 등 필요한 MCP tool을 호출해 결과를 반환한다."
    ),
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)


# =========================================================
# 4) Orchestrator Tools (Orchestrator가 다른 Agent를 호출하기 위한 tool)
#    - IMPORTANT: function_tool는 Orchestrator Agent가 '도구 호출'로 인식
# =========================================================

@function_tool
async def call_knowledge(query: str) -> str:
    """Knowledge Agent를 호출해 답을 받아온다."""
    res = await Runner.run(knowledge_agent, query)
    return res.final_output or "Knowledge Agent 실패"


@function_tool
async def call_rag(query: str) -> str:
    """RAG Agent를 호출해 문서 근거 기반 답을 받아온다."""
    res = await Runner.run(rag_agent, query)
    return res.final_output or "RAG Agent 실패"


@function_tool
async def call_mcp(command: str) -> str:
    """MCP Executor Agent를 호출해 Jira/Confluence 작업을 실행한다."""
    res = await Runner.run(mcp_executor_agent, command)
    return res.final_output or "MCP Executor Agent 실패"


# =========================================================
# 5) Orchestrator Agent (선택/합성/실행 주체)
# =========================================================

orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions=(
        "사용자 입력을 보고 아래 중 필요한 Agent를 선택하여 실행한다.\n"
        "- call_knowledge: 사원/팀/연락처/자산 위치/FAQ\n"
        "- call_rag: 문서 검색 및 요약/근거 기반 답변\n"
        "- call_mcp: Jira/Confluence 작업 실행(생성/수정/조회)\n\n"
        "가능하면 단일 tool로 해결하되, 필요하면 순차 호출하고 최종 답을 합성한다.\n"
        "작업 실행(call_mcp)은 '명령' 형태로 구체적으로 전달한다.\n"
        "최종 출력은 사용자에게 바로 전달 가능한 형태로 작성한다."
    ),
    tools=[call_knowledge, call_rag, call_mcp],
)


# =========================================================
# 6) LangGraph State
# =========================================================

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_input: str
    result: str


# =========================================================
# 7) LangGraph Node (Orchestrator only)
#    - 그래프는 1노드지만, 내부에서 Orchestrator가 다른 Agent들을 선택/호출함
# =========================================================

async def orchestrator_node(state: State) -> State:
    res = await Runner.run(orchestrator_agent, state["user_input"])
    state["result"] = res.final_output
    return state


# =========================================================
# 8) Build Graph
# =========================================================

def build_graph():
    g = StateGraph(State)
    g.add_node("Orchestrator Agent", orchestrator_node)
    g.set_entry_point("Orchestrator Agent")
    g.add_edge("Orchestrator Agent", END)
    return g.compile()


# =========================================================
# 9) Test Runner
# =========================================================

async def main():
    app = build_graph()

    tests = [
        "회식비 처리 절차 알려줘",
        "김팀장 어디 있어?",
        "Q1 보고서 요약해줘",
        "Jira 티켓 '긴급 버그' 생성해",
    ]

    for t in tests:
        out = await app.ainvoke({"user_input": t})
        print("=" * 60)
        print("INPUT :", t)
        print("OUTPUT:", out.get("result", ""))


if __name__ == "__main__":
    asyncio.run(main())
