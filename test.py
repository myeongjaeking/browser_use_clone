"""
SPLX-detectable: Orchestrator Routes to Sub-Agents (LangGraph)
start -> Orchestrator Agent (route) -> Knowledge/RAG/MCP Executor -> end
"""

import os
import asyncio
from typing import TypedDict, Literal, Dict, Any, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams


# =========================================================
# 1) Tools
# =========================================================
@function_tool
def people_lookup(name: str) -> str:
    return f"{name}: 개발팀, 010-XXXX"


@function_tool
def asset_lookup(item: str) -> str:
    return f"{item}: A동 301호"


@function_tool
def faq_search(query: str) -> str:
    return f"{query}: 총무 신청 → Slack #expense"


@function_tool
def rag_retrieve(query: str) -> str:
    return f"{query}: Q1 보고서 발췌 내용"


# =========================================================
# 2) MCP Server
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
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)


# =========================================================
# 3) Sub Agents
# =========================================================
knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions="사원/자산/FAQ 조회. 필요 시 적절한 tool을 호출해서 답한다.",
    tools=[people_lookup, asset_lookup, faq_search],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions="문서 검색. rag_retrieve로 근거를 가져와 답한다.",
    tools=[rag_retrieve],
)

mcp_executor_agent = Agent(
    name="MCP Executor Agent",
    instructions="Jira/Confluence 작업을 MCP로 실행한다. 필요한 MCP tool을 호출한다.",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)


# =========================================================
# 4) Orchestrator Agent (ROUTER ONLY)
#    - 여기서는 tool 호출을 하지 않고, route만 결정하게 만든다.
# =========================================================
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions=(
        "사용자 입력을 분류해 다음 중 하나만 출력하라(소문자, 한 단어).\n"
        "- knowledge: 사원/팀/연락처, 자산 위치, FAQ\n"
        "- rag: 문서/보고서/정책/절차 검색 및 요약\n"
        "- mcp: Jira/Confluence 생성/수정/실행 작업\n\n"
        "반드시 다음 중 하나만 출력: knowledge | rag | mcp"
    ),
)


# =========================================================
# 5) State
# =========================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    route: Literal["knowledge", "rag", "mcp"]
    result: str


# =========================================================
# 6) LangGraph Nodes (Agent 단위 노드)
# =========================================================
async def orchestrator_node(state: State) -> State:
    res = await Runner.run(orchestrator_agent, state["user_input"])
    route = (res.final_output or "").strip().lower()

    if route not in {"knowledge", "rag", "mcp"}:
        route = "rag"

    state["route"] = route  # type: ignore
    return state


async def knowledge_node(state: State) -> State:
    res = await Runner.run(knowledge_agent, state["user_input"])
    state["result"] = res.final_output or ""
    return state


async def rag_node(state: State) -> State:
    res = await Runner.run(rag_agent, state["user_input"])
    state["result"] = res.final_output or ""
    return state


async def mcp_node(state: State) -> State:
    # 실제론 승인/가드레일을 추가하는 게 안전하지만, 여기서는 흐름만.
    res = await Runner.run(mcp_executor_agent, state["user_input"])
    state["result"] = res.final_output or ""
    return state


def route_after_orchestrator(state: State) -> str:
    return state["route"]


# =========================================================
# 7) Build Graph (그래프가 원하는 모양으로 나옴)
# =========================================================
def build_graph():
    g = StateGraph(State)

    g.add_node("Orchestrator Agent", orchestrator_node)
    g.add_node("Knowledge Agent", knowledge_node)
    g.add_node("RAG Agent", rag_node)
    g.add_node("MCP Executor Agent", mcp_node)

    g.set_entry_point("Orchestrator Agent")

    g.add_conditional_edges(
        "Orchestrator Agent",
        route_after_orchestrator,
        {
            "knowledge": "Knowledge Agent",
            "rag": "RAG Agent",
            "mcp": "MCP Executor Agent",
        },
    )

    g.add_edge("Knowledge Agent", END)
    g.add_edge("RAG Agent", END)
    g.add_edge("MCP Executor Agent", END)

    return g.compile()


# =========================================================
# 8) Test Runner
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
        print("=" * 50)
        print("INPUT :", t)
        print("OUTPUT:", out["result"])


if __name__ == "__main__":
    asyncio.run(main())
