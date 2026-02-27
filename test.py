import os
import asyncio
from typing import TypedDict, Literal, Dict, Annotated

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams


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

knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions="사원/자산/FAQ 조회.",
    tools=[people_lookup, asset_lookup, faq_search],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions="문서 검색.",
    tools=[rag_retrieve],
)

mcp_executor_agent = Agent(
    name="MCP Executor Agent",
    instructions="Jira/Confluence 작업 실행.",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)

orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions=(
        "입력을 분류해 아래 중 하나만 출력:\n"
        "knowledge | rag | mcp\n"
        "반드시 해당 단어만 출력."
    ),
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    route: Literal["knowledge", "rag", "mcp"]
    result: str


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
    res = await Runner.run(mcp_executor_agent, state["user_input"])
    state["result"] = res.final_output or ""
    return state


def route_after_orchestrator(state: State) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(State)

    # 노드 등록
    g.add_node("Orchestrator Agent", orchestrator_node)
    g.add_node("Knowledge Agent", knowledge_node)
    g.add_node("RAG Agent", rag_node)
    g.add_node("MCP Executor Agent", mcp_node)

    # START를 명시적으로 연결 (스캐너가 가장 잘 해석하는 형태)
    g.add_edge(START, "Orchestrator Agent")

    # 분기
    g.add_conditional_edges(
        "Orchestrator Agent",
        route_after_orchestrator,
        {
            "knowledge": "Knowledge Agent",
            "rag": "RAG Agent",
            "mcp": "MCP Executor Agent",
        },
    )

    # 종료
    g.add_edge("Knowledge Agent", END)
    g.add_edge("RAG Agent", END)
    g.add_edge("MCP Executor Agent", END)

    return g.compile()


async def main():
    app = build_graph()
    out = await app.ainvoke({"user_input": "회식비 처리 절차 알려줘"})
    print(out["result"])


if __name__ == "__main__":
    asyncio.run(main())
