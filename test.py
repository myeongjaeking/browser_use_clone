"""
SPLX-detectable: Clean 3-Agent Flow
Orchestrator → Knowledge/RAG/MCP Executor
"""

import os
import asyncio
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams

# =========================================================
# 1) Tools
# =========================================================
@function_tool
def people_lookup(name: str) -> str: return f"{name}: 개발팀, 010-XXXX"

@function_tool
def asset_lookup(item: str) -> str: return f"{item}: A동 301호"

@function_tool
def faq_search(query: str) -> str: return f"{query}: 총무 신청 → Slack #expense"

@function_tool
def rag_retrieve(query: str) -> str: return f"{query}: Q1 보고서 발췌 내용"

# =========================================================
# 2) MCP Server
# =========================================================
ATLASSIAN_MCP_PARAMS: MCPServerStdioParams = {
    "command": "uvx", "args": ["mcp-atlassian"],
    "env": {"JIRA_URL": os.getenv("JIRA_URL"), "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
            "CONFLUENCE_URL": os.getenv("CONFLUENCE_URL"), "CONFLUENCE_API_TOKEN": os.getenv("CONFLUENCE_API_TOKEN")},
}
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)

# =========================================================
# 3) 3 Agents Only
# =========================================================
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
    instructions="Jira/Confluence 작업 직접 실행.",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)

# =========================================================
# 4) Orchestrator Tools (Agent 호출)
# =========================================================
@function_tool
async def call_knowledge(query: str) -> str:
    """Knowledge Agent 호출 (사원/FAQ/자산)."""
    res = await Runner.run(knowledge_agent, query)
    return res.final_output or "Knowledge 실패"

@function_tool
async def call_rag(query: str) -> str:
    """RAG Agent 호출 (문서)."""
    res = await Runner.run(rag_agent, query)
    return res.final_output or "RAG 실패"

@function_tool
async def call_mcp(command: str) -> str:
    """MCP Executor 호출 (Jira/Confluence)."""
    res = await Runner.run(mcp_executor_agent, command)
    return res.final_output or "MCP 실패"

# =========================================================
# 5) Orchestrator Agent (Router + Executor)
# =========================================================
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions="""
    쿼리 유형별 Agent 자동 선택/실행:
    - call_knowledge: 사원("김팀장"), 자산("노트북"), FAQ("회식비")
    - call_rag: 문서("Q1 보고서")
    - call_mcp: 작업("Jira 티켓 생성: 버그 수정")
    
    최적 tool 하나 선택 (또는 순차). 결과 명확히 합성.
    """,
    tools=[call_knowledge, call_rag, call_mcp],
)

# =========================================================
# 6) State & Single Node
# =========================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    result: str

async def orchestrator_node(state: State) -> State:
    """Orchestrator가 모든 결정/실행."""
    print(f"🤖 Processing: {state['user_input']}")
    res = await Runner.run(orchestrator_agent, state["user_input"])
    state["result"] = res.final_output
    print(f"✅ Result: {state['result'][:80]}...")
    return state

# =========================================================
# 7) Minimal Graph (1 Node!)
# =========================================================
def build_graph():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator_node)
    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", END)
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
        "Jira 티켓 '긴급 버그' 생성해"
    ]
    
    for test_input in tests:
        print(f"\n{'='*50}")
        print(f"🧪 '{test_input}'")
        out = await app.ainvoke({"user_input": test_input})
        print(f"📤 {out['result']}")

if __name__ == "__main__":
    asyncio.run(main())
