"""
SPLX-detectable: Pure Agent-to-Agent Orchestration (single file)
- Orchestrator calls sub-agents as tools
- Direct Agent objects in function_tools
"""

import os
import asyncio
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END

from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams

# =========================================================
# 1) Sub-Agents (독립 Agent 객체)
# =========================================================
knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions="사원/자산/FAQ 조회.",
    tools=[people_lookup, asset_lookup, faq_search],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions="문서 검색/답변.",
    tools=[rag_retrieve],
)

mcp_agent = Agent(
    name="MCP Agent",
    instructions="Atlassian MCP 실행 (Jira/Confluence).",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)

# 기존 도구들 (생략: 이전과 동일)
@function_tool def people_lookup(name: str) -> str: return "<TODO>"
@function_tool def asset_lookup(item: str) -> str: return "<TODO>"
@function_tool def faq_search(query: str) -> str: return "<TODO>"
@function_tool def rag_retrieve(query: str) -> str: return "<TODO>"

# =========================================================
# 2) MCP Server
# =========================================================
ATLASSIAN_MCP_PARAMS: MCPServerStdioParams = {
    "command": "uvx", "args": ["mcp-atlassian"],
    "env": { "JIRA_URL": os.getenv("JIRA_URL"), ... },  # 생략
}
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)

# =========================================================
# 3) Agent-as-Tool Functions (Orchestrator가 호출)
# =========================================================
@function_tool
async def call_knowledge_agent(query: str) -> str:
    """Knowledge Agent 호출."""
    res = await Runner.run(knowledge_agent, query)
    return res.final_output or "Knowledge 처리 실패"

@function_tool
async def call_rag_agent(query: str) -> str:
    """RAG Agent 호출."""
    res = await Runner.run(rag_agent, query)
    return res.final_output or "RAG 처리 실패"

@function_tool
async def call_mcp_agent(command: str) -> str:
    """MCP Agent 호출 (작업 명령)."""
    res = await Runner.run(mcp_agent, command)
    return res.final_output or "MCP 실행 실패"

# =========================================================
# 4) Orchestrator Agent (Sub-Agent 호출자)
# =========================================================
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions="""
    쿼리 분석 후 적합한 Agent 호출 (한 번에 하나, 또는 순차):
    - call_knowledge_agent: 사원/자산/FAQ ("김팀장", "회의실", "회식비")
    - call_rag_agent: 문서/보고서 ("Q1 실적 보고서")
    - call_mcp_agent: Jira/Confluence 작업 ("Jira 티켓 생성: 회식비 정산")
    
    결과 합성 후 최종 답변. 필요시 여러 호출.
    """,
    tools=[call_knowledge_agent, call_rag_agent, call_mcp_agent],
)

# =========================================================
# 5) State & Nodes
# =========================================================
class State(TypedDict):
    messages: Annotated[list, "add_messages"]
    route: Literal["knowledge", "rag", "mcp"]
    final_result: str

async def router_node(state: State) -> State:
    """Triage-like 라우팅 (Orchestrator 전처리)."""
    query = state["messages"][-1].content.lower()
    if any(kw in query for kw in ["사원", "팀", "자산", "위치", "faq", "절차"]):
        state["route"] = "knowledge"
    elif any(kw in query for kw in ["문서", "보고서", "파일"]):
        state["route"] = "rag"
    else:
        state["route"] = "mcp"
    return state

async def orchestrator_node(state: State) -> State:
    """Orchestrator 실행 + route 힌트."""
    hint = f"Route 힌트: {state['route']}. 쿼리: {state['messages'][-1].content}"
    res = await Runner.run(orchestrator_agent, hint)
    state["final_result"] = res.final_output
    return state

# =========================================================
# 6) Graph (라우팅 → Orchestrator)
# =========================================================
def build_graph():
    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("orchestrator", orchestrator_node)
    g.set_entry_point("router")
    g.add_edge("router", "orchestrator")
    g.add_edge("orchestrator", END)
    return g.compile()

# =========================================================
# 7) Runner
# =========================================================
async def main():
    app = build_graph()
    input_data = {"messages": [{"role": "user", "content": "회식비 처리 절차 알려줘"}]}
    out = await app.ainvoke(input_data)
    print("결과:", out["final_result"])

if __name__ == "__main__":
    asyncio.run(main())
