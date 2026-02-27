"""
SPLX-detectable: Orchestrator Pattern + LangGraph (single file)
- Orchestrator Agent: Central router/caller
- Sub-agents as tools
- MCP integration preserved
"""

import os
import asyncio
from typing import TypedDict, Dict, Any, Annotated, Literal
from langgraph.graph import StateGraph, END

# OpenAI Agents SDK
from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams

# =========================================================
# 1) Sub-Agent Tools (Orchestrator가 호출)
# =========================================================
@function_tool
async def knowledge_tool(query: str) -> str:
    """사내 지식 조회 (People/Asset/FAQ)."""
    agent = Agent(
        name="Knowledge",
        instructions="people/asset/faq 도구로 사내 조회.",
        tools=[people_lookup, asset_lookup, faq_search],  # 기존 도구들
    )
    res = await Runner.run(agent, query)
    return res.final_output or "조회 실패"

@function_tool
async def rag_tool(query: str) -> str:
    """RAG 문서 검색."""
    agent = Agent(
        name="RAG",
        instructions="rag_retrieve로 문서 기반 답변.",
        tools=[rag_retrieve],
    )
    res = await Runner.run(agent, query)
    return res.final_output or "RAG 실패"

# 기존 도구들 (knowledge_tool 내부 사용)
@function_tool
def people_lookup(name: str) -> str: return "<TODO>"
@function_tool
def asset_lookup(item: str) -> str: return "<TODO>"
@function_tool
def faq_search(query: str) -> str: return "<TODO>"
@function_tool
def rag_retrieve(query: str) -> str: return "<TODO>"

# =========================================================
# 2) MCP Server (unchanged)
# =========================================================
ATLASSIAN_MCP_PARAMS: MCPServerStdioParams = { ... }  # 이전과 동일
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)

# =========================================================
# 3) Orchestrator Agent (중앙 오케스트레이터)
# =========================================================
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions="""
    사용자 쿼리 분석 후 적합한 sub-agent 호출:
    - knowledge_tool: 사원/자산/FAQ (e.g. "김팀장", "회의실", "회식비")
    - rag_tool: 문서/보고서 (e.g. "최근 보고서")
    - 직접 MCP: Atlassian 작업 (Jira/Confluence)
    
    결과를 합성해 최종 답변. 여러 tool 병렬 가능.
    """,
    tools=[knowledge_tool, rag_tool],  # MCP는 직접 (또는 별도 tool)
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)

# =========================================================
# 4) State
# =========================================================
class State(TypedDict):
    messages: Annotated[list, "add_messages"]
    result: str

# =========================================================
# 5) Orchestrator Node (단일 노드!)
# =========================================================
async def orchestrator_node(state: State) -> State:
    """Orchestrator가 모든 호출 처리."""
    try:
        res = await Runner.run(orchestrator_agent, state["messages"][-1].content)
        state["result"] = res.final_output
    except Exception as e:
        state["result"] = f"오류: {str(e)}. 기본 처리: {state['messages'][-1].content}"
    return state

# =========================================================
# 6) Simplified Graph (1 노드!)
# =========================================================
def build_graph():
    g = StateGraph(State)
    g.add_node("orchestrator", orchestrator_node)
    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", END)
    return g.compile()

# =========================================================
# 7) Runner
# =========================================================
async def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "1"}}  # 세션 유지
    out = await app.ainvoke(
        {"messages": [{"role": "user", "content": "회식비 처리 절차 알려줘"}]},
        config
    )
    print(out["result"])

if __name__ == "__main__":
    asyncio.run(main())
