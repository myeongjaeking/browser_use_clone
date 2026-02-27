"""
SPLX-detectable: Orchestrator-Driven Agent Selection
- Orchestrator decides which sub-agent to call
"""

import os
import asyncio
from typing import TypedDict, Literal, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# OpenAI Agents SDK
from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams

# =========================================================
# 1) Tools
# =========================================================
@function_tool
def people_lookup(name: str) -> str: return "<TODO: people result>"
@function_tool
def asset_lookup(item: str) -> str: return "<TODO: asset result>"
@function_tool
def faq_search(query: str) -> str: return "<TODO: FAQ result>"
@function_tool
def rag_retrieve(query: str) -> str: return "<TODO: RAG result>"

# =========================================================
# 2) MCP Server
# =========================================================
ATLASSIAN_MCP_PARAMS: MCPServerStdioParams = {
    "command": "uvx", "args": ["mcp-atlassian"],
    "env": {"JIRA_URL": os.getenv("JIRA_URL", ""), "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN", ""),
            "CONFLUENCE_URL": os.getenv("CONFLUENCE_URL", ""), "CONFLUENCE_API_TOKEN": os.getenv("CONFLUENCE_API_TOKEN", "")},
}
ATLASSIAN_MCP_SERVER = MCPServerStdio(ATLASSIAN_MCP_PARAMS)

# =========================================================
# 3) 하위 Agents (Orchestrator가 호출)
# =========================================================
knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions="사원/자산/FAQ 전문.",
    tools=[people_lookup, asset_lookup, faq_search],
)

rag_agent = Agent(
    name="RAG Agent",
    instructions="문서 검색.",
    tools=[rag_retrieve],
)

action_agent = Agent(
    name="Action Agent",
    instructions="Jira/Confluence 작업 draft 생성.",
)

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="작업 승인/거부 결정.",
)

mcp_executor_agent = Agent(
    name="MCP Executor Agent",
    instructions="MCP로 작업 실행.",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)

# =========================================================
# 4) Agent-as-Tool (Orchestrator용)
# =========================================================
@function_tool
async def call_knowledge(query: str) -> str:
    res = await Runner.run(knowledge_agent, query)
    return res.final_output or "Knowledge 실패"

@function_tool
async def call_rag(query: str) -> str:
    res = await Runner.run(rag_agent, query)
    return res.final_output or "RAG 실패"

@function_tool
async def call_action(query: str) -> str:
    res = await Runner.run(action_agent, query)
    return f"Draft: {res.final_output}"  # 구조화 반환

@function_tool
async def call_guardrail(draft: str) -> str:
    res = await Runner.run(guardrail_agent, f"Review: {draft}")
    return "approved" if "승인" in res.final_output.lower() else "denied"

@function_tool
async def call_mcp(command: str) -> str:
    res = await Runner.run(mcp_executor_agent, command)
    return res.final_output or "MCP 실패"

# =========================================================
# 5) Orchestrator Agent (결정자)
# =========================================================
orchestrator_agent = Agent(
    name="Orchestrator Agent",
    instructions="""
    쿼리 분석 후 적절한 Agent 시퀀스 결정/실행:
    1. 조회: call_knowledge (사원/FAQ/자산), call_rag (문서)
    2. 작업: call_action → call_guardrail → call_mcp (Jira/Confluence)
    
    힌트 참고: {route}. 여러 호출 순차/병렬 가능. 최종 결과 합성.
    예: "회식비" → call_knowledge, "티켓 생성" → action→guard→mcp
    """,
    tools=[call_knowledge, call_rag, call_action, call_guardrail, call_mcp],
)

# =========================================================
# 6) State
# =========================================================
class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_input: str
    route: Literal["knowledge", "rag", "action"]
    result: str

# =========================================================
# 7) Nodes
# =========================================================
async def triage_hint_node(state: State) -> State:
    """경량 힌트 생성 (Orchestrator 보조)."""
    query = state["user_input"].lower()
    if any(kw in query for kw in ["사원", "팀", "자산", "faq", "절차", "회식"]):
        state["route"] = "knowledge"
    elif any(kw in query for kw in ["문서", "보고서"]):
        state["route"] = "rag"
    else:
        state["route"] = "action"
    print(f"💡 Hint route: {state['route']}")
    return state

async def orchestrator_node(state: State) -> State:
    """Orchestrator가 모든 결정/호출."""
    hint = f"Hint route: {state['route']}. User: {state['user_input']}"
    print("🤖 Orchestrator deciding...")
    res = await Runner.run(orchestrator_agent, hint)
    state["result"] = res.final_output
    print(f"✅ Orchestrator result: {state['result'][:100]}...")
    return state

# =========================================================
# 8) Graph (Hint → Orchestrator)
# =========================================================
def build_graph():
    g = StateGraph(State)
    g.add_node("triage_hint", triage_hint_node)
    g.add_node("orchestrator", orchestrator_node)
    g.set_entry_point("triage_hint")
    g.add_edge("triage_hint", "orchestrator")
    g.add_edge("orchestrator", END)
    return g.compile()

# =========================================================
# 9) Runner + Test
# =========================================================
async def main():
    app = build_graph()
    tests = [
        "회식비 처리 절차 알려줘",
        "김팀장 어디 있어?",
        "Q1 보고서 내용 요약",
        "Jira 티켓 '버그 수정' 생성해"
    ]
    
    for test_input in tests:
        print(f"\n{'='*60}")
        print(f"🧪 INPUT: {test_input}")
        out = await app.ainvoke({"user_input": test_input, "messages": [{"role": "user", "content": test_input}]})
        print(f"📤 OUTPUT: {out['result']}")

if __name__ == "__main__":
    asyncio.run(main())
