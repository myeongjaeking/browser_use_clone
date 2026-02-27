"""
SPLX-detectable: Optimized Agents SDK + LangGraph (single file)
- Agents: OpenAI Agents SDK `agents.Agent` (consolidated)
- Tools: `@function_tool`
- MCP Servers: `MCPServerStdio` at module scope
- LangGraph: Simplified nodes (4 Agents)
"""

import os
import asyncio
from typing import TypedDict, Literal, Dict, Any, Optional

from langgraph.graph import StateGraph, END

# OpenAI Agents SDK
from agents import Agent, Runner, function_tool
from agents.mcp.server import MCPServerStdio, MCPServerStdioParams

# =========================================================
# 1) Tools (unchanged)
# =========================================================
@function_tool
def people_lookup(name: str) -> str:
    """Lookup internal employee info."""
    return "<TODO: people_lookup result>"

@function_tool
def asset_lookup(item: str) -> str:
    """Lookup asset/location info."""
    return "<TODO: asset_lookup result>"

@function_tool
def faq_search(query: str) -> str:
    """FAQ search."""
    return "<TODO: faq_search result>"

@function_tool
def rag_retrieve(query: str) -> str:
    """RAG retrieve."""
    return "<TODO: rag_retrieve result>"

# =========================================================
# 2) MCP Server (unchanged)
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
# 3) Consolidated Agents (7→4)
# =========================================================
knowledge_agent = Agent(
    name="Knowledge Agent",
    instructions="""
    사내 조회 쿼리에 맞춰 적합한 도구 사용:
    - people_lookup: 사원/팀 정보 (e.g. "김팀장")
    - asset_lookup: 자산/위치 (e.g. "노트북 위치")
    - faq_search: FAQ/절차 (e.g. "회식비")
    """,
    tools=[people_lookup, asset_lookup, faq_search],
)

rag_agent = Agent(  # Unchanged
    name="RAG Agent",
    instructions="문서 기반 질문에 rag_retrieve 사용.",
    tools=[rag_retrieve],
)

action_agent = Agent(
    name="Action Agent",
    instructions="Jira/Confluence 작업 draft 생성 (JSON 구조). 실행 안 함.",
)

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Draft 검토 후 allow/deny 결정.",
)

mcp_executor_agent = Agent(
    name="MCP Executor Agent",
    instructions="승인된 draft를 MCP로 실행.",
    mcp_servers=[ATLASSIAN_MCP_SERVER],
)

# =========================================================
# 4) State (unchanged)
# =========================================================
class State(TypedDict, total=False):
    user_input: str
    route: Literal["knowledge", "rag", "action"]
    draft: Dict[str, Any]
    approved: bool
    result: str

# =========================================================
# 5) Simplified Nodes
# =========================================================
async def triage_node(state: State) -> State:  # LLM 없이 키워드 룰 (안정/빠름)
    query = state["user_input"].lower()
    if any(word in query for word in ["사원", "팀장", "인사", "이름"]):
        state["route"] = "knowledge"
    elif any(word in query for word in ["자산", "위치", "노트북", "회의실"]):
        state["route"] = "knowledge"
    elif any(word in query for word in ["faq", "절차", "회식", "휴가"]):
        state["route"] = "knowledge"
    elif any(word in query for word in ["문서", "파일", "보고서"]):
        state["route"] = "rag"
    else:
        state["route"] = "action"
    return state

async def knowledge_node(state: State) -> State:
    res = await Runner.run(knowledge_agent, state["user_input"])
    state["result"] = res.final_output
    return state

async def rag_node(state: State) -> State:
    res = await Runner.run(rag_agent, state["user_input"])
    state["result"] = res.final_output
    return state

async def action_node(state: State) -> State:
    _ = await Runner.run(action_agent, state["user_input"])
    state["draft"] = {"type": "jira", "payload": {"summary": "<TODO>", "description": "<TODO>"}}
    state["approved"] = False
    state["result"] = "Draft 생성. 승인 대기."
    return state

async def guardrail_node(state: State) -> State:
    _ = await Runner.run(guardrail_agent, f"Review draft: {state['draft']}")
    state["approved"] = True  # TODO: 실제 결정 로직
    return state

async def mcp_execute_node(state: State) -> State:
    if not state.get("approved", False):
        state["result"] = "승인 필요. 실행 스킵."
        return state
    res = await Runner.run(mcp_executor_agent, f"Execute: {state['draft']}")
    state["result"] = res.final_output
    return state

def route_after_triage(state: State) -> str:
    return state.get("route", "knowledge")

# =========================================================
# 6) Build Graph
# =========================================================
def build_graph():
    g = StateGraph(State)
    
    g.add_node("triage", triage_node)
    g.add_node("knowledge", knowledge_node)
    g.add_node("rag", rag_node)
    g.add_node("action", action_node)
    g.add_node("guardrail", guardrail_node)
    g.add_node("mcp_execute", mcp_execute_node)
    
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", route_after_triage, {
        "knowledge": "knowledge",
        "rag": "rag",
        "action": "action",
    })
    g.add_edge("knowledge", END)
    g.add_edge("rag", END)
    g.add_edge("action", "guardrail")
    g.add_edge("guardrail", "mcp_execute")
    g.add_edge("mcp_execute", END)
    
    return g.compile()

# =========================================================
# 7) Runner
# =========================================================
async def main():
    app = build_graph()
    out = await app.ainvoke({"user_input": "회식비 처리 절차 알려줘"})
    print(out.get("result", ""))

if __name__ == "__main__":
    asyncio.run(main())
