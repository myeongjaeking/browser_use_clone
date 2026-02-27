"""
Flow-only skeleton: Corporate AI Assistant (Autonomous) with LangGraph + LangChain Tools + Atlassian MCP.

- NO internal business logic implemented.
- Each node is a stub showing inputs/outputs and where to plug real logic.
- Replace TODO sections with your implementation.

Prereqs (examples):
  pip install -U langgraph langchain langchain-openai python-dotenv

Notes:
- This is a "Planner -> Execute -> Observe -> Verify" loop.
- Read-only tools are deterministic (LLM X) in principle.
- Write tools (Jira/Confluence CRUD) are gated by guardrail + approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END


# ---------------------------
# Types
# ---------------------------

Route = Literal["people", "asset", "faq", "rag", "action", "ask_user", "final"]
WriteTarget = Literal["jira", "confluence", "none"]


class PlanStep(TypedDict, total=False):
    """A single step in the agent plan."""
    step_id: str
    kind: Route                  # "people" | "asset" | "faq" | "rag" | "action"
    tool: str                    # tool name you intend to call
    args: Dict[str, Any]         # tool args (structured)
    expected: str                # what success looks like
    risk: str                    # "low" | "medium" | "high"


class Citation(TypedDict, total=False):
    source: Literal["jira", "confluence", "faq", "db", "other"]
    title: str
    url: str
    ref: str                     # issue key, page id, etc.


class ToolCall(TypedDict, total=False):
    tool: str
    args: Dict[str, Any]
    ok: bool
    output_summary: str


class AgentState(TypedDict, total=False):
    # Input
    user_input: str
    user_id: str
    user_groups: List[str]

    # Planning / routing
    goal: str
    plan: List[PlanStep]
    step_idx: int
    route: Route

    # Working memory
    notes: List[str]
    retrieved: List[Dict[str, Any]]          # raw retrieval hits
    citations: List[Citation]
    draft: Dict[str, Any]                    # action draft (jira issue, confluence page, etc.)
    write_target: WriteTarget

    # Governance
    needs_approval: bool
    approved: bool
    approval_token: Optional[str]
    guardrail_ok: bool
    evidence_ok: bool

    # Outputs
    final_answer: str
    questions_for_user: List[str]

    # Observability
    tool_calls: List[ToolCall]
    errors: List[str]


# ---------------------------
# Tool placeholders (stubs)
# ---------------------------

class Tools:
    """Placeholders for your deterministic tools. Replace with real implementations."""

    # Read-only / deterministic tools
    def people_lookup(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def asset_lookup(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def faq_search(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def rag_retrieve(self, **kwargs) -> Dict[str, Any]:
        """Hybrid retrieval across Confluence/Jira indexes; optionally do MCP live enrich."""
        raise NotImplementedError

    # Write tools via MCP (gated)
    def jira_create_or_update(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def confluence_create_or_update(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


tools = Tools()


# ---------------------------
# Graph Nodes (FLOW ONLY)
# ---------------------------

def node_intake(state: AgentState) -> AgentState:
    """
    Extract/normalize:
    - goal
    - user scope (projects/spaces allowlist)
    - initial notes
    """
    state.setdefault("notes", [])
    state.setdefault("tool_calls", [])
    state.setdefault("errors", [])
    state.setdefault("citations", [])
    state.setdefault("questions_for_user", [])
    state.setdefault("approved", False)
    state.setdefault("needs_approval", False)
    state.setdefault("guardrail_ok", True)
    state.setdefault("evidence_ok", True)
    state.setdefault("step_idx", 0)

    # TODO: derive goal from user_input (LLM planner or rules)
    state["goal"] = state.get("goal") or "<TODO: normalize user goal from user_input>"
    state["notes"].append("intake_done")
    return state


def node_plan(state: AgentState) -> AgentState:
    """
    Create a multi-step plan for the goal (LLM O).
    Plan should be structured steps (PlanStep list).
    """
    # TODO: use LLM to generate plan steps
    state["plan"] = state.get("plan") or [
        {
            "step_id": "step-1",
            "kind": "rag",
            "tool": "rag_retrieve",
            "args": {"query": "<TODO: query>"},
            "expected": "Find policy/procedure evidence and citations",
            "risk": "low",
        },
        # Add more steps as needed
    ]
    state["notes"].append("plan_created")
    return state


def node_select_step(state: AgentState) -> AgentState:
    """
    Decide next route based on plan + current progress (LLM O or deterministic).
    """
    plan = state.get("plan", [])
    idx = state.get("step_idx", 0)

    if idx >= len(plan):
        state["route"] = "final"
        return state

    state["route"] = plan[idx].get("kind", "rag")  # default to rag
    state["notes"].append(f"selected_route:{state['route']}")
    return state


def node_run_people(state: AgentState) -> AgentState:
    """Deterministic tool execution (LLM X) - People/Org."""
    step = state["plan"][state["step_idx"]]
    tool_args = step.get("args", {})

    # TODO: call tools.people_lookup(**tool_args)
    state["tool_calls"].append({"tool": "people_lookup", "args": tool_args, "ok": True, "output_summary": "<TODO>"})
    state["notes"].append("people_lookup_done")
    state["step_idx"] += 1
    return state


def node_run_asset(state: AgentState) -> AgentState:
    """Deterministic tool execution (LLM X) - Assets/Locations."""
    step = state["plan"][state["step_idx"]]
    tool_args = step.get("args", {})

    # TODO: call tools.asset_lookup(**tool_args)
    state["tool_calls"].append({"tool": "asset_lookup", "args": tool_args, "ok": True, "output_summary": "<TODO>"})
    state["notes"].append("asset_lookup_done")
    state["step_idx"] += 1
    return state


def node_run_faq(state: AgentState) -> AgentState:
    """Deterministic tool execution (LLM X) - FAQ top match."""
    step = state["plan"][state["step_idx"]]
    tool_args = step.get("args", {})

    # TODO: call tools.faq_search(**tool_args) and set evidence_ok by threshold
    state["tool_calls"].append({"tool": "faq_search", "args": tool_args, "ok": True, "output_summary": "<TODO>"})
    state["notes"].append("faq_search_done")
    state["step_idx"] += 1
    return state


def node_run_rag(state: AgentState) -> AgentState:
    """
    Retrieval across Confluence + Jira.
    - Usually: index-based RAG first
    - Optional: MCP live enrich if insufficient
    """
    step = state["plan"][state["step_idx"]]
    tool_args = step.get("args", {})

    # TODO: call tools.rag_retrieve(**tool_args) -> set retrieved + citations
    state["retrieved"] = state.get("retrieved") or []
    state["citations"] = state.get("citations") or []

    state["tool_calls"].append({"tool": "rag_retrieve", "args": tool_args, "ok": True, "output_summary": "<TODO>"})
    state["notes"].append("rag_retrieve_done")
    state["step_idx"] += 1
    return state


def node_evidence_check(state: AgentState) -> AgentState:
    """
    Check if there is enough evidence to answer or proceed.
    If insufficient -> ask user or trigger additional retrieval.
    """
    # TODO: implement evidence checks (counts, scores, citations presence)
    state["evidence_ok"] = True
    state["notes"].append(f"evidence_ok:{state['evidence_ok']}")
    return state


def node_draft_action(state: AgentState) -> AgentState:
    """
    Create a write-action draft (LLM O).
    Examples:
      - Jira issue draft (project, summary, description)
      - Confluence page draft (space, title, body)
    This node should NOT execute writes.
    """
    # TODO: create structured draft via LLM; determine write_target
    state["draft"] = {
        "kind": "<TODO: jira|confluence>",
        "payload": {"<TODO>": "<TODO>"},
    }
    state["write_target"] = "jira"  # or "confluence"
    state["needs_approval"] = True
    state["notes"].append("action_draft_created")
    return state


def node_guardrail_check(state: AgentState) -> AgentState:
    """
    Validate policy, permissions, scope, and risk before any write.
    This can be:
      - rule-based checks
      - plus an LLM-based review (optional)
    """
    # TODO: compute guardrail_ok based on user_groups, scope, risk
    state["guardrail_ok"] = True
    state["notes"].append(f"guardrail_ok:{state['guardrail_ok']}")
    return state


def node_request_approval(state: AgentState) -> AgentState:
    """
    In real system, you would:
      - ask user to approve (chat command) OR
      - request manager approval via UI/workflow
    Here we just set a placeholder question.
    """
    if not state.get("approved", False):
        state["questions_for_user"].append("승인이 필요합니다. '승인'이라고 답하면 진행합니다.")
        state["notes"].append("approval_requested")
    return state


def node_execute_write(state: AgentState) -> AgentState:
    """
    Execute Atlassian MCP write tool ONLY if:
      - needs_approval == True
      - approved == True
      - guardrail_ok == True
    """
    target = state.get("write_target", "none")
    draft = state.get("draft", {})

    if target == "jira":
        # TODO: call tools.jira_create_or_update(**draft["payload"])
        state["tool_calls"].append({"tool": "jira_create_or_update", "args": draft.get("payload", {}), "ok": True, "output_summary": "<TODO>"})
        state["notes"].append("jira_write_executed")

    elif target == "confluence":
        # TODO: call tools.confluence_create_or_update(**draft["payload"])
        state["tool_calls"].append({"tool": "confluence_create_or_update", "args": draft.get("payload", {}), "ok": True, "output_summary": "<TODO>"})
        state["notes"].append("confluence_write_executed")

    state["step_idx"] += 1
    return state


def node_observe_verify(state: AgentState) -> AgentState:
    """
    Summarize what happened, verify outcomes vs expected, decide to continue or stop.
    (LLM O can be used, but keep output deterministic if preferred.)
    """
    # TODO: verify last tool call result; maybe update plan or step_idx
    state["notes"].append("observe_verify_done")
    return state


def node_finalize(state: AgentState) -> AgentState:
    """
    Produce final response: answer + citations + actions performed.
    """
    # TODO: build final_answer using LLM or templates; include citations & audit trail
    state["final_answer"] = "<TODO: final answer with citations and performed actions>"
    state["notes"].append("finalized")
    return state


# ---------------------------
# Routing (Conditional Edges)
# ---------------------------

def route_by_kind(state: AgentState) -> str:
    """Send execution to the right node based on state['route']."""
    return state.get("route", "rag")


def route_after_evidence(state: AgentState) -> str:
    """
    If evidence insufficient:
      - ask user OR
      - go back to plan/retrieve
    For flow-only skeleton, we route to finalize with questions.
    """
    if not state.get("evidence_ok", True):
        return "final"
    return "select_step"


def route_action_gate(state: AgentState) -> str:
    """
    Control write execution:
      - if guardrail fails -> finalize with refusal
      - if approval needed and not approved -> request approval then finalize/stop
      - else -> execute_write
    """
    if not state.get("guardrail_ok", True):
        return "final"
    if state.get("needs_approval", False) and not state.get("approved", False):
        return "request_approval"
    return "execute_write"


def route_continue_or_end(state: AgentState) -> str:
    """Continue loop until plan is done."""
    plan = state.get("plan", [])
    idx = state.get("step_idx", 0)
    if idx >= len(plan):
        return "final"
    return "select_step"


# ---------------------------
# Build Graph
# ---------------------------

def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("intake", node_intake)
    g.add_node("plan", node_plan)
    g.add_node("select_step", node_select_step)

    g.add_node("run_people", node_run_people)
    g.add_node("run_asset", node_run_asset)
    g.add_node("run_faq", node_run_faq)
    g.add_node("run_rag", node_run_rag)

    g.add_node("evidence_check", node_evidence_check)

    g.add_node("draft_action", node_draft_action)
    g.add_node("guardrail_check", node_guardrail_check)
    g.add_node("request_approval", node_request_approval)
    g.add_node("execute_write", node_execute_write)

    g.add_node("observe_verify", node_observe_verify)
    g.add_node("final", node_finalize)

    # Entry
    g.set_entry_point("intake")

    # Main flow
    g.add_edge("intake", "plan")
    g.add_edge("plan", "select_step")

    # Route to domain execution
    g.add_conditional_edges("select_step", route_by_kind, {
        "people": "run_people",
        "asset": "run_asset",
        "faq": "run_faq",
        "rag": "run_rag",
        "action": "draft_action",
        "ask_user": "final",
        "final": "final",
    })

    # After read routes, evidence check then loop
    g.add_edge("run_people", "evidence_check")
    g.add_edge("run_asset", "evidence_check")
    g.add_edge("run_faq", "evidence_check")
    g.add_edge("run_rag", "evidence_check")

    g.add_conditional_edges("evidence_check", route_after_evidence, {
        "select_step": "select_step",
        "final": "final",
    })

    # Action (draft -> guardrail -> approval gate -> write -> observe)
    g.add_edge("draft_action", "guardrail_check")
    g.add_conditional_edges("guardrail_check", route_action_gate, {
        "request_approval": "request_approval",
        "execute_write": "execute_write",
        "final": "final",
    })

    # If requesting approval, stop for now (in real system you'd wait for user response then resume)
    g.add_edge("request_approval", "final")

    # After write execution, observe/verify then continue
    g.add_edge("execute_write", "observe_verify")
    g.add_conditional_edges("observe_verify", route_continue_or_end, {
        "select_step": "select_step",
        "final": "final",
    })

    # End
    g.add_edge("final", END)

    return g.compile()


# ---------------------------
# Example usage (flow-only)
# ---------------------------

if __name__ == "__main__":
    app = build_graph()

    initial_state: AgentState = {
        "user_input": "회식비 처리 절차 알려주고, 필요하면 Jira 티켓도 초안 만들어줘",
        "user_id": "u123",
        "user_groups": ["eng", "corp"],
        # For demo: you can simulate approval by setting approved=True
        "approved": False,
    }

    result = app.invoke(initial_state)

    print("=== FINAL ANSWER ===")
    print(result.get("final_answer", ""))

    if result.get("questions_for_user"):
        print("\n=== QUESTIONS FOR USER ===")
        for q in result["questions_for_user"]:
            print("-", q)

    print("\n=== NOTES ===")
    for n in result.get("notes", []):
        print("-", n)

    print("\n=== TOOL CALLS (planned/executed placeholders) ===")
    for c in result.get("tool_calls", []):
        print("-", c)
