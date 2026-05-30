"""
graph.py — RAI Compliance Agent graph with self-correction loop.

Execution flow:
  ingestion
    → pii_agent
      → bias_agent
        → explainability_agent
          → policy_agent
            → [should_correct?]
              YES → correction → pii_agent  (re-enters audit chain)
              NO  → scorecard → END

When correction_count >= max_corrections (default 3) the graph escalates
to human review via the escalation node rather than attempting another fix.

Every node transition is persisted to rai_audit.db via the SqliteSaver
checkpointer, providing the tamper-proof audit trail required by
EU AI Act Article 50 and DPDPA 2023 Section 11.
"""

import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import ComplianceState
from nodes import (
    ingestion_node,
    pii_agent_node,
    bias_agent_node,
    explainability_agent_node,
    policy_agent_node,
    scorecard_node,
    correction_node,
)


# ---------------------------------------------------------------------------
# Routing functions (pure — read state, return node name)
# ---------------------------------------------------------------------------

def _should_correct(state: ComplianceState) -> str:
    """
    Conditional edge: runs after policy_agent.

    Returns:
        "correction"  — violations exist and correction budget remains
        "escalation"  — correction budget exhausted (max_corrections reached)
        "scorecard"   — no violations; proceed to final scorecard
    """
    violations = state.get("violations") or []
    if not violations:
        return "scorecard"

    correction_count = state.get("correction_count", 0)
    max_corrections = state.get("max_corrections", 3)

    if correction_count >= max_corrections:
        return "escalation"

    return "correction"


def _after_correction(state: ComplianceState) -> str:
    """
    Conditional edge: runs after correction_node.

    After correction the full audit chain must re-run from pii_agent so
    that each pillar re-evaluates the corrected text independently.
    If we've just hit the max, hand off to escalation instead.
    """
    correction_count = state.get("correction_count", 0)
    max_corrections = state.get("max_corrections", 3)

    if correction_count >= max_corrections:
        return "escalation"

    return "pii_agent"


# ---------------------------------------------------------------------------
# Escalation node (inline — no separate file needed)
# ---------------------------------------------------------------------------

def _escalation_node(state: ComplianceState) -> dict:
    """
    Terminal node for cases where auto-correction could not resolve violations.
    Sets final_status = ESCALATED and documents the reason for human review.

    This node fulfils EU AI Act Article 14 (human oversight) and maps to the
    Governance & Accountability pillar — every escalation is audit-logged.
    """
    from datetime import datetime, timezone

    violations = list(set(state.get("violations") or []))
    count = state.get("correction_count", 0)

    reason = (
        f"Auto-correction exhausted after {count} attempt(s). "
        f"Unresolved violations: {', '.join(violations)}. "
        "Human review required per EU AI Act Article 14."
    )

    print(f"\n[ESCALATION] Human review required.")
    print(f"[ESCALATION] {reason}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "escalation",
        "action": "HUMAN_REVIEW_REQUIRED",
        "result": "escalated",
        "detail": {
            "correction_attempts": count,
            "unresolved_violations": violations,
            "reason": reason,
            "regulatory_basis": "EU AI Act Art.14 / NIST MANAGE 1.3",
        },
    }

    step_entry = {
        "step":  "escalation",
        "label": "Human Escalation",
        "status": "escalated",
        "prompt": None,
        "response": {
            "final_status": "ESCALATED",
            "correction_attempts": count,
            "unresolved_violations": violations,
        },
        "summary": reason,
    }

    return {
        "final_status": "ESCALATED",
        "escalation_reason": reason,
        "current_node": "escalation",
        "audit_log": [log_entry],
        "step_trace": [step_entry],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(ComplianceState)

    # --- Register all nodes ---
    workflow.add_node("ingestion",             ingestion_node)
    workflow.add_node("pii_agent",             pii_agent_node)
    workflow.add_node("bias_agent",            bias_agent_node)
    workflow.add_node("explainability_agent",  explainability_agent_node)
    workflow.add_node("policy_agent",          policy_agent_node)
    workflow.add_node("correction",            correction_node)
    workflow.add_node("escalation",            _escalation_node)
    workflow.add_node("scorecard",             scorecard_node)

    # --- Linear spine: ingestion through policy ---
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion",            "pii_agent")
    workflow.add_edge("pii_agent",            "bias_agent")
    workflow.add_edge("bias_agent",           "explainability_agent")
    workflow.add_edge("explainability_agent", "policy_agent")

    # --- Conditional routing after policy_agent ---
    # Routes to correction, escalation, or scorecard based on violation state
    workflow.add_conditional_edges(
        "policy_agent",
        _should_correct,
        {
            "correction": "correction",
            "escalation": "escalation",
            "scorecard":  "scorecard",
        },
    )

    # --- Correction loop: re-enter full audit chain after fix ---
    workflow.add_conditional_edges(
        "correction",
        _after_correction,
        {
            "pii_agent":  "pii_agent",
            "escalation": "escalation",
        },
    )

    # --- Terminal edges ---
    workflow.add_edge("scorecard",   END)
    workflow.add_edge("escalation",  END)

    # --- Persist every checkpoint to SQLite (audit trail) ---
    conn = sqlite3.connect("rai_audit.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    compiled = workflow.compile(checkpointer=checkpointer)

    print(
        "[GRAPH] Compiled with correction loop and escalation path. "
        "Audit trail → rai_audit.db"
    )
    return compiled


app = build_graph()
