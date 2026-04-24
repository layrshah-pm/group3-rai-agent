"""
graph.py
--------
Defines and compiles the LangGraph compliance graph.

This file wires together all nodes and edges, defines the conditional
routing logic, and compiles the graph with a SQLite checkpointer for
state persistence (the audit trail).

Graph topology:
                    ┌─────────────┐
                    │  INGESTION  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌─────│  PII AGENT  │◄────────────────┐
              │     └──────┬──────┘                 │
              │            │ pass                   │
              │     ┌──────▼──────┐                 │
              │     │ BIAS AGENT  │◄──────────┐     │
              │     └──────┬──────┘           │     │
              │            │ pass             │     │
              │     ┌──────▼────────────┐     │     │
              │     │EXPLAINABILITY AGT │     │     │
              │     └──────┬────────────┘     │     │
              │            │ (always pass)    │     │
              │     ┌──────▼──────┐           │     │
              │     │POLICY AGENT │◄────┐     │     │
              │     └──────┬──────┘     │     │     │
              │            │ pass       │     │     │
              │     ┌──────▼──────┐     │     │     │
              │     │ SCORECARD   │     │     │     │
              │     └─────────────┘     │     │     │
              │                         │     │     │
              └──── violation ──►  CORRECTION ──────┘
                                  (cycles back to
                                   PII agent)
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
    correction_node,
    scorecard_node,
)


# ---------------------------------------------------------------------------
# Routing functions
# Each returns a string key that maps to the next node name.
# ---------------------------------------------------------------------------

def route_after_pii(state: ComplianceState) -> str:
    """
    After PII agent runs:
      - If PII found AND corrections remaining → self-correct
      - If PII found AND no corrections left   → escalate (go to scorecard)
      - If PII clean                           → proceed to bias agent
    """
    pii = state.get("pii_result") or {}
    if not pii.get("passed", True):
        if state["correction_count"] < state["max_corrections"]:
            print("[ROUTER] PII violation → routing to correction node")
            return "correction"
        else:
            print("[ROUTER] PII violation + max corrections reached → escalating")
            return "scorecard"
    return "bias_agent"


def route_after_bias(state: ComplianceState) -> str:
    """
    After bias agent runs:
      - If bias detected AND corrections remaining → self-correct
      - If bias detected AND no corrections left   → escalate
      - If bias clean                              → proceed to policy agent
    """
    bias = state.get("bias_result") or {}
    if not bias.get("passed", True):
        if state["correction_count"] < state["max_corrections"]:
            print("[ROUTER] Bias violation → routing to correction node")
            return "correction"
        else:
            print("[ROUTER] Bias violation + max corrections reached → escalating")
            return "scorecard"
    return "explainability_agent"


def route_after_policy(state: ComplianceState) -> str:
    """
    After policy agent runs:
      - If violations found AND corrections remaining → self-correct
      - If violations found AND no corrections left   → escalate
      - If policy clean                              → proceed to scorecard
    """
    policy = state.get("policy_result") or {}
    if not policy.get("passed", True):
        if state["correction_count"] < state["max_corrections"]:
            print("[ROUTER] Policy violation → routing to correction node")
            # Set active_violation to the highest severity violation
            violations = policy.get("violations", [])
            if violations:
                # Sort by severity and pick the worst one to fix first
                severity_order = {"high": 0, "medium": 1, "low": 2}
                violations_sorted = sorted(
                    violations,
                    key=lambda v: severity_order.get(v.get("severity", "low"), 2)
                )
                return "correction"
        else:
            print("[ROUTER] Policy violation + max corrections reached → escalating")
            return "scorecard"
    return "scorecard"


def route_after_correction(state: ComplianceState) -> str:
    """
    After self-correction runs, always route back to PII agent
    so the full auditor chain re-runs on the corrected text.

    This is the cyclic edge that makes LangGraph necessary.
    A linear pipeline cannot do this.
    """
    print(f"[ROUTER] Correction complete → re-running full audit chain")
    return "pii_agent"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """
    Builds and returns the compiled LangGraph application.

    The returned `app` object is what you call with:
        result = app.invoke(initial_state, config=config)
    or stream with:
        for chunk in app.stream(initial_state, config=config):
            print(chunk)
    """

    # --- Initialise graph with our state schema ---
    workflow = StateGraph(ComplianceState)

    # --- Register all nodes ---
    workflow.add_node("ingestion",             ingestion_node)
    workflow.add_node("pii_agent",             pii_agent_node)
    workflow.add_node("bias_agent",            bias_agent_node)
    workflow.add_node("explainability_agent",  explainability_agent_node)
    workflow.add_node("policy_agent",          policy_agent_node)
    workflow.add_node("correction",            correction_node)
    workflow.add_node("scorecard",             scorecard_node)

    # --- Entry point ---
    workflow.set_entry_point("ingestion")

    # --- Fixed edges (no branching) ---
    workflow.add_edge("ingestion", "pii_agent")

    # --- Conditional edges (the routing logic lives here) ---
    workflow.add_conditional_edges(
        "pii_agent",
        route_after_pii,
        {
            "bias_agent":  "bias_agent",
            "correction":  "correction",
            "scorecard":   "scorecard",
        }
    )

    workflow.add_conditional_edges(
        "bias_agent",
        route_after_bias,
        {
            "explainability_agent": "explainability_agent",
            "correction":           "correction",
            "scorecard":            "scorecard",
        }
    )

    # Fixed edge: explainability is informational, always proceeds to policy
    workflow.add_edge("explainability_agent", "policy_agent")

    workflow.add_conditional_edges(
        "policy_agent",
        route_after_policy,
        {
            "scorecard":  "scorecard",
            "correction": "correction",
        }
    )

    # --- Cyclic edge: correction always loops back to PII agent ---
    workflow.add_conditional_edges(
        "correction",
        route_after_correction,
        {
            "pii_agent": "pii_agent",
        }
    )

    # --- Terminal node ---
    workflow.add_edge("scorecard", END)

    # --- Compile with SQLite checkpointer (the audit trail) ---
    # Each graph run is identified by a thread_id in the config.
    # LangGraph saves state at every node transition automatically.
    conn = sqlite3.connect("rai_audit.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    app = workflow.compile(checkpointer=checkpointer)

    print("[GRAPH] Compiled successfully. Audit trail → rai_audit.db")
    return app


# ---------------------------------------------------------------------------
# Convenience: build once at import time
# ---------------------------------------------------------------------------
app = build_graph()
