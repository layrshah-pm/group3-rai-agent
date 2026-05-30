"""
nodes/scorecard.py
------------------
Scorecard Generator — terminal node of the compliance graph.

Responsibilities:
  - Aggregate results from all auditor agents
  - Compute per-pillar RAI scores (0-3 maturity scale)
  - Set final_status: PASS or FAIL
  - Write the final audit log entry and step_trace entry
  - Print a human-readable summary to console

Scoring rubric (per pillar, 0-3):
  3 — Fully compliant, no violations
  2 — Mostly compliant, minor gaps
  1 — Partial compliance, violations found
  0 — Non-compliant, multiple violations

Pillar → Agent mapping:
  Strategic Alignment    → policy_result (accountability, oversight criteria)
  Data Governance        → bias_result + pii_result
  Model Governance       → bias_result + explainability_result
  Org Oversight          → policy_result (human oversight, accountability)
  Continuous Monitoring  → policy_result (robustness)
"""

from datetime import datetime, timezone
from state import ComplianceState


def scorecard_node(state: ComplianceState) -> dict:
    """
    Computes the final RAI scorecard and sets final_status.
    This is a terminal node — no further routing after this.
    """
    print(f"\n[SCORECARD] Computing final RAI scores...")

    violations = state.get("violations", [])

    # --- Compute per-pillar scores ---
    if state.get("input_type") == "policy_document":
        rai_scores = _compute_pillar_scores_policy_document(state)
    else:
        rai_scores = _compute_pillar_scores(state, violations)

    # --- Determine final status ---
    # CORRECTED: correction ran and all pillars now pass.
    # PASS:      no violations from the start.
    # FAIL:      violations remain after all correction attempts (or no correction).
    low_pillars = [k for k, v in rai_scores.items() if v < 2]
    if low_pillars or sum(rai_scores.values()) < 10:
        final_status = "FAIL"
    elif state.get("correction_count", 0) > 0:
        final_status = "CORRECTED"
    else:
        final_status = "PASS"

    # --- Final audit entry ---
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "scorecard",
        "action": "AUDIT_COMPLETE",
        "result": final_status,
        "detail": {
            "total_violations": len(violations),
            "rai_scores": rai_scores,
            "overall_score": sum(rai_scores.values()),
            "max_possible_score": 15,
        },
    }

    step_entry = {
        "step":  "scorecard",
        "label": "Scorecard",
        "status": final_status.lower(),
        "prompt": None,
        "response": {
            "final_status": final_status,
            "rai_scores": rai_scores,
            "overall_score": sum(rai_scores.values()),
            "max_possible": 15,
            "violations_total": len(violations),
        },
        "summary": (
            f"Final: {final_status}. Overall score {sum(rai_scores.values())}/15. "
            f"{len(violations)} violation(s) found."
        ),
    }

    # --- Print summary ---
    _print_summary(final_status, violations, rai_scores)

    return {
        "final_status": final_status,
        "rai_scores": rai_scores,
        "current_node": "scorecard",
        "audit_log": [log_entry],
        "step_trace": [step_entry],
    }


def _compute_pillar_scores(state: ComplianceState, violations: list) -> dict:
    """
    Maps agent results to the 5 RAI framework pillars.
    Returns a dict of pillar_name → score (0-3).
    """
    pii = state.get("pii_result") or {}
    bias = state.get("bias_result") or {}
    policy = state.get("policy_result") or {}

    policy_violations = {v["id"] for v in (policy.get("violations") or [])}

    # --- Strategic Alignment ---
    strategic = 3
    if "ACCOUNTABILITY" in policy_violations:
        strategic -= 2
    if "HUMAN_OVERSIGHT" in policy_violations:
        strategic -= 1
    strategic = max(0, strategic)

    # --- Data Governance ---
    data_governance = 3
    if not pii.get("passed", True):
        data_governance -= 2
    if not bias.get("passed", True):
        data_governance -= 1
    data_governance = max(0, data_governance)

    # --- Model Governance ---
    model_governance = 3
    if not bias.get("passed", True):
        model_governance -= 1
    if "TRANSPARENCY" in policy_violations:
        model_governance -= 1
    if "EXPLAINABILITY" in policy_violations:
        model_governance -= 1
    explainability = state.get("explainability_result") or {}
    if state["input_type"] == "model_output":
        if not explainability.get("top_features"):
            model_governance = max(0, model_governance - 1)
    model_governance = max(0, model_governance)

    # --- Org Oversight ---
    org_oversight = 3
    if "HUMAN_OVERSIGHT" in policy_violations:
        org_oversight -= 2
    if "ACCOUNTABILITY" in policy_violations:
        org_oversight -= 1
    org_oversight = max(0, org_oversight)

    # --- Continuous Monitoring ---
    continuous_monitoring = 3
    if "ROBUSTNESS" in policy_violations:
        continuous_monitoring -= 1
    continuous_monitoring = max(0, continuous_monitoring)

    return {
        "strategic_alignment": strategic,
        "data_governance": data_governance,
        "model_governance": model_governance,
        "org_oversight": org_oversight,
        "continuous_monitoring": continuous_monitoring,
    }


def _compute_pillar_scores_policy_document(state: ComplianceState) -> dict:
    """
    For policy_document mode: maps the 5 criterion compliance levels directly
    to the 5 RAI pillar keys. Level 0-3 from LLM becomes the pillar score.

    Defaults to 0 (absent) for any criterion the LLM did not score, so that
    assessment failures are conservative rather than optimistic.
    """
    pillar_gaps = state.get("pillar_gaps") or {}
    mapping = {
        "GOVERNANCE_ACCOUNTABILITY":    "strategic_alignment",
        "FAIRNESS_BIAS":                "model_governance",
        "TRANSPARENCY_EXPLAINABILITY":  "org_oversight",
        "ROBUSTNESS_MONITORING":        "continuous_monitoring",
        "PRIVACY_DATA_STEWARDSHIP":     "data_governance",
    }
    scores = {}
    for criterion_id, pillar_key in mapping.items():
        gap_data = pillar_gaps.get(criterion_id, {})
        # Default 0: an unassessed pillar must not pass by default
        scores[pillar_key] = gap_data.get("level", 0)
    return scores


def _print_summary(
    final_status: str,
    violations: list,
    rai_scores: dict,
) -> None:
    """Prints a formatted compliance summary to the console."""
    divider = "─" * 55
    print(f"\n{divider}")
    print(f"  RAI COMPLIANCE AUDIT — FINAL REPORT")
    print(divider)
    print(f"  Status          : {final_status}")
    print(f"  Violations found: {len(violations)}")
    print(f"\n  PILLAR SCORES (0-3):")
    for pillar, score in rai_scores.items():
        bar = "█" * score + "░" * (3 - score)
        print(f"  {pillar:<26} {bar}  {score}/3")
    total = sum(rai_scores.values())
    print(f"\n  OVERALL SCORE   : {total}/15")
    if violations:
        print(f"\n  VIOLATIONS:")
        for v in set(violations):
            print(f"    → {v}")
    print(divider)
