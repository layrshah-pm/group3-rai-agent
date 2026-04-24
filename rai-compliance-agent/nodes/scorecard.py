"""
nodes/scorecard.py
------------------
Scorecard Generator — terminal node of the compliance graph.

Responsibilities:
  - Aggregate results from all auditor agents
  - Compute per-pillar RAI scores (0-3 maturity scale)
  - Set final_status based on violations and correction history
  - Write the final audit log entry
  - Print a human-readable summary to console

Scoring rubric (per pillar, 0-3):
  3 — Fully compliant, no violations, best practices evident
  2 — Mostly compliant, minor gaps or informational findings
  1 — Partial compliance, violations found but auto-corrected
  0 — Non-compliant, uncorrected violations present

Pillar → Agent mapping:
  Strategic Alignment    → policy_result (accountability, oversight criteria)
  Data Governance        → bias_result + pii_result
  Model Governance       → bias_result + explainability_result
  Org Oversight          → policy_result (human oversight, accountability)
  Continuous Monitoring  → policy_result (robustness) + correction history
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
    correction_count = state.get("correction_count", 0)
    pii = state.get("pii_result")
    bias = state.get("bias_result")
    policy = state.get("policy_result")

    # --- Determine final status ---
    if not violations:
        final_status = "PASS"
    elif correction_count > 0 and _all_corrected(state):
        final_status = "CORRECTED"
    elif correction_count >= state.get("max_corrections", 3):
        final_status = "ESCALATED"
    else:
        final_status = "FAIL"

    # --- Compute per-pillar scores ---
    rai_scores = _compute_pillar_scores(state, violations)

    # --- Final audit entry ---
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "scorecard",
        "action": "AUDIT_COMPLETE",
        "result": final_status,
        "detail": {
            "total_violations": len(violations),
            "correction_count": correction_count,
            "rai_scores": rai_scores,
            "overall_score": sum(rai_scores.values()),
            "max_possible_score": 15,   # 5 pillars x 3 max
        },
        "correction_count": correction_count,
    }

    # --- Print summary ---
    _print_summary(final_status, violations, rai_scores, correction_count)

    return {
        "final_status": final_status,
        "rai_scores": rai_scores,
        "current_node": "scorecard",
        "audit_log": [log_entry],
    }


def _all_corrected(state: ComplianceState) -> bool:
    """
    Returns True if all agents passed AFTER correction cycles ran.
    Checks the latest agent results (which reflect corrected text).
    """
    pii_ok = (state.get("pii_result") or {}).get("passed", True)
    bias_ok = (state.get("bias_result") or {}).get("passed", True)
    policy_ok = (state.get("policy_result") or {}).get("passed", True)
    return pii_ok and bias_ok and policy_ok


def _compute_pillar_scores(state: ComplianceState, violations: list) -> dict:
    """
    Maps agent results to the 5 RAI framework pillars.
    Returns a dict of pillar_name → score (0-3).
    """
    pii = state.get("pii_result") or {}
    bias = state.get("bias_result") or {}
    policy = state.get("policy_result") or {}
    correction_count = state.get("correction_count", 0)

    policy_violations = {v["id"] for v in (policy.get("violations") or [])}

    # --- Strategic Alignment ---
    # Checks: accountability, human oversight defined in policy
    strategic = 3
    if "ACCOUNTABILITY" in policy_violations:
        strategic -= 2
    if "HUMAN_OVERSIGHT" in policy_violations:
        strategic -= 1
    strategic = max(0, strategic)

    # --- Data Governance ---
    # Checks: PII protection, bias in data
    data_governance = 3
    if not pii.get("passed", True):
        data_governance -= 2
    if not bias.get("passed", True):
        data_governance -= 1
    data_governance = max(0, data_governance)

    # --- Model Governance ---
    # Checks: bias metrics, explainability, transparency
    model_governance = 3
    if not bias.get("passed", True):
        model_governance -= 1
    if "TRANSPARENCY" in policy_violations:
        model_governance -= 1
    if "EXPLAINABILITY" in policy_violations:
        model_governance -= 1
    # EU AI Act Article 13: model outputs require meaningful explanation
    explainability = state.get("explainability_result") or {}
    if state["input_type"] == "model_output":
        if not explainability.get("top_features"):
            model_governance = max(0, model_governance - 1)
    model_governance = max(0, model_governance)

    # --- Org Oversight ---
    # Checks: human oversight, accountability structure
    org_oversight = 3
    if "HUMAN_OVERSIGHT" in policy_violations:
        org_oversight -= 2
    if "ACCOUNTABILITY" in policy_violations:
        org_oversight -= 1
    org_oversight = max(0, org_oversight)

    # --- Continuous Monitoring ---
    # Checks: robustness disclosure, correction history (proxy for monitoring)
    continuous_monitoring = 3
    if "ROBUSTNESS" in policy_violations:
        continuous_monitoring -= 1
    if correction_count >= 2:
        continuous_monitoring -= 1   # needed multiple corrections = monitoring gap
    if correction_count >= state.get("max_corrections", 3):
        continuous_monitoring -= 1
    continuous_monitoring = max(0, continuous_monitoring)

    return {
        "strategic_alignment": strategic,
        "data_governance": data_governance,
        "model_governance": model_governance,
        "org_oversight": org_oversight,
        "continuous_monitoring": continuous_monitoring,
    }


def _print_summary(
    final_status: str,
    violations: list,
    rai_scores: dict,
    correction_count: int,
) -> None:
    """Prints a formatted compliance summary to the console."""
    divider = "─" * 55
    print(f"\n{divider}")
    print(f"  RAI COMPLIANCE AUDIT — FINAL REPORT")
    print(divider)
    print(f"  Status          : {final_status}")
    print(f"  Violations found: {len(violations)}")
    print(f"  Corrections made: {correction_count}")
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
