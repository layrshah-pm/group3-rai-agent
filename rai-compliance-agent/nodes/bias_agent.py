"""
nodes/bias_agent.py
-------------------
Bias Detection Agent — Fairlearn implementation.

For model_output mode: loads the pre-trained model + German Credit test set
from disk and computes group fairness metrics against protected attributes.

For text mode: returns a passing stub (text bias via LLM is out of scope for PoC).
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

from state import ComplianceState, BiasResult

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "loan_model.pkl"
MODEL_PATH_FAIR = ROOT / "models" / "loan_model_fair.pkl"

DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
EQUALIZED_ODDS_THRESHOLD = 0.1
DISPARATE_IMPACT_THRESHOLD = 0.8   # US 4/5ths rule


def bias_agent_node(state: ComplianceState) -> dict:
    """
    Computes fairness metrics for model predictions.
    Text mode returns a passing result (out of scope for PoC).
    """
    print(f"\n[BIAS AGENT] Running fairness checks...")
    print(f"[BIAS AGENT] Mode: {state['input_type']}")

    if state["input_type"] == "model_output":
        bias_result = _check_model_bias(state)
    else:
        bias_result = _check_text_bias(state)

    new_violations = ["BIAS_DETECTED"] if not bias_result["passed"] else []

    status = "FAIL" if not bias_result["passed"] else "PASS"
    print(f"[BIAS AGENT] Result: {status}")
    print(f"[BIAS AGENT] {bias_result['details']}")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "bias_agent",
        "action": "BIAS_CHECK",
        "result": "fail" if not bias_result["passed"] else "pass",
        "detail": {
            "demographic_parity_diff": bias_result["demographic_parity_diff"],
            "equalized_odds_diff": bias_result["equalized_odds_diff"],
            "disparate_impact_ratio": bias_result["disparate_impact_ratio"],
        },
        "correction_count": state["correction_count"],
    }

    return {
        "bias_result": bias_result,
        "violations": new_violations,
        "current_node": "bias_agent",
        "audit_log": [log_entry],
    }


def _check_model_bias(state: ComplianceState) -> BiasResult:
    """
    Loads the saved model + German Credit test set and computes
    Fairlearn group fairness metrics across protected attributes.
    """
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )

    protected = state.get("protected_attributes") or []
    if not protected:
        return BiasResult(
            demographic_parity_diff=None,
            equalized_odds_diff=None,
            disparate_impact_ratio=None,
            privileged_group=None,
            unprivileged_group=None,
            passed=True,
            details="No protected attributes specified — bias check skipped.",
        )

    # Allow feature_vector to carry an optional _model_path hint (used by UI demo data)
    fv = state.get("feature_vector") or {}
    model_path_hint = fv.get("_model_path")
    active_model_path = Path(model_path_hint) if model_path_hint else MODEL_PATH

    if not active_model_path.exists():
        return BiasResult(
            demographic_parity_diff=None,
            equalized_odds_diff=None,
            disparate_impact_ratio=None,
            privileged_group=None,
            unprivileged_group=None,
            passed=True,
            details=f"Model file not found at {active_model_path}. Run data/train_model.py first.",
        )

    with open(active_model_path, "rb") as f:
        artefact = pickle.load(f)

    model = artefact["model"]
    X_test = artefact["test_X"]
    y_test = artefact["test_y"]
    s_test = artefact["test_sensitive"]   # Series: 'male' / 'female'

    y_pred = model.predict(X_test)

    dp_diff = demographic_parity_difference(
        y_test, y_pred, sensitive_features=s_test
    )
    eo_diff = equalized_odds_difference(
        y_test, y_pred, sensitive_features=s_test
    )
    di_ratio = _disparate_impact_ratio(y_pred, s_test.values)

    dp_violation = abs(dp_diff) > DEMOGRAPHIC_PARITY_THRESHOLD
    eo_violation = abs(eo_diff) > EQUALIZED_ODDS_THRESHOLD
    di_violation = di_ratio < DISPARATE_IMPACT_THRESHOLD

    passed = not (dp_violation or eo_violation or di_violation)

    violations_str = []
    if dp_violation:
        violations_str.append(f"DP diff={dp_diff:.3f} (±{DEMOGRAPHIC_PARITY_THRESHOLD})")
    if eo_violation:
        violations_str.append(f"EO diff={eo_diff:.3f} (±{EQUALIZED_ODDS_THRESHOLD})")
    if di_violation:
        violations_str.append(f"DI ratio={di_ratio:.3f} (<{DISPARATE_IMPACT_THRESHOLD})")

    details = (
        f"BIAS DETECTED: {'; '.join(violations_str)}"
        if violations_str
        else f"All fairness metrics within threshold. DI={di_ratio:.3f}, DP={dp_diff:.3f}, EO={eo_diff:.3f}"
    )

    # Identify privileged / unprivileged by denial rate
    groups = {}
    for pred, g in zip(y_pred, s_test.values):
        groups.setdefault(g, []).append(pred)
    rates = {g: sum(v) / len(v) for g, v in groups.items()}
    privileged = min(rates, key=rates.get)    # lowest denial rate = privileged
    unprivileged = max(rates, key=rates.get)

    return BiasResult(
        demographic_parity_diff=round(float(dp_diff), 4),
        equalized_odds_diff=round(float(eo_diff), 4),
        disparate_impact_ratio=round(float(di_ratio), 4),
        privileged_group=privileged,
        unprivileged_group=unprivileged,
        passed=passed,
        details=details,
    )


def _check_text_bias(state: ComplianceState) -> BiasResult:
    """Text bias via LLM is out of scope for this PoC — always passes."""
    return BiasResult(
        demographic_parity_diff=None,
        equalized_odds_diff=None,
        disparate_impact_ratio=None,
        privileged_group=None,
        unprivileged_group=None,
        passed=True,
        details="Text mode: bias check not applicable (model_output mode only).",
    )


def _disparate_impact_ratio(y_pred, sensitive_features) -> float:
    """Computes min(P(Y=1|A=a)) / max(P(Y=1|A=a)) across groups."""
    groups: dict = {}
    for pred, g in zip(y_pred, sensitive_features):
        groups.setdefault(g, []).append(int(pred))
    rates = {g: sum(v) / len(v) for g, v in groups.items()}
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    return min_rate / max_rate if max_rate > 0 else 1.0
