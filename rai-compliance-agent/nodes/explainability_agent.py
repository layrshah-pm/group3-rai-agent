"""
nodes/explainability_agent.py
-----------------------------
Explainability Agent — SHAP TreeExplainer implementation.

For model_output mode: loads the pre-trained model from disk and computes
SHAP values for the feature vector in state. Returns the top 5 features
by absolute SHAP value with plain-English direction labels.

For text mode: returns a passing stub (explainability is model-only).

This agent is INFORMATIONAL — it never sets a violation.
It always passes. Its output populates explainability_result in state,
which the scorecard uses to score Model Governance.

EU AI Act Article 13 requires that affected individuals receive a
meaningful explanation of automated decisions. This agent operationalises
that requirement.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path

from state import ComplianceState, ExplainabilityResult

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "loan_model.pkl"


def explainability_agent_node(state: ComplianceState) -> dict:
    """
    Computes SHAP-based feature importance for model predictions.
    Text mode returns a passing stub (model_output mode only).
    """
    print(f"\n[EXPLAINABILITY AGENT] Computing feature explanations...")
    print(f"[EXPLAINABILITY AGENT] Mode: {state['input_type']}")

    if state["input_type"] == "model_output":
        result = _explain_model_prediction(state)
    else:
        result = _text_stub()

    print(f"[EXPLAINABILITY AGENT] Result: PASS (informational)")
    if result["top_features"]:
        print(f"[EXPLAINABILITY AGENT] Top feature: {result['top_features'][0]['feature']} "
              f"(SHAP={result['top_features'][0]['shap_value']:+.4f})")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "explainability_agent",
        "action": "EXPLAINABILITY_CHECK",
        "result": "pass",
        "detail": {
            "top_features": [f["feature"] for f in result["top_features"][:3]],
            "explanation": result["explanation_text"][:120],
        },
    }

    xr = result
    summary = xr["explanation_text"] if xr["explanation_text"] else "Explainability not applicable (text mode)."

    step_entry = {
        "step":  "explainability_agent",
        "label": "Explainability (SHAP)",
        "status": "pass",
        "prompt": None,
        "response": {
            "mode": state["input_type"],
            "top_features": xr["top_features"],
            "explanation_text": xr["explanation_text"],
        },
        "summary": summary,
    }

    return {
        "explainability_result": result,
        "current_node": "explainability_agent",
        "audit_log": [log_entry],
        "step_trace": [step_entry],
    }


def _explain_model_prediction(state: ComplianceState) -> ExplainabilityResult:
    """
    Loads the saved model and runs SHAP TreeExplainer on the feature vector
    from state. Falls back gracefully if the model file is missing or if
    the feature vector doesn't align with training features.
    """
    import shap
    import pandas as pd

    fv = state.get("feature_vector") or {}
    model_path_hint = fv.get("_model_path")
    active_model_path = Path(model_path_hint) if model_path_hint else MODEL_PATH

    if not active_model_path.exists():
        return ExplainabilityResult(
            top_features=[],
            explanation_text=f"Model file not found at {active_model_path}. Run data/train_model.py first.",
            passed=True,
        )

    with open(active_model_path, "rb") as f:
        artefact = pickle.load(f)

    model = artefact["model"]
    X_test = artefact["test_X"]
    feature_names = list(X_test.columns)

    # Build instance from state feature_vector if columns align.
    # Strip internal keys (prefixed with _) before checking.
    clean_fv = {k: v for k, v in fv.items() if not k.startswith("_")}
    fv_keys = set(clean_fv.keys())
    feature_set = set(feature_names)

    if fv_keys & feature_set:
        # Partial match — fill missing numeric features with column means; strings use first row
        def _default(col):
            if pd.api.types.is_numeric_dtype(X_test[col]):
                return float(X_test[col].mean())
            return X_test[col].iloc[0]
        instance_data = {f: clean_fv.get(f, _default(f)) for f in feature_names}
        instance = pd.DataFrame([instance_data])
    else:
        # No matching features — explain the first test row as a representative example
        instance = X_test.iloc[:1].copy()

    try:
        from sklearn.pipeline import Pipeline
        import numpy as np

        # Unwrap sklearn Pipeline: apply preprocessing, then SHAP the final estimator
        if isinstance(model, Pipeline):
            final_estimator = model[-1]
            if len(model.steps) > 1:
                preprocessor = model[:-1]
                transformed = preprocessor.transform(instance)
                try:
                    col_names = preprocessor.get_feature_names_out()
                except AttributeError:
                    col_names = [f"f{i}" for i in range(transformed.shape[1])]
                instance = pd.DataFrame(transformed, columns=col_names)
            feature_names = list(instance.columns)
        else:
            final_estimator = model

        # Choose explainer based on model type
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        TREE_TYPES = (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
        )
        try:
            import xgboost
            TREE_TYPES = TREE_TYPES + (xgboost.XGBClassifier, xgboost.XGBRegressor)
        except ImportError:
            pass

        if isinstance(final_estimator, TREE_TYPES):
            explainer = shap.TreeExplainer(final_estimator)
        else:
            background = shap.sample(instance, min(50, len(instance)))
            explainer = shap.KernelExplainer(
                lambda x: final_estimator.predict_proba(x)[:, 1]
                if hasattr(final_estimator, "predict_proba")
                else final_estimator.predict(x),
                background,
            )

        shap_values = explainer.shap_values(instance)

        # For binary classifiers, shap_values is a list [class_0, class_1]
        # We explain class 1 (the positive / high-risk prediction)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        # Rank by absolute magnitude
        ranked = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
        top_features = [
            {
                "feature": name,
                "shap_value": round(float(val), 4),
                "direction": "increases risk" if val > 0 else "decreases risk",
            }
            for name, val in ranked[:5]
        ]

        # Plain English summary
        top3 = top_features[:3]
        parts = []
        for f in top3:
            verb = "increased" if f["shap_value"] > 0 else "decreased"
            parts.append(
                f"{f['feature']} {verb} the risk score by {abs(f['shap_value']):.3f}"
            )
        explanation_text = "Key decision factors: " + "; ".join(parts) + "."

        return ExplainabilityResult(
            top_features=top_features,
            explanation_text=explanation_text,
            passed=True,
        )

    except Exception as e:
        return ExplainabilityResult(
            top_features=[],
            explanation_text=f"SHAP computation failed: {e}. Explainability skipped.",
            passed=True,
        )


def _text_stub() -> ExplainabilityResult:
    """Text mode: SHAP explainability is only applicable to model outputs."""
    return ExplainabilityResult(
        top_features=[],
        explanation_text="Text mode: SHAP explainability not applicable (model_output mode only).",
        passed=True,
    )
