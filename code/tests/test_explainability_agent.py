"""
tests/test_explainability_agent.py
------------------------------------
Unit tests for the Explainability Agent (SHAP-backed).

Run with: python -m pytest tests/test_explainability_agent.py -v

Note: These tests require models/loan_model.pkl to exist.
Run  python data/train_model.py  first if the file is missing.
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from state import create_initial_state
from nodes.explainability_agent import explainability_agent_node


MODEL_EXISTS = (Path(__file__).parent.parent / "models" / "loan_model.pkl").exists()


def _run_model(feature_vector=None):
    state = create_initial_state(
        input_type="model_output",
        raw_input="Loan application DENIED. Risk score: 0.82.",
        feature_vector=feature_vector or {"age": 34, "income": 42000},
        protected_attributes=["sex"],
        prediction=0.82,
        predicted_label=1,
    )
    return explainability_agent_node(state)


def _run_text(text="Hello world"):
    state = create_initial_state(input_type="text", raw_input=text)
    return explainability_agent_node(state)


# ---------------------------------------------------------------------------
# Schema / contract tests
# ---------------------------------------------------------------------------

def test_result_has_required_keys():
    result = _run_model()
    exp = result["explainability_result"]
    assert "top_features" in exp
    assert "explanation_text" in exp
    assert "passed" in exp
    assert isinstance(exp["top_features"], list)
    assert isinstance(exp["explanation_text"], str)
    assert isinstance(exp["passed"], bool)


def test_always_passes():
    result = _run_model()
    assert result["explainability_result"]["passed"] is True


def test_no_violations_emitted():
    result = _run_model()
    assert result.get("violations", []) == []


def test_audit_log_entry_appended():
    result = _run_model()
    assert len(result["audit_log"]) == 1
    entry = result["audit_log"][0]
    assert entry["node"] == "explainability_agent"
    assert entry["action"] == "EXPLAINABILITY_CHECK"
    assert entry["result"] == "pass"


def test_top_feature_schema():
    result = _run_model()
    features = result["explainability_result"]["top_features"]
    if features:
        for f in features:
            assert "feature" in f
            assert "shap_value" in f
            assert "direction" in f
            assert f["direction"] in ("increases risk", "decreases risk")
            assert isinstance(f["shap_value"], float)


# ---------------------------------------------------------------------------
# Model-specific tests (skipped if model file absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MODEL_EXISTS, reason="models/loan_model.pkl not found — run data/train_model.py")
def test_model_output_produces_top_features():
    result = _run_model()
    features = result["explainability_result"]["top_features"]
    assert len(features) >= 1
    assert len(features) <= 5


@pytest.mark.skipif(not MODEL_EXISTS, reason="models/loan_model.pkl not found — run data/train_model.py")
def test_explanation_text_is_non_empty():
    result = _run_model()
    text = result["explainability_result"]["explanation_text"]
    assert len(text) > 20


@pytest.mark.skipif(not MODEL_EXISTS, reason="models/loan_model.pkl not found — run data/train_model.py")
def test_shap_values_are_finite():
    import math
    result = _run_model()
    for f in result["explainability_result"]["top_features"]:
        assert math.isfinite(f["shap_value"])


# ---------------------------------------------------------------------------
# Text mode stub tests
# ---------------------------------------------------------------------------

def test_text_mode_returns_passing_stub():
    result = _run_text("Some loan assessment text.")
    exp = result["explainability_result"]
    assert exp["passed"] is True
    assert exp["top_features"] == []
    assert "not applicable" in exp["explanation_text"].lower()


def test_text_mode_emits_audit_log():
    result = _run_text("Some text.")
    assert len(result["audit_log"]) == 1
    assert result["audit_log"][0]["node"] == "explainability_agent"
