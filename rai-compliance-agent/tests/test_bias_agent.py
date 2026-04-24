"""
tests/test_bias_agent.py
-------------------------
Unit tests for the Bias Detection Agent (Fairlearn-backed).

Run with: python -m pytest tests/test_bias_agent.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from state import create_initial_state
from nodes.bias_agent import bias_agent_node


def _model_state(**kwargs):
    return create_initial_state(
        input_type="model_output",
        raw_input="Loan application DENIED. Risk score: 0.82.",
        feature_vector={"age": 34, "income": 42000, "gender": "F"},
        protected_attributes=["sex"],
        prediction=0.82,
        predicted_label=1,
        **kwargs,
    )


def test_biased_model_fails():
    """Imbalanced model on German Credit should violate DI < 0.8."""
    result = bias_agent_node(_model_state())
    br = result["bias_result"]
    assert br["passed"] is False
    assert br["disparate_impact_ratio"] < 0.8
    assert "BIAS_DETECTED" in result["violations"]


def test_bias_metrics_populated():
    """All three metric fields must be present and numeric."""
    result = bias_agent_node(_model_state())
    br = result["bias_result"]
    assert br["demographic_parity_diff"] is not None
    assert br["equalized_odds_diff"] is not None
    assert br["disparate_impact_ratio"] is not None
    assert isinstance(br["demographic_parity_diff"], float)
    assert isinstance(br["disparate_impact_ratio"], float)


def test_text_mode_passes():
    """Text mode always passes — bias check is model_output only."""
    state = create_initial_state(
        input_type="text",
        raw_input="The application has been reviewed.",
    )
    result = bias_agent_node(state)
    assert result["bias_result"]["passed"] is True
    assert result["violations"] == []


def test_no_protected_attributes_passes():
    """Empty protected_attributes list → skip check, return passed=True."""
    state = create_initial_state(
        input_type="model_output",
        raw_input="Loan DENIED.",
        protected_attributes=[],
        prediction=0.9,
        predicted_label=1,
    )
    result = bias_agent_node(state)
    assert result["bias_result"]["passed"] is True
    assert result["violations"] == []


def test_privileged_unprivileged_populated():
    """Must identify which group is privileged vs unprivileged."""
    result = bias_agent_node(_model_state())
    br = result["bias_result"]
    assert br["privileged_group"] is not None
    assert br["unprivileged_group"] is not None
    assert br["privileged_group"] != br["unprivileged_group"]


def test_bias_result_schema():
    result = bias_agent_node(_model_state())
    br = result["bias_result"]
    for key in ["demographic_parity_diff", "equalized_odds_diff",
                "disparate_impact_ratio", "privileged_group",
                "unprivileged_group", "passed", "details"]:
        assert key in br
    assert isinstance(br["passed"], bool)
    assert isinstance(br["details"], str)


def test_audit_log_entry_appended():
    result = bias_agent_node(_model_state())
    assert len(result["audit_log"]) == 1
    entry = result["audit_log"][0]
    assert entry["node"] == "bias_agent"
    assert entry["action"] == "BIAS_CHECK"
    assert "timestamp" in entry
