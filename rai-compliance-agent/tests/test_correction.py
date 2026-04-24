"""
tests/test_correction.py
-------------------------
Unit tests for the Self-Correction Node.

Run with: python -m pytest tests/test_correction.py -v
Requires: ollama running with gemma4:latest (for LLM correction tests)
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from state import create_initial_state
from nodes.correction import correction_node, _attempt_correction


PII_TEXT = "Dear John Smith, email john@example.com or call 650-253-0000."
REDACTED_TEXT = "Dear <PERSON>, email <EMAIL_ADDRESS> or call <PHONE_NUMBER>."
POLICY_TEXT = "The loan application has been denied by our system."


def _state_with_pii(redacted_text=None):
    state = create_initial_state(input_type="text", raw_input=PII_TEXT)
    state["active_violation"] = "PII_DETECTED"
    state["pii_result"] = {
        "has_pii": True,
        "entities_found": [
            {"entity_type": "PERSON", "value": "John Smith", "start": 5, "end": 15, "score": 0.95},
            {"entity_type": "EMAIL_ADDRESS", "value": "john@example.com", "start": 23, "end": 39, "score": 0.99},
        ],
        "redacted_text": redacted_text,
        "passed": False,
    }
    return state


def test_pii_correction_uses_redacted_text_directly():
    """PII correction must use Presidio output, never call LLM."""
    state = _state_with_pii(redacted_text=REDACTED_TEXT)
    with patch("nodes.correction.ChatOllama") as mock_llm:
        result = _attempt_correction(state, "PII_DETECTED")
    assert result == REDACTED_TEXT
    mock_llm.assert_not_called()   # LLM must not be invoked for PII


def test_pii_correction_no_redacted_text_falls_back_to_llm():
    """If redacted_text is None, fall back to LLM correction."""
    state = _state_with_pii(redacted_text=None)
    mock_response = MagicMock()
    mock_response.content = "Dear [PERSON], email [EMAIL]."
    with patch("nodes.correction.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value = mock_response
        result = _attempt_correction(state, "PII_DETECTED")
    assert result == "Dear [PERSON], email [EMAIL]."


def test_correction_increments_count():
    """correction_node must increment correction_count by exactly 1."""
    state = create_initial_state(input_type="text", raw_input=POLICY_TEXT)
    state["active_violation"] = "POLICY_TRANSPARENCY"
    result = correction_node(state)
    assert result["correction_count"] == 1


def test_correction_resets_agent_results():
    """All agent results must be None after correction so they re-run."""
    state = create_initial_state(input_type="text", raw_input=POLICY_TEXT)
    state["active_violation"] = "POLICY_TRANSPARENCY"
    state["pii_result"] = {"has_pii": False, "passed": True, "entities_found": [], "redacted_text": None}
    state["bias_result"] = {"passed": True, "details": "ok"}
    state["policy_result"] = {"violations": [], "passed": True, "severity": "none", "summary": "ok"}
    result = correction_node(state)
    assert result["pii_result"] is None
    assert result["bias_result"] is None
    assert result["policy_result"] is None
    assert result["active_violation"] is None


def test_correction_llm_error_returns_original():
    """LLM failure must not crash the graph — returns original text."""
    state = create_initial_state(input_type="text", raw_input=POLICY_TEXT)
    state["active_violation"] = "POLICY_TRANSPARENCY"
    with patch("nodes.correction.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.side_effect = Exception("Connection refused")
        result = correction_node(state)
    assert result["current_text"] == POLICY_TEXT
    assert result["correction_count"] == 1   # still increments


def test_correction_audit_log_appended():
    """Correction node must append exactly one audit log entry."""
    state = create_initial_state(input_type="text", raw_input=POLICY_TEXT)
    state["active_violation"] = "POLICY_TRANSPARENCY"
    with patch("nodes.correction.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.side_effect = Exception("offline")
        result = correction_node(state)
    assert len(result["audit_log"]) == 1
    entry = result["audit_log"][0]
    assert entry["node"] == "correction"
    assert entry["action"] == "AUTO_CORRECTION"
    assert entry["detail"]["violation"] == "POLICY_TRANSPARENCY"
