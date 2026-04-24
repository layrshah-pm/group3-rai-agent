"""
nodes/ingestion.py
------------------
Ingestion Node — entry point of the compliance graph.

Responsibilities:
  - Validate the input is non-empty
  - Normalise whitespace and encoding
  - Log the ingestion event to the audit trail
  - Pass state forward unchanged (routing handled by graph edges)

This node does NOT call any LLM or external library.
It is pure Python and should always succeed.
"""

from datetime import datetime, timezone
from state import ComplianceState


def ingestion_node(state: ComplianceState) -> dict:
    """
    Validates and normalises the incoming input.
    Returns a partial state update (only the fields this node touches).
    """
    print(f"\n[INGESTION] Starting compliance check")
    print(f"[INGESTION] Input type : {state['input_type']}")
    print(f"[INGESTION] Input      : {state['raw_input'][:120]}...")

    # --- Normalise text ---
    normalised = state["raw_input"].strip()

    # Guard: empty input
    if not normalised:
        raise ValueError("raw_input cannot be empty. Nothing to audit.")

    # Guard: input type must be known
    if state["input_type"] not in ("text", "model_output"):
        raise ValueError(
            f"Unknown input_type '{state['input_type']}'. "
            "Must be 'text' or 'model_output'."
        )

    # --- Audit log entry ---
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": "ingestion",
        "action": "INPUT_RECEIVED",
        "result": "ok",
        "detail": {
            "input_type": state["input_type"],
            "char_count": len(normalised),
            "has_feature_vector": state.get("feature_vector") is not None,
        },
        "correction_count": state["correction_count"],
    }

    print(f"[INGESTION] Normalised {len(normalised)} characters. Passing to PII agent.")

    return {
        "current_text": normalised,
        "current_node": "ingestion",
        "audit_log": [log_entry],   # operator.add will append this
    }
