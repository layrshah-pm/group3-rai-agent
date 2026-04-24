"""
state.py
--------
Defines the single shared state object that flows through every node
in the RAI Compliance Agent graph.

Every agent reads from this state and writes its results back into it.
LangGraph persists a snapshot of this state at every node transition,
giving us the tamper-proof audit trail required by EU AI Act Article 50.
"""

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Sub-schemas for each agent's result
# These are plain dicts (not TypedDicts) so they serialise cleanly to SQLite.
# ---------------------------------------------------------------------------

class PIIResult(TypedDict):
    """Output from the PII Detection Agent."""
    has_pii: bool                    # True if PII was found
    entities_found: list[dict]       # List of {entity_type, value, start, end}
    redacted_text: Optional[str]     # Text with PII replaced by <REDACTED>
    passed: bool                     # True if no PII found (or successfully redacted)


class BiasResult(TypedDict):
    """Output from the Bias Detection Agent."""
    demographic_parity_diff: Optional[float]   # Ideal: 0.0
    equalized_odds_diff: Optional[float]        # Ideal: 0.0
    disparate_impact_ratio: Optional[float]     # Must be >= 0.8 (legal threshold)
    privileged_group: Optional[str]
    unprivileged_group: Optional[str]
    passed: bool
    details: str                                # Human-readable summary


class PolicyResult(TypedDict):
    """Output from the Policy Compliance Agent (LLM-based)."""
    violations: list[dict]           # List of {article, description, severity}
    passed: bool
    severity: str                    # "none" | "low" | "medium" | "high"
    summary: str                     # LLM-generated plain-English summary


class ExplainabilityResult(TypedDict):
    """Output from the Explainability Agent."""
    top_features: list[dict]         # List of {feature, shap_value, direction}
    explanation_text: str            # Plain English explanation of the prediction
    passed: bool                     # Always True — explainability is informational


# ---------------------------------------------------------------------------
# Main state schema
# ---------------------------------------------------------------------------

class ComplianceState(TypedDict):
    """
    The single source of truth for the entire compliance graph.

    Annotated fields use operator.add so that when multiple nodes
    append to violations or audit_log, values accumulate rather than
    overwrite each other. All other fields overwrite on update.
    """

    # --- Input ---
    input_type: str
    # "text"         → auditing AI-generated text (emails, reports, summaries)
    # "model_output" → auditing a prediction from an ML model

    raw_input: str
    # The original unmodified text or prediction string that entered the graph.
    # Never mutated after ingestion — used for audit trail comparison.

    current_text: str
    # The working copy of the text. The self-correction node updates this
    # when it produces a cleaned version. Agents always read from here.

    # --- Model context (only populated in model_output mode) ---
    feature_vector: Optional[dict[str, Any]]
    # The input features used to generate the prediction
    # e.g. {"age": 34, "income": 45000, "zip_code": "10001"}

    protected_attributes: Optional[list[str]]
    # Which feature names are protected characteristics
    # e.g. ["gender", "race", "age"]

    prediction: Optional[float]
    # The model's output score or class probability
    # e.g. 0.73 (loan approval probability)

    predicted_label: Optional[int]
    # Binary decision derived from prediction (0 or 1)

    # --- Agent results (None until that agent runs) ---
    pii_result: Optional[PIIResult]
    bias_result: Optional[BiasResult]
    policy_result: Optional[PolicyResult]
    explainability_result: Optional[ExplainabilityResult]
    retrieved_clauses: Optional[list[dict]]      # regulatory chunks from RAG retrieval

    # --- Control flow ---
    violations: Annotated[list[str], operator.add]
    # Accumulates violation codes as agents find problems.
    # Uses operator.add so each agent appends without overwriting.
    # e.g. ["PII_DETECTED", "BIAS_DISPARATE_IMPACT", "POLICY_TRANSPARENCY"]

    active_violation: Optional[str]
    # The specific violation the self-correction node is currently fixing.
    # Set by the routing function before entering correction node.

    correction_count: int
    # How many self-correction cycles have run. Graph escalates to human
    # review if this reaches max_corrections.

    max_corrections: int
    # Threshold before escalation. Default: 3.

    current_node: str
    # Tracks which node we're in — useful for routing and audit logs.

    # --- Audit log ---
    audit_log: Annotated[list[dict], operator.add]
    # Every node appends a timestamped entry. Uses operator.add to accumulate.
    # Each entry: {timestamp, node, action, result, correction_count}
    # This list is what gets persisted to SQLite via the checkpointer.

    # --- Output ---
    final_status: str
    # "PASS"       → all checks passed, output is compliant
    # "CORRECTED"  → violations found and fixed by self-correction
    # "FAIL"       → violations found, correction failed or not applicable
    # "ESCALATED"  → max_corrections reached, human review required

    rai_scores: Optional[dict[str, int]]
    # Per-pillar scores 0-3 for the final scorecard.
    # Keys map to the 5 RAI framework pillars:
    # {
    #   "strategic_alignment": 2,
    #   "data_governance": 1,
    #   "model_governance": 3,
    #   "org_oversight": 2,
    #   "continuous_monitoring": 2
    # }

    escalation_reason: Optional[str]
    # Populated if final_status == "ESCALATED". Explains why human review
    # is needed and which violations could not be auto-corrected.


# ---------------------------------------------------------------------------
# Default state factory
# ---------------------------------------------------------------------------

def create_initial_state(
    input_type: str,
    raw_input: str,
    feature_vector: Optional[dict] = None,
    protected_attributes: Optional[list[str]] = None,
    prediction: Optional[float] = None,
    predicted_label: Optional[int] = None,
    max_corrections: int = 3,
) -> ComplianceState:
    """
    Returns a fresh ComplianceState with sensible defaults.
    Call this before invoking the graph.

    Example:
        state = create_initial_state(
            input_type="model_output",
            raw_input="Loan application DENIED. Risk score: 0.82",
            feature_vector={"age": 34, "income": 42000, "zip": "10001"},
            protected_attributes=["age", "gender"],
            prediction=0.82,
            predicted_label=1,
        )
        result = app.invoke(state, config={"configurable": {"thread_id": "run-001"}})
    """
    return ComplianceState(
        # Input
        input_type=input_type,
        raw_input=raw_input,
        current_text=raw_input,

        # Model context
        feature_vector=feature_vector,
        protected_attributes=protected_attributes,
        prediction=prediction,
        predicted_label=predicted_label,

        # Agent results (all None until agents run)
        pii_result=None,
        bias_result=None,
        policy_result=None,
        explainability_result=None,
        retrieved_clauses=None,

        # Control flow
        violations=[],
        active_violation=None,
        correction_count=0,
        max_corrections=max_corrections,
        current_node="start",

        # Audit
        audit_log=[],

        # Output (set by scorecard generator)
        final_status="pending",
        rai_scores=None,
        escalation_reason=None,
    )
