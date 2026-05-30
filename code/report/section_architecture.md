# Section: Architecture and State Schema
**Person 1+2 · IIMA Capstone EPAIBBL01 · Group 3**

---

## 1. System Architecture

The RAI Compliance Agent is built on **LangGraph**, a framework for constructing stateful, cyclic computation graphs using directed edges and conditional routing. Unlike a linear pipeline, LangGraph allows nodes to loop back — a property essential for self-correction.

### Why LangGraph

Traditional compliance pipelines run once and flag violations as static reports. This system is designed to *correct* violations before output reaches the end user. To do so, it needs to:

1. detect a violation,
2. attempt a correction,
3. re-audit the corrected text, and
4. repeat up to `MAX_CORRECTIONS` times.

LangGraph's cyclic DAG architecture supports this loop natively. Each node returns a partial state update; the routing function decides the next node based on the current state.

### Graph topology

```
                ┌─────────────┐
         ┌─────►│  pii_agent  │──────────┐
         │      └─────────────┘          │
         │             │ violations?      │
         │             ▼                 │
INGESTION│      ┌─────────────┐          │
    ──────┤     │  bias_agent │          ├──► SCORECARD
         │      └─────────────┘          │
         │             │ violations?      │
         │             ▼                 │
         │      ┌─────────────┐          │
         │      │policy_agent │──────────┘
         │      └─────────────┘
         │             │ violations?
         │             ▼
         │      ┌─────────────┐
         └──────│  correction │
                └─────────────┘
                (max 3 cycles)
```

When no violations remain, or `correction_count` reaches `max_corrections`, routing directs to the `scorecard` terminal node.

---

## 2. State Schema

The central data structure is `ComplianceState`, a TypedDict. All nodes receive this state and return a *partial update* dictionary — they never mutate state directly.

```python
class ComplianceState(TypedDict):
    # Input
    input_type: str                    # "text" | "model_output"
    raw_input: str
    current_text: str                  # updated by ingestion + correction
    feature_vector: dict | None
    protected_attributes: list[str] | None
    prediction: float | None
    predicted_label: int | None

    # Agent results (reset to None between correction cycles)
    pii_result: PIIResult | None
    bias_result: BiasResult | None
    policy_result: PolicyResult | None
    explainability_result: dict | None

    # Violation tracking (operator.add — accumulates across cycles)
    violations: Annotated[list[str], operator.add]
    active_violation: str | None

    # Control flow
    correction_count: int
    max_corrections: int               # default: 3
    current_node: str

    # Output
    final_status: str                  # "PASS" | "CORRECTED" | "FAIL" | "ESCALATED"
    rai_scores: dict | None

    # Audit trail (operator.add — accumulates across all cycles)
    audit_log: Annotated[list[dict], operator.add]
    escalation_reason: str | None
```

### Key design decisions

**`operator.add` for accumulating fields.** LangGraph merges node return values into state. For scalar fields the new value replaces the old one. For `violations` and `audit_log`, we annotate with `operator.add` so each node's new entries are *appended* rather than overwritten. Without this, a correction cycle would erase the violation list from the previous audit pass.

**Agent results reset to `None` between correction cycles.** When `correction_node` runs, it sets `pii_result = None`, `bias_result = None`, `policy_result = None`. This forces the agents to re-execute on the corrected text rather than re-using stale results. The routing logic checks `.get("passed", True)` — a `None` result is treated as "not yet run", so the graph routes forward normally.

**`max_corrections` as a hard cap.** The routing function checks `correction_count >= max_corrections` and routes directly to `scorecard` (with `final_status = "ESCALATED"`) rather than back to `correction`. This prevents infinite cycles even if an LLM correction repeatedly fails.

---

## 3. Routing Logic

The conditional edge after each audit agent evaluates a priority order:

```python
def route_after_agents(state):
    pii_passed  = (state.get("pii_result")    or {}).get("passed", True)
    bias_passed = (state.get("bias_result")   or {}).get("passed", True)
    policy_passed = (state.get("policy_result") or {}).get("passed", True)

    at_max = state["correction_count"] >= state["max_corrections"]

    if not pii_passed or not bias_passed or not policy_passed:
        if at_max:
            return "scorecard"      # escalate — cannot correct further
        return "correction"         # attempt fix
    return "scorecard"              # all passed — finalise
```

Priority of violations for correction: **PII > Bias > worst-severity Policy**. This ensures the most privacy-critical issues are addressed first.

---

## 4. Audit Trail Persistence

The graph uses LangGraph's `SqliteSaver` checkpointer, which persists the full state after every node execution into `rai_audit.db`. This provides:

- **Tamper-evident record**: every intermediate state is stored with a unique `checkpoint_id`
- **Thread isolation**: each run has a UUID `thread_id`; runs do not interfere
- **Replay capability**: a past run can be resumed or inspected by re-supplying its `thread_id`

The `utils/audit_logger.py` module decodes the msgpack-serialised checkpoints and extracts the `audit_log` entries for display in the Streamlit UI.

---

## 5. Scalability Considerations

The current implementation runs all agents synchronously in a single process. In a production context:

- The `policy_agent` LLM call (Ollama/Gemma4) is the bottleneck: ~60–90s per invocation
- A production system would use an async graph, a faster model endpoint, or a caching layer for repeated identical inputs
- `SqliteSaver` is single-writer; a production deployment would swap in `PostgresSaver` for concurrent access
