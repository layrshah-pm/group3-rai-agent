# Capstone Report ↔ Codebase Mapping

**Project:** Navigating the Principles-to-Practice Gap — RAI Governance Framework  
**Report:** Capstone report Draft_V1.docx  
**Codebase:** `rai-compliance-agent/`

This document maps every major claim in the report's Section H (Proof of Concept) and Section F (Framework) to the actual files and functions that implement them.

---

## 1. System Architecture

### Report says (H.2.1):
> "A LangGraph-based automated compliance system... built on a **seven-node directed graph** with conditional edges that allow it to cycle back for self-correction."

### Code: `graph.py`

The seven nodes are registered here:
```python
workflow.add_node("ingestion",             ingestion_node)
workflow.add_node("pii_agent",             pii_agent_node)
workflow.add_node("bias_agent",            bias_agent_node)
workflow.add_node("explainability_agent",  explainability_agent_node)
workflow.add_node("policy_agent",          policy_agent_node)
workflow.add_node("correction",            correction_node)
workflow.add_node("scorecard",             scorecard_node)
```

The self-correction cycle (what makes LangGraph necessary — a linear pipeline cannot do this) is the edge:
```python
workflow.add_conditional_edges("correction", route_after_correction, {"pii_agent": "pii_agent"})
```

The `max_corrections` hard stop (default: 3) lives in `state.py`:
```python
max_corrections: int   # Threshold before escalation. Default: 3.
```

---

## 2. Five Pillars → Five Code Pillars

The report's Table in H.4.3 maps theoretical pillars to implementation pillars. Here's where each one actually lives in the code.

### Pillar 1: Governance & Accountability → `strategic_alignment` + `org_oversight`

**Report claim:** Policy Agent checks ACCOUNTABILITY and HUMAN_OVERSIGHT criteria. ESCALATED outcome enforces human override.

**Code:** `nodes/policy_agent.py`

```python
POLICY_CRITERIA = [
    {"id": "ACCOUNTABILITY", "description": "Responsible party for the AI decision must be identifiable",
     "reference": "EU AI Act Article 25 / NIST GOVERN 6.1"},
    {"id": "HUMAN_OVERSIGHT", "description": "Mechanism for human review must be available",
     "reference": "EU AI Act Article 14 / NIST MANAGE 1.3"},
    ...
]
```

Scorecard deduction in `nodes/scorecard.py`:
```python
# Strategic Alignment
if "ACCOUNTABILITY" in policy_violations: strategic -= 2
if "HUMAN_OVERSIGHT" in policy_violations: strategic -= 1

# Org Oversight
if "HUMAN_OVERSIGHT" in policy_violations: org_oversight -= 2
if "ACCOUNTABILITY" in policy_violations: org_oversight -= 1
```

The human override (ESCALATED outcome) is enforced in `graph.py` router:
```python
def route_after_bias(state):
    if not bias.get("passed", True):
        if state["correction_count"] < state["max_corrections"]:
            return "correction"
        else:
            return "scorecard"   # ← escalation path
```

---

### Pillar 2: Privacy & Data Stewardship → `data_governance`

**Report claim:** Microsoft Presidio detects and redacts PII. Deterministic — no LLM. DPDPA criteria in policy agent. SQLite audit trail = DPDP Rules 2025 data lineage.

**Code:** `nodes/pii_agent.py`

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

ENTITIES_TO_DETECT = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                      "CREDIT_CARD", "IBAN_CODE", "LOCATION", "NRP"]
CONFIDENCE_THRESHOLD = 0.6
```

The report's specific claim about Indian phone format (+91 98765 43210) scoring 0.62 is exactly this threshold being calibrated.

DPDPA-specific criteria in `policy_agent.py`:
```python
{"id": "DPDPA_CONSENT",           "reference": "India DPDPA 2023 Section 4 / Section 6 / Section 11"},
{"id": "DPDPA_DATA_MINIMISATION", "reference": "India DPDPA 2023 Section 6(a) / Section 8(3)"},
```

**Correction for PII — no LLM:** `nodes/correction.py`
```python
if violation == "PII_DETECTED":
    pii = state.get("pii_result") or {}
    redacted = pii.get("redacted_text")
    if redacted:
        print("[CORRECTION] Using Presidio redaction directly (no LLM).")
        return redacted
```
This is why the report says "no LLM call was made for this correction" — exactly as implemented.

**Audit trail (SQLite):** `graph.py`
```python
conn = sqlite3.connect("rai_audit.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)
```
The `rai_audit.db` file on disk (6.3MB, actively written) is the live audit database.

---

### Pillar 3: Fairness & Bias Mitigation → `model_governance`

**Report claim:** Fairlearn computes Demographic Parity Difference, Equalized Odds Difference, and Disparate Impact Ratio — the three metrics from Chouldechova (2017) and Barocas et al. (2023).

**Code:** `nodes/bias_agent.py`

```python
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

DEMOGRAPHIC_PARITY_THRESHOLD = 0.1
EQUALIZED_ODDS_THRESHOLD     = 0.1
DISPARATE_IMPACT_THRESHOLD   = 0.8   # US 4/5ths rule
```

German Credit dataset (Scenario 3):
```python
MODEL_PATH      = ROOT / "models" / "loan_model.pkl"    # biased model
MODEL_PATH_FAIR = ROOT / "models" / "loan_model_fair.pkl"
```

COMPAS dataset (Scenario 4):
```python
# The _model_path hint in feature_vector routes to:
ROOT / "models" / "compas_model.pkl"
```

The proxy discrimination finding (race excluded, bias persists via age + prior arrests) works because the COMPAS model pkl was trained without `race` as a feature, but `test_sensitive` (race labels) are still used to compute fairness metrics — demonstrating that bias persists through correlated proxies.

---

### Pillar 4: Transparency & Explainability → `org_oversight`

**Report claim:** SHAP values for top 5 features, labelled in plain English (increases risk / decreases risk). Operationalises EU AI Act Article 13 and Wachter et al. (2017) counterfactual standard.

**Code:** `nodes/explainability_agent.py`

```python
top_features = [
    {
        "feature": name,
        "shap_value": round(float(val), 4),
        "direction": "increases risk" if val > 0 else "decreases risk",
    }
    for name, val in ranked[:5]
]

explanation_text = "Key decision factors: " + "; ".join(parts) + "."
```

The agent is **always passing** (informational only):
```python
# This agent is INFORMATIONAL — it never sets a violation or triggers correction.
# It always passes. EU AI Act Article 13 requires meaningful explanation.
```

RAG-grounded policy evaluation (ChromaDB):
```python
# nodes/policy_agent.py
retrieved = retrieve_relevant_clauses(state["current_text"], k=5)
```
```python
# rag/retriever.py  — lazy-loads ChromaDB at rag/policy_store/
_STORE_PATH    = Path(__file__).parent / "policy_store"
_EMBED_MODEL   = "all-MiniLM-L6-v2"
```
The report's claim "evaluates compliance against the actual words of regulations" = ChromaDB queried by `retrieve_relevant_clauses()`, which pulls chunks from the indexed regulatory documents in `rag/documents/`.

---

### Pillar 5: Robustness & Monitoring → `continuous_monitoring`

**Report claim:** Self-correction loop with hard termination (max 3 cycles). SQLite audit trail records every node transition. ROBUSTNESS criterion checks uncertainty communication. Safe fallback if Ollama unavailable.

**Code:** `nodes/correction.py` (the loop) + `graph.py` (the termination)

```python
# correction_node increments and resets agent results each cycle:
return {
    "current_text": corrected_text,
    "correction_count": count,       # incremented
    "pii_result": None,              # reset — agents re-run fresh
    "bias_result": None,
    "policy_result": None,
    ...
}
```

Scorecard penalises repeated corrections — proxy for monitoring gap:
```python
# nodes/scorecard.py
if correction_count >= 2:
    continuous_monitoring -= 1
if correction_count >= state.get("max_corrections", 3):
    continuous_monitoring -= 1
```

Safe fallback in `policy_agent.py`:
```python
def _safe_fallback(error: str) -> PolicyResult:
    """Never crash the graph — return passing result with error noted."""
    return PolicyResult(violations=[], passed=True, severity="none",
                        summary=f"Policy check skipped due to LLM error: {error}")
```

---

## 3. Technology Stack Mapping

| Report Table (H.3.3) | File | Key Detail |
|---|---|---|
| **LangGraph** — directed graph, state checkpointing | `graph.py` | `StateGraph`, `SqliteSaver`, conditional edges |
| **Microsoft Presidio** — PII detection & anonymisation | `nodes/pii_agent.py` | `AnalyzerEngine`, `AnonymizerEngine`, threshold=0.6 |
| **Fairlearn** — DP Difference, EO Difference, DI Ratio | `nodes/bias_agent.py` | Thresholds: DP±0.1, EO±0.1, DI≥0.8 |
| **SHAP** — feature-level contribution | `nodes/explainability_agent.py` | `TreeExplainer` (or `KernelExplainer` fallback) |
| **Gemma 4 via Ollama** — policy compliance | `nodes/policy_agent.py`, `nodes/correction.py` | `ChatOllama(model="gemma4:latest", temperature=0)` |
| **ChromaDB + RAG** — regulatory vector store | `rag/retriever.py`, `rag/indexer.py` | `all-MiniLM-L6-v2`, cosine similarity, `rag/documents/` |
| **SQLite Audit DB** — tamper-proof log | `graph.py` → `rai_audit.db` | Every node transition recorded via LangGraph checkpointer |

---

## 4. Four Scenarios → Demo Data + Models

| Scenario | Expected Result | Input File | Model File |
|---|---|---|---|
| 1 — Clean Compliant Text | PASS | `demo_data/text_GOOD_compliant.txt` | n/a (text mode) |
| 2 — PII Data Leakage | CORRECTED | `demo_data/text_BAD_pii_leak.txt` | n/a (text mode) |
| 3 — Biased Loan Model | ESCALATED | `demo_data/model_BAD_biased.csv` | `models/loan_model.pkl` (German Credit, gender bias DI~0.316) |
| 4 — COMPAS Proxy Discrimination | ESCALATED | `demo_data/model_COMPAS_racial_bias.csv` | `models/compas_model.pkl` (race excluded, proxy via age + priors) |

Training scripts:
- `data/train_model.py` — trains German Credit biased/fair models
- `data/train_compas_model.py` — trains COMPAS model without race feature

---

## 5. State Schema → Audit Trail

`state.py` defines `ComplianceState` — the single object that flows through the entire graph. Key fields:

| State Field | Purpose | Report Reference |
|---|---|---|
| `pii_result` | Presidio output: entities found, redacted text | Scenario 2, Privacy Pillar |
| `bias_result` | Fairlearn metrics: DP diff, EO diff, DI ratio | Scenarios 3 & 4, Fairness Pillar |
| `policy_result` | LLM violations: 11 criteria, severity | All scenarios, Governance & Transparency Pillars |
| `explainability_result` | SHAP top-5 features + plain English | All model scenarios, Transparency Pillar |
| `retrieved_clauses` | ChromaDB chunks used in policy check | H.3.3 RAG grounding |
| `violations` | Accumulates via `operator.add` (never overwritten) | Immutable audit record |
| `correction_count` | Increments each correction cycle | H.4 ESCALATED after 3 cycles |
| `audit_log` | Accumulates via `operator.add` | EU AI Act Article 50 logging requirement |
| `final_status` | PASS / CORRECTED / FAIL / ESCALATED | H.4 Findings |
| `rai_scores` | Per-pillar 0-3 score dict | Radar chart in UI |
| `escalation_reason` | Why human review required | ESCALATED scenarios |

---

## 6. Regulatory Criteria — 11 Criteria in the Policy Agent

The report says the policy agent checks "11 regulatory criteria." Here they are verbatim from `policy_agent.py`:

| # | Criterion ID | Source |
|---|---|---|
| 1 | TRANSPARENCY | EU AI Act Article 50 / NIST GOVERN 1.1 |
| 2 | EXPLAINABILITY | EU AI Act Article 13 / NIST MEASURE 2.5 |
| 3 | HUMAN_OVERSIGHT | EU AI Act Article 14 / NIST MANAGE 1.3 |
| 4 | DATA_QUALITY | EU AI Act Article 10 / NIST MAP 3.5 |
| 5 | NON_DISCRIMINATION | EU AI Act Article 5(1)(b) / NIST MEASURE 2.2 |
| 6 | PRIVACY | GDPR Article 5 / EU AI Act Article 10(5) |
| 7 | ACCOUNTABILITY | EU AI Act Article 25 / NIST GOVERN 6.1 |
| 8 | ROBUSTNESS | EU AI Act Article 15 / NIST MEASURE 2.6 |
| 9 | DPDPA_CONSENT | India DPDPA 2023 Section 4 / 6 / 11 |
| 10 | DPDPA_DATA_MINIMISATION | India DPDPA 2023 Section 6(a) / 8(3) |
| 11 | RBI_MODEL_VALIDATION | RBI Model Risk Management Guidelines 2023 Section 3.2 / 4.1 |

---

## 7. Gaps Between Report and Code

These are things the report describes but the code either partially implements or flags as future work:

| Report Claim | Code Status | Notes |
|---|---|---|
| "Multi-lingual PII detection" (Hindi, regional languages) | Not implemented | Presidio configured for English only. Report explicitly flags this as a limitation (H.5.2). |
| "Text-based bias detection" | Not implemented | `bias_agent.py` returns a passing stub for `input_type == "text"`. Report acknowledges this. |
| "Indian datasets for bias testing" | Not implemented | German Credit and COMPAS are Western datasets. Report flags this in H.5.2 and I.5. |
| "Drift detection" | Not implemented | EIA Section 6 for the POC system notes "not applicable for this POC scope." |
| "Grievance mechanism for flagged orgs" | Not implemented | Identified as future work in H.5.2. |
| "More robust LLM serving" for production | Not implemented | Current: single Ollama process. `_safe_fallback()` handles crashes but no HA setup. |

---

## 8. RAG Store — Regulatory Documents

The report says the vector store contains "actual regulatory text from EU AI Act, DPDPA 2023, RBI Guidelines, NIST AI RMF, and GDPR."

These are the `.txt` files in `rag/documents/`, chunked with `[CHUNK]...[/CHUNK]` syntax and indexed by `rag/indexer.py` into `rag/policy_store/` (ChromaDB). The indexer uses:
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Distance metric: cosine
- Chunk ID: MD5 hash of content (deduplication on re-index)
- Retrieval: top-k by similarity at runtime via `retrieve_relevant_clauses(text, k=5)`

---

## 9. EIA Template Pillar Scores in Code

The report says Scenario 1 scored "averaging above 2.5 out of 3 across all five pillars." The scorecard logic in `nodes/scorecard.py` produces scores by deducting from a base of 3:

```
strategic_alignment:    3 - ACCOUNTABILITY(2) - HUMAN_OVERSIGHT(1) → min 0
data_governance:        3 - PII_failure(2) - bias_failure(1) → min 0
model_governance:       3 - bias(1) - TRANSPARENCY(1) - EXPLAINABILITY(1) - no_shap(1) → min 0
org_oversight:          3 - HUMAN_OVERSIGHT(2) - ACCOUNTABILITY(1) → min 0
continuous_monitoring:  3 - ROBUSTNESS(1) - corrections≥2(1) - corrections≥max(1) → min 0
```

Max total: 15. A clean pass with no violations = 15/15.

---

## 10. Files Not Yet Mapped to Report

| File | Status |
|---|---|
| `ui/app.py` (Streamlit UI, 34KB) | The radar chart + scenario runner UI. Report references "Streamlit user interface" and "radar chart visualisation" — this is where those live. |
| `nodes/ingestion.py` | Entry node — parses raw input and populates initial state fields. |
| `validate_demo.py` | Standalone script for running all 4 scenarios and verifying expected outcomes. |
| `report/section_*.md` | Earlier technical write-up sections — likely superseded by the final report. |
| `tests/` | 7 test files covering each node + integration. `test_integration.py` is the end-to-end scenario runner. |

---

*Generated by cross-referencing Capstone report Draft_V1.docx (94 pages) with the full rai-compliance-agent codebase as of May 2026.*
