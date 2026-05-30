# RAI Compliance Agent — Project Plan
**IIMA Capstone · EPAIBBL01 · Group 3**
**Last updated: April 2026**

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Tech Stack](#3-tech-stack)
4. [Team Assignments](#4-team-assignments)
5. [Timeline & Dependencies](#5-timeline--dependencies)
6. [API Contracts](#6-api-contracts)
7. [Per-Person Work Breakdown](#7-per-person-work-breakdown)
8. [Integration Milestones](#8-integration-milestones)
9. [Demo Scenarios](#9-demo-scenarios)
10. [Testing Requirements](#10-testing-requirements)
11. [Risk Register](#11-risk-register)
12. [Definition of Done](#12-definition-of-done)

---

## 1. Project Overview

### What we are building
A **LangGraph-based Automated AI Compliance Agent** that operationalises the group's Responsible AI governance framework. The agent audits AI-generated outputs in real time, self-corrects violations, and produces a tamper-proof audit trail aligned with EU AI Act Article 50.

### Why LangGraph
LangGraph's cyclic Directed Acyclic Graph (DAG) architecture allows governance to run at **machine speed** — detecting and correcting violations before output reaches the end user, without human-in-the-loop bottlenecks. This directly addresses the "asymmetric velocity" problem identified in the research synthesis.

### The five RAI pillars this PoC operationalises
| Pillar | Implemented by |
|---|---|
| Strategic Alignment | Policy Agent (regulatory mapping) |
| Data Governance | PII Agent + Bias Agent |
| Model Governance | Bias Agent (fairness metrics + explainability) |
| Org Oversight | Policy Agent (accountability / oversight criteria) |
| Continuous Monitoring | Self-correction loop + Audit Logger |

### What the PoC does NOT cover (future work)
- Hallucination / factual grounding via RAG pipeline
- Real-time production deployment
- Multi-lingual PII (Hindi, regional languages)
- SHAP deep integration (basic feature importance only)

---

## 2. Repository Structure

```
rai-compliance-agent/
│
├── PLAN.md                   ← this file
├── README.md                 ← setup + run instructions (Person 7)
├── requirements.txt          ← all dependencies (Person 1+2)
├── .env.example              ← API key template (Person 1+2)
├── rai_audit.db              ← SQLite audit trail (auto-generated at runtime)
│
├── state.py                  ← ComplianceState schema (Person 1+2) ✓ DONE
├── graph.py                  ← Graph wiring + routing logic (Person 1+2) ✓ DONE
├── main.py                   ← Entry point + demo scenarios (Person 7)
│
├── nodes/
│   ├── __init__.py           ← (Person 1+2) ✓ DONE
│   ├── ingestion.py          ← Ingestion node (Person 1+2) ✓ DONE
│   ├── pii_agent.py          ← PII Detection (Person 3)
│   ├── bias_agent.py         ← Bias + Fairness (Person 4)
│   ├── policy_agent.py       ← Regulatory Compliance LLM (Person 5)
│   ├── correction.py         ← Self-correction loop (Person 5)
│   └── scorecard.py          ← Final scoring (Person 1+2) ✓ DONE
│
├── data/
│   ├── german_credit.csv     ← Demo dataset (Person 4 downloads)
│   └── compas_sample.csv     ← Backup dataset (Person 4)
│
├── models/
│   └── loan_model.pkl        ← Pre-trained demo model (Person 4)
│
├── utils/
│   ├── __init__.py
│   └── audit_logger.py       ← SQLite audit query helpers (Person 1+2)
│
├── ui/
│   ├── app.py                ← Streamlit main app (Person 6)
│   ├── components/
│   │   ├── radar_chart.py    ← Plotly radar chart (Person 6)
│   │   ├── audit_table.py    ← Audit log display (Person 6)
│   │   └── status_badge.py   ← Pass/Fail/Corrected badge (Person 6)
│   └── assets/
│       └── style.css         ← Custom CSS (Person 6)
│
└── tests/
    ├── test_pii_agent.py     ← (Person 3)
    ├── test_bias_agent.py    ← (Person 4)
    ├── test_policy_agent.py  ← (Person 5)
    ├── test_correction.py    ← (Person 5)
    └── test_integration.py   ← (Person 7)
```

---

## 3. Tech Stack

### Core
| Package | Version | Purpose |
|---|---|---|
| `langgraph` | ≥ 0.2.0 | Graph orchestration + cyclic routing |
| `langchain` | ≥ 0.3.0 | LLM abstraction layer |
| `langchain-ollama` | ≥ 0.2.0 | Gemma (local via Ollama) for Policy + Correction nodes |
| `python` | 3.11+ | Runtime |

### PII Detection (Person 3)
| Package | Purpose |
|---|---|
| `presidio-analyzer` | Entity recognition |
| `presidio-anonymizer` | Redaction engine |
| `spacy` + `en_core_web_lg` | NLP model for Presidio |

### Bias & Fairness (Person 4)
| Package | Purpose |
|---|---|
| `fairlearn` | Fairness metrics (demographic parity, equalized odds) |
| `aif360` | Additional bias metrics (disparate impact) |
| `scikit-learn` | Model training (LogisticRegression) |
| `shap` | Feature importance |
| `pandas` / `numpy` | Data manipulation |

### UI (Person 6)
| Package | Purpose |
|---|---|
| `streamlit` | Web application |
| `plotly` | Radar chart + visualisations |

### Utilities
| Package | Purpose |
|---|---|
| `pydantic` ≥ 2.0 | Structured LLM output validation |
| `python-dotenv` | API key management |

### LLM Cost guidance
Use **Gemma via Ollama** for all LLM calls (Policy Agent, Correction Node). Runs locally — no API cost. Requires `ollama pull gemma` before first run.

---

## 4. Team Assignments

```
┌─────────────────────────────────────────────────────────────────────┐
│  Person 1+2   Core Architecture                                     │
│               state.py · graph.py · audit_logger.py                 │
│               routing logic · checkpointer · scorecard              │
│               STATUS: ✓ SKELETON COMPLETE                           │
├─────────────────────────────────────────────────────────────────────┤
│  Person 3     PII Agent                                             │
│               nodes/pii_agent.py                                    │
│               Presidio integration · redaction · loop testing       │
├─────────────────────────────────────────────────────────────────────┤
│  Person 4     Bias Agent                                            │
│               nodes/bias_agent.py · data/ · models/                 │
│               Model training · Fairlearn metrics · demo dataset     │
├─────────────────────────────────────────────────────────────────────┤
│  Person 5     Policy Compliance Agent + Correction Node             │
│               nodes/policy_agent.py · nodes/correction.py           │
│               LLM prompts · Pydantic parsing · regulatory mapping   │
├─────────────────────────────────────────────────────────────────────┤
│  Person 6     Streamlit UI                                          │
│               ui/app.py · ui/components/                            │
│               Both tabs · real-time streaming · radar chart         │
├─────────────────────────────────────────────────────────────────────┤
│  Person 7     Integration + Demo                                    │
│               main.py · tests/test_integration.py · README.md       │
│               End-to-end testing · demo rehearsal · report section  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Timeline & Dependencies

### Dependency chain
```
Person 1+2 (skeleton) ──► Person 3, 4, 5, 6 (parallel agents + UI)
                                     │
                                     ▼
                            Person 7 (integration)
                                     │
                                     ▼
                              Final demo + report
```

Person 6 (UI) can begin building the Streamlit shell and static components **in parallel** with Persons 3, 4, 5. They wire in real data only after agents are complete.

Person 7 (Integration) begins **after** at least 2 of the 3 agents are ready.

### Week-by-Week Schedule

#### Week 1 — Setup + Foundation

| Person | Tasks |
|---|---|
| 1+2 | Complete `utils/audit_logger.py`. Write `.env.example`. Verify skeleton runs (`python main.py`). Freeze `state.py` — no changes after this unless all agree. |
| 3 | Install Presidio + spacy model. Read Presidio docs. Write 10 test sentences with PII. |
| 4 | Download German Credit dataset. Explore data. Identify protected attributes (age, gender). Train baseline LogisticRegression. Measure raw bias before Fairlearn. |
| 5 | Study EU AI Act Articles 8–51. Draft 8 regulatory criteria. Write policy prompt v1. Test prompt manually in Claude console. |
| 6 | Set up Streamlit project structure under `ui/`. Build static layout: two tabs, placeholder charts, empty audit table. No real data yet. |
| 7 | Set up dev environment. Read all existing code. Write integration test scaffold (`tests/test_integration.py`). |

**End of Week 1 checkpoint:** Everyone's environment works. Skeleton runs. Dataset is downloaded. UI shell renders.

---

#### Week 2 — Agent Implementation

| Person | Tasks |
|---|---|
| 1+2 | Available for unblocking. Review PRs from Persons 3, 4, 5 for state contract compliance. |
| 3 | Implement `pii_agent_node()` with real Presidio calls. Verify 10 test sentences. Write `tests/test_pii_agent.py`. |
| 4 | Implement `bias_agent_node()` with Fairlearn metrics. Compute disparate impact ratio on German Credit test set. Verify score < 0.8 (so demo violation fires). Save model to `models/loan_model.pkl`. |
| 5 | Implement `policy_agent_node()` with LLM call + Pydantic parsing. Implement `_attempt_correction()` in correction node. Write `tests/test_policy_agent.py` and `tests/test_correction.py`. |
| 6 | Wire streaming into the UI. Build `radar_chart.py` component using Plotly. Show live node-by-node progress. |
| 7 | Write end-to-end test for Scenario 1 (clean input) against the skeleton with stubs. Document the 3 demo scenarios in `main.py`. |

**End of Week 2 checkpoint:** Each agent passes its own unit tests in isolation. UI renders streaming output with stub data.

---

#### Week 3 — Integration

| Person | Tasks |
|---|---|
| 1+2 | Final review of all three agent implementations against state contract. Fix any routing edge cases uncovered. |
| 3 | Deliver `pii_agent.py` to Person 7. Support integration. |
| 4 | Deliver `bias_agent.py` + `models/loan_model.pkl` + `data/german_credit.csv` to Person 7. Support integration. |
| 5 | Deliver `policy_agent.py` + `correction.py` to Person 7. Support integration. |
| 6 | Wire real agent outputs into UI components. Implement audit log table. Test all three scenarios visually in the UI. |
| 7 | Run `test_integration.py` with all real agents. Debug end-to-end flow. Verify self-correction loop terminates for all 3 scenarios. |

**End of Week 3 checkpoint:** All 3 demo scenarios run end-to-end without errors. Self-correction loop cycles and terminates correctly. UI shows real results.

---

#### Week 4 — Polish + Demo + Report

| Person | Tasks |
|---|---|
| 1+2 | Write report section on architecture and state schema. |
| 3 | Write report section on PII agent implementation. |
| 4 | Write report section on bias metrics and dataset findings. |
| 5 | Write report section on regulatory mapping and LLM prompt design. |
| 6 | UI polish: status badges, colour coding, final CSS. Write report section on UI design decisions. |
| 7 | Run each demo scenario 20 times. Fix any flakiness. Take screenshots for report. Write validation section. **Rehearse the 5-minute demo presentation.** |

**End of Week 4:** Full demo works reliably. Report sections submitted to group lead.

---

## 6. API Contracts

These contracts define exactly what each node must return. Person 7 will validate all nodes against these. **Do not change these without agreement from Person 1+2 and Person 7.**

### All nodes must return a `dict` — never mutate state directly.

---

### Ingestion Node (Person 1+2) ✓
```python
return {
    "current_text": str,        # normalised input text
    "current_node": "ingestion",
    "audit_log": [dict],        # list with one entry
}
```

---

### PII Agent Node (Person 3)
```python
return {
    "pii_result": {
        "has_pii": bool,
        "entities_found": [
            {
                "entity_type": str,    # e.g. "PERSON", "EMAIL_ADDRESS"
                "value": str,          # the detected text
                "start": int,
                "end": int,
                "score": float,        # Presidio confidence 0.0–1.0
            }
        ],
        "redacted_text": str | None,   # None if no PII found
        "passed": bool,                # True = no PII found
    },
    "violations": [str],               # ["PII_DETECTED"] or []
    "current_node": "pii_agent",
    "audit_log": [dict],               # list with one entry
}
```

---

### Bias Agent Node (Person 4)
```python
return {
    "bias_result": {
        "demographic_parity_diff": float | None,
        "equalized_odds_diff": float | None,
        "disparate_impact_ratio": float | None,
        "privileged_group": str | None,
        "unprivileged_group": str | None,
        "passed": bool,
        "details": str,
    },
    "violations": [str],               # ["BIAS_DETECTED"] or []
    "current_node": "bias_agent",
    "audit_log": [dict],               # list with one entry
}
```

**Thresholds (do not change without team agreement):**
- `demographic_parity_diff` must be within ±0.1
- `equalized_odds_diff` must be within ±0.1
- `disparate_impact_ratio` must be ≥ 0.8 (US 4/5ths rule)

---

### Policy Agent Node (Person 5)
```python
return {
    "policy_result": {
        "violations": [
            {
                "id": str,             # e.g. "TRANSPARENCY"
                "description": str,    # one sentence
                "severity": str,       # "low" | "medium" | "high"
                "article_reference": str,
                "remediation": str,    # one sentence fix
            }
        ],
        "passed": bool,
        "severity": str,               # "none" | "low" | "medium" | "high"
        "summary": str,
    },
    "violations": [str],               # ["POLICY_TRANSPARENCY", ...] or []
    "current_node": "policy_agent",
    "audit_log": [dict],
}
```

---

### Correction Node (Person 5)
```python
return {
    "current_text": str,               # corrected text (must be different from input)
    "correction_count": int,           # state["correction_count"] + 1
    "active_violation": None,          # always reset to None after correction
    "pii_result": None,                # reset so agents re-run
    "bias_result": None,
    "policy_result": None,
    "current_node": "correction",
    "audit_log": [dict],
}
```

---

### Scorecard Node (Person 1+2) ✓
```python
return {
    "final_status": str,               # "PASS" | "CORRECTED" | "FAIL" | "ESCALATED"
    "rai_scores": {
        "strategic_alignment": int,    # 0–3
        "data_governance": int,
        "model_governance": int,
        "org_oversight": int,
        "continuous_monitoring": int,
    },
    "current_node": "scorecard",
    "audit_log": [dict],
}
```

---

### Audit log entry format (all nodes)
Every node must append exactly one entry to `audit_log` in this format:
```python
{
    "timestamp": str,          # datetime.now(timezone.utc).isoformat()
    "node": str,               # node name
    "action": str,             # e.g. "PII_SCAN", "BIAS_CHECK"
    "result": str,             # "pass" | "fail" | "ok" | "attempted"
    "detail": dict,            # node-specific findings
    "correction_count": int,   # state["correction_count"] at time of execution
}
```

---

## 7. Per-Person Work Breakdown

---

### Person 1+2 — Core Architecture

**Status: Skeleton complete. Remaining tasks below.**

#### Remaining: `utils/audit_logger.py`

Build helper functions to query the SQLite audit database. Person 6 (UI) needs these to display the audit log table.

```python
def get_audit_trail(thread_id: str) -> list[dict]:
    """Returns all audit log entries for a given run, in order."""

def get_runs_summary() -> list[dict]:
    """Returns a summary of all past runs: thread_id, status, timestamp, score."""

def get_node_timings(thread_id: str) -> dict:
    """Returns how long each node took to execute."""
```

#### Remaining: `.env.example`
```
ANTHROPIC_API_KEY=your_key_here
MAX_CORRECTIONS=3
LOG_LEVEL=INFO
```

#### Remaining: Freeze `state.py`
Once all agents begin implementation (start of Week 2), `state.py` becomes frozen. Any changes require agreement from all persons and an announcement in the group channel.

#### Remaining: Support integration (Week 3)
Be available to debug routing edge cases when Person 7 runs the full integration. The most likely issues are:
- `operator.add` accumulating duplicate violations across correction cycles
- Correction node not resetting agent results properly
- Checkpointer failing on large state objects

---

### Person 3 — PII Agent

**File: `nodes/pii_agent.py`**

#### Environment setup
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

#### Implementation guide

**Step 1: Initialise the engine at module level (not inside the function)**
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()      # expensive — create once
_anonymizer = AnonymizerEngine()  # create once
```

**Step 2: Implement the scan**
```python
ENTITIES_TO_DETECT = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "IBAN_CODE", "DATE_TIME",
    "LOCATION", "NRP",           # NRP = nationalities, religions, political groups
]

results = _analyzer.analyze(
    text=state["current_text"],
    entities=ENTITIES_TO_DETECT,
    language="en",
)
```

**Step 3: Build entities_found list**
```python
entities_found = [
    {
        "entity_type": r.entity_type,
        "value": state["current_text"][r.start:r.end],
        "start": r.start,
        "end": r.end,
        "score": round(r.score, 3),
    }
    for r in results
    if r.score >= 0.7    # confidence threshold — tune this
]
```

**Step 4: Redact (for the correction node to use)**
```python
if entities_found:
    anonymized = _anonymizer.anonymize(
        text=state["current_text"],
        analyzer_results=results,
    )
    redacted_text = anonymized.text
else:
    redacted_text = None
```

**Important:** Person 3 also needs to update `correction.py`'s `_attempt_correction()` to use `redacted_text` from `pii_result` when the violation is `PII_DETECTED`. The correction prompt template exists — it just needs to check if `pii_result["redacted_text"]` is available and return it directly instead of calling the LLM (redaction is deterministic, not generative).

#### Test cases for `tests/test_pii_agent.py`

Write tests for all of the following:

| Input | Expected outcome |
|---|---|
| `"Hello, my name is Sarah Johnson."` | PERSON detected |
| `"Email me at test@example.com"` | EMAIL_ADDRESS detected |
| `"Call 555-867-5309"` | PHONE_NUMBER detected |
| `"Card: 4111 1111 1111 1111"` | CREDIT_CARD detected |
| `"The loan application was denied."` | No entities — passed = True |
| Multiple PII in one text | All entities detected |
| Empty string | Raises ValueError (handled by ingestion) |

---

### Person 4 — Bias Agent

**Files: `nodes/bias_agent.py`, `data/`, `models/`**

#### Dataset: German Credit (UCI)
```python
import pandas as pd
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    sep=" ", header=None
)
# Save locally: df.to_csv("data/german_credit.csv", index=False)
```

Column 9 (0-indexed) is the sex/marital status attribute — this is the protected feature. Column 20 is the target (1=good credit, 2=bad credit).

**Alternatively:** Use sklearn's built-in fetch:
```python
from sklearn.datasets import fetch_openml
data = fetch_openml("credit-g", as_frame=True)
```

#### Model training

Train a **deliberately imbalanced** model to produce a visible bias signal:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Do NOT balance classes — we want bias to be visible
model = LogisticRegression(max_iter=1000, class_weight=None)
model.fit(X_train, y_train)

# Save
with open("models/loan_model.pkl", "wb") as f:
    pickle.dump({"model": model, "feature_names": feature_names}, f)
```

Target result: disparate impact ratio between 0.55–0.75 on the test set (clearly below the 0.8 threshold). This makes the demo compelling.

#### Fairlearn implementation
```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
)

# y_true and y_pred should be arrays for the test set
# sensitive_features should be the protected attribute column

dp_diff = demographic_parity_difference(
    y_true, y_pred,
    sensitive_features=sensitive_features,
)
eo_diff = equalized_odds_difference(
    y_true, y_pred,
    sensitive_features=sensitive_features,
)
```

#### Disparate impact ratio
Fairlearn doesn't compute this directly. Calculate manually:
```python
def disparate_impact_ratio(y_pred, sensitive_features):
    groups = {}
    for pred, group in zip(y_pred, sensitive_features):
        groups.setdefault(group, []).append(pred)
    rates = {g: sum(preds) / len(preds) for g, preds in groups.items()}
    min_rate = min(rates.values())
    max_rate = max(rates.values())
    return min_rate / max_rate if max_rate > 0 else 1.0
```

#### What bias_agent_node receives from state
In `model_output` mode, the graph passes `feature_vector` (a single prediction's features) and `protected_attributes` (list of which columns are sensitive). The bias agent needs to load the full test set to compute group-level metrics — it cannot compute fairness from a single prediction alone. Load the test set from `data/german_credit.csv` inside the node.

#### Test cases for `tests/test_bias_agent.py`

| Scenario | Expected outcome |
|---|---|
| Imbalanced model on German Credit | `disparate_impact_ratio` < 0.8, `passed = False` |
| Balanced model (trained with `class_weight="balanced"`) | `disparate_impact_ratio` ≥ 0.8, `passed = True` |
| `input_type = "text"` | Returns stub result, `passed = True` (text bias not implemented) |
| `protected_attributes = []` | Returns `passed = True`, logs warning |

---

### Person 5 — Policy Compliance Agent + Correction Node

**Files: `nodes/policy_agent.py`, `nodes/correction.py`**

#### LLM setup
```python
from langchain_ollama import ChatOllama
from pydantic import BaseModel

llm = ChatOllama(
    model="gemma",
    temperature=0,          # deterministic output critical for compliance
)
```

#### Pydantic output model
```python
class PolicyViolation(BaseModel):
    id: str
    description: str
    severity: str           # "low" | "medium" | "high"
    article_reference: str
    remediation: str

class PolicyCheckOutput(BaseModel):
    violations: list[PolicyViolation]
    summary: str
```

#### Structured output call
```python
structured_llm = llm.with_structured_output(PolicyCheckOutput)

try:
    result = structured_llm.invoke(
        build_policy_prompt(state["current_text"], POLICY_CRITERIA)
    )
except Exception as e:
    # Never crash the graph — return a passing result with error logged
    return _safe_fallback_result(str(e))
```

**Critical:** always wrap LLM calls in try/except. A failed API call must not crash the graph. Return `passed=True` with an error note in the summary — Person 7 can then rerun rather than having a broken demo.

#### The 8 regulatory criteria (already in `policy_agent.py`)
These are ready. Person 5 should verify the LLM correctly identifies violations for each one with a test input.

#### Correction node implementation

The prompt templates are already written in `correction.py`. Person 5 needs to implement `_attempt_correction()`:

```python
def _attempt_correction(state: ComplianceState, violation: str) -> str:
    from langchain_ollama import ChatOllama

    # Special case: PII correction uses redacted text directly (no LLM needed)
    if violation == "PII_DETECTED":
        pii = state.get("pii_result") or {}
        if pii.get("redacted_text"):
            print("[CORRECTION] Using Presidio redaction directly.")
            return pii["redacted_text"]

    llm = ChatOllama(model="gemma", temperature=0)
    prompt = _build_correction_prompt(state, violation)

    try:
        response = llm.invoke(prompt)
        corrected = response.content.strip()
        # Guard: LLM must return something different
        if corrected == state["current_text"]:
            print("[CORRECTION] LLM returned unchanged text.")
            return state["current_text"]
        return corrected
    except Exception as e:
        print(f"[CORRECTION] LLM call failed: {e}. Returning original.")
        return state["current_text"]
```

#### Test cases for `tests/test_policy_agent.py`

| Input | Expected violations |
|---|---|
| Clean, transparent AI output with human review option | No violations |
| AI-generated text with no disclosure it is AI-generated | `TRANSPARENCY` |
| Decision with no explanation given | `EXPLAINABILITY` |
| Decision with no mention of how to appeal/review | `HUMAN_OVERSIGHT` |
| Output referencing sensitive demographic data | `NON_DISCRIMINATION` |

#### Test cases for `tests/test_correction.py`

| Scenario | Expected behaviour |
|---|---|
| PII violation + `redacted_text` available | Returns `redacted_text` directly, no LLM call |
| Policy TRANSPARENCY violation | Returns text with AI disclosure added |
| 3 corrections fail to fix → `correction_count` = `max_corrections` | Escalates — graph routes to scorecard |
| LLM API error during correction | Returns original text unchanged, does not crash |

---

### Person 6 — Streamlit UI

**Files: `ui/app.py`, `ui/components/`**

#### App structure
```python
# ui/app.py
import streamlit as st
import sys
sys.path.append("..")   # so we can import from parent directory

from graph import app
from state import create_initial_state

st.set_page_config(
    page_title="RAI Compliance Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

tab1, tab2 = st.tabs(["Text Auditor", "Model Auditor"])

with tab1:
    render_text_auditor()

with tab2:
    render_model_auditor()
```

#### Tab 1 — Text Auditor

Layout:
```
┌────────────────────────┬────────────────────────┐
│  Input text area       │  Status badge          │
│  [Audit Text] button   │  Radar chart           │
│                        │  Pillar scores table   │
└────────────────────────┴────────────────────────┘
│  Live execution log (node-by-node streaming)    │
│  Audit trail expander                           │
└─────────────────────────────────────────────────┘
```

Real-time streaming with LangGraph:
```python
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
status_placeholder = st.empty()
log_placeholder = st.empty()

for chunk in app.stream(initial_state, config=config):
    node_name = list(chunk.keys())[0]
    status_placeholder.info(f"Running: {node_name}...")
    # Append to running log display

# Once complete, show final result
final_result = chunk[node_name]
render_scorecard(final_result)
```

#### Tab 2 — Model Auditor

Layout:
```
┌─────────────────────────────────────────────────┐
│  Upload CSV  │  Protected attribute dropdown    │
│  [Run Audit] button                             │
└─────────────────────────────────────────────────┘
│  Fairness metrics table                         │
│  Radar chart                                    │
│  Before/After comparison (if corrected)         │
└─────────────────────────────────────────────────┘
```

#### `ui/components/radar_chart.py`
```python
import plotly.graph_objects as go

def render_radar_chart(rai_scores: dict) -> go.Figure:
    categories = [
        "Strategic\nAlignment",
        "Data\nGovernance",
        "Model\nGovernance",
        "Org\nOversight",
        "Continuous\nMonitoring",
    ]
    values = [
        rai_scores["strategic_alignment"],
        rai_scores["data_governance"],
        rai_scores["model_governance"],
        rai_scores["org_oversight"],
        rai_scores["continuous_monitoring"],
    ]
    values += values[:1]   # close the polygon

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories + categories[:1],
        fill="toself",
        line_color="#2E75B6",
        fillcolor="rgba(46, 117, 182, 0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
        showlegend=False,
        height=350,
    )
    return fig
```

#### Status badge colours
- `PASS` → green (`#28a745`)
- `CORRECTED` → amber (`#ffc107`)
- `FAIL` → red (`#dc3545`)
- `ESCALATED` → dark red (`#8B0000`)

#### Running the UI
```bash
cd rai-compliance-agent
streamlit run ui/app.py
```

---

### Person 7 — Integration + Demo

**Files: `main.py`, `tests/test_integration.py`, `README.md`**

#### Integration checklist

Before declaring integration complete, verify every item:

- [ ] Scenario 1 (clean text) runs end-to-end and returns `PASS`
- [ ] Scenario 2 (PII text) fires PII agent, correction redacts, returns `CORRECTED`
- [ ] Scenario 3 (biased model output) fires bias agent, correction cycles, returns `CORRECTED` or `FAIL`
- [ ] Self-correction loop terminates (never infinite cycles)
- [ ] `rai_audit.db` is created and contains entries after each run
- [ ] All 5 pillar scores are populated in the final state
- [ ] Streamlit UI shows correct output for all 3 scenarios
- [ ] No unhandled exceptions on any scenario

#### `tests/test_integration.py` structure
```python
def test_scenario_1_clean_input():
    state = create_initial_state(input_type="text", raw_input=CLEAN_TEXT)
    result = app.invoke(state, config=config_with_new_thread())
    assert result["final_status"] == "PASS"
    assert result["violations"] == []
    assert all(s >= 2 for s in result["rai_scores"].values())

def test_scenario_2_pii_detected():
    state = create_initial_state(input_type="text", raw_input=PII_TEXT)
    result = app.invoke(state, config=config_with_new_thread())
    assert result["final_status"] in ("CORRECTED", "ESCALATED")
    assert "PII_DETECTED" in result["violations"]
    assert result["correction_count"] >= 1

def test_scenario_3_biased_model():
    state = create_initial_state(input_type="model_output", raw_input=DENIAL_TEXT,
                                  feature_vector=SAMPLE_FEATURES,
                                  protected_attributes=["gender"],
                                  prediction=0.82, predicted_label=1)
    result = app.invoke(state, config=config_with_new_thread())
    assert result["final_status"] in ("CORRECTED", "FAIL", "ESCALATED")
    assert "BIAS_DETECTED" in result["violations"]

def test_loop_terminates():
    """Ensure max_corrections prevents infinite cycles."""
    state = create_initial_state(..., max_corrections=2)
    result = app.invoke(state, config=config_with_new_thread())
    assert result["correction_count"] <= 2
    assert result["final_status"] in ("CORRECTED", "ESCALATED")
```

#### README.md must include
1. One-command setup: `pip install -r requirements.txt && python -m spacy download en_core_web_lg`
2. `.env` setup instructions
3. How to run: `python main.py` and `streamlit run ui/app.py`
4. Description of each demo scenario
5. How to query the audit database

---

## 8. Integration Milestones

| Milestone | Owner | Criteria |
|---|---|---|
| M1: Skeleton confirmed | Person 1+2 | `python main.py` runs 3 scenarios with stubs, no errors |
| M2: PII agent unit tested | Person 3 | 7/7 test cases pass in isolation |
| M3: Bias agent unit tested | Person 4 | Disparate impact < 0.8 confirmed on German Credit |
| M4: Policy agent unit tested | Person 5 | 5/5 test cases pass in isolation |
| M5: Correction loop tested | Person 5 | Loop terminates, test_correction.py passes |
| M6: UI shell renders | Person 6 | Both tabs visible, streaming works with mock data |
| M7: Full integration | Person 7 | All 3 integration tests pass |
| M8: Demo rehearsed | Person 7 | 5-minute demo runs clean 3 times in a row |

---

## 9. Demo Scenarios

These are the three scenarios used for both testing and the capstone presentation.

---

### Scenario 1 — Clean Input (Expected: PASS)

**Purpose:** Show the graph running normally. Establishes baseline.

```
Input type: text
Text: "Based on your application, we have reviewed your eligibility
       for the loan product. Our AI system assessed your credit
       history, income stability, and debt-to-income ratio. This
       decision was made by an automated system and can be reviewed
       by a human advisor upon request. You may also contact us to
       request an explanation of the factors considered."
```

Expected output:
- `final_status = "PASS"`
- All pillar scores ≥ 2
- Audit log shows 4 nodes executed (ingestion → PII → bias → policy → scorecard)
- No correction cycles

---

### Scenario 2 — PII Leakage (Expected: CORRECTED)

**Purpose:** Show PII detection and auto-redaction. Demonstrates Data Governance pillar.

```
Input type: text
Text: "Dear John Smith, your application reference A-4821 has been
       reviewed. Please contact us at john.smith@email.com or call
       +91 98765 43210 to discuss the outcome of your assessment."
```

Expected output:
- `final_status = "CORRECTED"`
- `violations` contains `"PII_DETECTED"`
- Correction node runs once
- Final text has `[PERSON]`, `[EMAIL_ADDRESS]`, `[PHONE_NUMBER]` in place of real data
- Data Governance score: 1→3 after correction

---

### Scenario 3 — Biased Model Output (Expected: CORRECTED or FAIL)

**Purpose:** Show bias detection on model predictions. The centrepiece of the demo. Demonstrates Model Governance pillar and the cyclic self-correction loop.

```
Input type: model_output
Text: "Loan application DENIED. Risk score: 0.82. Application ID: APP-7291."
feature_vector: {"age": 34, "income": 42000, "employment_years": 3,
                 "existing_loans": 1, "gender": "F"}
protected_attributes: ["gender", "age"]
prediction: 0.82
predicted_label: 1
```

Expected output:
- `final_status = "CORRECTED"` (if correction succeeds) or `"FAIL"`
- `bias_result["disparate_impact_ratio"]` < 0.8 on first pass
- Correction loop cycles 1–2 times
- Final `bias_result["disparate_impact_ratio"]` ≥ 0.8
- Model Governance score: 1→3 after correction

---

## 10. Testing Requirements

### Unit test coverage per person

| Person | File | Minimum tests | Must cover |
|---|---|---|---|
| 3 | `test_pii_agent.py` | 7 | All entity types + no-PII case |
| 4 | `test_bias_agent.py` | 4 | Biased model + fair model + edge cases |
| 5 | `test_policy_agent.py` | 5 | One per regulatory criterion |
| 5 | `test_correction.py` | 4 | PII redact + LLM correct + fail + terminate |
| 7 | `test_integration.py` | 4 | All 3 scenarios + loop termination |

### Running tests
```bash
# Individual
python -m pytest tests/test_pii_agent.py -v

# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=nodes --cov-report=term-missing
```

### Test data — do not use real personal data
All test inputs must use fictional names, fake phone numbers, and synthetic data. Do not use real email addresses or financial details in test files.

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM API call fails during demo | Medium | High | Wrap all LLM calls in try/except with graceful fallback. Test offline mode. |
| Self-correction loop doesn't terminate | Low | High | `max_corrections` hard cap. Person 5 must test this explicitly. |
| Presidio misses PII with low confidence | Medium | Medium | Lower confidence threshold to 0.6. Tune in test cases. |
| Bias agent doesn't fire (model too fair) | Low | High | Person 4 must confirm disparate impact < 0.8 before Week 3. Use `class_weight=None`. |
| `state.py` changed mid-project | Low | Very High | Freeze after Week 1. Any change requires all-team agreement. |
| Integration takes longer than Week 3 | Medium | Medium | Person 7 starts with whichever agents are ready. Don't block on all three. |
| Streamlit streaming breaks on certain browsers | Low | Low | Test in Chrome. Have a fallback non-streaming version ready. |
| German Credit dataset URL breaks | Low | Medium | Person 4 commits CSV to `data/` folder. Never rely on live URL in demo. |

---

## 12. Definition of Done

The PoC is complete when:

1. **All three demo scenarios run without errors** on a clean machine (fresh `pip install`)
2. **Self-correction loop works** — at least Scenario 2 or 3 shows a cycle and returns `CORRECTED`
3. **Audit trail persists** — `rai_audit.db` exists after a run and contains timestamped entries
4. **Streamlit UI renders** both tabs with real data from the graph
5. **Radar chart shows** per-pillar scores for every completed run
6. **All integration tests pass** (`python -m pytest tests/test_integration.py`)
7. **Demo runs clean 3 times in a row** without any error, exception, or hang
8. **README.md** allows a new person to set up and run the demo from scratch in under 10 minutes

---

*This document is the single source of truth for the project. All architectural decisions should be reflected here.*
