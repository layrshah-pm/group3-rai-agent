# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Activate virtualenv first
source venv/bin/activate

# Install dependencies (Python 3.11+ required)
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Train demo models (first run only)
python data/train_model.py          # creates models/loan_model.pkl + loan_model_fair.pkl
python data/train_compas_model.py   # creates models/compas_model.pkl

# Build RAG vector store (run once, or after editing rag/documents/)
python rag/indexer.py
python rag/indexer.py --reset       # rebuild from scratch
python rag/indexer.py --verify      # inspect store + sample retrievals

# Run demo scenarios (4 total)
python main.py

# Streamlit UI
streamlit run ui/app.py

# Validation harness (runs each scenario N times, reports pass rate + timing)
python validate_demo.py --runs 5
python validate_demo.py --runs 3 --scenario 4

# Tests — fast, no Ollama required (~7s)
python -m pytest tests/test_pii_agent.py tests/test_bias_agent.py tests/test_correction.py tests/test_explainability_agent.py tests/test_rag_retriever.py -v

# Tests — require Ollama running with gemma4:latest
python -m pytest tests/test_policy_agent.py -v    # ~4 min
python -m pytest tests/test_integration.py -v     # ~15 min

# Single test file
python -m pytest tests/test_bias_agent.py -v

# With coverage
python -m pytest tests/ --cov=nodes --cov-report=term-missing
```

Ollama must be running (`ollama serve`) with `gemma4:latest` pulled for policy agent and correction node tests.

---

## Architecture

### LangGraph DAG

`graph.py` wires the compliance pipeline. State flows through nodes via `ComplianceState` (`state.py`). LangGraph persists a snapshot at every node transition into `rai_audit.db` — tamper-proof audit trail for EU AI Act Article 50.

**Current topology:**
```
Ingestion → PII Agent → Bias Agent → Explainability Agent → Policy Agent → Scorecard
                ↑            ↑                                    ↑
                └────────────┴────────────────────────────────────┘
                                      Correction
                                (cycles up to max_corrections=3)
```

Routing functions in `graph.py`: `_should_correct` (after policy_agent) and `_after_correction` (after correction). Correction always routes back to `pii_agent` so the full audit chain re-runs on corrected text.

**`_should_correct` routes on current agent results, not accumulated `violations`.** The violations list is a union across all passes and retains old entries; using it for routing would prevent the loop from ever exiting on a successful correction.

### State schema (`state.py`)

`ComplianceState` is the single source of truth. Key design constraints:
- `violations: Annotated[list[str], reduce_violations]` — agents append (deduped set-union), never overwrite; `audit_log` uses `operator.add`. `reduce_violations` supports `CLEAR_VIOLATIONS` sentinel but correction routing does NOT rely on it — routing checks agent results directly.
- `current_text` is the working copy mutated by correction; `raw_input` is immutable
- `retrieved_clauses: Optional[list[dict]]` — populated by policy agent after RAG retrieval; direct assignment (not `operator.add`), overwrites each run
- `_model_path` key inside `feature_vector` — bias agent and explainability agent both read this to switch model files (e.g. for COMPAS scenario)

### Nodes (`nodes/`)

| Node | Implementation | Key behaviour |
|---|---|---|
| `ingestion.py` | Validates + normalises input | Sets `input_type` routing |
| `pii_agent.py` | Microsoft Presidio | Redacts in-place; correction uses `redacted_text` directly |
| `bias_agent.py` | Fairlearn | `model_output` mode only; text mode always passes |
| `explainability_agent.py` | SHAP TreeExplainer | Informational only — never triggers violation or correction |
| `policy_agent.py` | Ollama/Gemma4 + Pydantic + RAG | 11 criteria; retrieves regulatory clauses from ChromaDB before LLM call |
| `correction.py` | Ollama/Gemma4 | PII uses Presidio redaction; others use targeted LLM prompts |
| `scorecard.py` | Deterministic | Maps violations to 5 RAI pillars, scores 0-3 each |

### Policy Agent — RAG layer

`policy_agent.py` calls `rag/retriever.py` before the LLM. It embeds `current_text` and retrieves the top 5 semantically relevant regulatory clauses from ChromaDB, then injects them into the prompt. Falls back to hardcoded criteria only if the store is missing (no crash).

RAG store lives at `rag/policy_store/` (ChromaDB PersistentClient). Documents in `rag/documents/` — one `.txt` per regulation, chunked with `[CHUNK]...[/CHUNK]` delimiters. Currently: EU AI Act (9 chunks), NIST AI RMF (7), GDPR (4), DPDPA 2023 (4), RBI Guidelines (3).

`VALID_IDS` filter in `_check_policy_compliance` blocks the LLM from inventing violation IDs from retrieved context. `_safe_fallback` is the LLM error path — never remove it; graph must not crash.

### Policy criteria (11 total)

8 core: `TRANSPARENCY`, `EXPLAINABILITY`, `HUMAN_OVERSIGHT`, `DATA_QUALITY`, `NON_DISCRIMINATION`, `PRIVACY`, `ACCOUNTABILITY`, `ROBUSTNESS`

3 India-specific: `DPDPA_CONSENT`, `DPDPA_DATA_MINIMISATION`, `RBI_MODEL_VALIDATION`

Guard rules in `build_policy_prompt()` prevent over-firing on the India criteria for non-Indian-context text.

### Model artefact format (`models/*.pkl`)

All pickle files must follow exactly:
```python
{
    "model":          sklearn_model,
    "test_X":         pd.DataFrame,   # column names must match training features
    "test_y":         pd.Series,
    "test_sensitive": pd.Series,      # string labels e.g. "male"/"female"
}
```
Both `bias_agent.py` and `explainability_agent.py` load this format.

### Fairness thresholds (`bias_agent.py`)

- Disparate Impact ratio ≥ 0.8 (US 4/5ths rule)
- Demographic Parity diff ≤ ±0.1
- Equalized Odds diff ≤ ±0.1

`loan_model.pkl` scores DI ≈ 0.316 (gender). `compas_model.pkl` scores DI ≈ 0.55–0.65 (race: African-American vs Caucasian). Both intentionally fail bias checks.

### Demo scenarios (`main.py`)

| Scenario | Input | Expected |
|---|---|---|
| 1 | Clean compliant text | PASS |
| 2 | PII leak (name, email, phone) | CORRECTED |
| 3 | Biased loan model (gender, DI ≈ 0.316) | ESCALATED |
| 4 | COMPAS racial bias (DI ≈ 0.55–0.65) | ESCALATED |

### UI (`ui/app.py`)

Two tabs: **Text Auditor** (`render_text_auditor`) and **Model Auditor** (`render_model_auditor`). Both show: status badge, RAI radar chart, violations with citations, bias metrics, SHAP expander (model tab), and "Regulatory Sources" expander listing RAG-retrieved clauses colour-coded by relevance (green ≥60%, amber ≥40%, grey <40%).

Model Auditor detects which model to load based on filename: `"fair"` → `loan_model_fair.pkl`, `"compas"` → `compas_model.pkl`, else default loan model.

### Audit database

`rai_audit.db` is SQLite written by `SqliteSaver` (msgpack). Query helpers in `utils/audit_logger.py`: `get_audit_trail(thread_id)`, `get_runs_summary()`, `get_node_timings(thread_id)`.

### Environment

Copy `.env.example` → `.env`. Default Ollama endpoint is `http://localhost:11434`. No changes needed for local Ollama.

---

## Design Constraints

Do not change without understanding downstream effects:

- **`reduce_violations` on `violations`** — set-union dedup across cycles; `operator.add` on `audit_log` — do not change either
- **`_should_correct` checks agent results** — not `state.violations`; violations accumulate and never clear between cycles so they cannot be used for routing
- **correction_node resets agent results** — returns `pii_result=None`, `bias_result=None`, `policy_result=None`, `explainability_result=None` so agents re-run on corrected text; do not remove
- **`correction → pii_agent` routing** — intentional; full chain re-runs on corrected text
- **`_safe_fallback` in policy_agent** — LLM error path must not crash the graph
- **`_model_path` in `feature_vector`** — internal key for switching model files; strip before passing to model features
- **RAG fallback `[]`** — `retrieve_relevant_clauses` must return empty list (not raise) when store missing
- **`VALID_IDS` filter** — blocks hallucinated violation IDs from reaching state
- **`retrieved_clauses` direct assignment** — not `operator.add`; only policy agent writes it; overwrites per run
