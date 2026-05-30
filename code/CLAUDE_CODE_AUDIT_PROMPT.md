# Claude Code: RAI Compliance Agent — Full System Audit & Fix Prompt

Paste this entire prompt into Claude Code (`claude` CLI) from inside the `rai-compliance-agent/` directory.

---

## CONTEXT

You are working on a LangGraph-based Responsible AI (RAI) compliance agent built as an IIMA capstone project. The system audits company AI policy documents and model outputs against the EU AI Act, DPDPA 2023, and RBI Model Risk Management Guidelines.

The app runs on Streamlit (`ui/app.py`). The LangGraph pipeline (`graph.py`) chains these nodes in order:

```
ingestion → pii_agent → bias_agent → explainability_agent → policy_agent → scorecard
```

The only LLM used is **Ollama Gemma 4** (`gemma4:latest`) — called ONLY inside `policy_agent.py` and `suggestion_agent.py`. All other nodes are rule-based libraries (Presidio, Fairlearn, SHAP, scikit-learn).

---

## TASK 1 — SYSTEM AUDIT: Verify every agent end-to-end

Read and audit ALL of the following files. For each, report: what it does, what LLM (if any) it uses, what passes/fails and why, and any bugs or gaps you find.

Files to audit:
- `graph.py` — LangGraph pipeline definition
- `state.py` — ComplianceState TypedDict
- `nodes/ingestion.py`
- `nodes/pii_agent.py`
- `nodes/bias_agent.py`
- `nodes/explainability_agent.py`
- `nodes/policy_agent.py`
- `nodes/scorecard.py`
- `nodes/suggestion_agent.py`
- `ui/app.py`
- `requirements.txt`

For each node, answer:
1. What does this node do?
2. Does it use an LLM? Which one? Show the exact model string.
3. What are the pass/fail conditions?
4. Are there any false positives, false negatives, or silent failures?
5. Is the audit_log and step_trace output complete and accurate?

---

## TASK 2 — FIX: PII Agent false positives

**Problem**: `nodes/pii_agent.py` uses Microsoft Presidio and is generating false positives on policy documents. Specifically:
- `━` (box-drawing horizontal rule characters) are being flagged as `NRP` (nationalities/religions/political groups)
- Formatting artifacts like `━\n1`, `━\n2` are flagged as `PERSON`
- Terms like `"Tier 1"`, `"Model Drift Detection"` are flagged as `LOCATION`
- Acronyms like `"AIEC"` are flagged as `PERSON`
- Nationality adjectives like `"Indian"` (in "Indian residents") are flagged as `NRP`

**Fix required**:
1. Add a `_is_false_positive(value: str) -> bool` function that filters entities where:
   - The entity string has fewer than 3 real alphanumeric characters
   - The entity string is composed entirely of non-alphanumeric characters
   - The lowercased stripped value is in a known governance-document false-positive denylist: `{"tier 1", "tier 2", "tier 3", "model drift detection", "aiec", "indian"}`
2. Raise `CONFIDENCE_THRESHOLD` from `0.6` to `0.75`
3. Apply the filter in the list comprehension that builds `entities_found`
4. Do NOT change the Presidio entity types being scanned — keep scanning for PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IBAN_CODE, LOCATION, NRP

---

## TASK 3 — FIX: Policy Agent — richer per-pillar LLM explanations

**Problem**: `nodes/policy_agent.py` uses the `PillarDocumentScore` Pydantic model which only has `id`, `compliance_level`, and `gap_description`. This means the UI cannot show:
- What text in the policy was actually evaluated (evidence)
- Why a pillar PASSED (not just why it failed)

**Fix required**:

### 3a. Update the Pydantic model

```python
class PillarDocumentScore(BaseModel):
    id: str
    compliance_level: int          # 0=absent, 1=minimal, 2=partial, 3=adequate
    gap_description: str           # what is MISSING (empty string if score == 3)
    evidence_from_policy: str      # direct quote or section reference evaluated
    pass_reason: str               # one sentence on why it passed (empty if score <= 1)
```

### 3b. Rewrite `_build_policy_document_prompt()`

The new prompt must:
- Instruct Gemma to provide `evidence_from_policy` — a short direct quote (max 60 words) from the policy text being evaluated, or `"Not found in policy."` if absent
- Instruct Gemma to provide `pass_reason` — one sentence explaining what the policy does well for this pillar (empty string if score 0 or 1)
- Make scoring rules explicit: score 3 ONLY if ALL three check questions are answered with specifics (named owners, defined thresholds, concrete procedures)
- Tell Gemma: "Return ONLY valid JSON. No markdown fences. No text outside the JSON object."

### 3c. Update `_check_policy_document_compliance()`

When building `pillar_gaps` dict, include the new fields:
```python
pillar_gaps[score.id] = {
    "level": level,
    "gap": score.gap_description,
    "pass_reason": getattr(score, "pass_reason", ""),
    "evidence": getattr(score, "evidence_from_policy", ""),
    "pillar_key": criterion["pillar_key"],
    "reference": criterion["reference"],
    "description": criterion["description"],
}
```

---

## TASK 4 — FIX: Suggestion Agent — actionable prompts

**Problem**: `nodes/suggestion_agent.py` prompt produces generic suggestions not tied to the specific policy text or regulatory framework.

**Fix required**: Rewrite the `prompt` variable inside `run_suggestion_agent()` to:

1. Pass each failing pillar's `evidence` field (from pillar_gaps) as `"existing_policy_text"` in the audit_findings_json
2. Instruct Gemma to EXTEND existing policy text rather than replace it when partial text exists
3. Require exactly 3 draft policy clauses per pillar — numbered, written in third person, with specifics (thresholds, timelines, named roles)
4. Explicitly forbid generic filler sentences like "adequate measures shall be taken"
5. Instruction: "If the policy already has partial text (see 'existing_policy_text'), EXTEND or STRENGTHEN it rather than replacing it"

---

## TASK 5 — FIX: UI — per-pillar breakdown with pass/fail reasoning

**Problem**: `ui/app.py` shows only a bar chart of pillar scores. Users cannot see:
- Which specific policy text was evaluated per pillar
- Why a pillar passed (what the policy got right)
- Why a pillar failed (exact gap + regulatory citation)
- A per-pillar "Suggest Fix" button

**Fix required**: Add a `_render_pillar_breakdown(scores, pillar_gaps, final_state)` function and replace the existing `_pillar_score_html` call in `render_text_auditor()` when `is_policy_doc and pillar_gaps`.

The function must render one `st.expander` per pillar (auto-expanded if failing) containing:

**Left column — "What the policy says":**
- Show `gap_data["evidence"]` in a blue left-border box (italic)
- If evidence is empty, show a yellow warning box: "No relevant text found in policy."

**Right column — "Why it passed / Why it failed":**
- If `score >= 2` and `pass_reason` is non-empty: show in a green left-border box with ✅
- If `score < 3` and `gap` is non-empty: show in a red left-border box with ❌

**Below both columns:**
- Regulatory citation (`reference` field) in small grey text with 📎
- A `st.button(f"💡 Suggest fix for this pillar", key=f"suggest_{criterion_id}")` — visible only when `score < 3`

**Also add `_run_single_pillar_suggestion(final_state, criterion_id, pillar_key)`**: slices state to only the target pillar's score, calls `run_suggestion_agent`, and merges the result into `st.session_state["suggestions_result"]`.

---

## TASK 6 — FIX: Bias agent — clarify why fairness is not run on policy documents

**Problem**: The bias agent step in the UI just says "Text mode: bias check not applicable (model_output mode only)" with no explanation. Users (and evaluators) see this and think the fairness check is broken or missing.

**Fix required**: In `ui/app.py`, inside the `bias_agent` step expander rendering (around the `elif step["step"] == "bias_agent":` block), replace the current `st.info(...)` with a rich explanation:

```python
st.markdown("#### Why no bias check here?")
st.markdown(
    """
    **Fairness analysis requires model prediction data** — specifically:
    - A set of predictions (0/1) from a trained model
    - A protected attribute column (e.g. `sex`, `race`, `age_group`)
    
    The bias agent uses **Fairlearn** (a statistical library) to compute:
    - **Demographic Parity Difference** — are positive outcomes equally likely across groups?
    - **Equalized Odds Difference** — are error rates equal across groups?
    - **Disparate Impact Ratio** — does the less-favoured group receive ≥ 80% of the favourable outcome rate?
    
    For **policy documents**, there are no predictions to measure. Instead, **fairness is audited by the 
    Policy Agent (Gemma 4)** via the `FAIRNESS_BIAS` pillar — which checks whether the policy itself 
    mandates bias testing, training data assessment, and proxy variable reviews before deployment.
    
    To run a statistical bias check: use the **Model Auditor** tab and upload a CSV with model predictions.
    """
)
```

---

## TASK 7 — FIX: PII Agent — LLM-generated explanation in the UI

**Problem**: The PII agent step shows a table of entities found but no plain-language explanation of what was scanned and why it matters.

**Fix required**: In `ui/app.py`, in the `pii_agent` step expander block, add a generated explanation section ABOVE the entity table:

```python
st.markdown("#### What is PII Detection?")
st.markdown(
    """
    **PII (Personally Identifiable Information)** is any data that can identify a specific individual.
    Exposing PII in AI training data, model outputs, or policy documents creates legal risk under:
    - **GDPR Article 5** — lawfulness, fairness, purpose limitation
    - **DPDPA 2023 Section 4** — processing must have lawful purpose and consent
    
    **What this agent scans for** (using Microsoft Presidio):
    | Entity Type | Example | Risk |
    |---|---|---|
    | `PERSON` | "Rahul Sharma" | Identity disclosure |
    | `EMAIL_ADDRESS` | "admin@company.com" | Direct contact exposure |
    | `PHONE_NUMBER` | "+91-98765-43210" | Contact tracing |
    | `CREDIT_CARD` | "4111 1111 1111 1111" | Financial fraud |
    | `IBAN_CODE` | "GB29 NWBK..." | Financial fraud |
    | `LOCATION` | "Mumbai, India" | Location profiling |
    | `NRP` | "Muslim", "BJP member" | Sensitive attribute exposure |
    
    **Confidence threshold**: 0.75 (entities below this score are suppressed to reduce false positives)
    
    **How to fix PII violations**: Remove or redact identified entities before using the document 
    in AI training pipelines. The redacted version is shown below.
    """
)
```

---

## TASK 8 — VERIFY: Check that `torchvision` is NOT a dependency

Search the entire codebase (`nodes/`, `ui/`, `rag/`, `data/`, `tests/`, `requirements.txt`) for any import or reference to `torchvision` or direct `torch` usage in YOUR code (not inside `venv/`).

**Expected result**: No references. `torch` appears only inside the `venv/` folder as a transitive dependency of `sentence-transformers` (used for ChromaDB embeddings in the RAG retriever). It is NOT used directly by any of your agents.

**If found**: Remove any unnecessary torch/torchvision imports and add `# torch is NOT a direct dependency — sentence-transformers pulls it in transitively` to `requirements.txt`.

---

## TASK 9 — VERIFY: End-to-end smoke test

Run this end-to-end test from the `rai-compliance-agent/` directory:

```bash
cd rai-compliance-agent
source venv/bin/activate  # or: python -m venv venv && pip install -r requirements.txt

python -c "
import sys, json
sys.path.insert(0, '.')
from graph import app
from state import create_initial_state

# Test 1: Policy PASS
state = create_initial_state(
    input_type='policy_document',
    raw_input='''
GOVERNANCE: An AI Ethics Committee (AIEC), chaired by the CRO, maintains a model registry.
Human override of AI decisions is mandatory within 48 hours.
FAIRNESS: All models undergo pre-deployment bias testing using disparate impact ratio >= 0.8.
Proxy variable reviews are mandatory.
TRANSPARENCY: Affected individuals are informed of AI decisions in plain language.
Model cards are maintained for all production systems.
ROBUSTNESS: Performance thresholds are monitored monthly with automated drift alerts.
An incident response procedure escalates failures within 24 hours.
PRIVACY: Personal data used for AI training requires documented consent per DPDPA 2023 Section 6.
Third-party AI vendors are subject to data processing agreements.
'''
)
result = app.invoke(state, config={'configurable': {'thread_id': 'smoke-test-1'}})
print('Test 1 PASS policy:', result['final_status'])
print('Violations:', result['violations'])
print('RAI Scores:', json.dumps(result['rai_scores'], indent=2))
assert result['final_status'] == 'PASS', f'Expected PASS, got {result[\"final_status\"]}'

# Test 2: Policy FAIL — missing fairness and privacy
state2 = create_initial_state(
    input_type='policy_document',
    raw_input='We use AI to improve services. A committee exists to review decisions.'
)
result2 = app.invoke(state2, config={'configurable': {'thread_id': 'smoke-test-2'}})
print('Test 2 minimal policy:', result2['final_status'])
assert result2['final_status'] == 'FAIL', f'Expected FAIL, got {result2[\"final_status\"]}'

# Test 3: PII false positive check — box-drawing chars should NOT trigger PII
state3 = create_initial_state(
    input_type='policy_document',
    raw_input='━━━━━━━━━━━━━━━━━━━━━\n1. Governance\n━━━━━━━━━━━━━━━━━━━━━\nTier 1 risk systems require review.'
)
result3 = app.invoke(state3, config={'configurable': {'thread_id': 'smoke-test-3'}})
pii = result3.get('pii_result', {})
entities = pii.get('entities_found', [])
fp_chars = [e for e in entities if all(not c.isalnum() for c in e['value'].replace(' ', ''))]
print('Test 3 false positives (should be 0):', len(fp_chars), fp_chars)
assert len(fp_chars) == 0, f'False positive PII entities found: {fp_chars}'

print('\\n✅ All smoke tests passed.')
"
```

If any test fails, diagnose the root cause and fix it.

---

## TASK 10 — VERIFY: Audit log completeness

After running the smoke test, check that the `step_trace` for a policy document audit contains ALL of the following keys in each entry:
- `step`, `label`, `status`, `summary`
- For `policy_agent`: `prompt` (the full prompt sent to Gemma), `response` (the structured JSON back)
- For `pii_agent`: `response.entities_found`, `response.confidence_threshold`, `response.entities_scanned_for`
- For `bias_agent`: `response.mode` and explanation of why bias check is mode-dependent

Print the full step_trace from smoke test 1 and confirm all fields are present.

---

## SUMMARY OF EXPECTED OUTCOMES AFTER ALL FIXES

| Node | LLM Used | Pass Condition | Fail Condition |
|---|---|---|---|
| Ingestion | None | Input non-empty, type valid | Empty input or unknown type |
| PII Agent | None (Presidio) | No real PII found (confidence ≥ 0.75, non-artifact) | EMAIL, PHONE, CREDIT_CARD, PERSON found |
| Bias Agent | None (Fairlearn) | Policy/text mode: always PASS (N/A) | Model mode: DI < 0.8, DP diff > 0.1, EO diff > 0.1 |
| Explainability | None (SHAP) | Always PASS — informational only | Never fails (no violation) |
| Policy Agent | **Gemma 4 (Ollama)** | All 5 pillars score ≥ 2 | Any pillar scores 0 or 1 |
| Scorecard | None | Overall ≥ 10/15, no pillars < 2 | Any pillar < 2 OR total < 10 |
| Suggestion Agent | **Gemma 4 (Ollama)** | Returns 3 policy clauses per failing pillar | Falls back to rule-based if LLM fails |

---

## IMPORTANT NOTES FOR SUBMITTER

1. **torchvision is NOT used anywhere in this project.** It appears as a transitive dependency of `sentence-transformers` (which powers the ChromaDB RAG embeddings), but none of the agents import or call it directly.

2. **The only LLM is Gemma 4 via Ollama** — running locally. No OpenAI, no Anthropic API. Make sure `ollama pull gemma4:latest` has been run before starting the app.

3. **Bias check design decision**: Statistical fairness (Fairlearn) requires model prediction data. For policy document mode, fairness compliance is evaluated by the LLM via the FAIRNESS_BIAS pillar. This is intentional — you cannot compute disparate impact ratio on a text document.

4. **PII false positive fix**: The `━` character is a Unicode box-drawing character (U+2501) used as a horizontal separator in formatted text. Presidio's NRP recogniser over-fires on it at low confidence. The fix raises the threshold and adds a structural filter.
