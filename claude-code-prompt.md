# Claude Code Prompt — RAI Policy Document Auditor

## Context

This is a LangGraph + Streamlit app called the **RAI Compliance Agent** (located in `rai-compliance-agent/`). The current app audits AI-generated text outputs (loan decisions, HR summaries, etc.) against 5 RAI pillars.

**We are pivoting the core use case.** Instead of auditing AI-generated text, we now audit **company policy documents** (PDFs, Word docs) to determine whether they adequately address the 5 Responsible AI governance pillars from our framework. Think: "Does your AI policy actually cover fairness? Does it address DPDPA consent? Does it name accountable owners?"

---

## The 5 RAI Pillars (from our framework — keep these exact names and criteria)

1. **Governance & Accountability** — Does the policy name a responsible party for AI systems? Is there an AI Ethics Committee or equivalent? Is there a human override protocol? (Regulatory anchors: EU AI Act Art.17, RBI MRM Sec.4, DPDPA 2023 Sec.8)

2. **Fairness & Bias Mitigation** — Does the policy require training data assessment for bias? Is there a proxy variable review process? Are fairness metrics defined pre-deployment? (Regulatory anchors: EU AI Act Art.10, RBI MRM Sec.7, DPDPA 2023 Sec.8)

3. **Transparency & Explainability** — Does the policy require disclosure when AI is used in decisions? Is there a counterfactual or plain-language explanation standard? Are model cards required? (Regulatory anchors: EU AI Act Art.13, DPDPA 2023 Sec.11)

4. **Robustness & Monitoring** — Does the policy define performance thresholds and drift detection? Is there a periodic review schedule? Is there an incident response playbook? (Regulatory anchors: EU AI Act Art.72, RBI MRM Sec.8)

5. **Privacy & Data Stewardship** — Does the policy require consent review for AI training data? Is there a purpose limitation enforcement clause? Are vendor AI data assessments required? (Regulatory anchors: DPDPA 2023 Sec.6, Sec.8, EU AI Act Art.10(5))

---

## Changes Required

### 1. New Primary Tab — Policy Document Auditor

Replace or rename the existing "Text Auditor" tab. The new primary tab should be called **"Policy Document Auditor"**.

**Input:** User uploads a PDF or Word (.docx) document — a company's AI policy, HR policy, data governance policy, or similar. The existing `utils/file_parser.py` already handles PDF and DOCX extraction.

**What it does:** The LangGraph pipeline reads the full extracted text and evaluates it against each of the 5 pillars. For each pillar, the policy agent determines:
- Is this pillar **addressed** (present and adequate)?
- Is it **partially addressed** (mentioned but lacks specifics)?
- Is it **absent** (not covered at all)?

This maps to the 0–3 scoring rubric already in `nodes/scorecard.py` (3=fully addressed, 2=partially addressed, 1=minimal mention, 0=absent).

**Changes to `nodes/policy_agent.py`:**

Add a new mode: `input_type == "policy_document"`. When in this mode, replace the 11 text-output criteria with the following 5 pillar-specific policy criteria. Use the existing Ollama/Gemma call structure — just change the prompt.

```python
POLICY_DOCUMENT_CRITERIA = [
    {
        "id": "GOVERNANCE_ACCOUNTABILITY",
        "pillar": "governance",
        "description": "Policy must name a responsible owner or AI Ethics Committee for AI systems, with a human override mechanism",
        "reference": "EU AI Act Art.17 / RBI MRM Sec.4 / DPDPA 2023 Sec.8",
        "check_questions": [
            "Is a named individual or committee accountable for AI governance?",
            "Is there a model inventory or register requirement?",
            "Is human override of AI decisions mandated?"
        ]
    },
    {
        "id": "FAIRNESS_BIAS",
        "pillar": "fairness",
        "description": "Policy must require pre-deployment bias testing, training data assessment, and fairness metric selection",
        "reference": "EU AI Act Art.10 / RBI MRM Sec.7 / DPDPA 2023 Sec.8",
        "check_questions": [
            "Does the policy require bias/fairness testing before deployment?",
            "Is training data assessment for historical bias required?",
            "Are proxy variable reviews mandated?"
        ]
    },
    {
        "id": "TRANSPARENCY_EXPLAINABILITY",
        "pillar": "transparency",
        "description": "Policy must require disclosure of AI use in decisions and provide an explanation standard for affected individuals",
        "reference": "EU AI Act Art.13 / DPDPA 2023 Sec.11",
        "check_questions": [
            "Does the policy require informing people when AI influences their decisions?",
            "Is there an explanation standard (e.g. counterfactual, plain-language)?",
            "Are model cards or documentation required?"
        ]
    },
    {
        "id": "ROBUSTNESS_MONITORING",
        "pillar": "robustness",
        "description": "Policy must define performance monitoring, drift detection, periodic review, and an incident response protocol",
        "reference": "EU AI Act Art.72 / RBI MRM Sec.8",
        "check_questions": [
            "Are performance thresholds and monitoring protocols defined?",
            "Is there a requirement for periodic model review?",
            "Is there an incident response or escalation procedure?"
        ]
    },
    {
        "id": "PRIVACY_DATA_STEWARDSHIP",
        "pillar": "privacy",
        "description": "Policy must require DPDPA-compliant consent for AI training data, purpose limitation, and vendor AI data assessments",
        "reference": "DPDPA 2023 Sec.6 / Sec.8 / EU AI Act Art.10(5)",
        "check_questions": [
            "Does the policy address consent for using personal data in AI training?",
            "Is purpose limitation for AI data use defined?",
            "Are third-party/vendor AI tools subject to data assessment?"
        ]
    }
]
```

The LLM prompt for policy document mode should instruct Gemma to read the full policy text and for each criterion, determine the compliance level (0=absent, 1=minimal, 2=partial, 3=adequate) and identify specific gaps.

**Scorecard mapping for policy document mode** (update `nodes/scorecard.py`):

When `input_type == "policy_document"`, map the 5 criteria directly to the 5 pillars (1:1 mapping). The pillar score equals the LLM-assigned compliance level for that criterion (0–3).

```
GOVERNANCE_ACCOUNTABILITY  → strategic_alignment
FAIRNESS_BIAS              → model_governance  
TRANSPARENCY_EXPLAINABILITY → org_oversight
ROBUSTNESS_MONITORING      → continuous_monitoring
PRIVACY_DATA_STEWARDSHIP   → data_governance
```

---

### 2. Fix the Scorecard — Always Show FAIL When It Fails

**Problem:** The current UI (`ui/app.py`) shows a status badge and pillar scores, but the scorecard node itself may not be setting `final_status = "FAIL"` reliably when policy violations exist in policy document mode.

**Fix in `nodes/scorecard.py`:** Ensure that `final_status` is set to `"FAIL"` if **any** pillar scores below 2, OR if the overall score is below 10/15. Currently violations are checked but a policy document could fail without generating the same violation strings. Add explicit logic:

```python
# In scorecard_node, after computing rai_scores:
low_pillars = [k for k, v in rai_scores.items() if v < 2]
if low_pillars or sum(rai_scores.values()) < 10:
    final_status = "FAIL"
else:
    final_status = "PASS"
```

**Fix in `ui/app.py`:** Make sure the scorecard section in `render_text_auditor()` (which will now serve policy docs too) renders the FAIL badge prominently. The `_status_badge_html()` function already has red for FAIL — just ensure it's called with the correct `final_status` from state and not overridden. Also update `_pillar_score_html()` to add a ❌ icon next to pillars scoring 0 or 1, and a ⚠️ for score 2. Currently it only colours the bar.

---

### 3. Add a "Suggest Changes" Button

After a policy document audit completes and `final_status == "FAIL"` (or any pillar < 3), show a **"Suggest Changes"** button below the scorecard.

**Location in `ui/app.py`:** After the `_pillar_score_html()` block in `render_text_auditor()`, add:

```python
if final_state and final_state.get("final_status") == "FAIL":
    if st.button("💡 Suggest Changes", type="secondary", key="suggest_changes_btn"):
        # Trigger suggestion generation
        _run_suggestions(final_state)
```

**New function `_run_suggestions(final_state)`** in `ui/app.py`:

This calls a new LangGraph node (or a direct Ollama call) that takes:
- The original policy document text
- The pillar scores and violation descriptions from the audit
- The POLICY_DOCUMENT_CRITERIA including `check_questions`

And returns **specific, actionable rewrite suggestions** per failing pillar. Format: for each pillar that scored < 3, provide:
- A plain-language description of what's missing
- 2–3 concrete sentences the policy could add to address the gap
- The regulatory reference that requires it

**New node `nodes/suggestion_agent.py`:**

Create this file. It takes `ComplianceState` and runs a single Ollama/Gemma call with a prompt like:

```
You are a responsible AI governance expert. Below is a company policy document and an audit showing which RAI pillars it fails to address.

POLICY TEXT:
{current_text[:3000]}

AUDIT FINDINGS:
{json of pillar scores and violations}

For each failing pillar (score < 3), provide:
1. What is missing from the policy
2. 2-3 specific sentences the policy should include to address this gap
3. The regulatory clause that requires it

Be concrete and specific. Write suggestions as if you are drafting policy language.
```

Return structured output:
```python
class PillarSuggestion(BaseModel):
    pillar_id: str
    pillar_name: str  
    current_score: int
    gap_description: str
    suggested_language: list[str]  # 2-3 draft sentences
    regulatory_basis: str

class SuggestionsOutput(BaseModel):
    suggestions: list[PillarSuggestion]
```

Store result in `ComplianceState` as `suggestions: list[dict]`.

**Render suggestions in `ui/app.py`** as an expandable section after the button click, showing each pillar's suggestions in a styled card:

```
🔴 [Pillar Name] — Score: 1/3
Gap: [gap_description]
Suggested language:
  • "Sentence 1..."
  • "Sentence 2..."
Regulatory basis: [regulatory_basis]
```

Use `st.session_state` to persist suggestions across reruns (key: `"suggestions_result"`).

---

### 4. Update Demo Scenarios

In `_DEMO_SCENARIOS`, replace the loan text examples with **policy document examples** relevant to the new use case. Add 3–4 scenarios:

- `"Complete AI Policy (→ PASS)"` — a well-written AI policy that covers all 5 pillars
- `"Vague Tech Policy — Missing Governance & Privacy (→ FAIL)"` — a policy that mentions AI broadly but has no accountability structure or consent language
- `"HR Policy with No Bias Testing Clause (→ FAIL)"` — mentions AI in hiring but no fairness testing requirement
- `"Data Policy Missing AI Training Consent (→ FAIL)"` — covers general data use but doesn't address AI model training consent per DPDPA

Write realistic 300–500 word policy excerpts for each. Make them sound like real corporate policy language.

---

### 5. Update State Schema

In `state.py`, add these fields to `ComplianceState`:

```python
suggestions: list[dict]           # output from suggestion_agent
source_doc_type: str              # "policy_document" | "text" | "model_output"  
pillar_gaps: dict                 # per-pillar gap descriptions from policy audit
```

Also update `create_initial_state()` to accept `input_type="policy_document"` (it may already accept it via `"text"` but the distinction matters for the policy agent routing).

---

### 6. Wire the New Node into the Graph

In `graph.py`, add `suggestion_agent` as an **optional terminal branch** — it does not run automatically, only when triggered by the UI button. One approach: keep it out of the main LangGraph flow and call it as a standalone invocation from `_run_suggestions()` in the UI (simpler and avoids re-running the full pipeline).

The main graph flow stays:
```
ingestion → pii_agent → bias_agent → explainability_agent → policy_agent → scorecard
```

The suggestion call is a separate direct Ollama invocation, not a graph node, unless you prefer to add it as a conditional edge after scorecard.

---

## Files to Create or Modify

| File | Action |
|------|--------|
| `nodes/policy_agent.py` | Add `policy_document` mode with 5-pillar criteria and new prompt |
| `nodes/scorecard.py` | Fix FAIL logic to trigger on low pillar scores, not just violation strings |
| `nodes/suggestion_agent.py` | **CREATE** — new node/function for generating policy rewrite suggestions |
| `ui/app.py` | Add "Suggest Changes" button, update demo scenarios, fix FAIL badge display, add `_run_suggestions()` |
| `state.py` | Add `suggestions`, `source_doc_type`, `pillar_gaps` fields |
| `graph.py` | Minor: ensure `policy_document` input_type routes correctly through policy_agent |

---

## Do NOT Change

- The existing Model Auditor tab (CSV bias audit) — keep it intact
- The existing PII detection logic in `nodes/pii_agent.py`
- The existing Bias Agent in `nodes/bias_agent.py`
- The RAG/ChromaDB retrieval setup
- The audit trail / SQLite logging
- The run history sidebar

---

## Constraints

- Use Ollama/Gemma (already configured as `OLLAMA_MODEL = "gemma4:latest"`) for all LLM calls — do not introduce new model dependencies
- Keep Pydantic structured output pattern already used in `policy_agent.py`
- The `utils/file_parser.py` already handles PDF and DOCX — use it as-is for document extraction
- Maintain the existing pillar name keys (`strategic_alignment`, `data_governance`, `model_governance`, `org_oversight`, `continuous_monitoring`) in `rai_scores` so the radar chart still works
