# Claude Code: Complete Fix Prompt — RAI Compliance Agent

Run this from inside `rai-compliance-agent/`. This is a self-contained prompt covering every
visible problem in the current frontend (as of 30 May 2026 screenshot).

---

## ROOT CAUSE DIAGNOSIS (read this first)

### Bug 1 — THE CRITICAL BUG: `_run_and_display` saves wrong state (ui/app.py line 1093)

```python
# CURRENT (BROKEN):
for chunk in app.stream(initial_state, config=config):
    node_name = list(chunk.keys())[0]
    node_state = chunk[node_name]
    ...
    final_state = node_state   # ← BUG: this is only the current node's partial update
```

In LangGraph, `app.stream()` yields one chunk per node. Each chunk contains ONLY that
node's partial state update — e.g. what `pii_agent_node()` returned in its `return {}` dict.
It does NOT contain the full accumulated state.

So by the time the loop ends, `final_state` is set to what ONLY the `scorecard` node returned:
`{final_status, rai_scores, current_node, audit_log: [one entry], step_trace: [one entry]}`.

**This explains every symptom in the screenshot:**
- "Audit Steps" shows only Scorecard — step_trace has only 1 entry (scorecard's)
- "Violations Found: 0" — violations list is empty (scorecard doesn't add violations)
- No pillar breakdown cards — pillar_gaps is None (set by policy_agent, not scorecard)
- No LLM explanations — policy_result is None in the saved state
- No "Suggest Changes" button — pillar_gaps empty so condition fails

### Bug 2 — torchvision console errors

`sentence-transformers` (used for ChromaDB RAG embeddings) imports `torch`, which on some
installs triggers torchvision deprecation/compatibility warnings. These flood the console.
Fix: suppress at module level in app.py.

### Bug 3 — No LLM explanation visible in Audit Steps

Even after fixing Bug 1, the Policy Agent step expander only shows the raw JSON prompt/response.
It needs a rendered section that shows per-pillar reasoning in human-readable format.

### Bug 4 — Bias Agent step shows unhelpful message

The step says "Text mode: bias check not applicable" with no explanation. Evaluators need
to understand WHY fairness isn't checked on text and WHERE it IS checked.

### Bug 5 — PII Agent step shows no explanation of what was scanned or why

The step just shows a table of entities (or "none found") with no context.

---

## FIX 1 — Critical: get full accumulated state after streaming (ui/app.py)

Find the `_run_and_display` function. Replace the entire function body with this:

```python
def _run_and_display(initial_state: dict):
    from graph import app

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    pipeline_ph = st.empty()
    completed: dict[str, str] = {}
    running_node = None
    all_steps = []
    # Track per-node results from stream chunks for the pipeline bar
    node_results: dict[str, str] = {}

    for chunk in app.stream(initial_state, config=config):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]   # partial update only — DO NOT use as final_state

        # Determine result for the node that just finished
        audit_entries = node_state.get("audit_log", [])
        entry = next((e for e in reversed(audit_entries) if e.get("node") == node_name), None)
        res = entry.get("result", "pass") if entry else "pass"
        node_results[node_name] = res

        # Mark previous running as complete, start this one
        if running_node and running_node not in completed:
            completed[running_node] = node_results.get(running_node, "pass")

        running_node = node_name
        pipeline_ph.markdown(_pipeline_bar_html(completed, running_node), unsafe_allow_html=True)

    # Mark final node complete
    if running_node:
        completed[running_node] = node_results.get(running_node, "pass")

    pipeline_ph.markdown(_pipeline_bar_html(completed, None), unsafe_allow_html=True)

    # ── CRITICAL FIX: get the FULL accumulated state from the checkpoint ──
    # app.stream() yields partial updates per node. We need the full state.
    try:
        full_state = app.get_state(config).values
    except Exception as e:
        st.error(f"Could not retrieve final state: {e}")
        return

    if not full_state:
        st.error("Graph returned no output.")
        return

    st.session_state["pipeline_completed"] = completed
    st.session_state["pipeline_log"] = all_steps
    st.session_state["text_audit_result"] = full_state   # full state, not partial
    st.session_state["history_thread_id"] = thread_id
```

---

## FIX 2 — Suppress torchvision/torch warnings (ui/app.py — top of file)

At the very top of `ui/app.py`, immediately after the module docstring and before any imports,
add:

```python
import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*torch.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)
```

---

## FIX 3 — Policy Agent step: render LLM explanation clearly (ui/app.py)

Find the block in `render_text_auditor()` that handles `step["step"] == "policy_agent"`.
Currently it dumps raw JSON for the prompt and response. Replace that block with:

```python
elif step["step"] == "policy_agent":
    resp = step.get("response", {})
    pillar_scores_raw = resp.get("pillar_scores", [])
    summary = resp.get("summary", "")

    # Show LLM summary
    if summary:
        st.markdown("**Gemma 4 overall assessment:**")
        st.info(summary)

    # Show per-pillar results from LLM response
    if pillar_scores_raw:
        st.markdown("**Per-pillar LLM findings:**")
        PILLAR_DISPLAY = {
            "GOVERNANCE_ACCOUNTABILITY":   "🏛️ Governance & Accountability",
            "FAIRNESS_BIAS":               "⚖️ Fairness & Bias",
            "TRANSPARENCY_EXPLAINABILITY": "🔍 Transparency & Explainability",
            "ROBUSTNESS_MONITORING":       "🛡️ Robustness & Monitoring",
            "PRIVACY_DATA_STEWARDSHIP":    "🔒 Privacy & Data Stewardship",
        }
        for ps in pillar_scores_raw:
            pid = ps.get("id", "")
            level = ps.get("compliance_level", 0)
            gap = ps.get("gap_description", "")
            evidence = ps.get("evidence_from_policy", "")
            pass_reason = ps.get("pass_reason", "")
            label = PILLAR_DISPLAY.get(pid, pid)
            colour = "#28a745" if level == 3 else ("#ffc107" if level >= 2 else "#dc3545")
            icon = "✅" if level == 3 else ("⚠️" if level >= 2 else "❌")

            st.markdown(
                f'<div style="border:1px solid {colour};border-radius:6px;'
                f'padding:10px 14px;margin:8px 0;">'
                f'<strong>{icon} {label}</strong> — Score: <strong>{level}/3</strong>',
                unsafe_allow_html=True,
            )
            if evidence and evidence.lower() not in ("not found in policy.", "", "none"):
                st.markdown(
                    f'<div style="background:#f0f4ff;border-left:3px solid #2E75B6;'
                    f'padding:8px 12px;margin:6px 0;border-radius:0 4px 4px 0;'
                    f'font-size:0.84rem;font-style:italic;">📄 Policy says: "{evidence}"</div>',
                    unsafe_allow_html=True,
                )
            if pass_reason and level >= 2:
                st.markdown(
                    f'<div style="background:#d4edda;border-left:3px solid #28a745;'
                    f'padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0;font-size:0.84rem;">'
                    f'✅ {pass_reason}</div>',
                    unsafe_allow_html=True,
                )
            if gap and level < 3:
                st.markdown(
                    f'<div style="background:#f8d7da;border-left:3px solid #dc3545;'
                    f'padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0;font-size:0.84rem;">'
                    f'❌ Missing: {gap}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    # Show raw prompt in a collapsed expander for transparency
    if step.get("prompt"):
        with st.expander("View raw Gemma 4 prompt", expanded=False):
            st.code(step["prompt"], language="text")
        with st.expander("View raw Gemma 4 JSON response", expanded=False):
            import json as _json
            st.code(_json.dumps(step.get("response", {}), indent=2), language="json")
```

---

## FIX 4 — Bias Agent step: explain WHY fairness is not run on text (ui/app.py)

Find the bias_agent section inside the step expander loop. Replace the `st.info(...)` for
text mode with:

```python
if mode == "text":
    st.markdown("#### Why isn't statistical fairness checked here?")
    st.markdown("""
**Statistical bias analysis needs model prediction data.** The Fairlearn library computes
these three metrics — but all three require a column of actual model predictions and a
protected attribute column (e.g. `sex`, `race`, `age_group`):

| Metric | What it measures | Threshold |
|---|---|---|
| **Demographic Parity Difference** | Are positive outcomes equally likely across groups? | ≤ 0.10 |
| **Equalized Odds Difference** | Are error rates equal across groups? | ≤ 0.10 |
| **Disparate Impact Ratio** | Does the disadvantaged group get ≥ 80% of favourable outcomes? | ≥ 0.80 |

A policy document contains no predictions — so these metrics cannot be computed.

**How fairness IS audited for policy documents:**
The Policy Agent (Gemma 4) checks the `FAIRNESS_BIAS` pillar — it reads the policy text
and verifies whether the company has committed to:
- Pre-deployment bias testing with defined metrics (EU AI Act Art.10)
- Training data assessment for historical bias (RBI MRM Sec.7)
- Proxy variable reviews before model deployment (DPDPA 2023 Sec.8)

**To run statistical bias analysis:** use the **Model Auditor** tab and upload a CSV
with model predictions and a protected attribute column.
""")
```

---

## FIX 5 — PII Agent step: explain what PII is and what was scanned (ui/app.py)

Find the `pii_agent` section inside the step expander loop. Before the entity table,
add this explanation block:

```python
st.markdown("#### What is PII Detection?")
st.markdown("""
**PII (Personally Identifiable Information)** is any data that can be used to identify
a specific individual. The presence of PII in an AI policy, training dataset, or model
output creates legal exposure under:
- **GDPR Article 5** — lawfulness, purpose limitation, data minimisation
- **DPDPA 2023 Section 4** — processing must have a documented lawful purpose and consent

**This agent uses Microsoft Presidio** (an NLP-based entity recogniser) to scan for:

| Entity | Example | Why it matters |
|---|---|---|
| `PERSON` | "Rahul Sharma" | Identity disclosure |
| `EMAIL_ADDRESS` | "admin@firm.com" | Direct contact exposure |
| `PHONE_NUMBER` | "+91-98765-43210" | Contact tracing risk |
| `CREDIT_CARD` | "4111 1111 1111 1111" | Financial fraud enablement |
| `IBAN_CODE` | "GB29 NWBK 6016 1331" | Financial fraud enablement |
| `LOCATION` | "Mumbai" | Location profiling |
| `NRP` | "Muslim", "BJP supporter" | Sensitive attribute leakage |

**Confidence threshold: 0.75** — entities below this are suppressed to reduce false positives
on governance documents (e.g. box-drawing separator characters `━`).

**If PII is found**: the agent generates a redacted version with entities replaced by
`<ENTITY_TYPE>` tags. Fix by removing or anonymising the flagged content before deployment.
""")
```

---

## FIX 6 — "Violations Found" metric shows 0 when violations exist

This is caused by Bug 1 (fixed above). But add a safety fallback in case the violations
list is empty but the status is FAIL:

Find the metrics section in `render_text_auditor()`:
```python
c1, c2 = st.columns(2)
c1.metric("Status", fs)
c2.metric("Violations Found", len(set(violations)))
```

Replace with:
```python
c1, c2, c3 = st.columns(3)
c1.metric("Status", fs)
# Count both violation codes AND failing pillars for completeness
policy_violations = len(policy.get("violations") or [])
pii_flag = 1 if (final_state.get("pii_result") or {}).get("has_pii") else 0
total_violations = len(set(violations)) or (policy_violations + pii_flag)
c2.metric("Violations Found", total_violations)
overall = sum(scores.values()) if scores else 0
c3.metric("Overall Score", f"{overall}/15")
```

---

## FIX 7 — Show "Suggest Changes" button and explanation always visible for failing policy

Currently the button is hidden behind a `not suggestions_result` check. Change it so the
button is always shown for failing policy docs, and the suggestions section replaces/updates
rather than hides:

Find:
```python
if is_policy_doc and fs == "FAIL" and not suggestions_result:
    if st.button("💡 Suggest Improvements for All Failing Pillars", ...):
```

Replace with:
```python
if is_policy_doc and fs == "FAIL":
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        st.markdown(
            "**This policy has gaps.** Click below to get Gemma 4's suggestions for "
            "specific policy language to add for each failing pillar."
        )
    with col_s2:
        if st.button("💡 Suggest Improvements", type="primary", key="suggest_all_btn"):
            st.session_state.pop("suggestions_result", None)
            _run_suggestions(final_state)
```

---

## FIX 8 — Add `_render_pillar_breakdown` guard for missing pillar_gaps

The `_render_pillar_breakdown` function requires pillar_gaps to be populated. If it's empty
(e.g. loaded from history before Fix 1 is applied), add a fallback:

Find in `render_text_auditor()`:
```python
if is_policy_doc and pillar_gaps:
    _render_pillar_breakdown(scores, pillar_gaps, final_state)
else:
    st.markdown("**RAI Pillar Scores**")
    st.markdown(_pillar_score_html(scores), unsafe_allow_html=True)
```

Replace with:
```python
if is_policy_doc and pillar_gaps:
    _render_pillar_breakdown(scores, pillar_gaps, final_state)
else:
    st.markdown("**RAI Pillar Scores**")
    st.markdown(_pillar_score_html(scores), unsafe_allow_html=True)
    if is_policy_doc and not pillar_gaps:
        st.caption(
            "ℹ️ Per-pillar reasoning not available for this run. "
            "Re-run the audit to see detailed Gemma 4 explanations per pillar."
        )
```

---

## FIX 9 — Show the Policy Agent in Audit Steps (currently skipped)

Check if `ingestion` is the only step being skipped. The current filter is:
```python
for step in step_trace:
    if step["step"] == "ingestion":
        continue
```

This is correct — but after Fix 1, all 5 agent steps will appear. Verify after applying Fix 1.

---

## VERIFICATION STEPS

After applying all fixes, run these checks:

### Check 1 — Smoke test (terminal)
```bash
cd rai-compliance-agent
source venv/bin/activate
python -c "
import sys; sys.path.insert(0, '.')
from graph import app
from state import create_initial_state

state = create_initial_state(
    input_type='policy_document',
    raw_input='We use AI. A committee reviews decisions.'
)
config = {'configurable': {'thread_id': 'verify-001'}}
result = app.invoke(state, config=config)

# Full state should have all fields
assert result.get('pii_result') is not None, 'pii_result missing'
assert result.get('bias_result') is not None, 'bias_result missing'
assert result.get('policy_result') is not None, 'policy_result missing'
assert len(result.get('step_trace', [])) >= 5, f'step_trace only has {len(result.get(\"step_trace\", []))} entries'
assert len(result.get('violations', [])) >= 0, 'violations missing'
print('✅ Full state check passed')
print('step_trace entries:', [s[\"step\"] for s in result.get(\"step_trace\", [])])
print('violations:', result.get('violations'))
print('pillar_gaps keys:', list((result.get('pillar_gaps') or {}).keys()))
"
```

### Check 2 — get_state works after stream
```bash
python -c "
import sys, uuid; sys.path.insert(0, '.')
from graph import app
from state import create_initial_state

state = create_initial_state(input_type='policy_document', raw_input='Test policy.')
thread_id = str(uuid.uuid4())
config = {'configurable': {'thread_id': thread_id}}

# Stream (mimics UI)
for chunk in app.stream(state, config=config):
    pass  # discard partial updates

# Get full state — this is what the UI should use
full = app.get_state(config).values
print('Full state keys:', list(full.keys()))
print('step_trace count:', len(full.get('step_trace', [])))
print('pillar_gaps:', full.get('pillar_gaps') is not None)
print('policy_result:', full.get('policy_result') is not None)
print('✅ get_state works correctly')
"
```

### Check 3 — torchvision warnings suppressed
```bash
python -W all -c "
import warnings
warnings.filterwarnings('ignore', message='.*torchvision.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
import sentence_transformers
print('✅ No torchvision warnings')
" 2>&1 | grep -i "torchvision" | wc -l
# Should output: 0
```

---

## WHAT THE UI SHOULD SHOW AFTER ALL FIXES

1. **Pipeline bar** — all 6 nodes (Ingestion → PII → Bias → Explainability → Policy → Scorecard)
   with individual PASS/FAIL status, animated during run.

2. **Status badge** — FAIL or PASS correctly.

3. **Metrics row** — Status | Violations Found (real count) | Overall Score (e.g. 11/15).

4. **RAI Pillar Assessment** (for policy docs with pillar_gaps):
   - One collapsible card per pillar, auto-expanded if failing
   - Left: direct quote from policy ("What the policy says")
   - Right: green box (why it passed) OR red box (exactly what's missing)
   - Regulatory citation (📎 EU AI Act Art.10...)
   - "💡 Suggest fix for this pillar" button on failing pillars

5. **Audit Steps** — all 5 agent steps visible, each expandable:
   - **PII**: explanation of what PII is + entity type table + results
   - **Bias**: explanation of why statistical fairness needs model data + where policy fairness IS checked
   - **Explainability**: SHAP not applicable for text (explanation shown)
   - **Policy**: Gemma 4 summary + per-pillar colour-coded findings + raw prompt/response
   - **Scorecard**: final scores + violation count

6. **Suggested Improvements**: "💡 Suggest Improvements" button always visible for failing
   policy docs. Per-pillar suggestions show 3 numbered policy clauses each.

7. **No torchvision errors** in console.

---

## SUMMARY TABLE: What was wrong and what fixed it

| Symptom | Root Cause | Fix |
|---|---|---|
| Only Scorecard in Audit Steps | `final_state = node_state` saves partial update only | Fix 1: `app.get_state(config).values` |
| Violations Found = 0 | Same — violations list empty in scorecard partial | Fix 1 + Fix 6 fallback |
| No pillar breakdown cards | `pillar_gaps` = None in partial state | Fix 1 |
| No LLM explanation | `policy_result` = None in partial state | Fix 1 + Fix 3 rendering |
| No Suggest Changes button | `pillar_gaps` empty so condition never met | Fix 1 + Fix 7 |
| torchvision console errors | sentence-transformers transitive dependency | Fix 2 |
| Bias step says "not applicable" | No explanation rendered | Fix 4 |
| PII step shows only table | No explanation rendered | Fix 5 |
