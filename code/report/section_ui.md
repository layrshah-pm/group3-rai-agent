# Section: UI Design Decisions
**Person 6 · IIMA Capstone EPAIBBL01 · Group 3**

---

## 1. Overview

The user interface is a **Streamlit** web application that exposes the RAI Compliance Agent to non-technical stakeholders. It provides real-time streaming output, visual compliance scoring, and a persistent run history. The interface was designed around two primary user journeys:

1. **Text Auditor**: a compliance analyst pastes AI-generated text to audit in real time
2. **Model Auditor**: a data scientist uploads a CSV with model predictions to check for demographic bias

---

## 2. Technology Choice: Streamlit

Streamlit was chosen over Flask/React for three reasons:

- **Zero frontend build overhead**: Python-native, no JavaScript, no npm — any team member can modify it
- **Native streaming support**: `app.stream()` yields LangGraph node outputs one at a time; Streamlit's placeholder API renders them incrementally
- **Plotly integration**: the radar chart is a first-class Streamlit component via `st.plotly_chart()`

The tradeoff is limited layout control (Streamlit imposes a vertical flow) and no fine-grained state persistence between interactions beyond session state.

---

## 3. Real-Time Streaming

The audit runs live in the browser as the graph executes. This was implemented by calling `app.stream()` instead of `app.invoke()`:

```python
for chunk in app.stream(state, config=config):
    node_name = list(chunk.keys())[0]
    label = NODE_LABELS.get(node_name, node_name)
    log_lines.append(f"✓ **{label}**")
    progress_ph.markdown(" → ".join(log_lines))
    final_state = chunk[node_name]
```

Each chunk contains the output of one node. The progress placeholder updates after each node completes, giving the user a live breadcrumb trail:

```
✓ Ingestion → ✓ PII Detection → ✓ Bias Analysis → ✓ Policy Compliance → ✓ Scorecard
```

---

## 4. Radar Chart

The five-pillar RAI scorecard is visualised as a Plotly `Scatterpolar` chart. Each axis represents one pillar, scaled 0–3:

```python
fig = go.Figure(data=go.Scatterpolar(
    r=values_closed,
    theta=categories_closed,
    fill="toself",
    line_color="#2E75B6",
    fillcolor="rgba(46, 117, 182, 0.2)",
))
fig.update_layout(
    polar=dict(radialaxis=dict(range=[0, 3], tickvals=[0, 1, 2, 3])),
    height=350,
)
```

The radar chart provides immediate visual intuition: a full pentagon means all pillars score 3/3; a collapsed shape in one direction instantly identifies the weak pillar. This is far more useful for a presentation than a table of numbers.

---

## 5. Status Badge and Colour Coding

The overall compliance status is displayed as a large coloured badge:

| Status | Colour | Meaning |
|---|---|---|
| `PASS` | Green `#28a745` | All criteria satisfied, no corrections needed |
| `CORRECTED` | Amber `#ffc107` | Violation detected and automatically fixed |
| `FAIL` | Red `#dc3545` | Violation detected, correction failed |
| `ESCALATED` | Dark red `#8B0000` | Max corrections reached, human review required |

Policy violation severity also uses colour chips (high = red, medium = orange, low = yellow), allowing a reviewer to triage findings at a glance without reading each description.

---

## 6. Pillar Score Bars

In addition to the radar chart, inline progress bars show each pillar's score with colour coding:

- **Green**: score = 3 (fully compliant)
- **Amber**: score = 2 (minor gaps)
- **Red**: score 0–1 (significant violations)

These bars are rendered as inline HTML because Streamlit's native `st.progress()` does not support per-bar colour customisation.

---

## 7. Sidebar Run History

The sidebar displays the last 15 runs from the `rai_audit.db` audit database, colour-coded by status. This allows a demo presenter to quickly show multiple runs and their outcomes without re-running the agent each time.

Each entry shows: status, total score, timestamp (truncated to second), and the first 8 characters of the thread ID for traceability.

```python
from utils.audit_logger import get_runs_summary
runs = get_runs_summary()
for r in runs[:15]:
    st.markdown(f'<div class="run-history-item run-{fs.lower()}">...')
```

---

## 8. Model Auditor Tab

The Model Auditor tab accepts a CSV upload and displays fairness metrics using Streamlit `st.metric()` components with delta indicators:

```python
c1.metric(
    "Disparate Impact",
    f"{di:.3f}",
    delta="✓ ≥0.8" if di_ok else "✗ <0.8 FAIL",
    delta_color="normal" if di_ok else "inverse",
)
```

Green delta = threshold met; red delta = threshold violated. This communicates pass/fail without requiring the user to know the threshold values.

---

## 9. Design Limitations

- **`st.rerun()` on completion**: After an audit finishes, `st.rerun()` refreshes the page to update the sidebar history. This causes a brief flicker. A future improvement would use `st.session_state` to persist partial results across reruns.
- **Streaming and Streamlit Cloud**: The live streaming works reliably in local development. Streamlit Community Cloud enforces a timeout on long-running requests; the policy agent's ~90s LLM calls may time out. A production deployment would use a background task queue.
- **No authentication**: The current app has no login. For a real compliance tool, access should be restricted to authorised personnel.
