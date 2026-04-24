"""
ui/app.py
---------
Streamlit application for the RAI Compliance Agent.

Run with:
  cd rai-compliance-agent
  streamlit run ui/app.py
"""

import sys
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="RAI Compliance Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.title("RAI Compliance Agent")
st.caption("Automated AI governance · EU AI Act Article 50 · IIMA Capstone Group 3")

NODE_LABELS = {
    "ingestion":    "Ingestion",
    "pii_agent":    "PII Detection",
    "bias_agent":   "Bias Analysis",
    "policy_agent": "Policy Compliance",
    "correction":   "Self-Correction",
    "scorecard":    "Scorecard",
}

SEVERITY_COLOURS = {
    "high":   ("#dc3545", "#fff"),
    "medium": ("#fd7e14", "#fff"),
    "low":    ("#ffc107", "#212529"),
}

STATUS_COLOURS = {
    "PASS":      "#28a745",
    "CORRECTED": "#ffc107",
    "FAIL":      "#dc3545",
    "ESCALATED": "#8B0000",
}

STATUS_TEXT_COLOURS = {
    "PASS":      "#fff",
    "CORRECTED": "#000",
    "FAIL":      "#fff",
    "ESCALATED": "#fff",
}


def _status_badge_html(status: str) -> str:
    bg = STATUS_COLOURS.get(status, "#6c757d")
    fg = STATUS_TEXT_COLOURS.get(status, "#fff")
    return (
        f'<span style="display:inline-block;padding:10px 28px;background:{bg};'
        f'color:{fg};border-radius:8px;font-size:1.3rem;font-weight:700;'
        f'letter-spacing:0.06em;">{status}</span>'
    )


def _severity_chip(severity: str) -> str:
    bg, fg = SEVERITY_COLOURS.get(severity, ("#6c757d", "#fff"))
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600;'
        f'margin-right:4px;">{severity.upper()}</span>'
    )


def _pillar_score_html(scores: dict) -> str:
    labels = {
        "strategic_alignment":   "Strategic Alignment",
        "data_governance":       "Data Governance",
        "model_governance":      "Model Governance",
        "org_oversight":         "Org Oversight",
        "continuous_monitoring": "Continuous Monitoring",
    }
    rows = []
    for key, label in labels.items():
        score = scores.get(key, 0)
        pct = int(score / 3 * 100)
        colour = "#28a745" if score == 3 else ("#ffc107" if score >= 2 else "#dc3545")
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
            f'<span style="width:160px;font-size:0.85rem;">{label}</span>'
            f'<div style="flex:1;height:12px;border-radius:6px;background:#dee2e6;overflow:hidden;">'
            f'<div style="width:{pct}%;height:100%;background:{colour};border-radius:6px;"></div></div>'
            f'<span style="width:30px;font-size:0.85rem;font-weight:600;">{score}/3</span>'
            f'</div>'
        )
    total = sum(scores.values())
    rows.append(
        f'<div style="margin-top:8px;font-weight:700;font-size:0.95rem;">'
        f'Overall: {total}/15</div>'
    )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Sidebar — run history
# ---------------------------------------------------------------------------

def _load_run_from_db(thread_id: str):
    """Load full state for a past run and populate session_state."""
    try:
        from utils.audit_logger import get_audit_trail, _connect, _channel_values
        conn = _connect()
        try:
            cursor = conn.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY rowid DESC LIMIT 1",
                (thread_id,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        if not row:
            st.sidebar.error("Run data not found.")
            return
        cv = _channel_values(row["checkpoint"])
        audit_log = get_audit_trail(thread_id)
        cv["audit_log"] = audit_log
        st.session_state["text_audit_result"] = cv
        st.session_state["history_thread_id"] = thread_id
        st.session_state.pop("pipeline_log", None)
        st.session_state.pop("pipeline_completed", None)
    except Exception as e:
        st.sidebar.error(f"Could not load run: {e}")


def _render_sidebar():
    with st.sidebar:
        st.markdown("### Run History")
        try:
            from utils.audit_logger import get_runs_summary
            runs = get_runs_summary()
            if not runs:
                st.caption("No runs yet.")
            for r in runs[:15]:
                fs = r["final_status"]
                css_cls = f"run-{fs.lower()}"
                score = r["total_score"]
                ts = (r["timestamp"] or "")[:19].replace("T", " ")
                tid = r["thread_id"][:8]
                full_tid = r["thread_id"]
                is_active = st.session_state.get("history_thread_id") == full_tid
                border_extra = "box-shadow:0 0 0 2px #2E75B6;" if is_active else ""
                st.markdown(
                    f'<div class="run-history-item {css_cls}" style="color:#1a1a1a;{border_extra}">'
                    f'<strong>{fs}</strong> &nbsp; {score}/15<br>'
                    f'<small style="color:#444;">{ts} · {tid}</small></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Load", key=f"load_{full_tid}", use_container_width=True):
                    _load_run_from_db(full_tid)
                    st.rerun()
        except FileNotFoundError:
            st.caption("No audit DB yet — run an audit first.")
        except Exception as e:
            st.caption(f"History unavailable: {e}")


_render_sidebar()


# ---------------------------------------------------------------------------
# Tab 1 — Text Auditor
# ---------------------------------------------------------------------------

def render_text_auditor():
    from ui.components.radar_chart import render_radar_chart

    st.subheader("Text Auditor")
    st.markdown("Paste AI-generated text to audit for PII, bias, and policy compliance.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        text_input = st.text_area(
            "Input text",
            height=240,
            placeholder="Paste AI-generated text here...",
        )
        run_btn = st.button("Audit Text", type="primary", use_container_width=True, key="audit_text_btn")

    if run_btn and text_input.strip():
        _run_text_audit_streaming(text_input)
    elif run_btn:
        st.warning("Enter some text first.")

    final_state = st.session_state.get("text_audit_result")
    if not final_state:
        return

    # Show static pipeline bar for history loads (no live animation)
    completed = st.session_state.get("pipeline_completed")
    if not completed:
        audit_log_entries = final_state.get("audit_log", [])
        completed = {}
        for entry in audit_log_entries:
            n = entry.get("node", "")
            r = entry.get("result", "pass")
            if n:
                completed[n] = r
    st.markdown(_pipeline_bar_html(completed, None), unsafe_allow_html=True)

    fs = final_state.get("final_status", "UNKNOWN")
    scores = final_state.get("rai_scores")
    violations = final_state.get("violations", [])
    correction_count = final_state.get("correction_count", 0)
    pii = final_state.get("pii_result") or {}
    policy = final_state.get("policy_result") or {}

    # Status + radar side by side
    with col_result:
        st.markdown(_status_badge_html(fs), unsafe_allow_html=True)
        if scores:
            fig = render_radar_chart(scores)
            st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", fs)
    c2.metric("Violations", len(set(violations)))
    c3.metric("Corrections", correction_count)

    if scores:
        st.markdown("**RAI Pillar Scores**")
        st.markdown(_pillar_score_html(scores), unsafe_allow_html=True)

    st.divider()

    # ── Annotated text view ──────────────────────────────────────────────────
    raw_text = final_state.get("raw_input", "")
    corrected_text = final_state.get("current_text", "")
    has_issues = pii.get("entities_found") or policy.get("violations")

    if raw_text:
        st.markdown("#### Input Text Analysis")
        legend_parts = []
        if pii.get("entities_found"):
            legend_parts.append('<span style="background:rgba(220,53,69,0.18);border-bottom:2px solid #dc3545;padding:1px 6px;border-radius:2px;font-size:0.8rem;">PII Detected</span>')
        if policy.get("violations"):
            legend_parts.append('<span style="background:rgba(253,126,20,0.15);border-bottom:2px solid #fd7e14;padding:1px 6px;border-radius:2px;font-size:0.8rem;">Policy Violation</span>')
        if legend_parts:
            st.markdown(" &nbsp; ".join(legend_parts), unsafe_allow_html=True)

        st.markdown(
            _annotate_text_html(raw_text, pii, policy, corrected_text if correction_count > 0 else None),
            unsafe_allow_html=True,
        )

        if correction_count > 0 and corrected_text and corrected_text != raw_text:
            st.markdown("**Corrected output:**")
            st.success(corrected_text)

    st.divider()

    # ── Policy violations with citations ─────────────────────────────────────
    if policy.get("violations"):
        pvs = policy["violations"]
        st.markdown(f"#### Policy Violations ({len(pvs)} found)")
        for i, v in enumerate(pvs, 1):
            sev = v.get("severity", "low")
            chip = _severity_chip(sev)
            ref = v.get("article_reference", "")
            remediation = v.get("remediation", "")
            st.markdown(
                f'{chip} <strong>[{i}] {v["id"]}</strong> — {v["description"]}',
                unsafe_allow_html=True,
            )
            col_ref, col_fix = st.columns([1, 1])
            with col_ref:
                st.markdown(
                    f'<div style="background:#e8f0fb;border-left:3px solid #2E75B6;padding:6px 10px;border-radius:0 4px 4px 0;font-size:0.8rem;">'
                    f'📎 <strong>Citation:</strong> {ref}</div>',
                    unsafe_allow_html=True,
                )
            with col_fix:
                st.markdown(
                    f'<div style="background:#d4edda;border-left:3px solid #28a745;padding:6px 10px;border-radius:0 4px 4px 0;font-size:0.8rem;">'
                    f'🔧 <strong>Fix:</strong> {remediation}</div>',
                    unsafe_allow_html=True,
                )
            if i < len(pvs):
                st.divider()
    elif policy.get("passed"):
        st.success("Policy: all criteria satisfied.")

    # ── PII detail table ─────────────────────────────────────────────────────
    if pii.get("entities_found"):
        with st.expander(f"PII Entities — {len(pii['entities_found'])} detected", expanded=False):
            df_pii = pd.DataFrame(pii["entities_found"])
            cols = [c for c in ["entity_type", "value", "score", "start", "end"] if c in df_pii.columns]
            st.dataframe(df_pii[cols], hide_index=True, use_container_width=True)
            if pii.get("redacted_text"):
                st.markdown("**Redacted version:**")
                st.code(pii["redacted_text"])

    # ── Bias metrics ─────────────────────────────────────────────────────────
    bias = final_state.get("bias_result") or {}
    if bias.get("disparate_impact_ratio") is not None:
        with st.expander("Bias Metrics", expanded=False):
            di = bias["disparate_impact_ratio"]
            di_ok = di >= 0.8
            c1b, c2b, c3b = st.columns(3)
            c1b.metric("Disparate Impact", f"{di:.3f}",
                       delta="✓ ≥0.8" if di_ok else "✗ <0.8",
                       delta_color="normal" if di_ok else "inverse")
            dp_val = bias.get("demographic_parity_diff")
            if dp_val is not None:
                dp_ok = abs(dp_val) <= 0.1
                c2b.metric("Dem. Parity Diff", f"{dp_val:.3f}",
                           delta="✓ ≤0.1" if dp_ok else "✗ >0.1",
                           delta_color="normal" if dp_ok else "inverse")
            eo_val = bias.get("equalized_odds_diff")
            if eo_val is not None:
                eo_ok = abs(eo_val) <= 0.1
                c3b.metric("Eq. Odds Diff", f"{eo_val:.3f}",
                           delta="✓ ≤0.1" if eo_ok else "✗ >0.1",
                           delta_color="normal" if eo_ok else "inverse")
            if bias.get("privileged_group"):
                st.caption(f"Privileged: {bias['privileged_group']} · Unprivileged: {bias.get('unprivileged_group')}")

    # ── Retrieved regulatory clauses (RAG evidence) ──────────────────────────
    retrieved = final_state.get("retrieved_clauses") or []
    if retrieved:
        with st.expander(f"Regulatory Sources — {len(retrieved)} clauses retrieved", expanded=False):
            st.caption(
                "These regulatory clauses were retrieved from the vector store based on semantic "
                "relevance to the audited text. They grounded the policy compliance analysis."
            )
            for i, clause in enumerate(retrieved, 1):
                sim_pct = int(clause.get("similarity_score", 0) * 100)
                colour = "#28a745" if sim_pct >= 60 else "#ffc107" if sim_pct >= 40 else "#6c757d"
                st.markdown(
                    f'<div style="border-left:3px solid {colour};padding:8px 12px;margin-bottom:8px;">'
                    f'<strong>{clause["regulation"]} — {clause["article_id"]}</strong> '
                    f'<span style="color:{colour};font-size:0.8rem;">({sim_pct}% relevance)</span><br>'
                    f'<span style="font-size:0.85rem;color:#555;">{clause["reference"]}</span><br><br>'
                    f'<span style="font-size:0.85rem;">{clause["text"][:350]}...</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Audit timeline ───────────────────────────────────────────────────────
    audit_log = final_state.get("audit_log", [])
    with st.expander("Audit Log", expanded=False):
        st.markdown(_audit_timeline_html(audit_log), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline progress renderer
# ---------------------------------------------------------------------------

PIPELINE_NODES_ORDER = [
    ("ingestion",    "📥", "Ingestion"),
    ("pii_agent",    "🔍", "PII Detection"),
    ("bias_agent",   "⚖️", "Bias Analysis"),
    ("policy_agent", "📋", "Policy"),
    ("correction",   "🔧", "Correction"),
    ("scorecard",    "📊", "Scorecard"),
]

NODE_ICONS = {k: icon for k, icon, _ in PIPELINE_NODES_ORDER}


def _pipeline_bar_html(completed: dict[str, str], running: str | None) -> str:
    """
    Renders an animated pipeline progress bar.
    completed: {node_name: result_class}  e.g. {"ingestion": "pass"}
    running: node_name currently executing, or None
    """
    parts = []
    for i, (node_key, icon, label) in enumerate(PIPELINE_NODES_ORDER):
        if node_key in completed:
            cls = completed[node_key]
            status_icon = {"pass": "✓", "fail": "✗", "corrected": "⚡", "escalated": "🚨"}.get(cls, "•")
            sub = cls.upper()
        elif node_key == running:
            cls = "running"
            status_icon = icon
            sub = "Running…"
        else:
            cls = "pending"
            status_icon = icon
            sub = ""

        parts.append(
            f'<div class="pipeline-node">'
            f'  <div class="pipeline-node-circle {cls}">{status_icon}</div>'
            f'  <div class="pipeline-node-label">{label}</div>'
            f'  <div class="pipeline-node-sub">{sub}</div>'
            f'</div>'
        )
        if i < len(PIPELINE_NODES_ORDER) - 1:
            conn_cls = "active" if node_key in completed and completed[node_key] not in ("fail", "escalated") else (
                "fail" if node_key in completed else ""
            )
            parts.append(f'<div class="pipeline-connector {conn_cls}"></div>')

    return f'<div class="pipeline-wrapper">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Text annotation helpers
# ---------------------------------------------------------------------------

def _annotate_text_html(raw_text: str, pii_result: dict, policy_result: dict, corrected_text: str | None) -> str:
    """
    Returns HTML with PII spans highlighted red, policy-flagged phrases orange,
    and corrected segments struck-through green.
    Falls back gracefully when spans can't be located.
    """
    import html as html_lib

    safe = html_lib.escape(raw_text)

    # Build annotation markers from PII entities
    annotations = []
    for ent in (pii_result.get("entities_found") or []):
        start = ent.get("start")
        end = ent.get("end")
        etype = ent.get("entity_type", "PII")
        val = ent.get("value", "")
        if start is not None and end is not None and val:
            annotations.append(("pii", start, end, etype, val))

    if not annotations:
        # Fallback: bold-search entity values in escaped text
        result = safe
        for ent in (pii_result.get("entities_found") or []):
            val = html_lib.escape(str(ent.get("value", "")))
            etype = ent.get("entity_type", "PII")
            if val and val in result:
                result = result.replace(
                    val,
                    f'<span class="rai-highlight-pii" title="{etype}">{val}'
                    f'<span class="rai-citation">{etype}</span></span>',
                    1,
                )
        # Policy violations — highlight key phrases from description
        for v in (policy_result.get("violations") or []):
            vid = v.get("id", "")
            ref = v.get("article_reference", "")
            # Highlight first sentence of description as a proxy
            desc_words = str(v.get("description", "")).split()[:4]
            phrase = html_lib.escape(" ".join(desc_words))
            if phrase and phrase in result:
                result = result.replace(
                    phrase,
                    f'<span class="rai-highlight-policy" title="{vid}: {ref}">{phrase}'
                    f'<span class="rai-citation">{vid}</span></span>',
                    1,
                )
        return f'<div style="font-family:monospace;white-space:pre-wrap;line-height:1.8;padding:12px;background:#fafafa;border-radius:6px;border:1px solid #e9ecef;">{result}</div>'

    # Sort by start offset, render with spans
    annotations.sort(key=lambda x: x[1])
    result_parts = []
    cursor = 0
    for kind, start, end, label, val in annotations:
        if start > cursor:
            result_parts.append(html_lib.escape(raw_text[cursor:start]))
        span_text = html_lib.escape(raw_text[start:end])
        result_parts.append(
            f'<span class="rai-highlight-pii" title="{label}">{span_text}'
            f'<span class="rai-citation">{label}</span></span>'
        )
        cursor = end
    if cursor < len(raw_text):
        result_parts.append(html_lib.escape(raw_text[cursor:]))

    body = "".join(result_parts)
    return f'<div style="font-family:monospace;white-space:pre-wrap;line-height:1.8;padding:12px;background:#fafafa;border-radius:6px;border:1px solid #e9ecef;">{body}</div>'


# ---------------------------------------------------------------------------
# Audit timeline renderer
# ---------------------------------------------------------------------------

def _audit_timeline_html(audit_log: list[dict]) -> str:
    if not audit_log:
        return "<p style='color:#868e96;font-size:0.85rem;'>No audit events recorded.</p>"

    events = []
    for entry in audit_log:
        node = entry.get("node", "")
        label = NODE_LABELS.get(node, node)
        ts = (entry.get("timestamp") or "")
        ts_display = ts[11:19] if len(ts) >= 19 else ts  # HH:MM:SS only
        action = entry.get("action", "")
        result = entry.get("result", "")
        detail = entry.get("detail", "")
        correction_n = entry.get("correction_count", 0)

        res_cls = result if result in ("pass", "fail", "corrected", "escalated") else ""
        icon = {"pass": "✓", "fail": "✗", "corrected": "⚡", "escalated": "🚨"}.get(result, "·")

        detail_html = ""
        citation_html = ""
        if isinstance(detail, dict):
            pairs = list(detail.items())
            detail_html = "<br>".join(
                f'<span style="color:#868e96">{k}:</span> <strong>{str(v)[:150]}</strong>'
                for k, v in pairs[:6]
            )
            # Extract citation / reference if present
            ref = detail.get("article_reference") or detail.get("reference") or detail.get("ref", "")
            if ref:
                citation_html = f'<div class="audit-event-citation">📎 {ref}</div>'
        elif isinstance(detail, str) and detail:
            detail_html = detail[:300]

        correction_badge = (
            f'<span style="background:#fff3cd;color:#856404;padding:1px 5px;border-radius:3px;font-size:0.68rem;margin-left:6px;">Attempt #{correction_n}</span>'
            if correction_n > 0 else ""
        )

        result_colour = (
            "28a745" if result == "pass"
            else "dc3545" if result == "fail"
            else "ffc107" if result == "corrected"
            else "8B0000" if result == "escalated"
            else "868e96"
        )
        result_label = result.upper() if result else ""
        events.append(
            f'<div class="audit-event {res_cls}">'
            f'  <div class="audit-event-header">'
            f'    <span class="audit-event-node">{icon} {label}{correction_badge}</span>'
            f'    <span class="audit-event-time">{ts_display}</span>'
            f'  </div>'
            f'  <span class="audit-event-action">{action}</span>'
            f'  <span style="font-size:0.75rem;color:#{result_colour};font-weight:600;">{result_label}</span>'
            + (f'  <div class="audit-event-detail">{detail_html}</div>' if detail_html else "")
            + citation_html
            + f'</div>'
        )

    return f'<div class="audit-timeline">{"".join(events)}</div>'


# ---------------------------------------------------------------------------
# Streaming runner
# ---------------------------------------------------------------------------

def _run_text_audit_streaming(text: str):
    from graph import app
    from state import create_initial_state

    thread_id = str(uuid.uuid4())
    state = create_initial_state(input_type="text", raw_input=text)
    config = {"configurable": {"thread_id": thread_id}}

    pipeline_ph = st.empty()
    completed: dict[str, str] = {}
    running_node = None
    final_state = None
    all_steps = []

    for chunk in app.stream(state, config=config):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]

        # Mark previous running as done
        if running_node and running_node not in completed:
            audit_entries = node_state.get("audit_log", [])
            prev_entry = next(
                (e for e in reversed(audit_entries) if e.get("node") == running_node),
                None,
            )
            res = prev_entry.get("result", "pass") if prev_entry else "pass"
            completed[running_node] = res
            detail = prev_entry.get("detail", "") if prev_entry else ""
            all_steps.append((running_node, res, detail))

        running_node = node_name
        pipeline_ph.markdown(
            _pipeline_bar_html(completed, running_node),
            unsafe_allow_html=True,
        )
        final_state = node_state

    # Mark final node complete
    if running_node and final_state:
        audit_entries = final_state.get("audit_log", [])
        last_entry = next(
            (e for e in reversed(audit_entries) if e.get("node") == running_node),
            None,
        )
        res = last_entry.get("result", "pass") if last_entry else "pass"
        completed[running_node] = res
        detail = last_entry.get("detail", "") if last_entry else ""
        all_steps.append((running_node, res, detail))

    pipeline_ph.markdown(
        _pipeline_bar_html(completed, None),
        unsafe_allow_html=True,
    )

    if not final_state:
        st.error("Graph returned no output.")
        return

    st.session_state["pipeline_completed"] = completed
    st.session_state["pipeline_log"] = all_steps
    st.session_state["text_audit_result"] = final_state
    st.session_state["history_thread_id"] = thread_id


# ---------------------------------------------------------------------------
# Tab 2 — Model Auditor
# ---------------------------------------------------------------------------

def render_model_auditor():
    from ui.components.radar_chart import render_radar_chart

    st.subheader("Model Auditor")
    st.markdown("Upload a CSV with model predictions to check for algorithmic bias.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        protected_attr = st.text_input(
            "Protected attribute column",
            placeholder="e.g. sex",
        )
        prediction_col = st.text_input(
            "Prediction column",
            placeholder="e.g. target",
        )
        run_btn = st.button("Run Bias Audit", type="primary", use_container_width=True, key="audit_model_btn")

    if run_btn and uploaded_file and protected_attr and prediction_col:
        _run_model_audit(uploaded_file, protected_attr, prediction_col)
    elif run_btn:
        st.warning("Upload CSV and fill in both column names.")

    # Render persisted result (survives across Streamlit reruns)
    final = st.session_state.get("model_audit_result")
    if not final:
        return

    fs = final.get("final_status", "UNKNOWN")
    st.markdown(_status_badge_html(fs), unsafe_allow_html=True)

    bias = final.get("bias_result") or {}
    if bias.get("disparate_impact_ratio") is not None:
        st.markdown("### Fairness Metrics")
        di = bias["disparate_impact_ratio"]
        di_ok = di >= 0.8
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Disparate Impact",
            f"{di:.3f}",
            delta="✓ ≥0.8" if di_ok else "✗ <0.8 FAIL",
            delta_color="normal" if di_ok else "inverse",
        )
        dp = bias.get("demographic_parity_diff")
        if dp is not None:
            dp_ok = abs(dp) <= 0.1
            c2.metric(
                "Dem. Parity Diff",
                f"{dp:.3f}",
                delta="✓" if dp_ok else "✗ FAIL",
                delta_color="normal" if dp_ok else "inverse",
            )
        eo = bias.get("equalized_odds_diff")
        if eo is not None:
            eo_ok = abs(eo) <= 0.1
            c3.metric(
                "Eq. Odds Diff",
                f"{eo:.3f}",
                delta="✓" if eo_ok else "✗ FAIL",
                delta_color="normal" if eo_ok else "inverse",
            )
        if bias.get("privileged_group"):
            st.caption(f"Privileged: {bias['privileged_group']} · Unprivileged: {bias.get('unprivileged_group')}")

    explainability = final.get("explainability_result") or {}
    top_features = explainability.get("top_features", [])
    if top_features:
        with st.expander("Explainability — SHAP Feature Importance", expanded=True):
            st.caption(explainability.get("explanation_text", ""))

            import plotly.graph_objects as go
            features = [f["feature"] for f in top_features]
            shap_vals = [f["shap_value"] for f in top_features]
            colours = ["#dc3545" if v > 0 else "#28a745" for v in shap_vals]

            fig = go.Figure(go.Bar(
                x=shap_vals,
                y=features,
                orientation="h",
                marker_color=colours,
            ))
            fig.update_layout(
                title="SHAP Values — Impact on Risk Score",
                xaxis_title="SHAP value (positive = increases risk)",
                yaxis=dict(autorange="reversed"),
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                pd.DataFrame(top_features)[["feature", "shap_value", "direction"]],
                use_container_width=True,
                hide_index=True,
            )

    retrieved_m = final.get("retrieved_clauses") or []
    if retrieved_m:
        with st.expander(f"Regulatory Sources — {len(retrieved_m)} clauses retrieved", expanded=False):
            st.caption(
                "These regulatory clauses were retrieved from the vector store based on semantic "
                "relevance to the audited text. They grounded the policy compliance analysis."
            )
            for i, clause in enumerate(retrieved_m, 1):
                sim_pct = int(clause.get("similarity_score", 0) * 100)
                colour = "#28a745" if sim_pct >= 60 else "#ffc107" if sim_pct >= 40 else "#6c757d"
                st.markdown(
                    f'<div style="border-left:3px solid {colour};padding:8px 12px;margin-bottom:8px;">'
                    f'<strong>{clause["regulation"]} — {clause["article_id"]}</strong> '
                    f'<span style="color:{colour};font-size:0.8rem;">({sim_pct}% relevance)</span><br>'
                    f'<span style="font-size:0.85rem;color:#555;">{clause["reference"]}</span><br><br>'
                    f'<span style="font-size:0.85rem;">{clause["text"][:350]}...</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    scores = final.get("rai_scores")
    if scores:
        st.markdown("### RAI Pillar Scores")
        st.markdown(_pillar_score_html(scores), unsafe_allow_html=True)
        fig = render_radar_chart(scores)
        st.plotly_chart(fig, use_container_width=True)

    correction_count = final.get("correction_count", 0)
    original_text = final.get("_original_text", "")
    corrected_text = final.get("current_text", "")
    if correction_count > 0 and corrected_text and corrected_text != original_text:
        st.markdown("### Before / After Correction")
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("**Before**")
            st.code(original_text)
        with col_after:
            st.markdown("**After**")
            st.code(corrected_text)

    audit_log = final.get("audit_log", [])
    if audit_log:
        with st.expander("Pipeline Execution Log", expanded=False):
            rows = []
            for entry in audit_log:
                rows.append({
                    "Time": (entry.get("timestamp") or "")[:19].replace("T", " "),
                    "Node": NODE_LABELS.get(entry.get("node", ""), entry.get("node", "")),
                    "Action": entry.get("action", ""),
                    "Result": entry.get("result", "").upper(),
                    "Detail": (
                        " · ".join(f"{k}: {v}" for k, v in list(entry["detail"].items())[:4])
                        if isinstance(entry.get("detail"), dict)
                        else str(entry.get("detail", ""))[:200]
                    ),
                    "Corrections": entry.get("correction_count", 0),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _run_model_audit(uploaded_file, protected_attr, prediction_col):
    from graph import app
    from state import create_initial_state

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if protected_attr not in df.columns:
        st.error(f"Column '{protected_attr}' not found. Available: {list(df.columns)}")
        return

    sample = df.head(1)
    feature_vector = sample.drop(columns=[prediction_col], errors="ignore").iloc[0].to_dict()
    prediction = float(sample[prediction_col].iloc[0]) if prediction_col in df.columns else 0.5

    fname = getattr(uploaded_file, "name", "")
    if "fair" in fname.lower():
        from pathlib import Path as _P
        feature_vector["_model_path"] = str(_P(__file__).parent.parent / "models" / "loan_model_fair.pkl")
    elif "compas" in fname.lower():
        from pathlib import Path as _P
        feature_vector["_model_path"] = str(_P(__file__).parent.parent / "models" / "compas_model.pkl")

    state = create_initial_state(
        input_type="model_output",
        raw_input=f"Model prediction: {prediction:.3f}",
        feature_vector=feature_vector,
        protected_attributes=[protected_attr],
        prediction=prediction,
        predicted_label=int(round(prediction)),
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    with st.spinner("Running bias audit..."):
        final = app.invoke(state, config=config)

    final["_original_text"] = f"Model prediction: {prediction:.3f}"
    st.session_state["model_audit_result"] = final


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Text Auditor", "Model Auditor"])

with tab1:
    render_text_auditor()

with tab2:
    render_model_auditor()
