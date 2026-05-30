"""
ui/app.py
---------
Streamlit application for the RAI Compliance Agent.

Run with:
  cd rai-compliance-agent
  streamlit run ui/app.py
"""

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*torch.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)

import json
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
    "FAIL":      "#dc3545",
    "CORRECTED": "#17a2b8",
    "ESCALATED": "#6f42c1",
}

STATUS_TEXT_COLOURS = {
    "PASS":      "#fff",
    "FAIL":      "#fff",
    "CORRECTED": "#fff",
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
        icon = "✅" if score == 3 else ("⚠️" if score == 2 else "❌")
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
            f'<span style="width:22px;font-size:0.9rem;">{icon}</span>'
            f'<span style="width:150px;font-size:0.85rem;">{label}</span>'
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

_DEMO_SCENARIOS = {
    "Complete AI Policy (→ PASS)": (
        "RESPONSIBLE AI GOVERNANCE POLICY — Version 3.1\n\n"
        "1. GOVERNANCE & ACCOUNTABILITY\n"
        "The Board of Directors has established an AI Ethics Committee (AIEC), chaired by the Chief Risk Officer, "
        "with representation from Legal, Technology, and Business Operations. The AIEC maintains a centralised AI "
        "model registry that documents every AI system in production use, including its purpose, owner, risk tier, "
        "and review schedule. Accountability for each AI system rests with a named Model Owner who is responsible "
        "for compliance with this policy. Human override of any AI-assisted decision is mandatory and must be "
        "executed within 48 hours of a formal request by an affected party.\n\n"
        "2. FAIRNESS & BIAS MITIGATION\n"
        "All AI models must undergo a pre-deployment fairness assessment covering disparate impact analysis, "
        "demographic parity, and equalized odds across protected characteristics. Training datasets are subject "
        "to a mandatory historical bias review; any identified proxy variables must be flagged and either removed "
        "or explicitly justified with Board-level sign-off. Fairness metrics and acceptance thresholds are defined "
        "in the model's Technical Specification Document prior to deployment. Models failing to meet the "
        "disparate impact threshold of 0.8 must not be deployed without remediation.\n\n"
        "3. TRANSPARENCY & EXPLAINABILITY\n"
        "Any individual whose application, eligibility, or standing is influenced by an AI system must be "
        "informed of this in plain language at the point of engagement. Upon request, a plain-language explanation "
        "citing the primary factors behind the decision must be provided within 15 business days. Model cards are "
        "maintained for all production systems and made available to relevant regulators on request. AI-generated "
        "outputs used in customer communications are labelled as such.\n\n"
        "4. ROBUSTNESS & MONITORING\n"
        "Performance thresholds are defined for every production model and monitored monthly. Automated alerts "
        "trigger when performance degrades beyond defined bounds. All models are subject to annual formal "
        "re-validation; models with high business impact undergo quarterly review. An incident response playbook "
        "governs detection, escalation, and remediation of model failures, with a target resolution time of 24 hours "
        "for high-severity incidents.\n\n"
        "5. PRIVACY & DATA STEWARDSHIP\n"
        "Personal data used in AI model training is processed only with a documented lawful basis; where the data "
        "originates from Indian residents, explicit consent per DPDPA 2023 Section 6 is obtained and logged. "
        "Purpose limitation is enforced — data collected for one purpose may not be repurposed for AI model "
        "training without a fresh consent review. All third-party AI tools and vendors are required to execute "
        "data processing agreements that include obligations on data minimisation and prohibitions on secondary use "
        "of training data."
    ),
    "Vague Tech Policy — Missing Governance & Privacy (→ FAIL)": (
        "TECHNOLOGY USE POLICY — Section 7: Artificial Intelligence\n\n"
        "7.1 The organisation recognises that artificial intelligence and machine learning technologies offer "
        "significant opportunities to improve operational efficiency and customer outcomes. We are committed to "
        "adopting AI in a responsible manner consistent with our values.\n\n"
        "7.2 AI tools deployed within the organisation must be evaluated by the Technology team prior to adoption. "
        "The evaluation process considers factors including vendor credibility, data security, integration "
        "complexity, and total cost of ownership.\n\n"
        "7.3 All AI deployments must comply with applicable laws and regulations. Business units wishing to "
        "adopt an AI solution must submit a use-case brief to Technology for review.\n\n"
        "7.4 Employees are reminded that AI tools may produce errors and that human judgement should be applied "
        "when interpreting AI outputs. Staff are encouraged to report any concerns regarding AI behaviour to "
        "their line manager.\n\n"
        "7.5 The organisation will continue to monitor developments in AI regulation and update this policy "
        "as required. This policy will be reviewed annually by the Technology Governance Committee.\n\n"
        "7.6 For questions regarding this policy, please contact the Technology Risk team at "
        "technology.risk@company.com."
    ),
    "HR Policy with No Bias Testing Clause (→ FAIL)": (
        "TALENT ACQUISITION POLICY — Section 4: Use of Technology in Hiring\n\n"
        "4.1 Purpose\n"
        "This section governs the use of technology-assisted tools in recruitment and hiring decisions. "
        "The organisation is committed to fair and efficient hiring practices that attract diverse talent.\n\n"
        "4.2 AI-Assisted Screening\n"
        "The organisation may use AI-powered applicant tracking and CV screening tools to manage high volumes "
        "of applications. These tools assist hiring managers by surfacing relevant candidates based on "
        "qualifications and experience. AI screening scores are provided as a guide only; final shortlisting "
        "decisions are made by a qualified HR Business Partner.\n\n"
        "4.3 Video Interview Technology\n"
        "Where asynchronous video interviews are used, AI may be used to transcribe responses and flag key "
        "competency indicators. All video assessments are reviewed by a human interviewer before any candidate "
        "is progressed or rejected.\n\n"
        "4.4 Data Retention\n"
        "Application data and assessment results are retained for 12 months in accordance with our Data "
        "Retention Schedule. Candidate data is not shared with third parties without explicit consent.\n\n"
        "4.5 Appeals\n"
        "Candidates who believe they have been unfairly treated during the recruitment process may raise a "
        "formal complaint with the HR Director within 30 days of notification of an outcome.\n\n"
        "4.6 Review\n"
        "This policy is reviewed annually and approved by the Chief People Officer."
    ),
    "Data Policy Missing AI Training Consent (→ FAIL)": (
        "DATA GOVERNANCE FRAMEWORK — Chapter 3: Data Use and Processing\n\n"
        "3.1 Principles\n"
        "The organisation collects and processes personal data in accordance with applicable data protection "
        "laws including the Information Technology Act 2000, the DPDPA 2023, and any sector-specific "
        "requirements from the Reserve Bank of India. We are committed to data minimisation, accuracy, "
        "storage limitation, and integrity.\n\n"
        "3.2 Lawful Basis\n"
        "All personal data processing activities must have a documented lawful basis prior to commencement. "
        "The primary lawful bases used are consent, contract performance, legal obligation, and legitimate "
        "interest. The Data Protection Officer maintains a processing activity register for all active "
        "data use cases.\n\n"
        "3.3 Data Subject Rights\n"
        "Data subjects have the right to access, correct, and erase their personal data in accordance with "
        "applicable law. Requests must be acknowledged within 72 hours and fulfilled within 30 days. "
        "The organisation will not charge for routine access requests.\n\n"
        "3.4 Third-Party Data Sharing\n"
        "Personal data may be shared with third-party service providers only under a Data Processing Agreement "
        "that mandates equivalent data protection standards. Cross-border transfers are subject to Transfer "
        "Impact Assessments.\n\n"
        "3.5 Retention and Deletion\n"
        "Personal data is retained only as long as necessary for its stated processing purpose. Retention "
        "schedules are maintained by the Data Governance team and reviewed annually. Secure deletion "
        "procedures apply to all end-of-life data.\n\n"
        "3.6 Security\n"
        "Technical and organisational measures are in place to protect personal data from unauthorised "
        "access, disclosure, and loss, including encryption at rest and in transit, access controls, "
        "and annual security reviews."
    ),
}


def render_text_auditor():
    from ui.components.radar_chart import render_radar_chart
    from state import create_initial_state

    st.subheader("Policy Document Auditor")
    st.markdown(
        "Upload a company AI, HR, or data governance policy to audit its coverage across "
        "the 5 Responsible AI pillars: Governance, Fairness, Transparency, Robustness, and Privacy."
    )

    col_input, col_result = st.columns([1, 1])

    with col_input:
        tab1, tab2, tab3 = st.tabs(["📋 Paste Text", "📁 Upload File", "🎯 Demo Scenarios"])

        with tab1:
            text_input = st.text_area(
                "Paste policy text",
                height=220,
                placeholder="Paste a policy document excerpt here...",
            )
            run_btn = st.button("Audit Policy", type="primary", use_container_width=True, key="audit_text_btn")
            if run_btn and text_input.strip():
                st.session_state.pop("suggestions_result", None)
                _run_and_display(create_initial_state(input_type="policy_document", raw_input=text_input))
            elif run_btn:
                st.warning("Enter some text first.")

        with tab2:
            st.markdown(
                "Upload a policy document for RAI pillar audit. Supported: PDF, Word (.docx), plain text (.txt), Markdown (.md)"
            )
            uploaded_file = st.file_uploader(
                label="Choose a policy document",
                type=["pdf", "docx", "txt", "md"],
                help="Text is extracted and evaluated against the 5 RAI governance pillars.",
            )
            if uploaded_file is not None:
                st.caption(f"File: {uploaded_file.name}  ·  Size: {uploaded_file.size / 1024:.1f} KB")
                try:
                    from utils.file_parser import extract_text
                    file_bytes = uploaded_file.read()
                    extracted_text, file_type = extract_text(file_bytes, uploaded_file.name)

                    with st.expander("Preview extracted text", expanded=False):
                        st.text(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
                        st.caption(f"Total: {len(extracted_text):,} characters extracted")

                    if st.button("Run Policy Audit", key="file_audit_btn", type="primary"):
                        st.session_state.pop("suggestions_result", None)
                        _run_and_display(create_initial_state(
                            input_type="policy_document",
                            raw_input=extracted_text,
                            source_filename=uploaded_file.name,
                            source_file_type=file_type,
                        ))
                except (ValueError, RuntimeError) as e:
                    st.error(f"Could not extract text: {e}")
                except Exception as e:
                    st.error(f"Unexpected error reading file: {e}")

        with tab3:
            st.markdown("Run a pre-built policy document scenario through the RAI pillar audit.")
            for label, text in _DEMO_SCENARIOS.items():
                if st.button(label, use_container_width=True, key=f"demo_{label[:20]}"):
                    st.session_state.pop("suggestions_result", None)
                    _run_and_display(create_initial_state(input_type="policy_document", raw_input=text))

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
    pii = final_state.get("pii_result") or {}
    policy = final_state.get("policy_result") or {}

    # Source file header (upload mode only)
    src_name = final_state.get("source_filename")
    src_type = final_state.get("source_file_type")
    raw_len = len(final_state.get("raw_input", ""))
    if src_name:
        st.caption(f"Source: {src_name} ({src_type}) — {raw_len:,} characters")

    # Status + radar side by side
    with col_result:
        st.markdown(_status_badge_html(fs), unsafe_allow_html=True)
        if scores:
            fig = render_radar_chart(scores)
            st.plotly_chart(fig, use_container_width=True)

    # Escalation banner — shown prominently when human review is required
    escalation_reason = final_state.get("escalation_reason")
    if fs == "ESCALATED" and escalation_reason:
        st.markdown(
            f'<div style="background:#f3e8ff;border-left:4px solid #6f42c1;padding:12px 16px;'
            f'border-radius:0 6px 6px 0;margin:10px 0;">'
            f'<strong style="color:#6f42c1">🚨 Human Review Required</strong><br>'
            f'<span style="font-size:0.9rem;color:#3d1f6e;">{escalation_reason}</span><br>'
            f'<span style="font-size:0.8rem;color:#555;margin-top:4px;display:block;">'
            f'Regulatory basis: EU AI Act Art. 14 (human oversight) / NIST MANAGE 1.3</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", fs)
    policy_violations = len(policy.get("violations") or [])
    pii_flag = 1 if (final_state.get("pii_result") or {}).get("has_pii") else 0
    total_violations = len(set(violations)) or (policy_violations + pii_flag)
    c2.metric("Violations Found", total_violations)
    correction_count = final_state.get("correction_count", 0)
    c3.metric(
        "Overall Score",
        f"{sum(scores.values()) if scores else 0}/15",
        delta=f"{correction_count} correction(s)" if correction_count > 0 else None,
    )

    is_policy_doc = final_state.get("input_type") == "policy_document"
    pillar_gaps = final_state.get("pillar_gaps") or {}

    if scores:
        if is_policy_doc and pillar_gaps:
            # Rich per-pillar breakdown with pass/fail reasoning
            st.markdown("### RAI Pillar Assessment")
            _render_pillar_breakdown(scores, pillar_gaps, final_state)
        else:
            st.markdown("**RAI Pillar Scores**")
            st.markdown(_pillar_score_html(scores), unsafe_allow_html=True)
            if is_policy_doc and not pillar_gaps:
                st.caption(
                    "ℹ️ Per-pillar reasoning not available for this run. "
                    "Re-run the audit to see detailed Gemma 4 explanations per pillar."
                )

    suggestions_result = st.session_state.get("suggestions_result")
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

    if suggestions_result:
        _render_suggestions(suggestions_result)

    st.divider()

    # ── Annotated text view ──────────────────────────────────────────────────
    raw_text = final_state.get("raw_input", "")

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
            _annotate_text_html(raw_text, pii, policy, None),
            unsafe_allow_html=True,
        )

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

    # ── Step-by-step audit trail ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Audit Steps")
    st.caption(
        "Each pillar check ran once in sequence. "
        "Expand any step to see the full input, prompt, and response."
    )

    step_trace = final_state.get("step_trace", [])

    STEP_ICONS = {
        "pii_agent":            "🔍",
        "bias_agent":           "⚖️",
        "explainability_agent": "💡",
        "policy_agent":         "📋",
        "scorecard":            "📊",
    }

    for step in step_trace:
        if step["step"] == "ingestion":
            continue

        icon = STEP_ICONS.get(step["step"], "•")
        label = step["label"]
        status = step["status"]
        summary = step["summary"]
        status_color = "#28a745" if status == "pass" else "#dc3545"
        status_label = "PASS" if status == "pass" else "FAIL"

        with st.expander(f"{icon} {label} — {status_label}", expanded=(status == "fail")):
            st.markdown(
                f'<span style="color:{status_color};font-weight:700;">{status_label}</span>'
                f' — {summary}',
                unsafe_allow_html=True,
            )

            resp = step.get("response", {})
            if step["step"] == "policy_agent":
                # FIX 3: human-readable policy agent rendering
                pillar_scores_raw = resp.get("pillar_scores", [])
                summary = resp.get("summary", "")
                if summary:
                    st.markdown("**Gemma 4 overall assessment:**")
                    st.info(summary)
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
                if step.get("prompt"):
                    with st.expander("View raw Gemma 4 prompt", expanded=False):
                        st.code(step["prompt"], language="text")
                    with st.expander("View raw Gemma 4 JSON response", expanded=False):
                        import json as _json
                        st.code(_json.dumps(resp, indent=2), language="json")

            elif step.get("prompt"):
                st.markdown("**Prompt sent to Gemma 4:**")
                st.code(step["prompt"], language="text")
                st.markdown("**Response from Gemma 4:**")
                import json as _json
                st.code(
                    _json.dumps(resp, indent=2) if isinstance(resp, dict) else str(resp),
                    language="json",
                )
            else:
                if step["step"] == "pii_agent":
                    # FIX 5: detailed PII explanation
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
                    entities = resp.get("entities_found", [])
                    st.markdown(f"**Scanned for:** {', '.join(resp.get('entities_scanned_for', []))}")
                    st.markdown(f"**Confidence threshold:** {resp.get('confidence_threshold')}")
                    if entities:
                        st.markdown(f"**Entities found ({len(entities)}):**")
                        df_pii = pd.DataFrame(entities)
                        cols = [c for c in ["entity_type", "value", "score"] if c in df_pii.columns]
                        st.dataframe(df_pii[cols], hide_index=True, use_container_width=True)
                        if resp.get("redacted_text"):
                            st.markdown("**Redacted output:**")
                            st.code(resp["redacted_text"])
                    else:
                        st.success("No PII entities found.")

                elif step["step"] == "bias_agent":
                    # FIX 4: detailed fairness explanation
                    mode = resp.get("mode", "text")
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
                    else:
                        metrics = resp.get("metrics", {})
                        thresholds = resp.get("thresholds", {})
                        bcols = st.columns(3)
                        di = metrics.get("disparate_impact_ratio")
                        dp = metrics.get("demographic_parity_diff")
                        eo = metrics.get("equalized_odds_diff")
                        if di is not None:
                            ok = di >= thresholds.get("disparate_impact", 0.8)
                            bcols[0].metric("Disparate Impact Ratio", f"{di:.3f}",
                                            delta=f"threshold ≥ {thresholds.get('disparate_impact', 0.8)}",
                                            delta_color="normal" if ok else "inverse")
                        if dp is not None:
                            ok = abs(dp) <= thresholds.get("demographic_parity", 0.1)
                            bcols[1].metric("Demographic Parity Diff", f"{dp:.3f}",
                                            delta=f"threshold ≤ {thresholds.get('demographic_parity', 0.1)}",
                                            delta_color="normal" if ok else "inverse")
                        if eo is not None:
                            ok = abs(eo) <= thresholds.get("equalized_odds", 0.1)
                            bcols[2].metric("Equalized Odds Diff", f"{eo:.3f}",
                                            delta=f"threshold ≤ {thresholds.get('equalized_odds', 0.1)}",
                                            delta_color="normal" if ok else "inverse")
                        pg = resp.get("privileged_group")
                        ug = resp.get("unprivileged_group")
                        if pg:
                            st.caption(f"Privileged group: {pg} · Unprivileged group: {ug}")

                elif step["step"] == "explainability_agent":
                    features = resp.get("top_features", [])
                    if features:
                        st.markdown("**Top features by SHAP value (|impact| ranked):**")
                        df_shap = pd.DataFrame(features)
                        st.dataframe(df_shap, hide_index=True, use_container_width=True)
                        st.markdown(f"**Plain-English explanation:** {resp.get('explanation_text', '')}")
                    else:
                        st.info(resp.get("explanation_text", "No SHAP values computed."))

                elif step["step"] == "scorecard":
                    scores_s = resp.get("rai_scores", {})
                    if scores_s:
                        st.markdown(_pillar_score_html(scores_s), unsafe_allow_html=True)
                    st.markdown(
                        f"**Overall:** {resp.get('overall_score', 0)}/{resp.get('max_possible', 15)} "
                        f"· **Violations:** {resp.get('violations_total', 0)}"
                    )

    # ── Audit timeline ───────────────────────────────────────────────────────
    audit_log = final_state.get("audit_log", [])
    with st.expander("Raw Audit Log", expanded=False):
        st.markdown(_audit_timeline_html(audit_log), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Suggestion helpers
# ---------------------------------------------------------------------------

def _render_pillar_breakdown(scores: dict, pillar_gaps: dict, final_state: dict):
    """
    Renders a per-pillar card showing score, evidence from the policy,
    pass/fail reasoning, and an individual 'Suggest Fix' button.
    Pillar order matches the 5 RAI governance criteria.
    """
    CRITERION_ORDER = [
        ("GOVERNANCE_ACCOUNTABILITY",    "strategic_alignment",   "🏛️ Governance & Accountability"),
        ("FAIRNESS_BIAS",                "model_governance",      "⚖️ Fairness & Bias"),
        ("TRANSPARENCY_EXPLAINABILITY",  "org_oversight",         "🔍 Transparency & Explainability"),
        ("ROBUSTNESS_MONITORING",        "continuous_monitoring", "🛡️ Robustness & Monitoring"),
        ("PRIVACY_DATA_STEWARDSHIP",     "data_governance",       "🔒 Privacy & Data Stewardship"),
    ]

    for criterion_id, pillar_key, display_name in CRITERION_ORDER:
        score = scores.get(pillar_key, 0)
        gap_data = pillar_gaps.get(criterion_id, {})
        gap = gap_data.get("gap", "")
        evidence = gap_data.get("evidence", "")
        pass_reason = gap_data.get("pass_reason", "")
        reference = gap_data.get("reference", "")

        # Header colour
        if score == 3:
            border_color = "#28a745"
            status_icon = "✅"
            status_label = "PASS"
        elif score == 2:
            border_color = "#ffc107"
            status_icon = "⚠️"
            status_label = "PARTIAL"
        elif score == 1:
            border_color = "#fd7e14"
            status_icon = "⚠️"
            status_label = "MINIMAL"
        else:
            border_color = "#dc3545"
            status_icon = "❌"
            status_label = "ABSENT"

        bar_pct = int(score / 3 * 100)
        bar_html = (
            f'<div style="height:8px;border-radius:4px;background:#dee2e6;margin:4px 0 8px 0;">'
            f'<div style="width:{bar_pct}%;height:100%;background:{border_color};border-radius:4px;"></div>'
            f'</div>'
        )

        with st.expander(
            f"{status_icon} {display_name} — **{score}/3** ({status_label})",
            expanded=(score < 3),
        ):
            st.markdown(bar_html, unsafe_allow_html=True)

            col_l, col_r = st.columns([1, 1])

            with col_l:
                if evidence and evidence.lower() not in ("not found in policy.", "", "none"):
                    st.markdown("**What the policy says:**")
                    st.markdown(
                        f'<div style="background:#f0f4ff;border-left:3px solid #2E75B6;'
                        f'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.85rem;'
                        f'font-style:italic;">"{evidence}"</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background:#fff3cd;border-left:3px solid #ffc107;'
                        'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.85rem;">'
                        '⚠️ No relevant text found in policy.</div>',
                        unsafe_allow_html=True,
                    )

            with col_r:
                if score >= 2 and pass_reason:
                    st.markdown("**Why it passed:**")
                    st.markdown(
                        f'<div style="background:#d4edda;border-left:3px solid #28a745;'
                        f'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.85rem;">'
                        f'✅ {pass_reason}</div>',
                        unsafe_allow_html=True,
                    )
                if score < 3 and gap:
                    st.markdown("**Why it failed / what's missing:**")
                    st.markdown(
                        f'<div style="background:#f8d7da;border-left:3px solid #dc3545;'
                        f'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.85rem;">'
                        f'❌ {gap}</div>',
                        unsafe_allow_html=True,
                    )

            if reference:
                st.markdown(
                    f'<div style="margin-top:6px;font-size:0.78rem;color:#555;">📎 <strong>Regulatory basis:</strong> {reference}</div>',
                    unsafe_allow_html=True,
                )

            # Per-pillar suggestion button (only for failing pillars)
            if score < 3:
                btn_key = f"suggest_{criterion_id}"
                if st.button(f"💡 Suggest fix for this pillar", key=btn_key, type="secondary"):
                    _run_single_pillar_suggestion(final_state, criterion_id, pillar_key)


def _run_single_pillar_suggestion(final_state: dict, criterion_id: str, pillar_key: str):
    """
    Generate and display improvement suggestion for a single failing pillar.
    Calls the audited wrapper so the action is recorded in rai_audit.db.
    """
    import copy
    # Build a sliced state with only the target pillar so the agent focuses on it
    sliced = copy.copy(final_state)
    sliced["rai_scores"] = {pillar_key: final_state.get("rai_scores", {}).get(pillar_key, 0)}
    thread_id = st.session_state.get("history_thread_id")
    with st.spinner(f"Generating suggestion for {criterion_id}..."):
        try:
            from nodes.suggestion_agent import run_suggestion_agent_with_audit
            suggestions = run_suggestion_agent_with_audit(sliced, thread_id=thread_id)
            # Merge with any existing suggestions for other pillars
            existing = st.session_state.get("suggestions_result") or []
            existing_ids = {s.get("pillar_id") for s in existing}
            for s in suggestions:
                pid = s.get("pillar_id")
                if pid not in existing_ids:
                    existing.append(s)
                else:
                    existing = [s if x.get("pillar_id") == pid else x for x in existing]
            st.session_state["suggestions_result"] = existing
        except Exception as e:
            st.error(f"Could not generate suggestion: {e}")
    st.rerun()


def _run_suggestions(final_state: dict):
    """
    Generate improvement suggestions for all failing pillars.
    Calls the audited wrapper so the action is recorded in rai_audit.db.
    """
    thread_id = st.session_state.get("history_thread_id")
    with st.spinner("Generating improvement suggestions..."):
        try:
            from nodes.suggestion_agent import run_suggestion_agent_with_audit
            suggestions = run_suggestion_agent_with_audit(final_state, thread_id=thread_id)
            st.session_state["suggestions_result"] = suggestions
        except Exception as e:
            st.error(f"Could not generate suggestions: {e}")
    st.rerun()


def _render_suggestions(suggestions: list[dict]):
    st.markdown("---")
    st.markdown("### 💡 Suggested Policy Improvements")
    st.caption(
        "These suggestions are generated by Gemma 4 and mapped to the RAI framework. "
        "Copy-paste the suggested language directly into your policy document."
    )
    if not suggestions:
        st.info("No gaps found — all audited pillars already meet suggested standards.")
        return
    for s in suggestions:
        score = s.get("current_score", 0)
        icon = "🔴" if score == 0 else ("🟠" if score == 1 else "🟡")
        pillar_name = s.get("pillar_name") or s.get("pillar_id", "Unknown Pillar")
        with st.expander(f"{icon} {pillar_name} — Current Score: {score}/3", expanded=True):
            gap = s.get("gap_description", "")
            if gap:
                st.markdown(
                    f'<div style="background:#f8d7da;border-left:3px solid #dc3545;'
                    f'padding:8px 12px;margin-bottom:10px;border-radius:0 4px 4px 0;font-size:0.87rem;">'
                    f'❌ <strong>Gap:</strong> {gap}</div>',
                    unsafe_allow_html=True,
                )
            lang_list = s.get("suggested_language", [])
            if lang_list:
                st.markdown("**Suggested policy language to add:**")
                for i, lang in enumerate(lang_list, 1):
                    st.markdown(
                        f'<div style="background:#f0f4ff;border-left:3px solid #2E75B6;'
                        f'padding:10px 14px;margin:6px 0;border-radius:0 4px 4px 0;">'
                        f'<span style="font-size:0.75rem;color:#2E75B6;font-weight:600;">CLAUSE {i}</span><br>'
                        f'<span style="font-style:italic;font-size:0.88rem;">&#8220;{lang}&#8221;</span></div>',
                        unsafe_allow_html=True,
                    )
            reg = s.get("regulatory_basis", "")
            if reg:
                st.markdown(
                    f'<div style="margin-top:8px;font-size:0.8rem;color:#555;">📎 <strong>Regulatory basis:</strong> {reg}</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Pipeline progress renderer
# ---------------------------------------------------------------------------

PIPELINE_NODES_ORDER = [
    ("ingestion",            "📥", "Ingestion"),
    ("pii_agent",            "🔍", "PII Detection"),
    ("bias_agent",           "⚖️",  "Bias Analysis"),
    ("explainability_agent", "💡", "Explainability"),
    ("policy_agent",         "📋", "Policy"),
    ("correction",           "🔧", "Correction"),
    ("escalation",           "🚨", "Escalation"),
    ("scorecard",            "📊", "Scorecard"),
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
            status_icon = {"pass": "✓", "fail": "✗"}.get(cls, "•")
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

        res_cls = result if result in ("pass", "fail") else ""
        icon = {"pass": "✓", "fail": "✗"}.get(result, "·")

        detail_html = ""
        citation_html = ""
        if isinstance(detail, dict):
            pairs = list(detail.items())
            detail_html = "<br>".join(
                f'<span style="color:#868e96">{k}:</span> <strong>{str(v)[:150]}</strong>'
                for k, v in pairs[:6]
            )
            ref = detail.get("article_reference") or detail.get("reference") or detail.get("ref", "")
            if ref:
                citation_html = f'<div class="audit-event-citation">📎 {ref}</div>'
        elif isinstance(detail, str) and detail:
            detail_html = detail[:300]

        result_colour = "28a745" if result == "pass" else "dc3545" if result == "fail" else "868e96"
        result_label = result.upper() if result else ""
        events.append(
            f'<div class="audit-event {res_cls}">'
            f'  <div class="audit-event-header">'
            f'    <span class="audit-event-node">{icon} {label}</span>'
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

def _run_and_display(initial_state: dict):
    from graph import app

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    pipeline_ph = st.empty()
    completed: dict[str, str] = {}
    running_node = None
    node_results: dict[str, str] = {}

    for chunk in app.stream(initial_state, config=config):
        node_name = list(chunk.keys())[0]
        node_state = chunk[node_name]   # partial update only — DO NOT use as final_state

        # Determine result for the node that just finished by inspecting its
        # audit_log entry — this is the authoritative per-node result.
        audit_entries = node_state.get("audit_log", [])
        entry = next((e for e in reversed(audit_entries) if e.get("node") == node_name), None)
        res = entry.get("result", "pass") if entry else "pass"
        node_results[node_name] = res

        # Mark previous running node as complete, advance to this one
        if running_node and running_node not in completed:
            completed[running_node] = node_results.get(running_node, "pass")

        running_node = node_name
        pipeline_ph.markdown(_pipeline_bar_html(completed, running_node), unsafe_allow_html=True)

    # Mark final node complete
    if running_node:
        completed[running_node] = node_results.get(running_node, "pass")

    pipeline_ph.markdown(_pipeline_bar_html(completed, None), unsafe_allow_html=True)

    # Retrieve the FULL accumulated state from the LangGraph checkpoint.
    # app.stream() yields partial updates per node — the final chunk contains
    # only what the terminal node returned, missing all prior agents' results.
    try:
        full_state = app.get_state(config).values
    except Exception as e:
        st.error(f"Could not retrieve final state: {e}")
        return

    if not full_state:
        st.error("Graph returned no output.")
        return

    st.session_state["pipeline_completed"] = completed
    st.session_state["pipeline_log"] = []
    st.session_state["text_audit_result"] = full_state
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

tab1, tab2 = st.tabs(["Policy Document Auditor", "Model Auditor"])

with tab1:
    render_text_auditor()

with tab2:
    render_model_auditor()
