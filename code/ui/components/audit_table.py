"""
ui/components/audit_table.py
-----------------------------
Renders the audit trail log as a Streamlit dataframe.
"""

import pandas as pd
import streamlit as st


def render_audit_table(audit_log: list[dict]) -> None:
    """
    Displays the audit log as an expandable table in Streamlit.

    Args:
        audit_log: list of audit log entry dicts from ComplianceState
    """
    if not audit_log:
        st.info("No audit events recorded yet.")
        return

    rows = []
    for entry in audit_log:
        rows.append({
            "Timestamp": entry.get("timestamp", ""),
            "Node": entry.get("node", ""),
            "Action": entry.get("action", ""),
            "Result": entry.get("result", ""),
        })

    df = pd.DataFrame(rows)

    with st.expander("Audit Trail", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
