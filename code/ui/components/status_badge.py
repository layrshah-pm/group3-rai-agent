"""
ui/components/status_badge.py
------------------------------
Pass/Fail status badge HTML.
"""

STATUS_COLOURS = {
    "PASS": "#28a745",
    "FAIL": "#dc3545",
}

STATUS_TEXT_COLOURS = {
    "PASS": "#ffffff",
    "FAIL": "#ffffff",
}


def render_status_badge(status: str) -> str:
    """
    Returns an HTML string for a coloured status badge.

    Args:
        status: one of "PASS", "FAIL"

    Returns:
        HTML string — render with st.markdown(..., unsafe_allow_html=True)
    """
    bg = STATUS_COLOURS.get(status, "#6c757d")
    fg = STATUS_TEXT_COLOURS.get(status, "#ffffff")
    return (
        f'<div style="display:inline-block; padding:8px 20px; '
        f'background:{bg}; color:{fg}; border-radius:6px; '
        f'font-size:1.2rem; font-weight:bold;">{status}</div>'
    )
