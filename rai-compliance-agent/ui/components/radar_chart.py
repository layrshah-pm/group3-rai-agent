"""
ui/components/radar_chart.py
----------------------------
Plotly radar chart for the 5-pillar RAI scorecard.
"""

import plotly.graph_objects as go


CATEGORIES = [
    "Strategic\nAlignment",
    "Data\nGovernance",
    "Model\nGovernance",
    "Org\nOversight",
    "Continuous\nMonitoring",
]

CATEGORY_KEYS = [
    "strategic_alignment",
    "data_governance",
    "model_governance",
    "org_oversight",
    "continuous_monitoring",
]


def render_radar_chart(rai_scores: dict) -> go.Figure:
    """
    Returns a Plotly Scatterpolar figure for the given rai_scores dict.

    Args:
        rai_scores: dict with keys matching CATEGORY_KEYS, values 0-3
    """
    values = [rai_scores.get(k, 0) for k in CATEGORY_KEYS]
    values_closed = values + values[:1]
    categories_closed = CATEGORIES + CATEGORIES[:1]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        line_color="#2E75B6",
        fillcolor="rgba(46, 117, 182, 0.2)",
        name="RAI Score",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                tickvals=[0, 1, 2, 3],
                ticktext=["0", "1", "2", "3"],
            )
        ),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig
