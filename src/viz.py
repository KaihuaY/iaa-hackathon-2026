"""Shared Plotly styling for EDA dashboards (renders fully offline)."""
import plotly.graph_objects as go
import plotly.io as pio

# Categorical slots, fixed order. One colour per line of business, everywhere.
LINE_COLOURS = {"motor": "#2a78d6", "buildings": "#eb6834", "contents": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#c3c2b7"  # exposure bars / context marks
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_axis = dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS,
             tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=INK2, size=12)), automargin=True)

pio.templates["iaa"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=INK2, size=12),
        title=dict(font=dict(color=INK, size=15), x=0, xanchor="left", y=0.98, yref="container", yanchor="top", pad=dict(l=4)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        xaxis=dict(_axis, showgrid=False), yaxis=dict(_axis),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0, font=dict(color=INK2), title_text=""),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK, size=12)),
        hovermode="closest", bargap=0.15, margin=dict(l=60, r=20, t=120, b=50),
    ),
    data=dict(
        scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=2, color=SURFACE)))],
        bar=[go.Bar(marker=dict(line=dict(width=0)))],
    ),
)


def setup():
    """Use the house template and embed plotly.js in the notebook (no CDN)."""
    pio.templates.default = "iaa"
    pio.renderers.default = "notebook"
