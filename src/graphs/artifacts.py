"""Shared Plotly graph artifact helpers for FSP portal output."""
import math
import os
from typing import Iterable, Optional

import pandas as pd


MODERN_FONT_FAMILY = "Inter, Segoe UI, Roboto, Arial, Helvetica, sans-serif"

SCIENTIFIC_COLORS = [
    "#2563eb",
    "#e11d48",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#475569",
]

MODERN_GRID_COLOR = "#e2e8f0"
MODERN_AXIS_COLOR = "#64748b"
MODERN_TEXT_COLOR = "#0f172a"
MODERN_MUTED_TEXT_COLOR = "#475569"
MODERN_PAPER_BG = "#f8fafc"
MODERN_PLOT_BG = "#ffffff"
MODERN_CONTROL_BG = "rgba(255, 255, 255, 0.94)"
MODERN_BORDER_COLOR = "rgba(148, 163, 184, 0.45)"
MODERN_SHADOW = "0 18px 42px rgba(15, 23, 42, 0.10)"
MODERN_HEAT_COLORSCALE = "Viridis"

PLOTLY_CONFIG = {"responsive": True, "displaylogo": False, "scrollZoom": True}


SLIP_PRESSURE_COLOR_SCALE = [
    [0.0, "#800000"],
    [0.08, "#ff0000"],
    [0.18, "#ff5a00"],
    [0.28, "#ffc300"],
    [0.35, "#ffff00"],
    [0.67, "#ffff00"],
    [0.78, "#aad400"],
    [0.88, "#61b000"],
    [1.0, "#007f00"],
]

# FSP scale: green at 0 (low/safe) -> red at 1 (high/dangerous). Reverse of the slip-pressure
# scale, so the Leaflet renderer (which normalizes value->[0,1] with no 1-x hook) matches the
# CDF graph's _interpolate_colorscale(SLIP_PRESSURE_COLOR_SCALE, 1 - fsp).
FSP_COLOR_SCALE = [[round(1.0 - stop, 2), color] for stop, color in reversed(SLIP_PRESSURE_COLOR_SCALE)]

# Backwards-compatible alias for older graph helpers/tests.
MOHR_COLOR_SCALE = SLIP_PRESSURE_COLOR_SCALE


def graph_artifacts_dir(helper) -> str:
    path = os.path.join(helper.scratchPath, "graphs")
    os.makedirs(path, exist_ok=True)
    return path


def remove_graph_artifact(helper, artifact_key: str) -> None:
    artifacts = helper.origArgsData.get("GraphArtifacts")
    if not artifacts:
        return
    helper.origArgsData["GraphArtifacts"] = [
        artifact for artifact in artifacts if artifact.get("key") != artifact_key
    ]


def remove_step_messages(helper, step_index: int, message_prefix: str) -> None:
    try:
        messages = helper.origArgsData["SessionState"]["StepState"][step_index]["Messages"]
    except (KeyError, IndexError, TypeError):
        return
    helper.origArgsData["SessionState"]["StepState"][step_index]["Messages"] = [
        message
        for message in messages
        if not str(message.get("MessageContent", "")).startswith(message_prefix)
    ]


def add_graph_warning(helper, step_index: int, message: str) -> None:
    helper.addMessageWithStepIndex(step_index, message, 1)


def is_empty_frame(df: Optional[pd.DataFrame]) -> bool:
    return df is None or df.empty


def has_columns(df: Optional[pd.DataFrame], columns: Iterable[str]) -> bool:
    return not is_empty_frame(df) and all(column in df.columns for column in columns)


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA)


def plotly_fragment_html(fig) -> str:
    return fig.to_html(
        include_plotlyjs="cdn",
        include_mathjax=False,
        full_html=False,
        default_width="100%",
        default_height="100%",
        config=PLOTLY_CONFIG,
    )


def html_document(
    fig,
    *,
    dark: bool = False,
    extra_head: str = "",
    extra_body: str = "",
    extra_shell_html: str = "",
) -> str:
    background = "#111827" if dark else MODERN_PAPER_BG
    color = "#f8fafc" if dark else "#111827"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: {background};
      color: {color};
      font-family: {MODERN_FONT_FAMILY};
      -webkit-font-smoothing: antialiased;
      text-rendering: geometricPrecision;
    }}
    .plot-shell {{
      width: 100%;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: {background};
      box-sizing: border-box;
      padding: 10px;
      gap: 10px;
    }}
    .plot-shell-content {{
      width: 100% !important;
      height: 100% !important;
      flex: 1 1 auto;
      min-height: 0;
      border: 1px solid {"rgba(148, 163, 184, 0.24)" if dark else MODERN_BORDER_COLOR};
      border-radius: 8px;
      box-shadow: {"none" if dark else MODERN_SHADOW};
      overflow: hidden;
      background: {"#111827" if dark else MODERN_PLOT_BG};
      box-sizing: border-box;
    }}
    .modebar {{
      right: 10px !important;
      top: 10px !important;
    }}
    .modebar-btn svg path {{
      fill: {"#e5e7eb" if dark else "#475569"} !important;
    }}
  </style>
  {extra_head}
</head>
<body>
  <div class="plot-shell">
    {extra_shell_html}
    <div class="plot-shell-content">
{plotly_fragment_html(fig)}
    </div>
  </div>
  {extra_body}
</body>
</html>
"""


def write_plotly_artifact(
    helper,
    fig,
    artifact_key: str,
    title: str,
    *,
    caption: str,
    display_order: int,
    preferred_height: int,
    dark: bool = False,
    extra_head: str = "",
    extra_body: str = "",
    extra_shell_html: str = "",
) -> str:
    output_path = os.path.join(graph_artifacts_dir(helper), f"{artifact_key}.html")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(
            html_document(
                fig,
                dark=dark,
                extra_head=extra_head,
                extra_body=extra_body,
                extra_shell_html=extra_shell_html,
            )
        )

    remove_graph_artifact(helper, artifact_key)
    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        renderer="html",
        path=output_path,
        contentType="text/html",
        caption=caption,
        displayOrder=display_order,
        preferredHeight=preferred_height,
    )
    return output_path


def apply_scientific_layout(fig, *, x_title: str, y_title: str, title: str = ""):
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"} if title else None,
        autosize=True,
        template="plotly_white",
        margin={"l": 74, "r": 34, "t": 58 if title else 34, "b": 64},
        font={"family": MODERN_FONT_FAMILY, "size": 12, "color": MODERN_TEXT_COLOR},
        xaxis={
            "title": {"text": x_title, "standoff": 10, "font": {"color": MODERN_MUTED_TEXT_COLOR}},
            "showgrid": True,
            "gridcolor": MODERN_GRID_COLOR,
            "linecolor": MODERN_GRID_COLOR,
            "tickcolor": MODERN_GRID_COLOR,
            "tickfont": {"color": MODERN_AXIS_COLOR},
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": {"text": y_title, "standoff": 12, "font": {"color": MODERN_MUTED_TEXT_COLOR}},
            "showgrid": True,
            "gridcolor": MODERN_GRID_COLOR,
            "linecolor": MODERN_GRID_COLOR,
            "tickcolor": MODERN_GRID_COLOR,
            "tickfont": {"color": MODERN_AXIS_COLOR},
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.04,
            "xanchor": "left",
            "yanchor": "bottom",
            "bgcolor": MODERN_CONTROL_BG,
            "bordercolor": MODERN_BORDER_COLOR,
            "borderwidth": 1,
            "font": {"color": MODERN_MUTED_TEXT_COLOR},
        },
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#0f172a",
            "bordercolor": "#334155",
            "font": {"family": MODERN_FONT_FAMILY, "color": "#f8fafc"},
        },
        paper_bgcolor=MODERN_PLOT_BG,
        plot_bgcolor=MODERN_PLOT_BG,
        colorway=SCIENTIFIC_COLORS,
        uniformtext={"mode": "hide", "minsize": 10},
    )


def modern_updatemenu(buttons, *, x: float = 1.0, y: float = 1.14, menu_type: str = "dropdown", direction: str = "down"):
    return {
        "type": menu_type,
        "direction": direction,
        "x": x,
        "xanchor": "right",
        "y": y,
        "yanchor": "top",
        "buttons": buttons,
        "bgcolor": MODERN_CONTROL_BG,
        "bordercolor": MODERN_BORDER_COLOR,
        "borderwidth": 1,
        "font": {"family": MODERN_FONT_FAMILY, "size": 12, "color": MODERN_TEXT_COLOR},
        "pad": {"r": 8, "t": 6, "b": 6, "l": 8},
    }


def modern_colorbar(title: str, *, horizontal: bool = False, dark: bool = False) -> dict:
    text_color = "#f8fafc" if dark else MODERN_MUTED_TEXT_COLOR
    outline = "#64748b" if dark else MODERN_BORDER_COLOR
    colorbar = {
        "title": {"text": title, "font": {"color": text_color}},
        "tickfont": {"color": text_color},
        "tickformat": ",.0f",
        "outlinecolor": outline,
        "thickness": 16,
    }
    if horizontal:
        colorbar.update({
            "orientation": "h",
            "title": {"text": title, "font": {"color": text_color}, "side": "top"},
            "x": 0.5,
            "y": -0.30,
            "xanchor": "center",
            "yanchor": "top",
            "len": 0.52,
            "thickness": 16,
        })
    return colorbar


def apply_modern_subplots_layout(fig, *, rows: int, title: str = ""):
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"} if title else None,
        template="plotly_white",
        bargap=0.05,
        margin={"l": 54, "r": 34, "t": 68 if title else 58, "b": 48},
        font={"family": MODERN_FONT_FAMILY, "size": 11, "color": MODERN_TEXT_COLOR},
        paper_bgcolor=MODERN_PLOT_BG,
        plot_bgcolor=MODERN_PLOT_BG,
        height=max(560, rows * 340),
        hoverlabel={
            "bgcolor": "#0f172a",
            "bordercolor": "#334155",
            "font": {"family": MODERN_FONT_FAMILY, "color": "#f8fafc"},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=MODERN_GRID_COLOR,
        linecolor=MODERN_GRID_COLOR,
        tickcolor=MODERN_GRID_COLOR,
        tickfont={"color": MODERN_AXIS_COLOR},
        title_font={"color": MODERN_MUTED_TEXT_COLOR},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=MODERN_GRID_COLOR,
        linecolor=MODERN_GRID_COLOR,
        tickcolor=MODERN_GRID_COLOR,
        tickfont={"color": MODERN_AXIS_COLOR},
        title_font={"color": MODERN_MUTED_TEXT_COLOR},
    )
