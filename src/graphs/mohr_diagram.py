"""Mohr diagram HTML graph artifacts."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from graphs.artifacts import (
    MODERN_AXIS_COLOR,
    MODERN_BORDER_COLOR,
    MODERN_FONT_FAMILY,
    MODERN_MUTED_TEXT_COLOR,
    SLIP_PRESSURE_COLOR_SCALE,
    add_graph_warning,
    has_columns,
    modern_colorbar,
    remove_step_messages,
    write_plotly_artifact,
)

MESSAGE_PREFIX = "Mohr diagram graph was not generated"


def _artifact_key_title(step_index: int, artifact_key=None, title=None, display_order=None):
    if artifact_key and title and display_order is not None:
        return artifact_key, title, display_order
    if step_index == 3:
        return (
            artifact_key or "fsp-deterministic-hydrology-mohr-diagram",
            title or "Hydrology Mohr Diagram",
            42 if display_order is None else display_order,
        )
    return (
        artifact_key or "fsp-deterministic-geomechanics-mohr-diagram",
        title or "Mohr Diagram",
        20 if display_order is None else display_order,
    )


def _clean_numeric(df: pd.DataFrame, columns):
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=columns)


def _range_upper(values, multiplier=1.08):
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return 1.0
    upper = max(clean)
    return 1.0 if upper <= 0.0 else upper * multiplier


def _circle_stress_labels(stress_regime: str = None):
    """Map Mohr circle IDs to the stress pair represented by that circle."""
    regime = str(stress_regime or "").strip().lower().replace("_", " ").replace("-", " ")
    if "normal" in regime:
        principal = ["sigmaV", "sigmaH", "sigmah"]
    elif "strike" in regime or "slip" in regime:
        principal = ["sigmaH", "sigmaV", "sigmah"]
    elif "reverse" in regime or "thrust" in regime:
        principal = ["sigmaH", "sigmah", "sigmaV"]
    else:
        return {
            "circle1": "Circle 1",
            "circle2": "Circle 2",
            "circle3": "Circle 3",
        }

    display = {
        "sigmaV": "σV",
        "sigmaH": "σH",
        "sigmah": "σh",
    }
    return {
        "circle1": f"{display[principal[0]]} - {display[principal[2]]}",
        "circle2": f"{display[principal[1]]} - {display[principal[2]]}",
        "circle3": f"{display[principal[0]]} - {display[principal[1]]}",
    }


def _principal_stress_x_positions(arcs_df: pd.DataFrame, stress_regime: str = None):
    """Return {label: x_position} for σh, σH, σV on the Mohr diagram x-axis.

    Derived from the circle extents rather than raw stress values so this
    works for both the geomechanics and hydrology Mohr diagrams.
    """
    regime = str(stress_regime or "").strip().lower().replace("_", " ").replace("-", " ")
    if not any(k in regime for k in ("normal", "strike", "slip", "reverse", "thrust")):
        return {}
    c1 = arcs_df[arcs_df["id"] == "circle1"]["x"]
    c2 = arcs_df[arcs_df["id"] == "circle2"]["x"]
    if c1.empty or c2.empty:
        return {}
    sigma_max = float(c1.max())   # largest effective principal stress
    sigma_min = float(c1.min())   # smallest
    sigma_mid = float(c2.max())   # middle
    if "normal" in regime:
        principal = ["sigmaV", "sigmaH", "sigmah"]
    elif "strike" in regime or "slip" in regime:
        principal = ["sigmaH", "sigmaV", "sigmah"]
    else:
        principal = ["sigmaH", "sigmah", "sigmaV"]
    display = {"sigmaV": "σV", "sigmaH": "σH", "sigmah": "σh"}
    return {
        display[principal[0]]: sigma_max,
        display[principal[1]]: sigma_mid,
        display[principal[2]]: sigma_min,
    }


def _mohr_controls_script(min_pressure, max_pressure, fault_trace_index, x_upper, y_upper):
    return f"""
  <style>
    .mohr-controls {{
      position: absolute;
      left: 120px;
      right: 96px;
      bottom: 14px;
      z-index: 10;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      color: #f8fafc;
      font-size: 12px;
      line-height: 1;
      pointer-events: none;
    }}
    .mohr-controls label {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
      pointer-events: auto;
    }}
    .mohr-controls input {{
      width: 82px;
      box-sizing: border-box;
      padding: 5px 7px;
      color: #f8fafc;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #64748b;
      border-radius: 6px;
      font-size: 12px;
    }}
  </style>
  <div class="mohr-controls" aria-label="Delta PP to slip PSI color range">
    <label>Min PSI <input id="mohr-min-psi" type="number" step="any" value="{min_pressure}"></label>
    <label>Max PSI <input id="mohr-max-psi" type="number" step="any" value="{max_pressure}"></label>
  </div>
  <script>
    (function () {{
      const originalMin = {min_pressure};
      const originalMax = {max_pressure};
      const faultTraceIndex = {fault_trace_index};
      const dataXUpper = {x_upper};
      const dataYUpper = {y_upper};
      const minInput = document.getElementById('mohr-min-psi');
      const maxInput = document.getElementById('mohr-max-psi');
      let aspectResizeObserver = null;

      function findPlot() {{
        return document.querySelector('.plot-shell .js-plotly-plot');
      }}

      function validRange(minValue, maxValue) {{
        return Number.isFinite(minValue) && Number.isFinite(maxValue) && minValue < maxValue;
      }}

      function applyRange() {{
        let minValue = Number.parseFloat(minInput.value);
        let maxValue = Number.parseFloat(maxInput.value);
        if (!validRange(minValue, maxValue)) {{
          minValue = originalMin;
          maxValue = originalMax;
          minInput.value = originalMin;
          maxInput.value = originalMax;
        }}
        const plot = findPlot();
        if (!plot || typeof Plotly === 'undefined') return;
        Plotly.restyle(plot, {{'marker.cmin': [minValue], 'marker.cmax': [maxValue]}}, [faultTraceIndex]);
      }}

      function updateMohrAspect() {{
        const plot = findPlot();
        if (!plot || typeof Plotly === 'undefined' || !plot._fullLayout) return;
        const plotSize = plot._fullLayout._size;
        if (!plotSize || !plotSize.w || !plotSize.h || !dataXUpper || !dataYUpper) return;
        const targetAspect = dataXUpper / dataYUpper;
        const domainHeight = Math.min(1, Math.max(0.2, plotSize.w / (plotSize.h * targetAspect)));
        Plotly.relayout(plot, {{
          'xaxis.autorange': false,
          'yaxis.autorange': false,
          'xaxis.range': [0, dataXUpper],
          'yaxis.range': [0, dataYUpper],
          'yaxis.domain': [1 - domainHeight, 1]
        }});
      }}

      function initializeMohrAspect() {{
        const plot = findPlot();
        if (!plot) {{
          window.setTimeout(initializeMohrAspect, 50);
          return;
        }}
        window.setTimeout(updateMohrAspect, 0);
        if (typeof ResizeObserver !== 'undefined' && !aspectResizeObserver) {{
          aspectResizeObserver = new ResizeObserver(function () {{
            window.requestAnimationFrame(updateMohrAspect);
          }});
          aspectResizeObserver.observe(plot);
        }} else {{
          window.addEventListener('resize', updateMohrAspect);
        }}
      }}

      minInput.addEventListener('input', applyRange);
      maxInput.addEventListener('input', applyRange);
      minInput.addEventListener('change', applyRange);
      maxInput.addEventListener('change', applyRange);
      initializeMohrAspect();
    }})();
  </script>
"""


def save_mohr_diagram_graph_artifact(
    helper,
    arcs_df: pd.DataFrame,
    slip_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    *,
    step_index: int = 1,
    artifact_key: str = None,
    title: str = None,
    display_order: int = None,
    stress_regime: str = None,
):
    """Generate a Plotly Mohr circle diagram and register it with the portal."""
    try:
        remove_step_messages(helper, step_index, MESSAGE_PREFIX)
        if not has_columns(arcs_df, ["id", "x", "y"]) or not has_columns(fault_df, ["id", "x", "y"]):
            add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX} because required Mohr diagram columns are missing.")
            return None

        arcs_df = _clean_numeric(arcs_df, ["x", "y"])
        fault_df = _clean_numeric(fault_df, ["x", "y"])
        if arcs_df.empty or fault_df.empty:
            add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX} because the Mohr diagram data is empty.")
            return None

        artifact_key, title, display_order = _artifact_key_title(step_index, artifact_key, title, display_order)
        color_column = "pore_pressure_slip" if "pore_pressure_slip" in fault_df.columns else None
        if color_column is None and "slip_pressure" in fault_df.columns:
            color_column = "slip_pressure"
        if color_column is None and "dp" in fault_df.columns:
            color_column = "dp"
        if color_column is None:
            fault_df["delta_pressure"] = 0.0
            color_column = "delta_pressure"
        fault_df[color_column] = pd.to_numeric(fault_df[color_column], errors="coerce").fillna(0.0)

        fig = go.Figure()
        all_x = []
        all_y = []
        circle_labels = _circle_stress_labels(stress_regime)

        for circle_id in [value for value in arcs_df["id"].astype(str).unique() if value != "friction_line"]:
            circle_df = arcs_df[arcs_df["id"].astype(str) == circle_id]
            circle_label = circle_labels.get(circle_id, circle_id.replace("_", " ").title())
            all_x.extend(circle_df["x"].tolist())
            all_y.extend(np.maximum(circle_df["y"].to_numpy(), 0.0).tolist())
            fig.add_trace(go.Scatter(
                x=circle_df["x"],
                y=circle_df["y"],
                mode="lines",
                name=circle_label,
                showlegend=False,
                line={"width": 2.2, "color": "#e5e7eb"},
                hovertemplate=f"{circle_label}<br>σ: %{{x:,.2f}} psi<br>τ: %{{y:,.2f}} psi<extra></extra>",
            ))

        slip_line = arcs_df[arcs_df["id"].astype(str) == "friction_line"]
        if not slip_line.empty:
            all_x.extend(slip_line["x"].tolist())
            all_y.extend(slip_line["y"].tolist())
            fig.add_trace(go.Scatter(
                x=slip_line["x"],
                y=slip_line["y"],
                mode="lines",
                name="Frictional Slip Line",
                line={"width": 2.2, "color": "#f43f5e"},
                hovertemplate="Frictional Slip Line<br>σ: %{x:,.2f} psi<br>τ: %{y:,.2f} psi<extra></extra>",
            ))

        pressures = fault_df[color_column].astype(float).to_numpy()
        cmin = 0.0
        cmax = float(np.nanmax(pressures)) if len(pressures) else 1.0
        if cmax <= cmin:
            cmax = cmin + 1.0

        hover_text = (
            "Fault: " + fault_df["id"].astype(str)
            + "<br>σ: " + fault_df["x"].map("{:,.2f}".format) + " psi"
            + "<br>τ: " + fault_df["y"].map("{:,.2f}".format) + " psi"
            + "<br>Delta PP to slip: " + fault_df[color_column].map("{:,.2f}".format) + " PSI"
        ).tolist()
        all_x.extend(fault_df["x"].tolist())
        all_y.extend(fault_df["y"].tolist())

        fig.add_trace(go.Scatter(
            x=[-1.0],
            y=[-1.0],
            mode="markers",
            name="Faults",
            marker={"size": 9, "symbol": "circle-open", "color": "#ffffff", "line": {"width": 2, "color": "#f8fafc"}},
            hoverinfo="skip",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=fault_df["x"],
            y=fault_df["y"],
            mode="markers",
            name="Faults",
            text=hover_text,
            showlegend=False,
            marker={
                "size": 9,
                "color": pressures,
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": SLIP_PRESSURE_COLOR_SCALE,
                "showscale": True,
                "colorbar": modern_colorbar("Delta PP to slip (PSI)", horizontal=True, dark=True),
                "line": {"width": 1, "color": "#111827"},
            },
            hovertemplate="%{text}<extra></extra>",
        ))
        fault_trace_index = len(fig.data) - 1

        x_upper = _range_upper(all_x, 1.05)
        y_upper = _range_upper(all_y, 1.08)
        x_upper = max(x_upper, y_upper * 2.8)

        annotations = []
        if stress_regime:
            annotations.append({
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.98,
                "text": f"Stress Regime: {stress_regime}",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(15, 23, 42, 0.86)",
                "bordercolor": "#64748b",
                "borderwidth": 1,
                "borderpad": 4,
                "font": {"size": 12, "color": "#f8fafc"},
            })

        for label, x_pos in _principal_stress_x_positions(arcs_df, stress_regime).items():
            if 0 <= x_pos <= x_upper * 1.05:
                annotations.append({
                    "x": x_pos,
                    "y": 0,
                    "xref": "x",
                    "yref": "y",
                    "text": label,
                    "showarrow": False,
                    "xanchor": "center",
                    "yanchor": "bottom",
                    "bgcolor": "rgba(17, 24, 39, 0.88)",
                    "borderpad": 3,
                    "font": {"size": 13, "color": "#06b6d4", "family": MODERN_FONT_FAMILY},
                })

        fig.update_layout(
            autosize=True,
            margin={"l": 78, "r": 38, "t": 48, "b": 150},
            font={"family": MODERN_FONT_FAMILY, "color": "#f8fafc"},
            xaxis={
                "title": {"text": "σ Effective Normal Stress (psi)", "standoff": 8},
                "range": [0.0, x_upper],
                "tickformat": ",.0f",
                "separatethousands": True,
                "automargin": True,
                "showgrid": True,
                "gridcolor": "#334155",
                "linecolor": "#f8fafc",
                "tickcolor": "#f8fafc",
                "zeroline": False,
            },
            yaxis={
                "title": {"text": "τ Shear Stress (psi)", "standoff": 10},
                "range": [0.0, y_upper],
                "tickformat": ",.0f",
                "separatethousands": True,
                "automargin": True,
                "scaleanchor": "x",
                "scaleratio": 1,
                "showgrid": True,
                "gridcolor": "#334155",
                "linecolor": "#f8fafc",
                "tickcolor": "#f8fafc",
                "zeroline": False,
            },
            legend={
                "orientation": "h",
                "x": 0,
                "y": 1.03,
                "xanchor": "left",
                "yanchor": "bottom",
                "bgcolor": "rgba(15, 23, 42, 0.86)",
                "bordercolor": "#475569",
                "borderwidth": 1,
            },
            annotations=annotations,
            hovermode="closest",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            hoverlabel={"bgcolor": "#111827", "bordercolor": "#64748b", "font": {"color": "#f8fafc"}},
        )

        return write_plotly_artifact(
            helper,
            fig,
            artifact_key,
            title,
            caption="Interactive Mohr diagram generated by FSP.",
            display_order=display_order,
            preferred_height=560,
            dark=True,
            extra_body=_mohr_controls_script(cmin, cmax, fault_trace_index, x_upper, y_upper),
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX}: {exc}")
        return None
