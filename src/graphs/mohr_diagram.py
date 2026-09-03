"""Mohr diagram HTML graph artifacts."""
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from graphs.artifacts import (
    MODERN_FONT_FAMILY,
    SLIP_PRESSURE_COLOR_SCALE,
    add_graph_warning,
    has_columns,
    modern_colorbar,
    remove_step_messages,
    write_plotly_artifact,
)

MESSAGE_PREFIX = "Mohr diagram graph was not generated"

# Dark-theme side panel for hydrology Mohr (2+ faults). Injected via extra_head
# so it only overrides .plot-shell when the selector is present.
_MOHR_SELECTOR_HEAD = """
  <style>
    .plot-shell {
      flex-direction: row;
      align-items: stretch;
    }
    .plot-shell-content {
      order: 1;
      position: relative;
      width: auto !important;
      height: auto !important;
      flex: 1 1 auto;
      min-width: 0;
      min-height: 0;
    }
    .mohr-fault-selector {
      order: 2;
      flex: 0 0 clamp(200px, 28%, 280px);
      width: clamp(200px, 28%, 280px);
      min-width: 200px;
      max-width: 280px;
      min-height: 0;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
      border: 1px solid #475569;
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.96);
      color: #f8fafc;
      overflow: hidden;
    }
    .mohr-fault-selector-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 10px 6px;
      border-bottom: 1px solid #334155;
      flex: 0 0 auto;
    }
    .mohr-fault-selector-title {
      font-weight: 700;
      font-size: 13px;
    }
    .mohr-fault-summary {
      color: #94a3b8;
      font-size: 11px;
      white-space: nowrap;
    }
    .mohr-fault-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      padding: 8px 10px 6px;
      flex: 0 0 auto;
    }
    .mohr-fault-tools button {
      font: inherit;
      font-size: 11px;
      font-weight: 600;
      color: #f8fafc;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #64748b;
      border-radius: 6px;
      padding: 5px 9px;
      cursor: pointer;
    }
    .mohr-fault-tools button:hover {
      background: #1e2937;
    }
    .mohr-fault-filter {
      margin: 0 10px 6px;
      width: calc(100% - 20px);
      box-sizing: border-box;
      padding: 6px 8px;
      color: #f8fafc;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid #64748b;
      border-radius: 6px;
      font: inherit;
      font-size: 12px;
      flex: 0 0 auto;
    }
    .mohr-fault-master {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0 10px 6px;
      padding-bottom: 6px;
      border-bottom: 1px solid #334155;
      font-size: 12px;
      color: #cbd5e1;
      flex: 0 0 auto;
    }
    .mohr-fault-list {
      flex: 1 1 auto;
      min-height: 0;
      overflow-y: auto;
      padding: 4px 6px 8px;
      box-sizing: border-box;
    }
    .mohr-fault-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 6px;
      border-radius: 6px;
      cursor: pointer;
      content-visibility: auto;
      contain-intrinsic-size: 32px;
    }
    .mohr-fault-item:hover {
      background: rgba(148, 163, 184, 0.12);
    }
    .mohr-fault-item input {
      margin: 0;
      flex: 0 0 auto;
    }
    .mohr-fault-swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(248, 250, 252, 0.35);
      flex: 0 0 auto;
    }
    .mohr-fault-label {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 12px;
      flex: 1 1 auto;
    }
    @media (max-width: 720px) {
      .plot-shell {
        flex-direction: column;
      }
      .mohr-fault-selector {
        order: 2;
        flex: 0 0 auto;
        width: 100%;
        min-width: 0;
        max-width: none;
        max-height: 38%;
        min-height: 160px;
      }
    }
  </style>
"""

_MOHR_SELECTOR_SHELL = """
    <aside class="mohr-fault-selector" aria-label="Fault legend and selector">
      <div class="mohr-fault-selector-header">
        <div class="mohr-fault-selector-title">Faults</div>
        <div id="mohr-fault-summary" class="mohr-fault-summary"></div>
      </div>
      <div class="mohr-fault-tools">
        <button type="button" id="mohr-show-all">Show All</button>
        <button type="button" id="mohr-hide-all">Hide All</button>
      </div>
      <input type="search" id="mohr-fault-filter" class="mohr-fault-filter" placeholder="Filter faults..." aria-label="Filter faults">
      <label class="mohr-fault-master">
        <input type="checkbox" id="mohr-fault-all" checked>
        <span>All faults</span>
      </label>
      <div id="mohr-fault-list" class="mohr-fault-list"></div>
    </aside>
"""


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


def _hex_to_rgb(hex_color: str):
    color = str(hex_color).lstrip("#")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _interpolate_slip_color(value: float, cmin: float, cmax: float) -> str:
    """Map remaining ΔP onto SLIP_PRESSURE_COLOR_SCALE for selector swatches."""
    span = float(cmax) - float(cmin)
    normalized = 0.5 if span <= 0 else (float(value) - float(cmin)) / span
    normalized = max(0.0, min(1.0, normalized))
    palette = [(float(stop), _hex_to_rgb(color)) for stop, color in SLIP_PRESSURE_COLOR_SCALE]
    if normalized <= palette[0][0]:
        red, green, blue = palette[0][1]
        return f"#{red:02x}{green:02x}{blue:02x}"
    for index in range(1, len(palette)):
        stop, color = palette[index]
        previous_stop, previous_color = palette[index - 1]
        if normalized <= stop:
            ratio = 0.0 if stop == previous_stop else (normalized - previous_stop) / (stop - previous_stop)
            blended = tuple(
                int(round(previous_color[channel] + (color[channel] - previous_color[channel]) * ratio))
                for channel in range(3)
            )
            return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"
    red, green, blue = palette[-1][1]
    return f"#{red:02x}{green:02x}{blue:02x}"


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


def _selector_faults_payload(fault_df: pd.DataFrame, color_column: str, cmin: float, cmax: float):
    """Compact {id, color} rows for the hydrology Mohr selector list."""
    payload = []
    seen = set()
    for _, row in fault_df.iterrows():
        fault_id = str(row["id"])
        if fault_id in seen:
            continue
        seen.add(fault_id)
        payload.append({
            "id": fault_id,
            "color": _interpolate_slip_color(float(row[color_column]), cmin, cmax),
        })
    return payload


def _fault_points_payload(fault_df: pd.DataFrame, hover_text, pressures):
    """Original fault-scatter coordinates so JS can filter without N marker traces."""
    return [
        {
            "id": str(row["id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "text": hover_text[index],
            "color": float(pressures[index]),
        }
        for index, (_, row) in enumerate(fault_df.iterrows())
    ]


def _mohr_controls_script(
    min_pressure,
    max_pressure,
    fault_trace_index,
    x_upper,
    y_upper,
    *,
    selector_faults=None,
    fault_points=None,
):
    """PSI color-range inputs plus optional hydrology fault-selector logic."""
    selector_json = json.dumps(selector_faults or [], separators=(",", ":"))
    points_json = json.dumps(fault_points or [], separators=(",", ":"))
    has_selector_js = "true" if selector_faults else "false"
    return f"""
  <style>
    .plot-shell-content {{
      position: relative;
    }}
    .mohr-controls {{
      position: absolute;
      left: 88px;
      right: 16px;
      bottom: 10px;
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
      const hasSelector = {has_selector_js};
      const faultItems = {selector_json};
      const faultPoints = {points_json};
      const knownFaultIds = new Set(faultItems.map(function (item) {{ return item.id; }}));
      const selectedFaultIds = new Set(knownFaultIds);
      const minInput = document.getElementById('mohr-min-psi');
      const maxInput = document.getElementById('mohr-max-psi');
      let aspectResizeObserver = null;

      function findPlot() {{
        return document.querySelector('.plot-shell .js-plotly-plot');
      }}

      function placeControls() {{
        const plotContent = document.querySelector('.plot-shell-content');
        const controls = document.querySelector('.mohr-controls');
        if (plotContent && controls && controls.parentElement !== plotContent) {{
          plotContent.appendChild(controls);
        }}
      }}

      function validRange(minValue, maxValue) {{
        return Number.isFinite(minValue) && Number.isFinite(maxValue) && minValue < maxValue;
      }}

      function colorRange() {{
        let minValue = Number.parseFloat(minInput.value);
        let maxValue = Number.parseFloat(maxInput.value);
        if (!validRange(minValue, maxValue)) {{
          minValue = originalMin;
          maxValue = originalMax;
          minInput.value = originalMin;
          maxInput.value = originalMax;
        }}
        return {{ minValue: minValue, maxValue: maxValue }};
      }}

      function applyRange() {{
        const range = colorRange();
        const plot = findPlot();
        if (!plot || typeof Plotly === 'undefined') return;
        Plotly.restyle(plot, {{'marker.cmin': [range.minValue], 'marker.cmax': [range.maxValue]}}, [faultTraceIndex]);
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

      function traceFaultId(trace) {{
        if (!trace) return null;
        let candidate = null;
        if (trace.meta && typeof trace.meta === 'object' && trace.meta.fault_id) {{
          candidate = String(trace.meta.fault_id);
        }} else if (typeof trace.meta === 'string' && trace.meta) {{
          candidate = String(trace.meta);
        }} else if (trace.legendgroup) {{
          candidate = String(trace.legendgroup);
        }}
        if (candidate && knownFaultIds.has(candidate)) return candidate;
        return null;
      }}

      function updateSummary() {{
        const summary = document.getElementById('mohr-fault-summary');
        if (summary) {{
          summary.textContent = selectedFaultIds.size + ' of ' + faultItems.length + ' faults visible';
        }}
        const master = document.getElementById('mohr-fault-all');
        if (master) {{
          master.checked = faultItems.length > 0 && selectedFaultIds.size === faultItems.length;
          master.indeterminate = selectedFaultIds.size > 0 && selectedFaultIds.size < faultItems.length;
        }}
      }}

      function applyFaultSelection() {{
        const plot = findPlot();
        if (!plot || typeof Plotly === 'undefined') return;
        const range = colorRange();
        const circleIndices = [];
        const visibilities = [];
        (plot.data || []).forEach(function (trace, index) {{
          const faultId = traceFaultId(trace);
          if (!faultId) return;
          circleIndices.push(index);
          visibilities.push(selectedFaultIds.has(faultId));
        }});
        if (circleIndices.length) {{
          Plotly.restyle(plot, {{ visible: visibilities }}, circleIndices);
        }}
        const xs = [];
        const ys = [];
        const texts = [];
        const colors = [];
        faultPoints.forEach(function (point) {{
          if (!selectedFaultIds.has(point.id)) return;
          xs.push(point.x);
          ys.push(point.y);
          texts.push(point.text);
          colors.push(point.color);
        }});
        Plotly.restyle(plot, {{
          x: [xs],
          y: [ys],
          text: [texts],
          'marker.color': [colors],
          'marker.cmin': [range.minValue],
          'marker.cmax': [range.maxValue]
        }}, [faultTraceIndex]);
      }}

      function populateLegend() {{
        const legend = document.getElementById('mohr-fault-list');
        const filterInput = document.getElementById('mohr-fault-filter');
        if (!legend) return;
        const query = ((filterInput && filterInput.value) || '').trim().toLowerCase();
        legend.innerHTML = '';
        faultItems.forEach(function (fault) {{
          const labelText = String(fault.id);
          if (query && labelText.toLowerCase().indexOf(query) === -1) return;
          const label = document.createElement('label');
          label.className = 'mohr-fault-item';
          const checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.checked = selectedFaultIds.has(fault.id);
          checkbox.addEventListener('change', function () {{
            if (checkbox.checked) {{
              selectedFaultIds.add(fault.id);
            }} else {{
              selectedFaultIds.delete(fault.id);
            }}
            updateSummary();
            applyFaultSelection();
          }});
          const swatch = document.createElement('span');
          swatch.className = 'mohr-fault-swatch';
          swatch.style.backgroundColor = fault.color || '#e5e7eb';
          const text = document.createElement('span');
          text.className = 'mohr-fault-label';
          text.textContent = labelText;
          label.appendChild(checkbox);
          label.appendChild(swatch);
          label.appendChild(text);
          legend.appendChild(label);
        }});
        updateSummary();
      }}

      function setAllSelection(checked) {{
        selectedFaultIds.clear();
        if (checked) {{
          faultItems.forEach(function (fault) {{
            selectedFaultIds.add(fault.id);
          }});
        }}
        populateLegend();
        applyFaultSelection();
      }}

      function initializeSelector() {{
        if (!hasSelector) return;
        const filterInput = document.getElementById('mohr-fault-filter');
        const showAll = document.getElementById('mohr-show-all');
        const hideAll = document.getElementById('mohr-hide-all');
        const master = document.getElementById('mohr-fault-all');
        if (filterInput) {{
          filterInput.addEventListener('input', populateLegend);
        }}
        if (showAll) {{
          showAll.addEventListener('click', function () {{ setAllSelection(true); }});
        }}
        if (hideAll) {{
          hideAll.addEventListener('click', function () {{ setAllSelection(false); }});
        }}
        if (master) {{
          master.addEventListener('change', function () {{
            setAllSelection(master.checked);
          }});
        }}
        populateLegend();
      }}

      placeControls();
      minInput.addEventListener('input', applyRange);
      maxInput.addEventListener('input', applyRange);
      minInput.addEventListener('change', applyRange);
      maxInput.addEventListener('change', applyRange);
      initializeSelector();
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
        is_hydrology_mohr = "dp" in fault_df.columns and "slip_pressure" in fault_df.columns
        color_column = "pore_pressure_slip" if "pore_pressure_slip" in fault_df.columns else None
        if color_column is None and "slip_pressure" in fault_df.columns:
            color_column = "slip_pressure"
        if color_column is None and "dp" in fault_df.columns:
            color_column = "dp"
        if color_column is None:
            fault_df["delta_pressure"] = 0.0
            color_column = "delta_pressure"
        if "dp" in fault_df.columns:
            fault_df["dp"] = pd.to_numeric(fault_df["dp"], errors="coerce").fillna(0.0)
        fault_df[color_column] = pd.to_numeric(fault_df[color_column], errors="coerce").fillna(0.0)
        pressure_colorbar_title = "Remaining ΔP to slip (PSI)" if is_hydrology_mohr else "Delta PP to slip (PSI)"
        show_fault_selector = (
            is_hydrology_mohr
            and "fault_id" in arcs_df.columns
            and fault_df["id"].astype(str).nunique() >= 2
        )

        fig = go.Figure()
        all_x = []
        all_y = []
        circle_labels = _circle_stress_labels(stress_regime)

        circle_ids = [value for value in arcs_df["id"].astype(str).unique() if value != "friction_line"]
        for circle_id in circle_ids:
            circle_df = arcs_df[arcs_df["id"].astype(str) == circle_id]
            circle_label = circle_labels.get(circle_id, circle_id.replace("_", " ").title())
            grouped_circles = (
                list(circle_df.groupby("fault_id", sort=False))
                if "fault_id" in circle_df.columns
                else [(None, circle_df)]
            )
            for group_key, circle_group in grouped_circles:
                all_x.extend(circle_group["x"].tolist())
                all_y.extend(np.maximum(circle_group["y"].to_numpy(), 0.0).tolist())
                trace_kwargs = {
                    "x": circle_group["x"],
                    "y": circle_group["y"],
                    "mode": "lines",
                    "name": circle_label,
                    "showlegend": False,
                    "line": {"width": 2.2, "color": "#e5e7eb"},
                    "hovertemplate": f"{circle_label}<br>σ: %{{x:,.2f}} psi<br>τ: %{{y:,.2f}} psi<extra></extra>",
                }
                # Hydrology arcs are one triad per fault; tag them so the selector can hide/show.
                if group_key is not None and pd.notna(group_key):
                    fault_id = str(group_key)
                    trace_kwargs["meta"] = {"fault_id": fault_id}
                    trace_kwargs["legendgroup"] = fault_id
                fig.add_trace(go.Scatter(**trace_kwargs))

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
        )
        if is_hydrology_mohr:
            hover_text = (
                hover_text
                + "<br>Hydrology ΔP applied: " + fault_df["dp"].map("{:,.2f}".format) + " PSI"
                + "<br>Remaining ΔP to slip: " + fault_df[color_column].map("{:,.2f}".format) + " PSI"
            )
        else:
            hover_text = hover_text + "<br>Delta PP to slip: " + fault_df[color_column].map("{:,.2f}".format) + " PSI"
        hover_text = hover_text.tolist()
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
                "colorbar": modern_colorbar(pressure_colorbar_title, horizontal=True, dark=True),
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

        selector_faults = (
            _selector_faults_payload(fault_df, color_column, cmin, cmax)
            if show_fault_selector
            else None
        )
        fault_points = (
            _fault_points_payload(fault_df, hover_text, pressures)
            if show_fault_selector
            else None
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
            extra_head=_MOHR_SELECTOR_HEAD if show_fault_selector else "",
            extra_shell_html=_MOHR_SELECTOR_SHELL if show_fault_selector else "",
            extra_body=_mohr_controls_script(
                cmin,
                cmax,
                fault_trace_index,
                x_upper,
                y_upper,
                selector_faults=selector_faults,
                fault_points=fault_points,
            ),
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX}: {exc}")
        return None
