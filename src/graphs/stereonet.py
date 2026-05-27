"""Stereonet HTML graph artifact for deterministic geomechanics."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fsp.geomechanics.slip import calculate_fault_effective_stresses, calculate_slip_pressure
from fsp.models.stress import StressState
from graphs.artifacts import (
    MODERN_AXIS_COLOR,
    MODERN_BORDER_COLOR,
    MODERN_CONTROL_BG,
    MODERN_FONT_FAMILY,
    MODERN_GRID_COLOR,
    MODERN_MUTED_TEXT_COLOR,
    MODERN_TEXT_COLOR,
    SLIP_PRESSURE_COLOR_SCALE,
    add_graph_warning,
    has_columns,
    modern_colorbar,
    modern_updatemenu,
    remove_step_messages,
    write_plotly_artifact,
)

MESSAGE_PREFIX = "Stereonet graph was not generated"


def fault_normal_projection(strike, dip):
    """Return lower-hemisphere fault-normal stereonet x/y coordinates."""
    strike_rad = (90.0 - np.asarray(strike, dtype=float)) * np.pi / 180.0
    dip_values = np.asarray(dip, dtype=float)
    theta = strike_rad + np.pi / 2.0
    rho = dip_values / 90.0
    return rho * np.cos(theta), rho * np.sin(theta)


def projected_curve(strike, dip, rake_count=181):
    """Return lower-hemisphere projected fault curve x/y coordinates."""
    strike_rad = (90.0 - float(strike)) * np.pi / 180.0
    dip_rad = float(dip) * np.pi / 180.0
    rake = np.linspace(0.0, np.pi, int(rake_count))
    plunge = np.arcsin(np.sin(dip_rad) * np.sin(rake))
    trend = strike_rad + np.arctan2(np.cos(dip_rad) * np.sin(rake), np.cos(rake))
    rho = np.tan(np.pi / 4.0 - plunge / 2.0)
    theta = trend + np.pi
    return rho * np.cos(theta), rho * np.sin(theta)


def normal_composite_grid(stress_state: StressState, p0: float, friction: float, grid_size: int = 50) -> pd.DataFrame:
    """Evaluate slip pressure over a regular strike/dip grid for the composite stereonet."""
    strikes = np.linspace(0.0, 360.0, int(grid_size))
    dips = np.linspace(0.0, 90.0, int(grid_size))
    strike_grid, dip_grid = np.meshgrid(strikes, dips)

    sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(
        strike_grid.ravel(), dip_grid.ravel(), stress_state, p0, 0.0
    )
    slip_pressure = calculate_slip_pressure(
        sig, tau, friction, p0, 1.0, 0.5, 0.0, s11, s22, s33, s12, n1, n2
    )
    slip_pressure = np.maximum(np.asarray(slip_pressure, dtype=float), 0.0)

    theta = strike_grid.ravel() * np.pi / 180.0
    rho = dip_grid.ravel() / 90.0
    return pd.DataFrame({
        "strike": strike_grid.ravel(),
        "dip": dip_grid.ravel(),
        "x": rho * np.cos(theta),
        "y": rho * np.sin(theta),
        "slip_pressure": slip_pressure,
    })


def _composite_stress_state(stress_state: StressState) -> StressState:
    """Match MATLAB's normal-composite SHmax convention without changing fault results."""
    return StressState(
        np.asarray(stress_state.principal_stresses, dtype=float).copy(),
        (360.0 - float(stress_state.sH_azimuth)) % 360.0,
    )


def _normal_composite_heatmap(stress_state: StressState, p0: float, friction: float, grid_size: int = 90):
    coords = np.linspace(-1.0, 1.0, int(grid_size))
    x_grid, y_grid = np.meshgrid(coords, coords)
    rho = np.sqrt(x_grid ** 2 + y_grid ** 2)
    strike_grid = (np.degrees(np.arctan2(y_grid, x_grid)) + 360.0) % 360.0
    dip_grid = rho * 90.0
    inside = rho <= 1.0

    z = np.full_like(x_grid, np.nan, dtype=float)
    sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(
        strike_grid[inside], dip_grid[inside], stress_state, p0, 0.0
    )
    slip_pressure = calculate_slip_pressure(
        sig, tau, friction, p0, 1.0, 0.5, 0.0, s11, s22, s33, s12, n1, n2
    )
    z[inside] = np.maximum(np.asarray(slip_pressure, dtype=float), 0.0)
    return coords, coords, z


def _clean_faults(faults_df: pd.DataFrame) -> pd.DataFrame:
    required = ["FaultID", "Strike", "Dip", "slip_pressure"]
    if not has_columns(faults_df, required):
        return pd.DataFrame(columns=required)
    result = faults_df.copy()
    for column in ["Strike", "Dip", "slip_pressure"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["Strike", "Dip", "slip_pressure"])


def _pressure_range(*frames):
    values = []
    for frame in frames:
        if frame is not None and not frame.empty and "slip_pressure" in frame.columns:
            values.extend(pd.to_numeric(frame["slip_pressure"], errors="coerce").dropna().tolist())
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return 0.0, 1.0
    cmin = 0.0
    cmax = max(clean)
    if cmax <= cmin:
        cmax = cmin + 1.0
    return cmin, cmax


def _hex_to_rgb(hex_color: str):
    color = str(hex_color).lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(round(channel)))) for channel in rgb])


def _color_for_value(value: float, cmin: float, cmax: float):
    if cmax <= cmin:
        normalized = 0.0
    else:
        normalized = (float(value) - cmin) / (cmax - cmin)
    normalized = max(0.0, min(1.0, normalized))
    stops = sorted((float(stop), _hex_to_rgb(color)) for stop, color in SLIP_PRESSURE_COLOR_SCALE)
    if normalized <= stops[0][0]:
        return _rgb_to_hex(stops[0][1])
    for index in range(1, len(stops)):
        stop, color = stops[index]
        previous_stop, previous_color = stops[index - 1]
        if normalized <= stop:
            span = stop - previous_stop or 1.0
            ratio = (normalized - previous_stop) / span
            return _rgb_to_hex([
                previous_color[channel] + (color[channel] - previous_color[channel]) * ratio
                for channel in range(3)
            ])
    return _rgb_to_hex(stops[-1][1])


def _colorbar_marker(cmin: float, cmax: float):
    return {
        "size": 0.1,
        "opacity": 0.0,
        "color": [cmin, cmax],
        "cmin": cmin,
        "cmax": cmax,
        "colorscale": SLIP_PRESSURE_COLOR_SCALE,
        "showscale": True,
        "colorbar": modern_colorbar("Delta PP to slip (PSI)"),
    }


def _stereonet_controls_script(cmin: float, cmax: float):
    scale_js = [
        [float(stop), _hex_to_rgb(color)]
        for stop, color in SLIP_PRESSURE_COLOR_SCALE
    ]
    return f"""
  <style>
    .stereonet-controls {{
      position: absolute;
      left: 74px;
      right: 90px;
      bottom: 14px;
      z-index: 10;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      color: {MODERN_TEXT_COLOR};
      font-size: 12px;
      line-height: 1;
      pointer-events: none;
    }}
    .stereonet-controls label {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
      pointer-events: auto;
    }}
    .stereonet-controls input {{
      width: 88px;
      box-sizing: border-box;
      padding: 5px 7px;
      color: {MODERN_TEXT_COLOR};
      background: {MODERN_CONTROL_BG};
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      font-size: 12px;
    }}
  </style>
  <div class="stereonet-controls" aria-label="Delta PP to slip PSI color range">
    <label>Min PSI <input id="stereonet-min-psi" type="number" step="any" value="{cmin:.6g}"></label>
    <label>Max PSI <input id="stereonet-max-psi" type="number" step="any" value="{cmax:.6g}"></label>
  </div>
  <script>
    (function () {{
      const originalMin = {cmin};
      const originalMax = {cmax};
      const scale = {scale_js};
      const minInput = document.getElementById('stereonet-min-psi');
      const maxInput = document.getElementById('stereonet-max-psi');
      let initialAfterplotApplied = false;

      function findPlot() {{
        return document.querySelector('.plot-shell .js-plotly-plot');
      }}

      function toHex(rgb) {{
        return '#' + rgb.map(function (v) {{
          return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
        }}).join('');
      }}

      function colorForValue(value, minValue, maxValue) {{
        let t = maxValue > minValue ? (Number(value) - minValue) / (maxValue - minValue) : 0;
        t = Math.max(0, Math.min(1, t));
        if (t <= scale[0][0]) return toHex(scale[0][1]);
        for (let i = 1; i < scale.length; i++) {{
          if (t <= scale[i][0]) {{
            const span = scale[i][0] - scale[i - 1][0] || 1;
            const ratio = (t - scale[i - 1][0]) / span;
            return toHex([0, 1, 2].map(function (channel) {{
              return scale[i - 1][1][channel] + (scale[i][1][channel] - scale[i - 1][1][channel]) * ratio;
            }}));
          }}
        }}
        return toHex(scale[scale.length - 1][1]);
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

        const markerTraceIndexes = [];
        const markerUpdates = {{'marker.cmin': [], 'marker.cmax': []}};
        const contourTraceIndexes = [];
        const contourUpdates = {{zmin: [], zmax: []}};

        plot.data.forEach(function (trace, index) {{
          if (trace.marker && trace.marker.colorscale) {{
            markerTraceIndexes.push(index);
            markerUpdates['marker.cmin'].push(minValue);
            markerUpdates['marker.cmax'].push(maxValue);
          }}
          if (trace.type === 'contour') {{
            contourTraceIndexes.push(index);
            contourUpdates.zmin.push(minValue);
            contourUpdates.zmax.push(maxValue);
          }}
        }});
        if (markerTraceIndexes.length) {{
          Plotly.restyle(plot, markerUpdates, markerTraceIndexes);
        }}
        if (contourTraceIndexes.length) {{
          Plotly.restyle(plot, contourUpdates, contourTraceIndexes);
        }}
      }}

      minInput.addEventListener('input', applyRange);
      maxInput.addEventListener('input', applyRange);
      minInput.addEventListener('change', applyRange);
      maxInput.addEventListener('change', applyRange);

      function applyWhenReady(attempt) {{
        const plot = findPlot();
        if (plot && typeof Plotly !== 'undefined') {{
          applyRange();
          if (plot.on) {{
            plot.on('plotly_afterplot', function () {{
              if (initialAfterplotApplied) return;
              initialAfterplotApplied = true;
              window.setTimeout(applyRange, 0);
            }});
          }}
          return;
        }}
        if (attempt < 20) {{
          window.setTimeout(function () {{
            applyWhenReady(attempt + 1);
          }}, 100);
        }}
      }}

      applyWhenReady(0);
    }})();
  </script>
"""


def _circle_trace():
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    return go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode="lines",
        line={"color": "#0f172a", "width": 1.7},
        hoverinfo="skip",
        showlegend=False,
        name="Stereonet Boundary",
    )


def _grid_traces():
    traces = []
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    for dip in range(15, 90, 15):
        radius = dip / 90.0
        traces.append(go.Scatter(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            mode="lines",
            line={"color": MODERN_GRID_COLOR, "width": 1, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        ))
    for azimuth in range(0, 360, 30):
        angle = azimuth * np.pi / 180.0
        traces.append(go.Scatter(
            x=[0.0, math.cos(angle)],
            y=[0.0, math.sin(angle)],
            mode="lines",
            line={"color": "#cbd5e1", "width": 1, "dash": "dot"},
            hoverinfo="skip",
            showlegend=False,
        ))
    traces.append(_circle_trace())
    return traces


def _stress_arrow_trace(max_stress_azimuth):
    azimuth = float(max_stress_azimuth or 0.0)
    trig_angle = math.radians((90.0 - azimuth + 180.0) % 360.0)
    dx = math.cos(trig_angle)
    dy = math.sin(trig_angle)
    return go.Scatter(
        x=[-1.34 * dx, 1.34 * dx],
        y=[-1.34 * dy, 1.34 * dy],
        mode="lines+markers",
        line={"color": MODERN_AXIS_COLOR, "width": 3},
        marker={"symbol": ["arrow", "arrow"], "size": 14, "angleref": "previous", "color": MODERN_AXIS_COLOR},
        hoverinfo="skip",
        showlegend=False,
        name="SHmax Azimuth",
    )


def _label_annotations():
    labels = [
        ("N", 90), ("30", 60), ("60", 30), ("E", 0), ("120", 330), ("150", 300),
        ("S", 270), ("210", 240), ("240", 210), ("W", 180), ("300", 150), ("330", 120),
    ]
    annotations = []
    for text, angle_deg in labels:
        angle = math.radians(angle_deg)
        annotations.append({
            "xref": "x",
            "yref": "y",
            "x": 1.1 * math.cos(angle),
            "y": 1.1 * math.sin(angle),
            "text": text,
            "showarrow": False,
            "font": {"size": 15 if text in {"N", "E", "S", "W"} else 12, "color": MODERN_TEXT_COLOR},
            "xanchor": "center",
            "yanchor": "middle",
        })
    return annotations


def _mode_buttons(base_count, normal_count, curve_count, composite_count):
    total = base_count + normal_count + curve_count + composite_count

    def visible_for(mode_start, mode_count, *, composite=False):
        if composite and base_count >= 2:
            base_visible = [False] * base_count
            base_visible[base_count - 2] = True  # boundary
            base_visible[base_count - 1] = True  # SHmax arrow
        else:
            base_visible = [True] * base_count
        visible = base_visible + [False] * (total - base_count)
        for index in range(mode_start, mode_start + mode_count):
            visible[index] = True
        return visible

    normal_start = base_count
    curve_start = normal_start + normal_count
    composite_start = curve_start + curve_count
    return [
        {"label": "Fault Normals", "method": "update", "args": [{"visible": visible_for(normal_start, normal_count)}]},
        {"label": "Projected Curves", "method": "update", "args": [{"visible": visible_for(curve_start, curve_count)}]},
        {
            "label": "Normal Composite",
            "method": "update",
            "args": [{"visible": visible_for(composite_start, composite_count, composite=True)}],
        },
    ]


def save_stereonet_graph_artifact(
    helper,
    faults_df: pd.DataFrame,
    stress_state: StressState,
    p0: float,
    friction: float,
    max_stress_azimuth,
    *,
    step_index: int = 1,
    artifact_key: str = "fsp-deterministic-geomechanics-stereonet",
    title: str = "Stereonet",
    display_order: int = 22,
):
    """Generate and register the deterministic geomechanics stereonet artifact."""
    try:
        remove_step_messages(helper, step_index, MESSAGE_PREFIX)
        faults = _clean_faults(faults_df)
        if faults.empty:
            add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX} because required fault columns are missing or empty.")
            return None

        composite_state = _composite_stress_state(stress_state)
        composite = normal_composite_grid(composite_state, p0, friction)
        composite_x, composite_y, composite_z = _normal_composite_heatmap(composite_state, p0, friction)
        cmin, cmax = _pressure_range(faults, composite)

        fig = go.Figure()
        base_traces = _grid_traces()
        for trace in base_traces:
            fig.add_trace(trace)
        fig.add_trace(_stress_arrow_trace(max_stress_azimuth))
        base_count = len(base_traces) + 1

        normal_x, normal_y = fault_normal_projection(faults["Strike"], faults["Dip"])
        hover = [
            f"Fault: {row['FaultID']}<br>Strike: {float(row['Strike']):.1f} deg<br>"
            f"Dip: {float(row['Dip']):.1f} deg<br>Delta PP to slip: {float(row['slip_pressure']):,.2f} PSI"
            for _, row in faults.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=normal_x,
            y=normal_y,
            mode="markers",
            name="Fault Normals",
            text=hover,
            marker={
                "size": 11,
                "color": faults["slip_pressure"],
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": SLIP_PRESSURE_COLOR_SCALE,
                "showscale": True,
                "colorbar": {
                    **modern_colorbar("Delta PP to slip (PSI)"),
                },
                "line": {"width": 1, "color": "#0f172a"},
            },
            hovertemplate="%{text}<extra></extra>",
        ))
        normal_count = 1

        curve_x = []
        curve_y = []
        curve_pressure = []
        curve_hover = []
        for _, row in faults.iterrows():
            x, y = projected_curve(row["Strike"], row["Dip"], rake_count=361)
            slip_pressure = float(row["slip_pressure"])
            curve_x.extend(x.tolist())
            curve_y.extend(y.tolist())
            curve_pressure.extend([slip_pressure] * len(x))
            curve_hover.extend([
                f"Fault: {row['FaultID']}<br>Strike: {float(row['Strike']):.1f} deg<br>"
                f"Dip: {float(row['Dip']):.1f} deg<br>Delta PP to slip: {slip_pressure:,.2f} PSI"
            ] * len(x))
        fig.add_trace(go.Scattergl(
            x=curve_x,
            y=curve_y,
            mode="markers",
            name="Projected Curves",
            text=curve_hover,
            visible=False,
            marker={
                "size": 3.6,
                "symbol": "circle",
                "color": curve_pressure,
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": SLIP_PRESSURE_COLOR_SCALE,
                "showscale": True,
                "colorbar": {
                    **modern_colorbar("Delta PP to slip (PSI)"),
                },
            },
            hovertemplate="%{text}<extra></extra>",
        ))
        curve_count = 1

        fig.add_trace(go.Contour(
            x=composite_x,
            y=composite_y,
            z=composite_z,
            name="Normal Composite",
            visible=False,
            zmin=cmin,
            zmax=cmax,
            colorscale=SLIP_PRESSURE_COLOR_SCALE,
            showscale=True,
            colorbar={
                **modern_colorbar("Delta PP to slip (PSI)"),
            },
            contours={"coloring": "heatmap", "showlines": False},
            line={"width": 0},
            connectgaps=False,
            hovertemplate=(
                "x: %{x:.2f}<br>y: %{y:.2f}<br>"
                "Delta PP to slip: %{z:,.2f} PSI<extra></extra>"
            ),
        ))
        composite_count = 1

        fig.update_layout(
            autosize=True,
            margin={"l": 42, "r": 88, "t": 70, "b": 34},
            font={"family": MODERN_FONT_FAMILY, "size": 12, "color": MODERN_TEXT_COLOR},
            xaxis={
                "range": [-1.42, 1.42],
                "scaleanchor": "y",
                "scaleratio": 1,
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "visible": False,
            },
            yaxis={
                "range": [-1.42, 1.42],
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "visible": False,
            },
            updatemenus=[modern_updatemenu(
                _mode_buttons(base_count, normal_count, curve_count, composite_count),
                x=0.0,
                y=1.12,
            )],
            annotations=_label_annotations(),
            hovermode="closest",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            hoverlabel={
                "bgcolor": "#0f172a",
                "bordercolor": "#334155",
                "font": {"family": MODERN_FONT_FAMILY, "color": "#f8fafc"},
            },
            showlegend=False,
        )

        return write_plotly_artifact(
            helper,
            fig,
            artifact_key,
            title,
            caption="Interactive deterministic geomechanics stereonet showing fault normals, projected curves, and composite slip pressure.",
            display_order=display_order,
            preferred_height=620,
            dark=False,
            extra_body=_stereonet_controls_script(cmin, cmax),
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX}: {exc}")
        return None
