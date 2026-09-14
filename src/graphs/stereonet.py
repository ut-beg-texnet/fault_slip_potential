"""Stereonet HTML graph artifact for deterministic geomechanics."""
import html
import json
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
    MODERN_PLOT_BG,
    MODERN_SHADOW,
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
CURVE_RAKE_COUNT = 361
SELECTOR_ROW_HEIGHT = 32

# Toolbar + overlay combobox. Injected via extra_head so the dropdown can paint
# over the plot without growing the shell as fault count changes.
_STEREONET_SELECTOR_HEAD = f"""
  <style>
    .plot-shell {{
      overflow: visible;
    }}
    .plot-shell-content {{
      position: relative;
      z-index: 1;
      overflow: hidden;
    }}
    .stereonet-toolbar {{
      position: relative;
      z-index: 20;
      flex: 0 0 auto;
      display: flex;
      align-items: center;
      min-width: 0;
    }}
    .stereonet-fault-picker {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      flex: 1 1 auto;
      color: {MODERN_TEXT_COLOR};
      font-size: 12px;
    }}
    .stereonet-combobox {{
      position: relative;
      min-width: 0;
      flex: 1 1 auto;
      max-width: 420px;
    }}
    .stereonet-fault-toggle {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
      padding: 6px 10px;
      color: {MODERN_TEXT_COLOR};
      background: {MODERN_CONTROL_BG};
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }}
    .stereonet-fault-label {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .stereonet-fault-menu {{
      position: absolute;
      left: 0;
      right: 0;
      top: calc(100% + 4px);
      z-index: 30;
      display: flex;
      flex-direction: column;
      max-height: min(280px, 45vh);
      overflow: hidden;
      background: {MODERN_PLOT_BG};
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      box-shadow: {MODERN_SHADOW};
    }}
    .stereonet-fault-menu[hidden] {{
      display: none;
    }}
    .stereonet-fault-filter {{
      flex: 0 0 auto;
      margin: 8px 8px 6px;
      width: calc(100% - 16px);
      box-sizing: border-box;
      padding: 6px 8px;
      color: {MODERN_TEXT_COLOR};
      background: {MODERN_CONTROL_BG};
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      font: inherit;
      font-size: 12px;
    }}
    .stereonet-fault-viewport {{
      flex: 1 1 auto;
      min-height: 0;
      max-height: 232px;
      overflow-y: auto;
    }}
    .stereonet-fault-list {{
      position: relative;
    }}
    .stereonet-fault-item {{
      display: block;
      width: 100%;
      height: {SELECTOR_ROW_HEIGHT}px;
      box-sizing: border-box;
      padding: 0 10px;
      border: 0;
      background: transparent;
      color: {MODERN_TEXT_COLOR};
      font: inherit;
      font-size: 12px;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      cursor: pointer;
    }}
    .stereonet-fault-item:hover,
    .stereonet-fault-item[aria-selected="true"] {{
      background: rgba(148, 163, 184, 0.16);
    }}
    .stereonet-fault-empty {{
      padding: 10px;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 12px;
    }}
  </style>
"""

_STEREONET_SELECTOR_SHELL = """
    <div class="stereonet-toolbar">
      <label class="stereonet-fault-picker">
        <span>Fault</span>
        <div class="stereonet-combobox" id="stereonet-combobox">
          <button type="button" class="stereonet-fault-toggle" id="stereonet-fault-toggle"
                  aria-haspopup="listbox" aria-expanded="false" aria-controls="stereonet-fault-menu">
            <span class="stereonet-fault-label" id="stereonet-fault-label">__DEFAULT_FAULT_LABEL__</span>
          </button>
          <div class="stereonet-fault-menu" id="stereonet-fault-menu" hidden>
            <input type="search" class="stereonet-fault-filter" id="stereonet-fault-filter"
                   placeholder="Search faults..." aria-label="Search faults" autocomplete="off">
            <div class="stereonet-fault-viewport" id="stereonet-fault-viewport">
              <div id="stereonet-fault-spacer">
                <div class="stereonet-fault-list" id="stereonet-fault-list" role="listbox"></div>
              </div>
            </div>
          </div>
        </div>
      </label>
    </div>
"""


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


def _fault_hover(fault_id, strike, dip, slip) -> str:
    """Hover HTML for a single selected fault (pole or curve)."""
    return (
        f"Fault: {fault_id}<br>"
        f"Strike: {float(strike):.1f} deg<br>"
        f"Dip: {float(dip):.1f} deg<br>"
        f"Delta PP to slip: {float(slip):,.2f} PSI"
    )


def _fault_payload(faults: pd.DataFrame):
    """Compact per-fault records. Geometry is computed client-side for the selection."""
    payload = []
    for row in faults.itertuples(index=False):
        payload.append({
            "id": str(row.FaultID),
            "strike": float(row.Strike),
            "dip": float(row.Dip),
            "slip": float(row.slip_pressure),
        })
    return payload


def _stereonet_controls_script(
    cmin: float,
    cmax: float,
    faults_payload,
    *,
    rake_count: int,
    normal_index: int,
    curve_index: int,
    composite_index: int,
):
    """PSI range controls plus selected-fault geometry / combobox behavior."""
    scale_json = json.dumps(
        [[float(stop), list(_hex_to_rgb(color))] for stop, color in SLIP_PRESSURE_COLOR_SCALE],
        separators=(",", ":"),
    )
    script = """
  <style>
    .stereonet-controls {
      position: absolute;
      left: 74px;
      right: 90px;
      bottom: 14px;
      z-index: 10;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      color: __TEXT_COLOR__;
      font-size: 12px;
      line-height: 1;
      pointer-events: none;
    }
    .stereonet-controls label {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
      pointer-events: auto;
    }
    .stereonet-controls input {
      width: 88px;
      box-sizing: border-box;
      padding: 5px 7px;
      color: __TEXT_COLOR__;
      background: __CONTROL_BG__;
      border: 1px solid __BORDER_COLOR__;
      border-radius: 6px;
      font-size: 12px;
    }
  </style>
  <div class="stereonet-controls" aria-label="Delta PP to slip PSI color range">
    <label>Min PSI <input id="stereonet-min-psi" type="number" step="any" value="__CMIN_ATTR__"></label>
    <label>Max PSI <input id="stereonet-max-psi" type="number" step="any" value="__CMAX_ATTR__"></label>
  </div>
  <script>
    (function () {
      const faults = __FAULTS_JSON__;
      const originalMin = __CMIN__;
      const originalMax = __CMAX__;
      const scale = __SCALE_JSON__;
      const rakeCount = __RAKE_COUNT__;
      const normalIndex = __NORMAL_INDEX__;
      const curveIndex = __CURVE_INDEX__;
      const compositeIndex = __COMPOSITE_INDEX__;
      const rowHeight = __ROW_HEIGHT__;
      const minInput = document.getElementById('stereonet-min-psi');
      const maxInput = document.getElementById('stereonet-max-psi');
      const toggle = document.getElementById('stereonet-fault-toggle');
      const label = document.getElementById('stereonet-fault-label');
      const menu = document.getElementById('stereonet-fault-menu');
      const filterInput = document.getElementById('stereonet-fault-filter');
      const viewport = document.getElementById('stereonet-fault-viewport');
      const spacer = document.getElementById('stereonet-fault-spacer');
      const list = document.getElementById('stereonet-fault-list');
      const combobox = document.getElementById('stereonet-combobox');
      let selectedIndex = 0;
      let filtered = faults.map(function (_, index) { return index; });
      let initialAfterplotApplied = false;

      function findPlot() {
        return document.querySelector('.plot-shell .js-plotly-plot');
      }

      function placeControls() {
        const plotContent = document.querySelector('.plot-shell-content');
        const controls = document.querySelector('.stereonet-controls');
        if (plotContent && controls && controls.parentElement !== plotContent) {
          plotContent.appendChild(controls);
        }
      }

      function toHex(rgb) {
        return '#' + rgb.map(function (v) {
          return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
        }).join('');
      }

      function colorForValue(value, minValue, maxValue) {
        let t = maxValue > minValue ? (Number(value) - minValue) / (maxValue - minValue) : 0;
        t = Math.max(0, Math.min(1, t));
        if (t <= scale[0][0]) return toHex(scale[0][1]);
        for (let i = 1; i < scale.length; i++) {
          if (t <= scale[i][0]) {
            const span = scale[i][0] - scale[i - 1][0] || 1;
            const ratio = (t - scale[i - 1][0]) / span;
            return toHex([0, 1, 2].map(function (channel) {
              return scale[i - 1][1][channel] + (scale[i][1][channel] - scale[i - 1][1][channel]) * ratio;
            }));
          }
        }
        return toHex(scale[scale.length - 1][1]);
      }

      function validRange(minValue, maxValue) {
        return Number.isFinite(minValue) && Number.isFinite(maxValue) && minValue < maxValue;
      }

      function currentRange() {
        let minValue = Number.parseFloat(minInput.value);
        let maxValue = Number.parseFloat(maxInput.value);
        if (!validRange(minValue, maxValue)) {
          minValue = originalMin;
          maxValue = originalMax;
          minInput.value = originalMin;
          maxInput.value = originalMax;
        }
        return {minValue: minValue, maxValue: maxValue};
      }

      // Match Python fault_normal_projection / projected_curve (linspace 0..pi).
      function faultNormalProjection(strike, dip) {
        const strikeRad = (90.0 - strike) * Math.PI / 180.0;
        const theta = strikeRad + Math.PI / 2.0;
        const rho = dip / 90.0;
        return [rho * Math.cos(theta), rho * Math.sin(theta)];
      }

      function projectedCurve(strike, dip) {
        const strikeRad = (90.0 - strike) * Math.PI / 180.0;
        const dipRad = dip * Math.PI / 180.0;
        const xs = new Array(rakeCount);
        const ys = new Array(rakeCount);
        const denom = Math.max(rakeCount - 1, 1);
        for (let i = 0; i < rakeCount; i++) {
          const rake = Math.PI * i / denom;
          const plunge = Math.asin(Math.sin(dipRad) * Math.sin(rake));
          const trend = strikeRad + Math.atan2(Math.cos(dipRad) * Math.sin(rake), Math.cos(rake));
          const rho = Math.tan(Math.PI / 4.0 - plunge / 2.0);
          const theta = trend + Math.PI;
          xs[i] = rho * Math.cos(theta);
          ys[i] = rho * Math.sin(theta);
        }
        return [xs, ys];
      }

      function faultHover(fault) {
        const slipText = Number(fault.slip).toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        });
        return 'Fault: ' + fault.id +
          '<br>Strike: ' + Number(fault.strike).toFixed(1) + ' deg' +
          '<br>Dip: ' + Number(fault.dip).toFixed(1) + ' deg' +
          '<br>Delta PP to slip: ' + slipText + ' PSI';
      }

      function escapeHtml(value) {
        return String(value)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;');
      }

      function applyRangeAndSelection() {
        const plot = findPlot();
        if (!plot || typeof Plotly === 'undefined' || !faults.length) return;
        const range = currentRange();
        const fault = faults[selectedIndex];
        const pole = faultNormalProjection(fault.strike, fault.dip);
        const curve = projectedCurve(fault.strike, fault.dip);
        const hover = faultHover(fault) + '<extra></extra>';
        const lineColor = colorForValue(fault.slip, range.minValue, range.maxValue);

        Plotly.restyle(plot, {
          x: [[pole[0]]],
          y: [[pole[1]]],
          hovertemplate: hover,
          'marker.color': [[fault.slip]],
          'marker.cmin': range.minValue,
          'marker.cmax': range.maxValue
        }, [normalIndex]);
        Plotly.restyle(plot, {
          x: [curve[0]],
          y: [curve[1]],
          hovertemplate: hover,
          'marker.color': [[fault.slip]],
          'marker.cmin': range.minValue,
          'marker.cmax': range.maxValue,
          'line.color': lineColor
        }, [curveIndex]);
        Plotly.restyle(plot, {
          zmin: range.minValue,
          zmax: range.maxValue
        }, [compositeIndex]);
      }

      function applyRange() {
        applyRangeAndSelection();
      }

      function setSelectedIndex(index) {
        if (index < 0 || index >= faults.length) return;
        selectedIndex = index;
        label.textContent = faults[index].id;
        applyRangeAndSelection();
      }

      function rebuildFiltered() {
        const query = String(filterInput.value || '').toLowerCase();
        filtered = [];
        for (let i = 0; i < faults.length; i++) {
          if (!query || String(faults[i].id).toLowerCase().indexOf(query) !== -1) {
            filtered.push(i);
          }
        }
      }

      function renderList() {
        if (!filtered.length) {
          spacer.style.height = '40px';
          list.style.transform = 'translateY(0)';
          list.innerHTML = '<div class="stereonet-fault-empty">No matching faults</div>';
          return;
        }
        const viewportHeight = viewport.clientHeight || 232;
        const buffer = 6;
        const start = Math.max(0, Math.floor(viewport.scrollTop / rowHeight) - buffer);
        const visibleCount = Math.ceil(viewportHeight / rowHeight) + buffer * 2;
        const end = Math.min(filtered.length, start + visibleCount);
        spacer.style.height = (filtered.length * rowHeight) + 'px';
        list.style.transform = 'translateY(' + (start * rowHeight) + 'px)';
        let html = '';
        for (let i = start; i < end; i++) {
          const index = filtered[i];
          const selected = index === selectedIndex ? ' aria-selected="true"' : '';
          html += '<button type="button" class="stereonet-fault-item" role="option" data-index="' +
            index + '"' + selected + '>' + escapeHtml(faults[index].id) + '</button>';
        }
        list.innerHTML = html;
      }

      function openMenu() {
        menu.hidden = false;
        toggle.setAttribute('aria-expanded', 'true');
        filterInput.value = '';
        rebuildFiltered();
        const selectedPos = filtered.indexOf(selectedIndex);
        viewport.scrollTop = selectedPos >= 0 ? selectedPos * rowHeight : 0;
        renderList();
        filterInput.focus();
      }

      function closeMenu() {
        menu.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
      }

      minInput.addEventListener('input', applyRange);
      maxInput.addEventListener('input', applyRange);
      minInput.addEventListener('change', applyRange);
      maxInput.addEventListener('change', applyRange);

      toggle.addEventListener('click', function () {
        if (menu.hidden) openMenu();
        else closeMenu();
      });
      filterInput.addEventListener('input', function () {
        rebuildFiltered();
        viewport.scrollTop = 0;
        renderList();
      });
      viewport.addEventListener('scroll', renderList);
      list.addEventListener('click', function (event) {
        const item = event.target.closest('[data-index]');
        if (!item) return;
        setSelectedIndex(Number(item.getAttribute('data-index')));
        closeMenu();
      });
      document.addEventListener('mousedown', function (event) {
        if (!menu.hidden && combobox && !combobox.contains(event.target)) {
          closeMenu();
        }
      });
      document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape') closeMenu();
      });

      function applyWhenReady(attempt) {
        placeControls();
        const plot = findPlot();
        if (plot && typeof Plotly !== 'undefined') {
          applyRangeAndSelection();
          if (plot.on) {
            plot.on('plotly_afterplot', function () {
              if (initialAfterplotApplied) return;
              initialAfterplotApplied = true;
              window.setTimeout(applyRangeAndSelection, 0);
            });
          }
          return;
        }
        if (attempt < 20) {
          window.setTimeout(function () {
            applyWhenReady(attempt + 1);
          }, 100);
        }
      }

      if (faults.length) {
        label.textContent = faults[0].id;
      }
      applyWhenReady(0);
    })();
  </script>
"""
    replacements = {
        "__TEXT_COLOR__": MODERN_TEXT_COLOR,
        "__CONTROL_BG__": MODERN_CONTROL_BG,
        "__BORDER_COLOR__": MODERN_BORDER_COLOR,
        "__CMIN_ATTR__": f"{cmin:.6g}",
        "__CMAX_ATTR__": f"{cmax:.6g}",
        "__CMIN__": json.dumps(cmin),
        "__CMAX__": json.dumps(cmax),
        "__SCALE_JSON__": scale_json,
        "__RAKE_COUNT__": str(int(rake_count)),
        "__NORMAL_INDEX__": str(int(normal_index)),
        "__CURVE_INDEX__": str(int(curve_index)),
        "__COMPOSITE_INDEX__": str(int(composite_index)),
        "__ROW_HEIGHT__": str(int(SELECTOR_ROW_HEIGHT)),
        "__FAULTS_JSON__": json.dumps(faults_payload, separators=(",", ":")),
    }
    for token, value in replacements.items():
        script = script.replace(token, value)
    return script


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
        payload = _fault_payload(faults)
        default_fault = payload[0]
        hover = _fault_hover(
            default_fault["id"], default_fault["strike"], default_fault["dip"], default_fault["slip"]
        )

        fig = go.Figure()
        base_traces = _grid_traces()
        for trace in base_traces:
            fig.add_trace(trace)
        fig.add_trace(_stress_arrow_trace(max_stress_azimuth))
        base_count = len(base_traces) + 1

        # Seed only the default fault; JS restyles these traces on selection.
        normal_x, normal_y = fault_normal_projection([default_fault["strike"]], [default_fault["dip"]])
        fig.add_trace(go.Scatter(
            x=[float(normal_x[0])],
            y=[float(normal_y[0])],
            mode="markers",
            name="Fault Normals",
            hovertemplate=hover + "<extra></extra>",
            marker={
                "size": 11,
                "color": [default_fault["slip"]],
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": SLIP_PRESSURE_COLOR_SCALE,
                "showscale": True,
                "colorbar": {
                    **modern_colorbar("Delta PP to slip (PSI)"),
                },
                "line": {"width": 1, "color": "#0f172a"},
            },
        ))
        normal_count = 1
        normal_index = base_count

        curve_x, curve_y = projected_curve(
            default_fault["strike"], default_fault["dip"], rake_count=CURVE_RAKE_COUNT
        )
        fig.add_trace(go.Scatter(
            x=curve_x,
            y=curve_y,
            mode="lines",
            name="Projected Curves",
            hovertemplate=hover + "<extra></extra>",
            visible=False,
            line={
                "color": _color_for_value(default_fault["slip"], cmin, cmax),
                "width": 2.8,
            },
            marker={
                "size": 0.1,
                "opacity": 0.0,
                "color": [default_fault["slip"]],
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": SLIP_PRESSURE_COLOR_SCALE,
                "showscale": True,
                "colorbar": {
                    **modern_colorbar("Delta PP to slip (PSI)"),
                },
            },
        ))
        curve_count = 1
        curve_index = normal_index + normal_count

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
        composite_index = curve_index + curve_count

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
            extra_head=_STEREONET_SELECTOR_HEAD,
            extra_shell_html=_STEREONET_SELECTOR_SHELL.replace(
                "__DEFAULT_FAULT_LABEL__",
                html.escape(default_fault["id"], quote=True),
            ),
            extra_body=_stereonet_controls_script(
                cmin,
                cmax,
                payload,
                rake_count=CURVE_RAKE_COUNT,
                normal_index=normal_index,
                curve_index=curve_index,
                composite_index=composite_index,
            ),
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{MESSAGE_PREFIX}: {exc}")
        return None
