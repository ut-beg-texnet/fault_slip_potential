"""Direct scientific graph artifacts for probabilistic FSP outputs."""
import html
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from graphs.artifacts import (
    MODERN_AXIS_COLOR,
    MODERN_BORDER_COLOR,
    MODERN_CONTROL_BG,
    MODERN_FONT_FAMILY,
    MODERN_GRID_COLOR,
    MODERN_HEAT_COLORSCALE,
    MODERN_MUTED_TEXT_COLOR,
    MODERN_PAPER_BG,
    MODERN_PLOT_BG,
    MODERN_SHADOW,
    MODERN_TEXT_COLOR,
    PLOTLY_CONFIG,
    SCIENTIFIC_COLORS,
    SLIP_PRESSURE_COLOR_SCALE,
    add_graph_warning,
    apply_modern_subplots_layout,
    apply_scientific_layout,
    graph_artifacts_dir,
    has_columns,
    modern_colorbar,
    modern_updatemenu,
    remove_step_messages,
    remove_graph_artifact,
    write_plotly_artifact,
)


def _warn_prefix(title):
    return f"{title} graph was not generated"


def _write_html_artifact(
    helper,
    *,
    artifact_key: str,
    title: str,
    html_text: str,
    caption: str,
    display_order: int,
    preferred_height: int,
):
    output_path = os.path.join(graph_artifacts_dir(helper), f"{artifact_key}.html")
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_text)

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


def _clean(df, numeric_columns):
    result = df.copy()
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=numeric_columns)


def _visibility_mask(trace_count: int, visible_index=None):
    if visible_index is None:
        return [True] * trace_count
    if visible_index == "none":
        return [False] * trace_count
    return [index == visible_index for index in range(trace_count)]


def _hex_to_rgb(hex_color: str):
    color = str(hex_color).lstrip("#")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interpolate_colorscale(stops, normalized: float) -> str:
    palette = [(float(stop), _hex_to_rgb(color)) for stop, color in stops]
    if not palette:
        return SCIENTIFIC_COLORS[0]

    normalized = max(0.0, min(1.0, float(normalized)))
    if normalized <= palette[0][0]:
        return _rgb_to_hex(palette[0][1])

    for index in range(1, len(palette)):
        stop, color = palette[index]
        previous_stop, previous_color = palette[index - 1]
        if normalized <= stop:
            span = stop - previous_stop or 1.0
            ratio = (normalized - previous_stop) / span
            blended = tuple(
                int(round(previous_color[channel] + (color[channel] - previous_color[channel]) * ratio))
                for channel in range(3)
            )
            return _rgb_to_hex(blended)

    return _rgb_to_hex(palette[-1][1])


def _series_colors_from_payload(series_payload: dict) -> dict:
    deterministic_pressures = {}
    for series_id, payload in series_payload.items():
        value = payload.get("detSlipPressure")
        if value is None:
            continue
        try:
            deterministic_pressures[series_id] = float(value)
        except (TypeError, ValueError):
            continue

    if deterministic_pressures:
        min_value = min(deterministic_pressures.values())
        max_value = max(deterministic_pressures.values())
        span = max_value - min_value
        colors = {}
        for series_id in series_payload:
            value = deterministic_pressures.get(series_id)
            if value is None:
                colors[series_id] = SCIENTIFIC_COLORS[len(colors) % len(SCIENTIFIC_COLORS)]
                continue
            normalized = 0.5 if span <= 0 else (value - min_value) / span
            colors[series_id] = _interpolate_colorscale(SLIP_PRESSURE_COLOR_SCALE, normalized)
        return colors

    return {
        series_id: SCIENTIFIC_COLORS[index % len(SCIENTIFIC_COLORS)]
        for index, series_id in enumerate(series_payload)
    }


def _deterministic_pressure_range_from_payload(series_payload: dict):
    values = []
    for payload in series_payload.values():
        value = payload.get("detSlipPressure")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        values.append(numeric_value)

    if not values:
        return None, None

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    return round(min_value, 6), round(max_value, 6)


def _single_series_plotly_html(
    *,
    title: str,
    series_payload: dict,
    default_id: str,
    x_label: str,
    y_label: str,
    auto_color_min=None,
    auto_color_max=None,
    y_range=None,
    x_type: str = "linear",
    y_tickformat: Optional[str] = None,
    year_of_interest: Optional[float] = None,
    show_fsp_background: bool = False,
    show_color_tab: bool = True,
    color_tab_label: str = "Color Range",
    default_color_min: Optional[float] = None,
    default_color_max: Optional[float] = None,
):
    title_html = html.escape(title)
    payload_json = json.dumps(series_payload, separators=(",", ":"))
    default_id_json = json.dumps(default_id)
    title_json = json.dumps(title)
    x_label_json = json.dumps(x_label)
    y_label_json = json.dumps(y_label)
    y_range_json = json.dumps(y_range)
    x_type_json = json.dumps(x_type)
    y_tickformat_json = json.dumps(y_tickformat)
    config_json = json.dumps(PLOTLY_CONFIG)
    series_ids_json = json.dumps(list(series_payload.keys()))
    auto_color_min_json = json.dumps(auto_color_min)
    auto_color_max_json = json.dumps(auto_color_max)
    color_scale_json = json.dumps(SLIP_PRESSURE_COLOR_SCALE, separators=(",", ":"))
    year_of_interest_json = json.dumps(year_of_interest)
    show_fsp_background_json = json.dumps(show_fsp_background)
    show_color_tab_json = json.dumps(show_color_tab)
    # Use explicit defaults when provided; fall back to auto_color_min/max
    init_color_min = default_color_min if default_color_min is not None else auto_color_min
    init_color_max = default_color_max if default_color_max is not None else auto_color_max
    init_color_min_json = json.dumps(init_color_min)
    init_color_max_json = json.dumps(init_color_max)
    # Derive input unit label: empty for FSP (0-1), "psi" for pressure
    color_unit = "" if show_fsp_background else " PSI"
    color_tab_desc = (
        "Adjust the FSP range to control the gradient background coloring. Min and max default to 0 and 1."
        if show_fsp_background
        else "Adjust the deterministic slip-pressure range to dynamically recolor the CDF curves and legend entries."
    )
    color_tab_desc_html = html.escape(color_tab_desc)
    color_tab_label_html = html.escape(color_tab_label)
    # Pre-render conditional HTML fragments to avoid complex f-string logic
    year_toolbar_html = (
        f'<label class="toolbar-year-control">'
        f'<span class="toolbar-year-label">Year of Interest</span>'
        f'<input id="year-input" type="number" class="toolbar-year-input" '
        f'value="{int(year_of_interest)}" oninput="onYearInput()">'
        f'</label>'
    ) if year_of_interest is not None else ""
    color_tab_button_html = (
        '<button type="button" id="colors-tab-button" class="tab-button"'
        ' onclick="switchControlTab(\'colors\')">'
        + color_tab_label_html
        + "</button>"
    ) if show_color_tab else ""
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
      background: {MODERN_PAPER_BG};
      color: {MODERN_TEXT_COLOR};
      font-family: {MODERN_FONT_FAMILY};
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }}
    .viewer {{
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: {MODERN_PAPER_BG};
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 48px;
      padding: 8px 12px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      box-sizing: border-box;
      background: {MODERN_CONTROL_BG};
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
    }}
    .toolbar-title {{
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }}
    .toolbar-status {{
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 13px;
      white-space: nowrap;
    }}
    .content {{
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .plot-panel {{
      min-width: 0;
      min-height: 0;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
      background: {MODERN_PLOT_BG};
    }}
    .plot-stage {{
      width: 100%;
      height: 100%;
      min-height: 0;
    }}
    .legend-panel {{
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
    }}
    .legend-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 12px 6px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      flex: 0 0 auto;
    }}
    .legend-title {{
      font-weight: 700;
    }}
    .tab-strip {{
      display: inline-flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .tab-button {{
      font: inherit;
      font-size: 11.5px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 999px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_MUTED_TEXT_COLOR};
      padding: 4px 10px;
      cursor: pointer;
    }}
    .tab-button.is-active {{
      background: rgba(37, 99, 235, 0.10);
      color: {MODERN_TEXT_COLOR};
      border-color: rgba(37, 99, 235, 0.28);
    }}
    .panel-body {{
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .tab-panel {{
      display: none;
      flex: 1 1 auto;
      min-height: 0;
      overflow: hidden;
    }}
    .tab-panel.is-active {{
      display: flex;
      flex-direction: column;
    }}
    .control-section {{
      padding: 8px 12px;
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
    }}
    .control-title {{
      margin: 0 0 4px;
      font-weight: 700;
      font-size: 12.5px;
    }}
    .legend-all {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .fault-tools {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 12px 6px;
      flex: 0 0 auto;
    }}
    .legend-note {{
      padding: 0 12px 6px;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 11.5px;
      line-height: 1.4;
    }}
    .range-row {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 6px;
    }}
    .range-field {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      font-size: 11.5px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .range-field input {{
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 6px 8px;
    }}
    .colorbar {{
      margin-top: 6px;
    }}
    .colorbar-ramp {{
      height: 8px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      background: linear-gradient(90deg, #800000 0%, #ff0000 8%, #ff5a00 18%, #ffc300 28%, #ffff00 35%, #ffff00 67%, #aad400 78%, #61b000 88%, #007f00 100%);
    }}
    .colorbar-values {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 4px;
      font-size: 11px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .control-actions {{
      display: flex;
      justify-content: flex-end;
      margin-top: 6px;
    }}
    .series-legend {{
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 2px 6px 8px;
      box-sizing: border-box;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 8px;
      border-radius: 8px;
      cursor: pointer;
    }}
    .legend-item:hover {{
      background: rgba(148, 163, 184, 0.12);
    }}
    .legend-item input {{
      margin: 0;
      flex: 0 0 auto;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      flex: 0 0 auto;
    }}
    .legend-label {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }}
    .legend-pressure {{
      margin-left: auto;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 12px;
      white-space: nowrap;
    }}
    .legend-button {{
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 5px 10px;
      cursor: pointer;
    }}
    #plot {{
      width: 100%;
      height: 100%;
    }}
    .toolbar-year-control {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-shrink: 0;
    }}
    .toolbar-year-label {{
      font-size: 12.5px;
      color: {MODERN_MUTED_TEXT_COLOR};
      white-space: nowrap;
    }}
    .toolbar-year-input {{
      width: 80px;
      font: inherit;
      font-size: 13px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 4px 8px;
      box-sizing: border-box;
    }}
    @media (max-width: 780px) {{
      .content {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(260px, 1fr) minmax(180px, 42%);
      }}
      .plot-panel {{
        min-height: 260px;
      }}
      .series-legend {{
        max-height: none;
      }}
    }}
    @media (max-width: 680px) {{
      .toolbar {{
        align-items: flex-start;
        flex-wrap: wrap;
      }}
      .toolbar-status {{
        width: 100%;
      }}
      .range-row {{
        grid-template-columns: 1fr;
      }}
      .legend-header {{
        align-items: flex-start;
        flex-direction: column;
      }}
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="viewer">
    <div class="toolbar">
      <div class="toolbar-title">{title_html}</div>
      {year_toolbar_html}
      <div id="selection-summary" class="toolbar-status"></div>
    </div>
    <div class="content">
      <div class="plot-panel">
        <div class="plot-stage">
          <div id="plot"></div>
        </div>
      </div>
      <aside class="legend-panel" aria-label="Fault legend and selector">
        <div class="legend-header">
          <div class="legend-title">Controls</div>
          <div class="tab-strip">
            <button type="button" id="faults-tab-button" class="tab-button is-active" onclick="switchControlTab('faults')">Fault Curves</button>
            {color_tab_button_html}
          </div>
        </div>
        <div class="panel-body">
          <section id="faults-tab" class="tab-panel is-active" aria-label="Fault Curves">
            <div class="fault-tools">
              <label class="legend-all">
                <input type="checkbox" id="legend-all" checked onchange="toggleAllFromMaster(this.checked)">
                <span>All faults</span>
              </label>
              <div style="display:flex;gap:6px;">
                <button type="button" class="legend-button" onclick="setAllSelection(true)">Show All</button>
                <button type="button" class="legend-button" onclick="setAllSelection(false)">Clear All</button>
              </div>
            </div>
            <div class="legend-note">Use the checkboxes to compare one or more faults. The list below is scrollable so large fault runs stay manageable.</div>
            <div id="series-legend" class="series-legend"></div>
          </section>
          <section id="colors-tab" class="tab-panel" aria-label="{color_tab_label_html} Color Range">
            <div class="control-section">
              <div class="control-title">{color_tab_label_html}</div>
              <div class="legend-note">{color_tab_desc_html}</div>
              <div class="colorbar">
                <div class="colorbar-ramp" aria-hidden="true"></div>
                <div class="colorbar-values"><span id="colorbar-min"></span><span id="colorbar-max"></span></div>
              </div>
              <div class="range-row">
                <label class="range-field">Min{color_unit}
                  <input id="pressure-min" type="number" step="any" oninput="handleColorRangeInput()">
                </label>
                <label class="range-field">Max{color_unit}
                  <input id="pressure-max" type="number" step="any" oninput="handleColorRangeInput()">
                </label>
              </div>
              <div class="control-actions">
                <button type="button" class="legend-button" onclick="resetColorRange()">Reset Range</button>
              </div>
            </div>
          </section>
        </div>
      </aside>
    </div>
  </div>
  <script>
    const title = {title_json};
    const seriesPayload = {payload_json};
    const seriesIds = {series_ids_json};
    const defaultId = {default_id_json};
    const xLabel = {x_label_json};
    const yLabel = {y_label_json};
    const yRange = {y_range_json};
    const xType = {x_type_json};
    const yTickformat = {y_tickformat_json};
    const plotConfig = {config_json};
    const colorScale = {color_scale_json};
    const autoColorMin = {auto_color_min_json};
    const autoColorMax = {auto_color_max_json};
    const showFspBackground = {show_fsp_background_json};
    const showColorTab = {show_color_tab_json};
    const initColorMin = {init_color_min_json};
    const initColorMax = {init_color_max_json};
    let currentYearOfInterest = {year_of_interest_json};
    const selectedSeriesIds = new Set(seriesIds.length ? seriesIds : [defaultId]);

    function formatPressure(value) {{
      return Number.isFinite(value) ? value.toLocaleString(undefined, {{ maximumFractionDigits: 2 }}) + ' psi' : '';
    }}

    function switchControlTab(tabName) {{
      const faultsTab = document.getElementById('faults-tab');
      const colorsTab = document.getElementById('colors-tab');
      const faultsButton = document.getElementById('faults-tab-button');
      const colorsButton = document.getElementById('colors-tab-button');
      const showFaults = tabName === 'faults';
      if (faultsTab) faultsTab.classList.toggle('is-active', showFaults);
      if (colorsTab) colorsTab.classList.toggle('is-active', !showFaults);
      if (faultsButton) faultsButton.classList.toggle('is-active', showFaults);
      if (colorsButton) colorsButton.classList.toggle('is-active', !showFaults);
    }}

    function hexToRgb(hex) {{
      const cleaned = String(hex).replace('#', '');
      return [
        parseInt(cleaned.substring(0, 2), 16),
        parseInt(cleaned.substring(2, 4), 16),
        parseInt(cleaned.substring(4, 6), 16)
      ];
    }}

    function mixColor(start, end, ratio) {{
      const a = hexToRgb(start);
      const b = hexToRgb(end);
      return 'rgb(' +
        Math.round(a[0] + (b[0] - a[0]) * ratio) + ',' +
        Math.round(a[1] + (b[1] - a[1]) * ratio) + ',' +
        Math.round(a[2] + (b[2] - a[2]) * ratio) + ')';
    }}

    function initializeColorRange() {{
      const minInput = document.getElementById('pressure-min');
      const maxInput = document.getElementById('pressure-max');
      if (!minInput) return;
      // Prefer explicit initColorMin/Max, fall back to autoColorMin/Max
      const useMin = Number.isFinite(initColorMin) ? initColorMin : autoColorMin;
      const useMax = Number.isFinite(initColorMax) ? initColorMax : autoColorMax;
      minInput.value = Number.isFinite(useMin) ? Number(useMin).toFixed(showFspBackground ? 2 : 3) : '';
      maxInput.value = Number.isFinite(useMax) ? Number(useMax).toFixed(showFspBackground ? 2 : 3) : '';
      updateColorbarLabels();
    }}

    function selectedColorRange() {{
      const minInput = document.getElementById('pressure-min');
      const maxInput = document.getElementById('pressure-max');
      if (!minInput) return {{ minValue: null, maxValue: null, custom: false }};
      const minValue = Number.parseFloat(minInput.value);
      const maxValue = Number.parseFloat(maxInput.value);
      if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {{
        return {{ minValue: minValue, maxValue: maxValue, custom: true }};
      }}
      // Fall back to init defaults
      const fbMin = Number.isFinite(initColorMin) ? initColorMin : autoColorMin;
      const fbMax = Number.isFinite(initColorMax) ? initColorMax : autoColorMax;
      if (Number.isFinite(fbMin) && Number.isFinite(fbMax) && fbMax > fbMin) {{
        return {{ minValue: fbMin, maxValue: fbMax, custom: false }};
      }}
      return {{ minValue: null, maxValue: null, custom: false }};
    }}

    /** Interpolate the colorScale at a normalized 0-1 position. */
    function interpolateColorScale(scale, t) {{
      for (let i = 1; i < scale.length; i++) {{
        if (t <= scale[i][0]) {{
          const stopSpan = scale[i][0] - scale[i - 1][0] || 1;
          const ratio = (t - scale[i - 1][0]) / stopSpan;
          return mixColor(scale[i - 1][1], scale[i][1], ratio);
        }}
      }}
      return scale[scale.length - 1][1];
    }}

    function scaledColor(value) {{
      const colorRange = selectedColorRange();
      const raw = Number(value);
      if (!Number.isFinite(raw) || colorRange.minValue === null || colorRange.maxValue === null) {{
        return '#2563eb';
      }}
      const span = colorRange.maxValue - colorRange.minValue || 1;
      const normalized = Math.max(0, Math.min(1, (raw - colorRange.minValue) / span));
      return interpolateColorScale(colorScale, normalized);
    }}

    function currentSeriesColor(series) {{
      // On FSP background graphs all curves are plotted in dark so they
      // stand out against the colored background.
      if (showFspBackground) return '#1e293b';
      if (Number.isFinite(series.detSlipPressure)) {{
        return scaledColor(series.detSlipPressure);
      }}
      return series.color || '#2563eb';
    }}

    /** Build 50 thin background rect shapes spanning the y-axis gradient. */
    function buildBackgroundShapes() {{
      if (!showFspBackground) return [];
      const colorRange = selectedColorRange();
      if (colorRange.minValue === null) return [];
      const N = 50;
      const shapes = [];
      const span = colorRange.maxValue - colorRange.minValue;
      for (let i = 0; i < N; i++) {{
        const t0 = i / N;
        const t1 = (i + 1) / N;
        const y0 = colorRange.minValue + span * t0;
        const y1 = colorRange.minValue + span * t1;
        shapes.push({{
          type: 'rect',
          xref: 'paper',
          yref: 'y',
          x0: 0, x1: 1,
          y0: y0, y1: y1,
          fillcolor: interpolateColorScale(colorScale, (t0 + t1) / 2),
          opacity: 0.82,
          line: {{ width: 0 }},
          layer: 'below'
        }});
      }}
      return shapes;
    }}

    /** Build a vertical dashed line shape for the year of interest. */
    function buildYearLine() {{
      if (currentYearOfInterest === null || !Number.isFinite(currentYearOfInterest)) return [];
      return [{{
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: currentYearOfInterest,
        x1: currentYearOfInterest,
        y0: 0,
        y1: 1,
        line: {{ color: '#16a34a', width: 2, dash: 'dot' }}
      }}];
    }}

    /** Move the year-of-interest line when the toolbar input changes. */
    function onYearInput() {{
      const val = Number.parseFloat(document.getElementById('year-input').value);
      currentYearOfInterest = Number.isFinite(val) ? val : null;
      renderSeries();
    }}

    function formatColorValue(value) {{
      if (!Number.isFinite(value)) return '';
      // FSP background mode: dimensionless 0-1 values; otherwise show "psi"
      return showFspBackground
        ? value.toLocaleString(undefined, {{ maximumFractionDigits: 2 }})
        : formatPressure(value);
    }}

    function updateColorbarLabels() {{
      const colorRange = selectedColorRange();
      const minEl = document.getElementById('colorbar-min');
      const maxEl = document.getElementById('colorbar-max');
      if (!minEl) return;
      minEl.textContent = colorRange.minValue !== null ? formatColorValue(colorRange.minValue) : '';
      maxEl.textContent = colorRange.maxValue !== null ? formatColorValue(colorRange.maxValue) : '';
    }}

    function handleColorRangeInput() {{
      updateColorbarLabels();
      refreshLegendColors();
      renderSeries();
    }}

    function resetColorRange() {{
      initializeColorRange();
      refreshLegendColors();
      renderSeries();
    }}

    function updateSummary() {{
      const summary = document.getElementById('selection-summary');
      summary.textContent = selectedSeriesIds.size + ' of ' + seriesIds.length + ' faults visible';
      const master = document.getElementById('legend-all');
      if (master) {{
        master.checked = seriesIds.length > 0 && selectedSeriesIds.size === seriesIds.length;
        master.indeterminate = selectedSeriesIds.size > 0 && selectedSeriesIds.size < seriesIds.length;
      }}
    }}

    function setAllSelection(checked) {{
      selectedSeriesIds.clear();
      if (checked) {{
        seriesIds.forEach(function(seriesId) {{
          selectedSeriesIds.add(seriesId);
        }});
      }}
      document.querySelectorAll('.series-toggle').forEach(function(input) {{
        input.checked = checked;
      }});
      updateSummary();
      renderSeries();
    }}

    function toggleAllFromMaster(checked) {{
      setAllSelection(checked);
    }}

    function toggleSeries(seriesId, checked) {{
      if (checked) {{
        selectedSeriesIds.add(seriesId);
      }} else {{
        selectedSeriesIds.delete(seriesId);
      }}
      updateSummary();
      renderSeries();
    }}

    function refreshLegendColors() {{
      document.querySelectorAll('.legend-item').forEach(function(item) {{
        const seriesId = item.getAttribute('data-series-id');
        const swatch = item.querySelector('.legend-swatch');
        const series = seriesPayload[seriesId] || {{}};
        if (swatch) {{
          swatch.style.backgroundColor = currentSeriesColor(series);
        }}
      }});
    }}

    function populateLegend() {{
      const legend = document.getElementById('series-legend');
      legend.innerHTML = '';
      seriesIds.forEach(function(seriesId) {{
        const series = seriesPayload[seriesId] || {{}};
        const label = document.createElement('label');
        label.className = 'legend-item';
        label.setAttribute('data-series-id', seriesId);

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'series-toggle';
        checkbox.checked = selectedSeriesIds.has(seriesId);
        checkbox.addEventListener('change', function() {{
          toggleSeries(seriesId, checkbox.checked);
        }});

        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.backgroundColor = currentSeriesColor(series);

        const text = document.createElement('span');
        text.className = 'legend-label';
        text.textContent = series.label || seriesId;

        label.appendChild(checkbox);
        label.appendChild(swatch);
        label.appendChild(text);

        if (Number.isFinite(series.detSlipPressure)) {{
          const pressure = document.createElement('span');
          pressure.className = 'legend-pressure';
          pressure.textContent = formatPressure(series.detSlipPressure);
          label.appendChild(pressure);
        }}

        legend.appendChild(label);
      }});
      updateSummary();
    }}

    function renderSeries() {{
      const activeSeriesIds = seriesIds.filter(function(seriesId) {{
        return selectedSeriesIds.has(seriesId);
      }});
      const traces = activeSeriesIds.map(function(seriesId) {{
        const series = seriesPayload[seriesId] || seriesPayload[defaultId];
        return {{
          x: series.x,
          y: series.y,
          mode: series.mode || 'lines',
          type: 'scatter',
          name: series.label || seriesId,
          line: {{ width: activeSeriesIds.length > 8 ? 2.1 : 2.7, color: currentSeriesColor(series) }},
          marker: {{ size: activeSeriesIds.length > 8 ? 0 : 5.5, line: {{ width: 1, color: '#ffffff' }} }},
          hovertemplate: series.hovertemplate || '%{{x}}<br>%{{y}}<extra></extra>'
        }};
      }});
      const layout = {{
        autosize: true,
        template: 'plotly_white',
        margin: {{ l: 74, r: 34, t: 34, b: 64 }},
        font: {{ family: '{MODERN_FONT_FAMILY}', size: 12, color: '{MODERN_TEXT_COLOR}' }},
        paper_bgcolor: '{MODERN_PLOT_BG}',
        plot_bgcolor: '{MODERN_PLOT_BG}',
        showlegend: false,
        hovermode: 'closest',
        annotations: activeSeriesIds.length ? [] : [{{
          text: 'Select at least one fault from the legend to display its curve.',
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          showarrow: false,
          font: {{ size: 14, color: '{MODERN_MUTED_TEXT_COLOR}' }}
        }}],
        xaxis: {{
          title: {{ text: xLabel, standoff: 10, font: {{ color: '{MODERN_MUTED_TEXT_COLOR}' }} }},
          type: xType,
          showgrid: true,
          gridcolor: '{MODERN_GRID_COLOR}',
          linecolor: '{MODERN_GRID_COLOR}',
          tickcolor: '{MODERN_GRID_COLOR}',
          tickfont: {{ color: '{MODERN_AXIS_COLOR}' }},
          zeroline: false,
          automargin: true
        }},
        yaxis: {{
          title: {{ text: yLabel, standoff: 12, font: {{ color: '{MODERN_MUTED_TEXT_COLOR}' }} }},
          range: yRange || undefined,
          tickformat: yTickformat || undefined,
          showgrid: true,
          gridcolor: '{MODERN_GRID_COLOR}',
          linecolor: '{MODERN_GRID_COLOR}',
          tickcolor: '{MODERN_GRID_COLOR}',
          tickfont: {{ color: '{MODERN_AXIS_COLOR}' }},
          zeroline: false,
          automargin: true
        }},
        shapes: [...buildBackgroundShapes(), ...buildYearLine()]
      }};
      Plotly.react(document.getElementById('plot'), traces, layout, plotConfig);
    }}

    initializeColorRange();
    populateLegend();
    renderSeries();
    window.addEventListener('resize', function() {{
      Plotly.Plots.resize(document.getElementById('plot'));
    }});
  </script>
</body>
</html>
"""


def _multi_curve_selector_plotly_html(
    *,
    title: str,
    series_payload: dict,
    x_label: str,
    y_label: str,
    selector_title: str,
    selector_subject_plural: str,
    selector_empty_message: str,
    preferred_plot_height: str = "clamp(360px, 54vh, 500px)",
):
    title_html = html.escape(title)
    payload_json = json.dumps(series_payload, separators=(",", ":"))
    title_json = json.dumps(title)
    x_label_json = json.dumps(x_label)
    y_label_json = json.dumps(y_label)
    selector_subject_plural_json = json.dumps(selector_subject_plural)
    selector_empty_message_json = json.dumps(selector_empty_message)
    config_json = json.dumps(PLOTLY_CONFIG)
    series_ids_json = json.dumps(list(series_payload.keys()))
    plot_height_css = preferred_plot_height
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
      background: {MODERN_PAPER_BG};
      color: {MODERN_TEXT_COLOR};
      font-family: {MODERN_FONT_FAMILY};
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }}
    .viewer {{
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: {MODERN_PAPER_BG};
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 48px;
      padding: 8px 12px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      box-sizing: border-box;
      background: {MODERN_CONTROL_BG};
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
    }}
    .toolbar-title {{
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }}
    .toolbar-status {{
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 13px;
      white-space: nowrap;
    }}
    .content {{
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: row;
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .plot-panel {{
      flex: 1 1 auto;
      min-width: 0;
      min-height: 0;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
      background: {MODERN_PLOT_BG};
    }}
    .plot-stage {{
      width: 100%;
      height: 100%;
      min-height: 100%;
    }}
    .selector-panel {{
      flex: 0 0 clamp(300px, 28vw, 360px);
      min-width: clamp(300px, 28vw, 360px);
      min-height: 0;
      display: flex;
      flex-direction: column;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
    }}
    .selector-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 12px 6px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      flex: 0 0 auto;
    }}
    .selector-title {{
      font-weight: 700;
    }}
    .selector-body {{
      flex: 1 1 auto;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}
    .selector-tools {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 12px 6px;
      flex: 0 0 auto;
    }}
    .selector-actions {{
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .selector-note {{
      padding: 0 12px 6px;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 11.5px;
      line-height: 1.4;
      flex: 0 0 auto;
    }}
    .selector-master {{
      padding: 0 12px 6px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      flex: 0 0 auto;
    }}
    .legend-all {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .legend-button {{
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 5px 10px;
      cursor: pointer;
    }}
    .series-legend {{
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 6px;
      box-sizing: border-box;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 8px;
      border-radius: 8px;
      cursor: pointer;
    }}
    .legend-item:hover {{
      background: rgba(148, 163, 184, 0.12);
    }}
    .legend-item input {{
      margin: 0;
      flex: 0 0 auto;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      flex: 0 0 auto;
    }}
    .legend-label {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      flex: 1 1 auto;
    }}
    #plot {{
      width: 100%;
      height: 100%;
    }}
    @media (max-width: 780px) {{
      .content {{
        flex-direction: column;
      }}
      .plot-stage {{
        min-height: 260px;
      }}
      .plot-panel {{
        flex: 1 1 55%;
        min-height: 260px;
        max-height: 62%;
      }}
      .selector-panel {{
        flex: 1 1 45%;
        min-height: 200px;
        min-width: 0;
        width: 100%;
      }}
      .selector-header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .selector-note {{
        padding-bottom: 4px;
      }}
      .legend-item {{
        padding: 7px 8px;
      }}
      .series-legend {{
        flex: 1 1 auto;
        min-height: 140px;
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
      }}
    }}
    @media (max-width: 680px) {{
      .toolbar {{
        align-items: flex-start;
        flex-wrap: wrap;
      }}
      .toolbar-status {{
        width: 100%;
      }}
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="viewer">
    <div class="toolbar">
      <div class="toolbar-title">{title_html}</div>
      <div id="selection-summary" class="toolbar-status"></div>
    </div>
    <div class="content">
      <div class="plot-panel">
        <div class="plot-stage">
          <div id="plot"></div>
        </div>
      </div>
      <aside class="selector-panel" aria-label="{html.escape(selector_title)}">
        <div class="selector-header">
          <div class="selector-title">{html.escape(selector_title)}</div>
        </div>
        <div class="selector-body">
          <div class="selector-tools">
            <div class="selector-actions">
              <button type="button" class="legend-button" onclick="setAllSelection(true)">Show All</button>
              <button type="button" class="legend-button" onclick="setAllSelection(false)">Hide All</button>
            </div>
          </div>
          <div class="selector-note">Use the checkboxes to compare one or more {html.escape(selector_subject_plural)}.</div>
          <div class="selector-master">
            <label class="legend-all">
              <input type="checkbox" id="legend-all" checked onchange="toggleAllFromMaster(this.checked)">
              <span>All {html.escape(selector_subject_plural)}</span>
            </label>
          </div>
          <div id="series-legend" class="series-legend"></div>
        </div>
      </aside>
    </div>
  </div>
  <script>
    const title = {title_json};
    const seriesPayload = {payload_json};
    const seriesIds = {series_ids_json};
    const xLabel = {x_label_json};
    const yLabel = {y_label_json};
    const selectorSubjectPlural = {selector_subject_plural_json};
    const selectorEmptyMessage = {selector_empty_message_json};
    const plotConfig = {config_json};
    const selectedSeriesIds = new Set(seriesIds);

    function updateSummary() {{
      const summary = document.getElementById('selection-summary');
      summary.textContent = selectedSeriesIds.size + ' of ' + seriesIds.length + ' ' + selectorSubjectPlural + ' visible';
      const master = document.getElementById('legend-all');
      if (master) {{
        master.checked = seriesIds.length > 0 && selectedSeriesIds.size === seriesIds.length;
        master.indeterminate = selectedSeriesIds.size > 0 && selectedSeriesIds.size < seriesIds.length;
      }}
    }}

    function setAllSelection(checked) {{
      selectedSeriesIds.clear();
      if (checked) {{
        seriesIds.forEach(function(seriesId) {{
          selectedSeriesIds.add(seriesId);
        }});
      }}
      document.querySelectorAll('.series-toggle').forEach(function(input) {{
        input.checked = checked;
      }});
      updateSummary();
      renderSeries();
    }}

    function toggleAllFromMaster(checked) {{
      setAllSelection(checked);
    }}

    function toggleSeries(seriesId, checked) {{
      if (checked) {{
        selectedSeriesIds.add(seriesId);
      }} else {{
        selectedSeriesIds.delete(seriesId);
      }}
      updateSummary();
      renderSeries();
    }}

    function populateLegend() {{
      const legend = document.getElementById('series-legend');
      legend.innerHTML = '';
      seriesIds.forEach(function(seriesId) {{
        const series = seriesPayload[seriesId] || {{}};
        const label = document.createElement('label');
        label.className = 'legend-item';
        label.setAttribute('data-series-id', seriesId);

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'series-toggle';
        checkbox.checked = selectedSeriesIds.has(seriesId);
        checkbox.addEventListener('change', function() {{
          toggleSeries(seriesId, checkbox.checked);
        }});

        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.backgroundColor = series.color || '#2563eb';

        const text = document.createElement('span');
        text.className = 'legend-label';
        text.textContent = series.label || seriesId;

        label.appendChild(checkbox);
        label.appendChild(swatch);
        label.appendChild(text);
        legend.appendChild(label);
      }});
      updateSummary();
    }}

    function renderSeries() {{
      const activeSeriesIds = seriesIds.filter(function(seriesId) {{
        return selectedSeriesIds.has(seriesId);
      }});
      const traces = activeSeriesIds.map(function(seriesId) {{
        const series = seriesPayload[seriesId] || {{}};
        return {{
          x: series.x,
          y: series.y,
          mode: series.mode || 'lines',
          type: 'scatter',
          name: series.label || seriesId,
          line: {{ width: activeSeriesIds.length > 8 ? 2.1 : 2.7, color: series.color || '#2563eb' }},
          hovertemplate: series.hovertemplate || '%{{x}}<br>%{{y}}<extra></extra>'
        }};
      }});
      const layout = {{
        autosize: true,
        template: 'plotly_white',
        margin: {{ l: 74, r: 34, t: 34, b: 64 }},
        font: {{ family: '{MODERN_FONT_FAMILY}', size: 12, color: '{MODERN_TEXT_COLOR}' }},
        paper_bgcolor: '{MODERN_PLOT_BG}',
        plot_bgcolor: '{MODERN_PLOT_BG}',
        showlegend: false,
        hovermode: 'closest',
        annotations: activeSeriesIds.length ? [] : [{{
          text: selectorEmptyMessage,
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          showarrow: false,
          font: {{ size: 14, color: '{MODERN_MUTED_TEXT_COLOR}' }}
        }}],
        xaxis: {{
          title: {{ text: xLabel, standoff: 10, font: {{ color: '{MODERN_MUTED_TEXT_COLOR}' }} }},
          showgrid: true,
          gridcolor: '{MODERN_GRID_COLOR}',
          linecolor: '{MODERN_GRID_COLOR}',
          tickcolor: '{MODERN_GRID_COLOR}',
          tickfont: {{ color: '{MODERN_AXIS_COLOR}' }},
          zeroline: false,
          automargin: true,
          rangemode: 'tozero'
        }},
        yaxis: {{
          title: {{ text: yLabel, standoff: 12, font: {{ color: '{MODERN_MUTED_TEXT_COLOR}' }} }},
          showgrid: true,
          gridcolor: '{MODERN_GRID_COLOR}',
          linecolor: '{MODERN_GRID_COLOR}',
          tickcolor: '{MODERN_GRID_COLOR}',
          tickfont: {{ color: '{MODERN_AXIS_COLOR}' }},
          zeroline: false,
          automargin: true,
          rangemode: 'tozero'
        }}
      }};
      Plotly.react(document.getElementById('plot'), traces, layout, plotConfig);
    }}

    populateLegend();
    renderSeries();
    window.addEventListener('resize', function() {{
      Plotly.Plots.resize(document.getElementById('plot'));
    }});
  </script>
</body>
</html>
"""


def _paired_hydrology_geomechanics_cdf_html(
    *,
    title: str,
    series_payload: dict,
    default_id: str,
):
    title_html = html.escape(title)
    title_json = json.dumps(title)
    payload_json = json.dumps(series_payload, separators=(",", ":"))
    series_ids_json = json.dumps(list(series_payload.keys()))
    default_id_json = json.dumps(default_id)
    config_json = json.dumps(PLOTLY_CONFIG)
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
      background: {MODERN_PAPER_BG};
      color: {MODERN_TEXT_COLOR};
      font-family: {MODERN_FONT_FAMILY};
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }}
    .viewer {{
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: {MODERN_PAPER_BG};
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 48px;
      padding: 8px 12px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      box-sizing: border-box;
      background: {MODERN_CONTROL_BG};
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
    }}
    .toolbar-title {{
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }}
    .toolbar-status {{
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 13px;
      white-space: nowrap;
    }}
    .toolbar-controls {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex: 0 0 auto;
    }}
    .ctrl-label {{
      font-size: 12px;
      color: {MODERN_MUTED_TEXT_COLOR};
      white-space: nowrap;
    }}
    .ctrl-input {{
      width: 90px;
      padding: 3px 6px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 5px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      font-size: 12px;
    }}
    .content {{
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
      grid-template-rows: minmax(0, 1fr);
      grid-template-areas:
        "plot selector";
      gap: 12px;
      padding: 12px;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .plot-panel {{
      grid-area: plot;
      min-width: 0;
      min-height: 0;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
      background: {MODERN_PLOT_BG};
    }}
    .plot-stage {{
      width: 100%;
      height: 100%;
      min-height: 0;
    }}
    .selector-panel {{
      grid-area: selector;
      min-height: 0;
      display: flex;
      flex-direction: column;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
    }}
    .selector-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 12px 6px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      flex: 0 0 auto;
    }}
    .selector-title {{
      font-weight: 700;
    }}
    .selector-actions {{
      display: inline-flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .sort-control {{
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 5px 8px;
      min-width: 170px;
    }}
    .legend-button {{
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 5px 10px;
      cursor: pointer;
    }}
    .selector-note {{
      padding: 8px 12px 4px;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 11.5px;
      line-height: 1.4;
      flex: 0 0 auto;
    }}
    .series-legend {{
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 2px 6px 8px;
      box-sizing: border-box;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 8px;
      border-radius: 8px;
      cursor: pointer;
    }}
    .legend-item:hover {{
      background: rgba(148, 163, 184, 0.12);
    }}
    .legend-item input {{
      margin: 0;
      flex: 0 0 auto;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      background: #2563eb;
      flex: 0 0 auto;
    }}
    .legend-label {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
      flex: 1 1 auto;
    }}
    .legend-fsp {{
      margin-left: auto;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 12px;
      white-space: nowrap;
    }}
    #plot {{
      width: 100%;
      height: 100%;
    }}
    @media (max-width: 780px) {{
      .content {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(260px, 1fr) auto minmax(180px, 36%);
        grid-template-areas:
          "plot"
          "summary"
          "selector";
      }}
      .plot-panel {{
        min-height: 260px;
      }}
      .series-legend {{
        max-height: none;
      }}
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="viewer">
    <div class="toolbar">
      <div class="toolbar-title">{title_html}</div>
      <div class="toolbar-controls">
        <label class="ctrl-label" for="max-delta-pp">Max Delta PP (psi):</label>
        <input type="number" id="max-delta-pp" class="ctrl-input"
               placeholder="Auto" min="0" step="100"
               onchange="applyMaxDeltaPP()">
        <button type="button" class="legend-button" onclick="resetXAxisRange()">Reset</button>
      </div>
      <div id="selection-summary" class="toolbar-status"></div>
    </div>
    <div class="content">
      <div class="plot-panel">
        <div class="plot-stage">
          <div id="plot"></div>
        </div>
      </div>
      <aside class="selector-panel" aria-label="Fault legend and selector">
        <div class="selector-header">
          <div class="selector-title">Fault Curves</div>
          <div class="selector-actions">
            <select id="sort-control" class="sort-control" onchange="handleSortChange(this.value)" aria-label="Sort faults">
              <option value="fsp">Sort by FSP</option>
              <option value="hydrologyMax">Sort by max pressure</option>
              <option value="geoMedian">Sort by P50 slip pressure</option>
              <option value="label">Sort by fault ID</option>
            </select>
            <button type="button" class="legend-button" onclick="showFocusedOnly()">Selected Only</button>
            <button type="button" class="legend-button" onclick="setAllSelection(true)">Show All</button>
            <button type="button" class="legend-button" onclick="setAllSelection(false)">Clear All</button>
          </div>
        </div>
        <div class="selector-note">Solid blue curves show hydrology pressure exceedance. Dashed curves show the Step 3 geomechanics fault-slip CDF.</div>
        <div id="series-legend" class="series-legend"></div>
      </aside>
    </div>
  </div>
  <script>
    const title = {title_json};
    const seriesPayload = {payload_json};
    const seriesIds = {series_ids_json};
    const defaultId = {default_id_json};
    const plotConfig = {config_json};
    const selectedSeriesIds = new Set(defaultId ? [defaultId] : seriesIds.slice(0, 1));
    let focusedSeriesId = defaultId || (seriesIds.length ? seriesIds[0] : null);
    let currentSort = 'fsp';
    let maxDeltaPP = null;

    function updateSummary() {{
      const summary = document.getElementById('selection-summary');
      summary.textContent = selectedSeriesIds.size + ' of ' + seriesIds.length + ' faults visible';
    }}

    function formatFsp(value) {{
      return Number.isFinite(value) ? value.toFixed(2) : '0.00';
    }}

    function formatPsi(value) {{
      return Number.isFinite(value) ? value.toLocaleString(undefined, {{ maximumFractionDigits: 1 }}) + ' psi' : 'n/a';
    }}

    function applyMaxDeltaPP() {{
      const val = parseFloat(document.getElementById('max-delta-pp').value);
      maxDeltaPP = (Number.isFinite(val) && val > 0) ? val : null;
      renderSeries();
    }}

    function resetXAxisRange() {{
      maxDeltaPP = null;
      document.getElementById('max-delta-pp').value = '';
      renderSeries();
    }}

    function sortedSeriesIds() {{
      const ids = seriesIds.slice();
      ids.sort(function(a, b) {{
        const sa = seriesPayload[a] || {{}};
        const sb = seriesPayload[b] || {{}};
        const ast = sa.stats || {{}};
        const bst = sb.stats || {{}};
        if (currentSort === 'hydrologyMax') return (bst.hydrologyMax || 0) - (ast.hydrologyMax || 0);
        if (currentSort === 'geoMedian') return (ast.geomechanicsP50 || 0) - (bst.geomechanicsP50 || 0);
        if (currentSort === 'label') return String(sa.label || a).localeCompare(String(sb.label || b));
        return (sb.fsp || 0) - (sa.fsp || 0);
      }});
      return ids;
    }}

    function handleSortChange(value) {{
      currentSort = value || 'fsp';
      populateLegend();
    }}

    function setAllSelection(checked) {{
      selectedSeriesIds.clear();
      if (checked) {{
        seriesIds.forEach(function(seriesId) {{
          selectedSeriesIds.add(seriesId);
        }});
      }}
      document.querySelectorAll('.series-toggle').forEach(function(input) {{
        input.checked = checked;
      }});
      updateSummary();
      renderSeries();
    }}

    function showFocusedOnly() {{
      selectedSeriesIds.clear();
      if (focusedSeriesId) selectedSeriesIds.add(focusedSeriesId);
      document.querySelectorAll('.series-toggle').forEach(function(input) {{
        input.checked = input.getAttribute('data-series-id') === focusedSeriesId;
      }});
      updateSummary();
      renderSeries();
    }}

    function toggleSeries(seriesId, checked) {{
      if (checked) {{
        selectedSeriesIds.add(seriesId);
        focusedSeriesId = seriesId;
      }} else {{
        selectedSeriesIds.delete(seriesId);
      }}
      updateSummary();
      renderSeries();
    }}

    function populateLegend() {{
      const legend = document.getElementById('series-legend');
      legend.innerHTML = '';
      sortedSeriesIds().forEach(function(seriesId) {{
        const series = seriesPayload[seriesId] || {{}};
        const label = document.createElement('label');
        label.className = 'legend-item';
        label.setAttribute('data-series-id', seriesId);

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'series-toggle';
        checkbox.checked = selectedSeriesIds.has(seriesId);
        checkbox.setAttribute('data-series-id', seriesId);
        checkbox.addEventListener('change', function() {{
          toggleSeries(seriesId, checkbox.checked);
        }});

        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';

        const text = document.createElement('span');
        text.className = 'legend-label';
        text.textContent = series.label || seriesId;

        const fsp = document.createElement('span');
        fsp.className = 'legend-fsp';
        fsp.textContent = 'FSP ' + formatFsp(series.fsp);

        label.appendChild(checkbox);
        label.appendChild(swatch);
        label.appendChild(text);
        label.appendChild(fsp);
        legend.appendChild(label);
      }});
      updateSummary();
    }}

    function renderSeries() {{
      const activeSeriesIds = seriesIds.filter(function(seriesId) {{
        return selectedSeriesIds.has(seriesId);
      }});
      const traces = [];
      activeSeriesIds.forEach(function(seriesId) {{
        const series = seriesPayload[seriesId] || {{}};
        if (series.hydrology) {{
          traces.push({{
            x: series.hydrology.x,
            y: series.hydrology.y,
            mode: 'lines',
            type: 'scatter',
            name: 'Hydrology pressure exceedance - ' + (series.label || seriesId),
            line: {{ width: activeSeriesIds.length > 8 ? 2.0 : 2.7, color: '#2563eb' }},
            hovertemplate: 'Fault: ' + (series.label || seriesId) + '<br>Hydrology pressure: %{{x:,.2f}} psi<br>Exceedance probability: %{{y:.3f}}<br>FSP: ' + formatFsp(series.fsp) + '<extra></extra>'
          }});
        }}
        if (series.geomechanics) {{
          traces.push({{
            x: series.geomechanics.x,
            y: series.geomechanics.y,
            mode: 'lines',
            type: 'scatter',
            name: 'Geomechanics fault-slip CDF - ' + (series.label || seriesId),
            line: {{ width: activeSeriesIds.length > 8 ? 1.8 : 2.4, color: '#475569', dash: 'dash' }},
            hovertemplate: 'Fault: ' + (series.label || seriesId) + '<br>Geomechanics slip pressure: %{{x:,.2f}} psi<br>Slip potential: %{{y:.3f}}<br>FSP: ' + formatFsp(series.fsp) + '<extra></extra>'
          }});
        }}
      }});
      const layout = {{
        autosize: true,
        template: 'plotly_white',
        margin: {{ l: 74, r: 34, t: 34, b: 64 }},
        font: {{ family: '{MODERN_FONT_FAMILY}', size: 12, color: '{MODERN_TEXT_COLOR}' }},
        paper_bgcolor: '{MODERN_PLOT_BG}',
        plot_bgcolor: '{MODERN_PLOT_BG}',
        showlegend: false,
        hovermode: 'closest',
        annotations: activeSeriesIds.length ? [] : [{{
          text: 'Select at least one fault from the legend to display its curves.',
          x: 0.5,
          y: 0.5,
          xref: 'paper',
          yref: 'paper',
          showarrow: false,
          font: {{ size: 14, color: '{MODERN_MUTED_TEXT_COLOR}' }}
        }}],
        xaxis: {{
          title: {{ text: 'Pore Pressure on fault (psi)', standoff: 10, font: {{ color: '{MODERN_MUTED_TEXT_COLOR}' }} }},
          showgrid: true,
          gridcolor: '{MODERN_GRID_COLOR}',
          linecolor: '{MODERN_GRID_COLOR}',
          tickcolor: '{MODERN_GRID_COLOR}',
          tickfont: {{ color: '{MODERN_AXIS_COLOR}' }},
          zeroline: false,
          automargin: true,
          ...(maxDeltaPP !== null ? {{ range: [0, maxDeltaPP] }} : {{}})
        }},
        yaxis: {{
          title: {{ text: 'Probability', standoff: 12, font: {{ color: '{MODERN_MUTED_TEXT_COLOR}' }} }},
          range: [0, 1],
          tickformat: '.2f',
          showgrid: true,
          gridcolor: '{MODERN_GRID_COLOR}',
          linecolor: '{MODERN_GRID_COLOR}',
          tickcolor: '{MODERN_GRID_COLOR}',
          tickfont: {{ color: '{MODERN_AXIS_COLOR}' }},
          zeroline: false,
          automargin: true
        }}
      }};
      Plotly.react(document.getElementById('plot'), traces, layout, plotConfig);
    }}

    populateLegend();
    renderSeries();
    window.addEventListener('resize', function() {{
      Plotly.Plots.resize(document.getElementById('plot'));
    }});
  </script>
</body>
</html>
"""


def save_cdf_artifact(
    helper,
    step_index: int,
    cdf_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    pressure_label: str,
    probability_label: str,
    display_order: int,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(cdf_df, ["ID", "slip_pressure", "probability"]):
            add_graph_warning(helper, step_index, f"{prefix} because required CDF columns are missing.")
            return None

        graph_df = _clean(cdf_df, ["slip_pressure", "probability"])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid CDF records were available.")
            return None

        series_payload = {}
        for fault_id, fault_df in graph_df.groupby(graph_df["ID"].astype(str), sort=True):
            fault_df = fault_df.sort_values("slip_pressure")
            det_slip_pressure = None
            if "det_slip_pressure" in fault_df.columns:
                det_values = pd.to_numeric(fault_df["det_slip_pressure"], errors="coerce").dropna()
                if not det_values.empty:
                    det_slip_pressure = round(float(det_values.iloc[0]), 6)
            series_payload[str(fault_id)] = {
                "label": str(fault_id),
                "x": fault_df["slip_pressure"].round(6).tolist(),
                "y": fault_df["probability"].round(6).tolist(),
                "mode": "lines",
                "hovertemplate": "Pressure: %{x:,.2f} psi<br>Probability: %{y:.3f}<extra></extra>",
                "detSlipPressure": det_slip_pressure,
            }

        for fault_id, color in _series_colors_from_payload(series_payload).items():
            series_payload[fault_id]["color"] = color

        auto_color_min, auto_color_max = _deterministic_pressure_range_from_payload(series_payload)

        default_id = next(iter(series_payload))
        html_text = _single_series_plotly_html(
            title=title,
            series_payload=series_payload,
            default_id=default_id,
            x_label=pressure_label,
            y_label=probability_label,
            auto_color_min=auto_color_min,
            auto_color_max=auto_color_max,
            y_range=[0, 1],
            y_tickformat=".2f",
            color_tab_label="Pore Pressure to Slip",
        )
        return _write_html_artifact(
            helper,
            artifact_key=artifact_key,
            title=title,
            html_text=html_text,
            caption=f"Interactive {title.lower()} generated by FSP.",
            display_order=display_order,
            preferred_height=700,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_probabilistic_hydrology_cdf_artifact(
    helper,
    step_index: int,
    hydrology_cdf_df: pd.DataFrame,
    geomechanics_cdf_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    display_order: int,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        required_columns = ["ID", "slip_pressure", "probability"]
        if not has_columns(hydrology_cdf_df, required_columns):
            add_graph_warning(helper, step_index, f"{prefix} because required hydrology CDF columns are missing.")
            return None
        if not has_columns(geomechanics_cdf_df, required_columns):
            add_graph_warning(helper, step_index, f"{prefix} because required geomechanics CDF columns are missing.")
            return None

        hyd_df = _clean(hydrology_cdf_df, ["slip_pressure", "probability"])
        geo_df = _clean(geomechanics_cdf_df, ["slip_pressure", "probability"])
        if hyd_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid hydrology CDF records were available.")
            return None
        if geo_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid geomechanics CDF records were available.")
            return None

        geo_groups = {
            str(fid): group.sort_values("slip_pressure")
            for fid, group in geo_df.assign(ID=geo_df["ID"].astype(str)).groupby("ID", sort=False)
        }

        series_payload = {}
        for fault_id, hyd_fault_df in hyd_df.assign(ID=hyd_df["ID"].astype(str)).groupby("ID", sort=True):
            hyd_fault_df = hyd_fault_df.sort_values("slip_pressure")
            geo_fault_df = geo_groups.get(str(fault_id))
            if geo_fault_df is None or geo_fault_df.empty:
                continue

            hydro_pressures = hyd_fault_df["slip_pressure"].to_numpy(dtype=float)
            geo_pressures = geo_fault_df["slip_pressure"].to_numpy(dtype=float)
            geo_sorted = np.sort(geo_pressures[np.isfinite(geo_pressures)])
            if len(geo_sorted) and len(hydro_pressures):
                fsp_values = np.searchsorted(geo_sorted, hydro_pressures, side="right").astype(float) / float(len(geo_sorted))
                fsp = round(float(np.mean(fsp_values)), 6)
            else:
                fsp = 0.0

            hyd_clean = hydro_pressures[np.isfinite(hydro_pressures)]
            geo_clean = geo_pressures[np.isfinite(geo_pressures)]
            stats = {
                "hydrologyMedian": round(float(np.quantile(hyd_clean, 0.50)), 6) if len(hyd_clean) else None,
                "hydrologyP95": round(float(np.quantile(hyd_clean, 0.95)), 6) if len(hyd_clean) else None,
                "hydrologyMax": round(float(np.max(hyd_clean)), 6) if len(hyd_clean) else None,
                "geomechanicsP10": round(float(np.quantile(geo_clean, 0.10)), 6) if len(geo_clean) else None,
                "geomechanicsP50": round(float(np.quantile(geo_clean, 0.50)), 6) if len(geo_clean) else None,
                "geomechanicsP90": round(float(np.quantile(geo_clean, 0.90)), 6) if len(geo_clean) else None,
            }

            series_payload[str(fault_id)] = {
                "label": str(fault_id),
                "fsp": fsp,
                "stats": stats,
                "hydrology": {
                    "x": hyd_fault_df["slip_pressure"].round(6).tolist(),
                    "y": hyd_fault_df["probability"].round(6).tolist(),
                },
                "geomechanics": {
                    "x": geo_fault_df["slip_pressure"].round(6).tolist(),
                    "y": geo_fault_df["probability"].round(6).tolist(),
                },
            }

        if not series_payload:
            add_graph_warning(helper, step_index, f"{prefix} because no faults had both hydrology and geomechanics CDF data.")
            return None

        default_id = max(
            series_payload,
            key=lambda fid: (
                float(series_payload[fid].get("fsp") or 0.0),
                float((series_payload[fid].get("stats") or {}).get("hydrologyMax") or 0.0),
            ),
        )

        html_text = _paired_hydrology_geomechanics_cdf_html(
            title=title,
            series_payload=series_payload,
            default_id=default_id,
        )
        return _write_html_artifact(
            helper,
            artifact_key=artifact_key,
            title=title,
            html_text=html_text,
            caption=f"Interactive {title.lower()} generated by FSP.",
            display_order=display_order,
            preferred_height=700,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_histogram_artifact(
    helper,
    step_index: int,
    histogram_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    x_label: str,
    display_order: int,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(histogram_df, ["label", "count"]):
            add_graph_warning(helper, step_index, f"{prefix} because required histogram columns are missing.")
            return None

        graph_df = _clean(histogram_df, ["count"])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid histogram records were available.")
            return None

        id_col = "ID" if "ID" in graph_df.columns else "id" if "id" in graph_df.columns else None
        if id_col:
            grouped = graph_df.groupby("label", sort=False)["count"].sum().reset_index()
        else:
            grouped = graph_df

        fig = go.Figure(go.Bar(
            x=grouped["label"].astype(str),
            y=grouped["count"],
            marker={"color": "#2563eb", "line": {"color": "rgba(30, 64, 175, 0.45)", "width": 0.7}},
            hovertemplate=f"{x_label}: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
        ))
        apply_scientific_layout(fig, x_title=x_label, y_title="Count")
        fig.update_layout(showlegend=False, bargap=0.16)
        return write_plotly_artifact(
            helper,
            fig,
            artifact_key,
            title,
            caption=f"Interactive {title.lower()} generated by FSP.",
            display_order=display_order,
            preferred_height=480,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def _input_histogram_specs(sample_inputs_df: pd.DataFrame):
    candidates = [
        ("vertical_stress_gradient", "S<sub>vertical</sub>", "[PSI/ft]"),
        ("min_horizontal_stress_gradient", "S<sub>hmin</sub>", "[PSI/ft]"),
        ("max_horizontal_stress_gradient", "S<sub>Hmax</sub>", "[PSI/ft]"),
        ("aphi_value", "A-Phi", ""),
        ("initial_pore_pressure_gradient", "Natural Pore Pressure", "[PSI/ft]"),
        ("strike_angle", "Strike of fault", "[Degrees]"),
        ("dip_angle", "Dip of fault", "[Degrees]"),
        ("max_stress_azimuth", "S<sub>Hmax</sub> Azimuth", "[Degrees]"),
        ("friction_coefficient", "Mu", "Coefficient of Friction"),
    ]
    return [spec for spec in candidates if spec[0] in sample_inputs_df.columns]


def _hydrology_input_histogram_specs(sample_inputs_df: pd.DataFrame):
    candidates = [
        ("aquifer_thickness", "Injection Formation Thickness", "[ft]"),
        ("porosity", "Porosity", "[%]"),
        ("permeability", "Permeability", "[mD]"),
        ("fluid_density", "Fluid Density", "[kg/m^3]"),
        ("dynamic_viscosity", "Dynamic Viscosity", "[Pa.s]"),
        ("fluid_compressibility", "Fluid Compressibility", "[Pa^-1]"),
        ("rock_compressibility", "Rock Compressibility", "[Pa^-1]"),
    ]
    return [spec for spec in candidates if spec[0] in sample_inputs_df.columns]


def _histogram_panels_for_fault(
    sample_inputs_df: pd.DataFrame,
    mc_results_df: pd.DataFrame,
    fault_id: str,
    specs
):
    sample_fault_df = sample_inputs_df
    result_fault_df = mc_results_df

    panels = []
    for column, title, unit in specs:
        if column not in sample_fault_df.columns:
            continue
        values = pd.to_numeric(sample_fault_df[column], errors="coerce").dropna()
        if not values.empty:
            panels.append({"title": title, "unit": unit, "values": values})

    if "SlipPressure" in result_fault_df.columns:
        slip_values = pd.to_numeric(result_fault_df["SlipPressure"], errors="coerce").dropna()
        if not slip_values.empty:
            panels.append({"title": f"result: pore pressure to slip for fault {fault_id}", "unit": "[PSI]", "values": slip_values})

    return panels


def _figure_json_for_hydrology_inputs(sample_inputs_df: pd.DataFrame, specs, bins: int) -> str:
    panels = []
    for column, title, unit in specs:
        values = pd.to_numeric(sample_inputs_df[column], errors="coerce").dropna()
        if not values.empty:
            panels.append({"title": title, "unit": unit, "values": values})

    if not panels:
        fig = go.Figure()
        fig.update_layout(title="No hydrology input histogram data available")
        return fig.to_json()

    cols = 4
    rows = int((len(panels) + cols - 1) / cols)
    specs_grid = [[{} for _ in range(cols)] for _ in range(rows)]
    if len(panels) % cols:
        for col in range(len(panels) % cols, cols):
            specs_grid[-1][col] = None

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[panel["title"] for panel in panels],
        specs=specs_grid,
        horizontal_spacing=0.055,
        vertical_spacing=0.24 if rows > 1 else 0.12,
    )

    for index, panel in enumerate(panels):
        row = int(index / cols) + 1
        col = (index % cols) + 1
        values = panel["values"]
        fig.add_trace(
            _prebinned_histogram_trace(values, bins),
            row=row,
            col=col,
        )
        expected_count = len(values) / float(bins) if bins > 0 else 0.0
        fig.add_hline(
            y=expected_count,
            line={"color": "#e11d48", "width": 2, "dash": "dash"},
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=panel["unit"], row=row, col=col, automargin=True)
        fig.update_yaxes(title_text="Number of Realizations" if col == 1 else "", row=row, col=col, automargin=True)

    apply_modern_subplots_layout(fig, rows=rows)
    fig.update_layout(bargap=0)
    return fig.to_json()


def _prebinned_histogram_trace(values, bins: int) -> go.Bar:
    v = np.asarray(values, dtype=float)
    v_min, v_max = float(v.min()), float(v.max())
    hist_range = (v_min, v_max) if v_min != v_max else (v_min - 0.5, v_max + 0.5)
    counts, bin_edges = np.histogram(v, bins=bins, range=hist_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
    return go.Bar(
        x=bin_centers.tolist(),
        y=counts.tolist(),
        width=bin_width,
        marker={"color": "#2563eb", "line": {"color": "rgba(30, 64, 175, 0.35)", "width": 0.6}},
        hovertemplate="%{x:.4g}<br>Count: %{y}<extra></extra>",
        showlegend=False,
    )


def _figure_json_for_fault(
    sample_inputs_df: pd.DataFrame,
    mc_results_df: pd.DataFrame,
    fault_id: str,
    specs,
    bins: int,
) -> str:
    panels = _histogram_panels_for_fault(sample_inputs_df, mc_results_df, fault_id, specs)

    if not panels:
        fig = go.Figure()
        fig.update_layout(title=f"No histogram data available for fault {fault_id}")
        return fig.to_json()

    cols = 5
    rows = int((len(panels) + cols - 1) / cols)
    specs_grid = [[{} for _ in range(cols)] for _ in range(rows)]
    if len(panels) % cols:
        for col in range(len(panels) % cols, cols):
            specs_grid[-1][col] = None

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[panel["title"] for panel in panels],
        specs=specs_grid,
        horizontal_spacing=0.045,
        vertical_spacing=0.20 if rows > 1 else 0.12,
    )

    for index, panel in enumerate(panels):
        row = int(index / cols) + 1
        col = (index % cols) + 1
        values = panel["values"]
        fig.add_trace(
            _prebinned_histogram_trace(values, bins),
            row=row,
            col=col,
        )
        expected_count = len(values) / float(bins) if bins > 0 else 0.0
        fig.add_hline(
            y=expected_count,
            line={"color": "#e11d48", "width": 2, "dash": "dash"},
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=panel["unit"], row=row, col=col, automargin=True)
        fig.update_yaxes(title_text="Number of Realizations", row=row, col=col, automargin=True)

    apply_modern_subplots_layout(fig, rows=rows)
    fig.update_layout(bargap=0)
    return fig.to_json()


def _hydrology_input_distribution_histograms_html(title: str, figure_json: str, metadata_labels) -> str:
    title_json = json.dumps(title)
    figure_json = json.dumps(figure_json)
    title_html = html.escape(title)
    metadata_html = " ".join(html.escape(str(label)) for label in metadata_labels)
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
      background: {MODERN_PAPER_BG};
      color: {MODERN_TEXT_COLOR};
      font-family: {MODERN_FONT_FAMILY};
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }}
    #plot {{
      width: 100%;
      height: 100vh;
      box-sizing: border-box;
      padding: 10px;
    }}
    .metadata {{
      display: none;
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="metadata">{metadata_html}</div>
  <div id="plot"></div>
  <script>
    const title = {title_json};
    const figure = JSON.parse({figure_json});
    Plotly.react(document.getElementById('plot'), figure.data || [], figure.layout || {{}}, {json.dumps(PLOTLY_CONFIG)});
  </script>
</body>
</html>
"""


def _input_distribution_histograms_html(title: str, fault_figures: dict, default_fault: str, metadata_labels) -> str:
    title_json = json.dumps(title)
    default_fault_json = json.dumps(default_fault)
    fault_figures_json = json.dumps(fault_figures)
    title_html = html.escape(title)
    metadata_html = " ".join(html.escape(str(label)) for label in metadata_labels)
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
      background: {MODERN_PAPER_BG};
      color: {MODERN_TEXT_COLOR};
      font-family: {MODERN_FONT_FAMILY};
      -webkit-font-smoothing: antialiased;
    }}
    body {{
      overflow: hidden;
    }}
    .compact {{
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: {MODERN_PAPER_BG};
    }}
    .compact-panel {{
      width: min(520px, calc(100% - 32px));
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      padding: 24px;
      text-align: center;
      box-sizing: border-box;
      box-shadow: {MODERN_SHADOW};
      background: {MODERN_PLOT_BG};
    }}
    .compact-title {{
      margin: 0 0 10px 0;
      font-size: 18px;
      font-weight: 700;
    }}
    .compact-copy {{
      margin: 0;
      color: {MODERN_MUTED_TEXT_COLOR};
      line-height: 1.35;
    }}
    .viewer {{
      display: none;
      height: 100vh;
      flex-direction: column;
      background: {MODERN_PAPER_BG};
    }}
    .fault-select {{
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_CONTROL_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 9px 12px;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-height: 48px;
      padding: 8px 12px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      box-sizing: border-box;
      background: {MODERN_CONTROL_BG};
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
    }}
    .toolbar-title {{
      flex: 1;
      font-weight: 700;
    }}
    #plot {{
      flex: 1;
      min-height: 0;
    }}
    .metadata {{
      display: none;
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="metadata">{metadata_html}</div>
  <div id="compact" class="compact">
    <div class="compact-panel">
      <h1 class="compact-title">{title_html}</h1>
      <p class="compact-copy">Use the graph fullscreen button to view the per-fault input distribution histograms.</p>
    </div>
  </div>
  <div id="viewer" class="viewer" aria-hidden="true">
    <div class="toolbar">
      <div class="toolbar-title">{title_html}</div>
      <label>Fault
        <select id="fault-select" class="fault-select" onchange="renderFault(this.value)"></select>
      </label>
    </div>
    <div id="plot"></div>
  </div>
  <script>
    const title = {title_json};
    const defaultFault = {default_fault_json};
    const faultFigures = {fault_figures_json};
    let renderedFault = null;

    function populateFaultSelect(targetDocument) {{
      const select = targetDocument.getElementById('fault-select');
      select.innerHTML = '';
      Object.keys(faultFigures).forEach(function(faultId) {{
        const option = targetDocument.createElement('option');
        option.value = faultId;
        option.textContent = 'Fault ' + faultId;
        if (faultId === defaultFault) option.selected = true;
        select.appendChild(option);
      }});
    }}

    function renderInto(targetDocument, faultId) {{
      const plot = targetDocument.getElementById('plot');
      const figure = JSON.parse(faultFigures[faultId]);
      Plotly.react(plot, figure.data || [], figure.layout || {{}}, {json.dumps(PLOTLY_CONFIG)});
    }}

    function renderFault(faultId) {{
      renderedFault = faultId;
      renderInto(document, faultId);
    }}

    function isLargeEnoughForFullHistogram() {{
      return window.innerWidth >= 1100 && window.innerHeight >= 650;
    }}

    function syncViewMode() {{
      const compact = document.getElementById('compact');
      const viewer = document.getElementById('viewer');
      if (isLargeEnoughForFullHistogram()) {{
        compact.style.display = 'none';
        viewer.style.display = 'flex';
        viewer.setAttribute('aria-hidden', 'false');
        populateFaultSelect(document);
        renderFault(renderedFault || defaultFault);
      }} else {{
        viewer.style.display = 'none';
        viewer.setAttribute('aria-hidden', 'true');
        compact.style.display = 'flex';
      }}
    }}

    window.addEventListener('load', syncViewMode);
    window.addEventListener('resize', syncViewMode);
  </script>
</body>
</html>
"""


def save_hydrology_input_distribution_histograms_artifact(
    helper,
    step_index: int,
    sample_inputs_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    display_order: int,
    bins: int = 20,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if sample_inputs_df is None or sample_inputs_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no sampled hydrology input records were available.")
            return None

        graph_inputs = sample_inputs_df.copy()
        specs = _hydrology_input_histogram_specs(graph_inputs)
        if not specs:
            add_graph_warning(helper, step_index, f"{prefix} because no sampled hydrology input columns were available.")
            return None

        figure_json = _figure_json_for_hydrology_inputs(graph_inputs, specs, bins)
        metadata_labels = [spec[1] for spec in specs] + ["Number of Realizations"]
        html_text = _hydrology_input_distribution_histograms_html(title, figure_json, metadata_labels)
        output_path = os.path.join(graph_artifacts_dir(helper), f"{artifact_key}.html")
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_text)

        remove_graph_artifact(helper, artifact_key)
        helper.saveGraphArtifact(
            key=artifact_key,
            title=title,
            renderer="html",
            path=output_path,
            contentType="text/html",
            caption=f"Interactive {title.lower()} generated by FSP.",
            displayOrder=display_order,
            preferredHeight=740,
        )
        return output_path
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_input_distribution_histograms_artifact(
    helper,
    step_index: int,
    sample_inputs_df: pd.DataFrame,
    mc_results_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    display_order: int,
    bins: int = 20,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(sample_inputs_df, ["FaultID"]) or not has_columns(mc_results_df, ["FaultID", "SlipPressure"]):
            add_graph_warning(helper, step_index, f"{prefix} because required input distribution columns are missing.")
            return None

        graph_inputs = sample_inputs_df.copy()
        graph_results = mc_results_df.copy()
        graph_inputs["FaultID"] = graph_inputs["FaultID"].astype(str)
        graph_results["FaultID"] = graph_results["FaultID"].astype(str)

        specs = _input_histogram_specs(graph_inputs)
        if not specs:
            add_graph_warning(helper, step_index, f"{prefix} because no sampled input columns were available.")
            return None

        fault_ids = sorted(set(graph_inputs["FaultID"].dropna().astype(str)) & set(graph_results["FaultID"].dropna().astype(str)))
        if not fault_ids:
            add_graph_warning(helper, step_index, f"{prefix} because no fault-specific histogram data were available.")
            return None

        input_groups = {str(fid): group for fid, group in graph_inputs.groupby("FaultID", sort=False)}
        result_groups = {str(fid): group for fid, group in graph_results.groupby("FaultID", sort=False)}
        _empty_df = pd.DataFrame()
        fault_figures = {
            fault_id: _figure_json_for_fault(
                input_groups.get(fault_id, _empty_df),
                result_groups.get(fault_id, _empty_df),
                fault_id, specs, bins,
            )
            for fault_id in fault_ids
        }
        metadata_labels = []
        for fault_id in fault_ids:
            metadata_labels.append(f"Fault {fault_id}")
            metadata_labels.append(f"result: pore pressure to slip for fault {fault_id}")
        metadata_labels.extend(spec[1] for spec in specs)
        metadata_labels.append("Number of Realizations")

        html_text = _input_distribution_histograms_html(title, fault_figures, fault_ids[0], metadata_labels)
        output_path = os.path.join(graph_artifacts_dir(helper), f"{artifact_key}.html")
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_text)

        remove_graph_artifact(helper, artifact_key)
        helper.saveGraphArtifact(
            key=artifact_key,
            title=title,
            renderer="html",
            path=output_path,
            contentType="text/html",
            caption=f"Interactive {title.lower()} generated by FSP.",
            displayOrder=display_order,
            preferredHeight=220,
        )
        return output_path
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def _fault_sensitivity_tornado_html(
    *,
    title: str,
    fault_payload: dict,
    default_fault: str,
    x_label: str,
    color_min: float,
    color_max: float,
):
    payload_json = json.dumps(fault_payload, separators=(",", ":"))
    default_fault_json = json.dumps(default_fault)
    x_label_json = json.dumps(x_label)
    config_json = json.dumps(PLOTLY_CONFIG)
    color_scale_json = json.dumps(SLIP_PRESSURE_COLOR_SCALE, separators=(",", ":"))
    color_min_json = json.dumps(color_min)
    color_max_json = json.dumps(color_max)
    fault_options = "\n".join(
        f'<option value="{html.escape(str(fault_id))}"{" selected" if str(fault_id) == str(default_fault) else ""}>Fault {html.escape(str(fault_id))}</option>'
        for fault_id in fault_payload.keys()
    )
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
      background: {MODERN_PAPER_BG};
      color: {MODERN_TEXT_COLOR};
      font-family: {MODERN_FONT_FAMILY};
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }}
    .viewer {{
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: {MODERN_PAPER_BG};
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: flex-start;
      gap: 12px;
      min-height: 46px;
      padding: 8px 12px;
      border-bottom: 1px solid {MODERN_BORDER_COLOR};
      box-sizing: border-box;
      background: {MODERN_CONTROL_BG};
      box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
    }}
    .fault-select {{
      min-width: 180px;
      max-width: min(100%, 520px);
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 6px 8px;
      font: inherit;
    }}
    .content {{
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(250px, 280px);
      gap: 10px;
      padding: 10px;
      box-sizing: border-box;
      overflow: hidden;
    }}
    .plot-panel, .control-panel {{
      min-width: 0;
      min-height: 0;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 10px;
      box-shadow: {MODERN_SHADOW};
      overflow: hidden;
      background: {MODERN_PLOT_BG};
    }}
    #plot {{
      width: 100%;
      height: 100%;
    }}
    .control-panel {{
      background: {MODERN_CONTROL_BG};
      padding: 10px;
      box-sizing: border-box;
      overflow: auto;
    }}
    .control-title {{
      margin-bottom: 6px;
      font-weight: 700;
      font-size: 13px;
    }}
    .legend-note {{
      margin-bottom: 6px;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-size: 11px;
      line-height: 1.3;
    }}
    .colorbar-ramp {{
      height: 9px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      background: linear-gradient(90deg, #800000 0%, #ff0000 8%, #ff5a00 18%, #ffc300 28%, #ffff00 35%, #ffff00 67%, #aad400 78%, #61b000 88%, #007f00 100%);
    }}
    .colorbar-values {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 4px;
      font-size: 11px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .range-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 6px;
    }}
    .range-field {{
      display: flex;
      flex-direction: column;
      gap: 3px;
      font-size: 11.5px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .range-field input {{
      width: 100%;
      min-width: 0;
      box-sizing: border-box;
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 5px 7px;
    }}
    .legend-button {{
      margin-top: 6px;
      width: 100%;
      font: inherit;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: {MODERN_PLOT_BG};
      color: {MODERN_TEXT_COLOR};
      padding: 5px 10px;
      cursor: pointer;
    }}
    @media (max-width: 780px) {{
      .toolbar {{ align-items: flex-start; flex-wrap: wrap; }}
      .content {{ grid-template-columns: 1fr; grid-template-rows: minmax(260px, 1fr) minmax(170px, auto); }}
    }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div class="viewer">
    <div class="toolbar">
      <select id="fault-select" class="fault-select" onchange="renderSelectedFault()">
        {fault_options}
      </select>
    </div>
    <div class="content">
      <div class="plot-panel"><div id="plot"></div></div>
      <aside class="control-panel" aria-label="Pore Pressure to Slip color controls">
        <div class="control-title">Pore Pressure to Slip</div>
        <div class="legend-note">Adjust the pore-pressure-to-slip range to recolor the selected fault's tornado bars.</div>
        <div class="colorbar">
          <div class="colorbar-ramp" aria-hidden="true"></div>
          <div class="colorbar-values"><span id="colorbar-min"></span><span id="colorbar-max"></span></div>
        </div>
        <div class="range-row">
          <label class="range-field">Min PSI
            <input id="pressure-min" type="number" step="any" oninput="handleColorRangeInput()">
          </label>
          <label class="range-field">Max PSI
            <input id="pressure-max" type="number" step="any" oninput="handleColorRangeInput()">
          </label>
        </div>
        <button type="button" class="legend-button" onclick="resetColorRange()">Reset Range</button>
      </aside>
    </div>
  </div>
  <script>
    const faultPayload = {payload_json};
    const defaultFault = {default_fault_json};
    const xLabel = {x_label_json};
    const plotConfig = {config_json};
    const colorScale = {color_scale_json};
    const autoColorMin = {color_min_json};
    const autoColorMax = {color_max_json};

    function formatPressure(value) {{
      return Number.isFinite(value) ? value.toLocaleString(undefined, {{ maximumFractionDigits: 2 }}) + ' psi' : '';
    }}

    function hexToRgb(hex) {{
      const cleaned = String(hex).replace('#', '');
      return [
        parseInt(cleaned.substring(0, 2), 16),
        parseInt(cleaned.substring(2, 4), 16),
        parseInt(cleaned.substring(4, 6), 16)
      ];
    }}

    function mixColor(start, end, ratio) {{
      const a = hexToRgb(start);
      const b = hexToRgb(end);
      return 'rgb(' +
        Math.round(a[0] + (b[0] - a[0]) * ratio) + ',' +
        Math.round(a[1] + (b[1] - a[1]) * ratio) + ',' +
        Math.round(a[2] + (b[2] - a[2]) * ratio) + ')';
    }}

    function selectedColorRange() {{
      const minValue = Number(document.getElementById('pressure-min').value);
      const maxValue = Number(document.getElementById('pressure-max').value);
      if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {{
        return {{ minValue, maxValue }};
      }}
      return {{ minValue: autoColorMin, maxValue: autoColorMax }};
    }}

    function scaledColor(value) {{
      const colorRange = selectedColorRange();
      const raw = Number(value);
      if (!Number.isFinite(raw) || colorRange.maxValue <= colorRange.minValue) return '#16a34a';
      const normalized = Math.max(0, Math.min(1, (raw - colorRange.minValue) / (colorRange.maxValue - colorRange.minValue)));
      for (let i = 1; i < colorScale.length; i++) {{
        const previous = colorScale[i - 1];
        const current = colorScale[i];
        if (normalized <= current[0]) {{
          const span = current[0] - previous[0] || 1;
          return mixColor(previous[1], current[1], (normalized - previous[0]) / span);
        }}
      }}
      return colorScale[colorScale.length - 1][1];
    }}

    function selectedFaultId() {{
      return document.getElementById('fault-select').value || defaultFault;
    }}

    function selectedBaselineSlipPressure() {{
      const fault = faultPayload[selectedFaultId()];
      return fault ? Number(fault.baselineSlipPressure) : NaN;
    }}

    function updateColorbarLabels() {{
      const colorRange = selectedColorRange();
      document.getElementById('colorbar-min').textContent = formatPressure(colorRange.minValue);
      document.getElementById('colorbar-max').textContent = formatPressure(colorRange.maxValue);
    }}

    function handleColorRangeInput() {{
      updateColorbarLabels();
      renderSelectedFault();
    }}

    function resetColorRange() {{
      document.getElementById('pressure-min').value = Number(autoColorMin).toFixed(3);
      document.getElementById('pressure-max').value = Number(autoColorMax).toFixed(3);
      handleColorRangeInput();
    }}

    function renderSelectedFault() {{
      const faultId = selectedFaultId();
      const fault = faultPayload[faultId] || faultPayload[defaultFault];
      if (!fault) return;
      const baselineSlipPressure = Number(fault.baselineSlipPressure);
      const barColor = scaledColor(baselineSlipPressure);
      const customdata = fault.labels.map((label, index) => [
        fault.lowSlipPressure[index],
        fault.highSlipPressure[index],
        baselineSlipPressure,
        fault.impact[index],
        fault.method[index]
      ]);
      const trace = {{
        type: 'bar',
        orientation: 'h',
        name: 'Sensitivity range',
        x: fault.rangeWidth,
        base: fault.lowSlipPressure,
        y: fault.labels,
        marker: {{ color: barColor, line: {{ color: 'rgba(15, 23, 42, 0.28)', width: 0.8 }} }},
        customdata,
        hovertemplate: 'Fault: ' + faultId + '<br>%{{y}}<br>' +
          'Low slip pressure: %{{customdata[0]:,.2f}} psi<br>' +
          'High slip pressure: %{{customdata[1]:,.2f}} psi<br>' +
          'Baseline slip pressure: %{{customdata[2]:,.2f}} psi<br>' +
          'Impact: %{{customdata[3]:+,.2f}} psi<br>' +
          'Method: %{{customdata[4]}}<extra></extra>'
      }};
      const layout = {{
        paper_bgcolor: '{MODERN_PAPER_BG}',
        plot_bgcolor: '{MODERN_PLOT_BG}',
        font: {{ family: '{MODERN_FONT_FAMILY}', color: '{MODERN_TEXT_COLOR}' }},
        margin: {{ l: 150, r: 30, t: 24, b: 58 }},
        xaxis: {{ title: xLabel, gridcolor: '{MODERN_GRID_COLOR}', zerolinecolor: '{MODERN_AXIS_COLOR}' }},
        yaxis: {{ title: '', automargin: true }},
        barmode: 'overlay',
        bargap: 0.22,
        showlegend: false,
        shapes: [{{
          type: 'line',
          xref: 'x',
          yref: 'paper',
          x0: baselineSlipPressure,
          x1: baselineSlipPressure,
          y0: 0,
          y1: 1,
          line: {{ color: '#111827', width: 1 }}
        }}]
      }};
      Plotly.react(document.getElementById('plot'), [trace], layout, plotConfig);
    }}

    resetColorRange();
  </script>
</body>
</html>
"""


def save_uncertainty_tornado_artifact(
    helper,
    step_index: int,
    tornado_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    x_label: str,
    display_order: int,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if has_columns(tornado_df, [
            "FaultID",
            "label",
            "low_slip_pressure",
            "high_slip_pressure",
            "low_delta",
            "high_delta",
            "impact",
            "method",
        ]):
            graph_df = _clean(tornado_df, ["low_delta", "high_delta", "impact"])
            if graph_df.empty:
                add_graph_warning(helper, step_index, f"{prefix} because no valid tornado records were available.")
                return None

            graph_df["FaultID"] = graph_df["FaultID"].astype(str)
            graph_df["abs_impact"] = graph_df["impact"].abs()
            default_fault = (
                graph_df.groupby("FaultID")["abs_impact"].max().sort_values(ascending=False).index[0]
            )
            fault_ids = list(graph_df.groupby("FaultID", sort=True).groups.keys())
            baseline_by_fault = {}
            if "baseline_slip_pressure" in graph_df.columns:
                baseline_series = pd.to_numeric(graph_df["baseline_slip_pressure"], errors="coerce")
                graph_df = graph_df.assign(baseline_slip_pressure=baseline_series)
                baseline_by_fault = (
                    graph_df.dropna(subset=["baseline_slip_pressure"])
                    .groupby("FaultID")["baseline_slip_pressure"]
                    .first()
                    .to_dict()
                )

            fault_payload = {}
            baseline_values = []
            for fault_id in fault_ids:
                fault_df = graph_df[graph_df["FaultID"] == fault_id].sort_values("abs_impact", ascending=True)
                baseline_slip_pressure = baseline_by_fault.get(fault_id)
                if baseline_slip_pressure is None or not np.isfinite(baseline_slip_pressure):
                    baseline_slip_pressure = float(
                        ((fault_df["low_slip_pressure"] + fault_df["high_slip_pressure"]) / 2.0).median()
                    )
                baseline_values.append(float(baseline_slip_pressure))
                range_width = fault_df["high_slip_pressure"] - fault_df["low_slip_pressure"]
                fault_payload[str(fault_id)] = {
                    "labels": [str(value) for value in fault_df["label"]],
                    "lowSlipPressure": [float(value) for value in fault_df["low_slip_pressure"]],
                    "highSlipPressure": [float(value) for value in fault_df["high_slip_pressure"]],
                    "rangeWidth": [float(value) for value in range_width],
                    "impact": [float(value) for value in fault_df["impact"]],
                    "method": [str(value) for value in fault_df["method"]],
                    "baselineSlipPressure": float(baseline_slip_pressure),
                }

            color_min = min(baseline_values)
            color_max = max(baseline_values)
            if color_min == color_max:
                color_min -= 1.0
                color_max += 1.0

            html_text = _fault_sensitivity_tornado_html(
                title=title,
                fault_payload=fault_payload,
                default_fault=str(default_fault),
                x_label=x_label,
                color_min=round(float(color_min), 6),
                color_max=round(float(color_max), 6),
            )
            return _write_html_artifact(
                helper,
                artifact_key=artifact_key,
                title=title,
                html_text=html_text,
                caption=f"Interactive {title.lower()} generated by FSP.",
                display_order=display_order,
                preferred_height=600,
            )

        if not has_columns(tornado_df, ["label", "min", "max"]):
            add_graph_warning(helper, step_index, f"{prefix} because required tornado columns are missing.")
            return None

        graph_df = _clean(tornado_df, ["min", "max"])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid tornado records were available.")
            return None

        graph_df["span"] = (graph_df["max"] - graph_df["min"]).abs()
        graph_df = graph_df.sort_values("span", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=graph_df["max"],
            y=graph_df["label"],
            orientation="h",
            name="High",
            marker={"color": "#e11d48", "line": {"color": "rgba(159, 18, 57, 0.35)", "width": 0.6}},
            hovertemplate="%{y}<br>High: %{x:,.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=graph_df["min"],
            y=graph_df["label"],
            orientation="h",
            name="Low",
            marker={"color": "#2563eb", "line": {"color": "rgba(30, 64, 175, 0.35)", "width": 0.6}},
            hovertemplate="%{y}<br>Low: %{x:,.2f}<extra></extra>",
        ))
        apply_scientific_layout(fig, x_title=x_label, y_title="")
        fig.update_layout(barmode="overlay", bargap=0.22)
        fig.add_vline(x=0, line={"color": "#334155", "width": 1})
        return write_plotly_artifact(
            helper,
            fig,
            artifact_key,
            title,
            caption=f"Interactive {title.lower()} generated by FSP.",
            display_order=display_order,
            preferred_height=540,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_fault_sensitivity_artifact(
    helper,
    step_index: int,
    sensitivity_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    display_order: int,
):
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(sensitivity_df, ["FaultID", "slip_pressure", "probability"]):
            add_graph_warning(helper, step_index, f"{prefix} because required sensitivity columns are missing.")
            return None

        graph_df = _clean(sensitivity_df, ["slip_pressure", "probability"])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid sensitivity records were available.")
            return None

        graph_df = graph_df.sort_values("slip_pressure", ascending=True)
        fig = go.Figure(go.Bar(
            x=graph_df["slip_pressure"],
            y=graph_df["FaultID"].astype(str),
            orientation="h",
            marker={
                "color": graph_df["probability"],
                "colorscale": MODERN_HEAT_COLORSCALE,
                "showscale": True,
                "colorbar": modern_colorbar("Probability"),
            },
            customdata=graph_df["probability"],
            hovertemplate="Fault: %{y}<br>P10-P90 span: %{x:,.2f} psi<br>Probability: %{customdata:.3f}<extra></extra>",
        ))
        apply_scientific_layout(fig, x_title="P10-P90 Slip Pressure Span (psi)", y_title="Fault ID")
        return write_plotly_artifact(
            helper,
            fig,
            artifact_key,
            title,
            caption=f"Interactive {title.lower()} generated by FSP.",
            display_order=display_order,
            preferred_height=540,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_radial_curves_artifact(helper, step_index: int, radial_df: pd.DataFrame):
    prefix = "Hydrology radial pressure graph was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(radial_df, ["ID", "Distance_km", "Pressure_psi"]):
            add_graph_warning(helper, step_index, f"{prefix} because required radial curve columns are missing.")
            return None

        graph_df = _clean(radial_df, ["Distance_km", "Pressure_psi"])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid radial curve records were available.")
            return None

        series_payload = {}
        well_groups = list(graph_df.groupby(graph_df["ID"].astype(str), sort=True))
        for index, (well_id, well_df) in enumerate(well_groups):
            well_df = well_df.sort_values("Distance_km")
            color = SCIENTIFIC_COLORS[index % len(SCIENTIFIC_COLORS)]
            series_payload[str(well_id)] = {
                "label": f"Well {well_id}",
                "x": [round(float(value), 6) for value in well_df["Distance_km"]],
                "y": [round(float(value), 6) for value in well_df["Pressure_psi"]],
                "mode": "lines",
                "color": color,
                "hovertemplate": (
                    "Well: Well "
                    + str(well_id)
                    + "<br>Distance: %{x:,.2f} km<br>Pressure: %{y:,.2f} psi<extra></extra>"
                ),
            }

        html_text = _multi_curve_selector_plotly_html(
            title="Hydrology Radial Pressure",
            series_payload=series_payload,
            x_label="Distance from Injection Well (km)",
            y_label="Pressure Change (psi)",
            selector_title="Injection Wells",
            selector_subject_plural="wells",
            selector_empty_message="Select one or more injection wells from the legend to display their curves.",
            preferred_plot_height="clamp(360px, 54vh, 500px)",
        )
        return _write_html_artifact(
            helper,
            artifact_key="fsp-deterministic-hydrology-radial-pressure",
            title="Hydrology Radial Pressure",
            html_text=html_text,
            caption="Interactive deterministic hydrology radial pressure chart generated by FSP.",
            display_order=40,
            preferred_height=800,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_slip_potential_artifact(helper, step_index: int, slip_df: pd.DataFrame):
    prefix = "Slip potential graph was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(slip_df, ["ID", "probability", "Pressure", "slip_pressure"]):
            add_graph_warning(helper, step_index, f"{prefix} because required slip potential columns are missing.")
            return None

        graph_df = _clean(slip_df, ["probability", "Pressure", "slip_pressure"])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid slip potential records were available.")
            return None

        graph_df = graph_df.sort_values("probability", ascending=False)
        fig = go.Figure(go.Bar(
            x=graph_df["ID"].astype(str),
            y=graph_df["probability"],
            marker={
                "color": graph_df["Pressure"],
                "colorscale": MODERN_HEAT_COLORSCALE,
                "showscale": True,
                "colorbar": modern_colorbar("Mean Pressure (psi)"),
            },
            customdata=graph_df[["Pressure", "slip_pressure"]],
            hovertemplate=(
                "Fault: %{x}<br>Slip probability: %{y:.3f}<br>"
                "Mean pressure: %{customdata[0]:,.2f} psi<br>Slip threshold: %{customdata[1]:,.2f} psi<extra></extra>"
            ),
        ))
        apply_scientific_layout(fig, x_title="Fault ID", y_title="Slip Probability")
        fig.update_yaxes(range=[0, 1], tickformat=".2f")
        fig.update_layout(showlegend=False)
        return write_plotly_artifact(
            helper,
            fig,
            "fsp-probabilistic-hydrology-slip-potential",
            "Slip Potential",
            caption="Interactive probabilistic hydrology slip potential chart generated by FSP.",
            display_order=53,
            preferred_height=520,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_summary_artifacts(
    helper,
    step_index: int,
    fsp_df: pd.DataFrame,
    pressure_df: pd.DataFrame,
    year_of_interest: Optional[float] = None,
):
    """Save FSP Through Time and Pressure Through Time graph artifacts for Step 6."""
    save_time_series_artifact(
        helper,
        step_index,
        fsp_df,
        value_column="FSP",
        artifact_key="fsp-summary-fsp-through-time",
        title="FSP Through Time",
        y_label="Fault Slip Potential",
        display_order=60,
        y_range=[0, 1],
        show_fsp_background=True,
        show_color_tab=True,
        color_tab_label="FSP Range",
        default_color_min=0.0,
        default_color_max=1.0,
        year_of_interest=year_of_interest,
    )
    save_time_series_artifact(
        helper,
        step_index,
        pressure_df,
        value_column="Pressure",
        artifact_key="fsp-summary-pressure-through-time",
        title="Pressure Through Time",
        y_label="Pressure Change (psi)",
        display_order=61,
        y_range=None,
        show_fsp_background=False,
        show_color_tab=False,
        year_of_interest=year_of_interest,
    )


def save_time_series_artifact(
    helper,
    step_index: int,
    df: pd.DataFrame,
    *,
    value_column: str,
    artifact_key: str,
    title: str,
    y_label: str,
    display_order: int,
    y_range,
    show_fsp_background: bool = False,
    show_color_tab: bool = True,
    color_tab_label: str = "Color Range",
    default_color_min: Optional[float] = None,
    default_color_max: Optional[float] = None,
    year_of_interest: Optional[float] = None,
):
    """Build and register an interactive time-series graph artifact."""
    prefix = _warn_prefix(title)
    try:
        remove_step_messages(helper, step_index, prefix)
        if not has_columns(df, ["ID", "Year", value_column]):
            add_graph_warning(helper, step_index, f"{prefix} because required time-series columns are missing.")
            return None

        graph_df = _clean(df, ["Year", value_column])
        if graph_df.empty:
            add_graph_warning(helper, step_index, f"{prefix} because no valid time-series records were available.")
            return None

        series_payload = {}
        for fault_id, fault_df in graph_df.groupby(graph_df["ID"].astype(str), sort=True):
            fault_df = fault_df.sort_values("Year")
            series_payload[str(fault_id)] = {
                "x": fault_df["Year"].astype(int).tolist(),
                "y": fault_df[value_column].round(6).tolist(),
                "mode": "lines+markers",
                "hovertemplate": f"Fault: %{{fullData.name}}<br>Year: %{{x:.0f}}<br>{y_label}: %{{y:,.3f}}<extra></extra>",
            }

        default_id = next(iter(series_payload))
        html_text = _single_series_plotly_html(
            title=title,
            series_payload=series_payload,
            default_id=default_id,
            x_label="Year",
            y_label=y_label,
            y_range=y_range,
            y_tickformat=".2f" if y_range is not None else None,
            show_fsp_background=show_fsp_background,
            show_color_tab=show_color_tab,
            color_tab_label=color_tab_label,
            default_color_min=default_color_min,
            default_color_max=default_color_max,
            year_of_interest=year_of_interest,
        )
        return _write_html_artifact(
            helper,
            artifact_key=artifact_key,
            title=title,
            html_text=html_text,
            caption=f"Interactive {title.lower()} generated by FSP.",
            display_order=display_order,
            preferred_height=520,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None
