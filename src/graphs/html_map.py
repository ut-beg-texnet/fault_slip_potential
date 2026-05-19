"""Standalone Leaflet map artifacts for FSP portal output.

These maps intentionally use the portal's existing ``html`` artifact renderer.
For large FSP runs this avoids hundreds of portal-managed Leaflet layer
manifests and sidecar CSV requests while preserving per-fault/per-well toggles.
"""
import html
import json
import math
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from fsp.io.coords import latlon_to_wkt
from graphs.artifacts import (
    MODERN_BORDER_COLOR,
    MODERN_CONTROL_BG,
    MODERN_FONT_FAMILY,
    MODERN_MUTED_TEXT_COLOR,
    MODERN_PAPER_BG,
    MODERN_SHADOW,
    MODERN_TEXT_COLOR,
    SLIP_PRESSURE_COLOR_SCALE,
    add_graph_warning,
    graph_artifacts_dir,
    has_columns,
    remove_graph_artifact,
    remove_step_messages,
)


def _finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return round(result, 6)


def _clean_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    return value


def _with_fault_wkt(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "wkt" in df.columns:
        return df
    required = ["Latitude(WGS84)", "Longitude(WGS84)", "Strike", "LengthKm"]
    if not has_columns(df, required):
        return df
    return latlon_to_wkt(df)


def _numeric_range(df: pd.DataFrame, column: Optional[str]):
    if not column or column not in df.columns:
        return None, None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None, None
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    return min_value, max_value


def _bounds_from_points(points):
    valid = [
        (point[0], point[1])
        for point in points
        if point and len(point) == 2 and point[0] is not None and point[1] is not None
    ]
    if not valid:
        return None
    return [
        [min(lat for lat, _ in valid), min(lon for _, lon in valid)],
        [max(lat for lat, _ in valid), max(lon for _, lon in valid)],
    ]


def _merge_bounds(bounds_list):
    valid = [bounds for bounds in bounds_list if bounds]
    if not valid:
        return [[31.0, -100.0], [31.0, -100.0]]
    return [
        [min(bounds[0][0] for bounds in valid), min(bounds[0][1] for bounds in valid)],
        [max(bounds[1][0] for bounds in valid), max(bounds[1][1] for bounds in valid)],
    ]


def fault_payload(
    fault_df: pd.DataFrame,
    *,
    popup_fields: Iterable[str],
    value_column: Optional[str] = None,
) -> list:
    fault_df = _with_fault_wkt(fault_df)
    if fault_df is None or fault_df.empty:
        return []

    fields = list(dict.fromkeys(["FaultID", "ID"] + [field for field in popup_fields if field in fault_df.columns]))
    if value_column and value_column in fault_df.columns and value_column not in fields:
        fields.append(value_column)

    rows = []
    for index, row in fault_df.iterrows():
        props = {
            field: _clean_json_value(row.get(field))
            for field in fields
            if field in fault_df.columns
        }
        label = str(props.get("FaultID") or props.get("ID") or f"Fault {index + 1}")
        item = {
            "id": label,
            "label": label,
            "properties": props,
        }

        wkt = str(row.get("wkt", "") or "")
        if wkt.upper().startswith("LINESTRING"):
            item["wkt"] = wkt
        else:
            lat = _finite_float(row.get("Latitude(WGS84)"))
            lon = _finite_float(row.get("Longitude(WGS84)"))
            if lat is None or lon is None:
                continue
            item["lat"] = lat
            item["lon"] = lon
        rows.append(item)
    return rows


def well_payload(
    well_df: Optional[pd.DataFrame],
    *,
    id_column: str,
    latitude_column: str,
    longitude_column: str,
    popup_fields: Iterable[str],
) -> list:
    if well_df is None or well_df.empty or not has_columns(well_df, [latitude_column, longitude_column]):
        return []

    fields = [field for field in popup_fields if field in well_df.columns]
    rows = []
    for index, row in well_df.iterrows():
        lat = _finite_float(row.get(latitude_column))
        lon = _finite_float(row.get(longitude_column))
        if lat is None or lon is None:
            continue
        well_id = str(row.get(id_column, "") or f"Well {index + 1}")
        rows.append({
            "id": well_id,
            "label": well_id,
            "lat": lat,
            "lon": lon,
            "properties": {
                field: _clean_json_value(row.get(field))
                for field in fields
            },
        })
    return rows


def _deterministic_geomechanics_leaflet_html(
    *,
    title: str,
    faults: list,
    wells: list,
    bounds: list,
    fault_color: str,
    value_column: Optional[str],
    legend_title: str,
    value_min: Optional[float],
    value_max: Optional[float],
    field_labels: Optional[dict] = None,
) -> str:
    title_html = html.escape(title)
    title_json = json.dumps(title)
    faults_json = json.dumps(faults, separators=(",", ":"))
    wells_json = json.dumps(wells, separators=(",", ":"))
    bounds_json = json.dumps(bounds, separators=(",", ":"))
    color_scale_json = json.dumps(SLIP_PRESSURE_COLOR_SCALE, separators=(",", ":"))
    value_column_json = json.dumps(value_column)
    legend_title_html = html.escape(legend_title)
    value_min_json = json.dumps(value_min)
    value_max_json = json.dumps(value_max)
    fault_color_json = json.dumps(fault_color)
    field_labels_json = json.dumps(field_labels or {}, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      font-family: {MODERN_FONT_FAMILY};
      color: {MODERN_TEXT_COLOR};
      -webkit-font-smoothing: antialiased;
    }}
    .leaflet-container {{ background: {MODERN_PAPER_BG}; }}
    .leaflet-popup-content-wrapper {{
      border-radius: 8px;
      box-shadow: {MODERN_SHADOW};
    }}
    .leaflet-popup-content {{
      margin: 12px 14px;
      color: {MODERN_TEXT_COLOR};
      line-height: 1.45;
    }}
    .title-chip {{
      position: absolute;
      top: 12px;
      left: 54px;
      z-index: 1000;
      max-width: calc(100% - 390px);
      padding: 8px 12px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      font-weight: 700;
      font-size: 13px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
      backdrop-filter: blur(10px);
    }}
    .toolbar {{
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 1000;
      width: 306px;
      max-height: calc(100% - 24px);
      overflow: auto;
      box-sizing: border-box;
      padding: 12px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      font-size: 12px;
      backdrop-filter: blur(10px);
    }}
    .toolbar-title {{
      margin: 0 0 8px;
      font-weight: 700;
      font-size: 13px;
      color: {MODERN_TEXT_COLOR};
    }}
    .toolbar-section {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid {MODERN_BORDER_COLOR};
    }}
    .toolbar-row {{
      display: flex;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .toolbar button {{
      flex: 1;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: #ffffff;
      color: {MODERN_TEXT_COLOR};
      padding: 6px 9px;
      font: inherit;
      cursor: pointer;
    }}
    .toolbar input[type="search"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      padding: 7px 8px;
      margin-bottom: 8px;
      font: inherit;
      color: {MODERN_TEXT_COLOR};
    }}
    .summary {{
      margin-bottom: 8px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .map-option {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 0;
      color: {MODERN_MUTED_TEXT_COLOR};
      white-space: nowrap;
    }}
    .map-option input {{ accent-color: #2563eb; }}
    .option-list {{
      max-height: 184px;
      overflow: auto;
      padding-right: 3px;
    }}
    .range-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      margin-top: 8px;
    }}
    .range-field {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: {MODERN_MUTED_TEXT_COLOR};
      font-weight: 600;
    }}
    .range-field input {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      padding: 6px 7px;
      font: inherit;
      color: {MODERN_TEXT_COLOR};
    }}
    .legend {{
      position: absolute;
      left: 12px;
      bottom: 12px;
      z-index: 1000;
      min-width: 220px;
      padding: 10px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      font-size: 12px;
      color: {MODERN_MUTED_TEXT_COLOR};
      backdrop-filter: blur(10px);
    }}
    .legend-ramp {{
      height: 14px;
      margin: 7px 0;
      border: 1px solid rgba(17, 24, 39, 0.2);
      border-radius: 999px;
      background: linear-gradient(to right, #800000, #ff0000, #ff5a00, #ffc300, #ffff00, #aad400, #61b000, #007f00);
    }}
    .legend-values {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    @media (max-width: 760px) {{
      .toolbar {{
        left: 12px;
        right: 12px;
        top: auto;
        bottom: 12px;
        width: auto;
        max-height: 48%;
      }}
      .legend {{ bottom: calc(48% + 24px); }}
      .title-chip {{ max-width: calc(100% - 96px); }}
    }}
  </style>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
  <div id="map"></div>
  <div class="title-chip">{title_html}</div>
  <div class="toolbar">
    <div class="toolbar-title">Faults</div>
    <div class="summary"><span id="fault-count">0</span> visible of <span id="fault-total">0</span></div>
    <input id="fault-search" type="search" placeholder="Search faults" oninput="renderFaultControls()">
    <div class="toolbar-row">
      <button type="button" onclick="setAllFaults(true)">All</button>
      <button type="button" onclick="setAllFaults(false)">None</button>
    </div>
    <div id="fault-controls" class="option-list"></div>
    <div class="toolbar-section">
      <div class="toolbar-title">Injection Wells</div>
      <div class="summary"><span id="well-count">0</span> visible of <span id="well-total">0</span></div>
      <input id="well-search" type="search" placeholder="Search wells" oninput="renderWellControls()">
      <div class="toolbar-row">
        <button type="button" onclick="setAllWells(true)">All</button>
        <button type="button" onclick="setAllWells(false)">None</button>
      </div>
      <div id="well-controls" class="option-list"></div>
    </div>
    <div class="toolbar-section">
      <div class="toolbar-title">{legend_title_html}</div>
      <div class="range-row">
        <label class="range-field">Min PSI
          <input id="pressure-min" type="number" step="any" oninput="updateFaultColors()">
        </label>
        <label class="range-field">Max PSI
          <input id="pressure-max" type="number" step="any" oninput="updateFaultColors()">
        </label>
      </div>
    </div>
  </div>
  <div id="legend" class="legend">
    <div>{legend_title_html}</div>
    <div class="legend-ramp"></div>
    <div class="legend-values"><span id="legend-min"></span><span id="legend-max"></span></div>
  </div>
  <script>
    const title = {title_json};
    const faults = {faults_json};
    const wells = {wells_json};
    const initialBounds = {bounds_json};
    const faultColor = {fault_color_json};
    const valueColumn = {value_column_json};
    const valueMin = {value_min_json};
    const valueMax = {value_max_json};
    const colorScale = {color_scale_json};
    const fieldLabels = {field_labels_json};
    const faultLayers = [];
    const wellLayers = [];
    let selectedFaults = new Set();
    let selectedWells = new Set();

    const map = L.map('map', {{ preferCanvas: true }});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    function parseLineString(wkt) {{
      const match = /^LINESTRING\\s*\\((.*)\\)$/i.exec(wkt || '');
      if (!match) return [];
      return match[1].split(',').map(function(pair) {{
        const parts = pair.trim().split(/\\s+/).map(Number);
        return [parts[1], parts[0]];
      }}).filter(function(point) {{
        return Number.isFinite(point[0]) && Number.isFinite(point[1]);
      }});
    }}

    function escapeHtml(value) {{
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }}

    function popupHtml(properties) {{
      return Object.keys(properties || {{}}).map(function(key) {{
        const value = properties[key];
        return '<div><strong>' + escapeHtml(fieldLabels[key] || key) + ':</strong> ' + escapeHtml(value) + '</div>';
      }}).join('');
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
      const minInput = document.getElementById('pressure-min');
      const maxInput = document.getElementById('pressure-max');
      const minValue = Number(minInput.value);
      const maxValue = Number(maxInput.value);
      if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {{
        return {{ minValue, maxValue, custom: true }};
      }}
      return {{ minValue: valueMin, maxValue: valueMax, custom: false }};
    }}

    function scaledColor(value) {{
      if (!valueColumn || valueMin === null || valueMax === null || valueMax <= valueMin) return faultColor;
      const colorRange = selectedColorRange();
      const raw = Number(value);
      if (!Number.isFinite(raw)) return faultColor;
      const span = colorRange.maxValue - colorRange.minValue || 1;
      const normalized = Math.max(0, Math.min(1, (raw - colorRange.minValue) / span));
      for (let i = 1; i < colorScale.length; i++) {{
        const previous = colorScale[i - 1];
        const current = colorScale[i];
        if (normalized <= current[0]) {{
          const stopSpan = current[0] - previous[0] || 1;
          return mixColor(previous[1], current[1], (normalized - previous[0]) / stopSpan);
        }}
      }}
      return colorScale[colorScale.length - 1][1];
    }}

    function applyFaultStyle(item) {{
      const color = scaledColor(item.value);
      if (item.isLine) {{
        item.layer.setStyle({{ color: color, weight: 3.5, opacity: 0.92 }});
      }} else {{
        item.layer.setStyle({{
          radius: 5,
          color: color,
          fillColor: color,
          fillOpacity: 0.78,
          weight: 1
        }});
      }}
    }}

    function updateFaultColors() {{
      const colorRange = selectedColorRange();
      if (!colorRange.custom) {{
        document.getElementById('pressure-min').value = valueMin !== null ? Number(valueMin).toFixed(3) : '';
        document.getElementById('pressure-max').value = valueMax !== null ? Number(valueMax).toFixed(3) : '';
      }}
      document.getElementById('legend-min').textContent =
        colorRange.minValue !== null ? Number(colorRange.minValue).toLocaleString(undefined, {{ maximumFractionDigits: 2 }}) : '';
      document.getElementById('legend-max').textContent =
        colorRange.maxValue !== null ? Number(colorRange.maxValue).toLocaleString(undefined, {{ maximumFractionDigits: 2 }}) : '';
      faultLayers.forEach(applyFaultStyle);
    }}

    function makeFaultLayer(fault, index) {{
      const line = fault.wkt ? parseLineString(fault.wkt) : [];
      let layer = null;
      let isLine = false;
      if (line.length >= 2) {{
        isLine = true;
        layer = L.polyline(line, {{ color: faultColor, weight: 3.5, opacity: 0.92 }});
      }} else if (Number.isFinite(Number(fault.lat)) && Number.isFinite(Number(fault.lon))) {{
        layer = L.circleMarker([fault.lat, fault.lon], {{
          radius: 5,
          color: faultColor,
          fillColor: faultColor,
          fillOpacity: 0.78,
          weight: 1
        }});
      }}
      if (!layer) return null;
      layer.bindPopup(popupHtml(fault.properties));
      return {{
        key: String(index),
        label: fault.label || ('Fault ' + (index + 1)),
        layer,
        isLine,
        value: fault.properties ? fault.properties[valueColumn] : null
      }};
    }}

    function makeWellLayer(well, index) {{
      if (!Number.isFinite(Number(well.lat)) || !Number.isFinite(Number(well.lon))) return null;
      const layer = L.circleMarker([well.lat, well.lon], {{
        radius: 6,
        color: '#1d4ed8',
        fillColor: '#3b82f6',
        fillOpacity: 0.9,
        weight: 1
      }});
      layer.bindPopup(popupHtml(well.properties));
      return {{ key: String(index), label: well.label || ('Well ' + (index + 1)), layer }};
    }}

    function applyVisibility(items, selectedSet) {{
      items.forEach(function(item) {{
        const enabled = selectedSet.has(item.key);
        if (enabled && !map.hasLayer(item.layer)) item.layer.addTo(map);
        if (!enabled && map.hasLayer(item.layer)) map.removeLayer(item.layer);
      }});
    }}

    function filteredItems(items, inputId) {{
      const query = String(document.getElementById(inputId).value || '').toLowerCase();
      if (!query) return items;
      return items.filter(function(item) {{ return String(item.label).toLowerCase().includes(query); }});
    }}

    function renderControls(containerId, countId, totalId, inputId, items, selectedSet, updateFn) {{
      const container = document.getElementById(containerId);
      const visibleItems = filteredItems(items, inputId);
      container.innerHTML = '';
      visibleItems.forEach(function(item) {{
        const label = document.createElement('label');
        label.className = 'map-option';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = selectedSet.has(item.key);
        input.onchange = function() {{
          if (input.checked) selectedSet.add(item.key);
          else selectedSet.delete(item.key);
          updateFn();
        }};
        label.appendChild(input);
        label.appendChild(document.createTextNode(item.label));
        container.appendChild(label);
      }});
      document.getElementById(countId).textContent = selectedSet.size;
      document.getElementById(totalId).textContent = items.length;
    }}

    function updateFaultLayers() {{
      applyVisibility(faultLayers, selectedFaults);
      renderFaultControls();
    }}

    function updateWellLayers() {{
      applyVisibility(wellLayers, selectedWells);
      renderWellControls();
    }}

    function renderFaultControls() {{
      renderControls('fault-controls', 'fault-count', 'fault-total', 'fault-search', faultLayers, selectedFaults, updateFaultLayers);
    }}

    function renderWellControls() {{
      renderControls('well-controls', 'well-count', 'well-total', 'well-search', wellLayers, selectedWells, updateWellLayers);
    }}

    function setAllFaults(enabled) {{
      selectedFaults = new Set(enabled ? faultLayers.map(function(item) {{ return item.key; }}) : []);
      updateFaultLayers();
    }}

    function setAllWells(enabled) {{
      selectedWells = new Set(enabled ? wellLayers.map(function(item) {{ return item.key; }}) : []);
      updateWellLayers();
    }}

    const layerBounds = [];
    faults.forEach(function(fault, index) {{
      const item = makeFaultLayer(fault, index);
      if (!item) return;
      faultLayers.push(item);
      const bounds = item.layer.getBounds ? item.layer.getBounds() : item.layer.getLatLng().toBounds(1000);
      if (bounds && bounds.isValid && bounds.isValid()) layerBounds.push(bounds);
    }});
    wells.forEach(function(well, index) {{
      const item = makeWellLayer(well, index);
      if (!item) return;
      wellLayers.push(item);
      layerBounds.push(item.layer.getLatLng().toBounds(1000));
    }});

    setAllFaults(true);
    setAllWells(true);
    updateFaultColors();
    const initial = L.latLngBounds(initialBounds);
    if (initial.isValid()) layerBounds.push(initial);
    const combined = layerBounds.reduce(function(bounds, next) {{
      return bounds ? bounds.extend(next) : L.latLngBounds(next);
    }}, null);
    if (combined && combined.isValid()) map.fitBounds(combined.pad(0.06));
    else map.setView([31, -100], 6);
  </script>
</body>
</html>
"""


def _standalone_leaflet_html(
    *,
    title: str,
    faults: list,
    wells: list,
    bounds: list,
    fault_color: str,
    value_column: Optional[str],
    legend_title: str,
    value_min: Optional[float],
    value_max: Optional[float],
    field_labels: Optional[dict] = None,
) -> str:
    title_html = html.escape(title)
    title_json = json.dumps(title)
    faults_json = json.dumps(faults, separators=(",", ":"))
    wells_json = json.dumps(wells, separators=(",", ":"))
    bounds_json = json.dumps(bounds, separators=(",", ":"))
    color_scale_json = json.dumps(SLIP_PRESSURE_COLOR_SCALE, separators=(",", ":"))
    value_column_json = json.dumps(value_column)
    legend_title_html = html.escape(legend_title)
    legend_title_json = json.dumps(legend_title)
    value_min_json = json.dumps(value_min)
    value_max_json = json.dumps(value_max)
    fault_color_json = json.dumps(fault_color)
    field_labels_json = json.dumps(field_labels or {}, separators=(",", ":"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      font-family: {MODERN_FONT_FAMILY};
      color: {MODERN_TEXT_COLOR};
      -webkit-font-smoothing: antialiased;
    }}
    .leaflet-container {{ background: {MODERN_PAPER_BG}; }}
    .leaflet-popup-content-wrapper {{
      border-radius: 8px;
      box-shadow: {MODERN_SHADOW};
    }}
    .leaflet-popup-content {{
      margin: 12px 14px;
      color: {MODERN_TEXT_COLOR};
      line-height: 1.45;
    }}
    .title-chip {{
      position: absolute;
      top: 12px;
      left: 54px;
      z-index: 1000;
      max-width: calc(100% - 390px);
      padding: 8px 12px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      font-weight: 700;
      font-size: 13px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
      backdrop-filter: blur(10px);
    }}
    .toolbar {{
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 1000;
      width: 306px;
      max-height: calc(100% - 24px);
      overflow: auto;
      box-sizing: border-box;
      padding: 12px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      font-size: 12px;
      backdrop-filter: blur(10px);
    }}
    .toolbar-title {{
      margin: 0 0 8px;
      font-weight: 700;
      font-size: 13px;
      color: {MODERN_TEXT_COLOR};
    }}
    .toolbar-section {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid {MODERN_BORDER_COLOR};
    }}
    .toolbar-row {{
      display: flex;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .toolbar button {{
      flex: 1;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      background: #ffffff;
      color: {MODERN_TEXT_COLOR};
      padding: 6px 9px;
      font: inherit;
      cursor: pointer;
    }}
    .toolbar input[type="search"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 6px;
      padding: 7px 8px;
      margin-bottom: 8px;
      font: inherit;
      color: {MODERN_TEXT_COLOR};
    }}
    .option-list {{
      max-height: 184px;
      overflow: auto;
      padding-right: 3px;
    }}
    .map-option {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 0;
      color: {MODERN_MUTED_TEXT_COLOR};
      white-space: nowrap;
    }}
    .map-option input {{ accent-color: #2563eb; }}
    .summary {{
      margin-bottom: 8px;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .legend {{
      display: none;
      position: absolute;
      left: 12px;
      bottom: 12px;
      z-index: 1000;
      min-width: 220px;
      padding: 10px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      box-shadow: {MODERN_SHADOW};
      font-size: 12px;
      color: {MODERN_MUTED_TEXT_COLOR};
      backdrop-filter: blur(10px);
    }}
    .legend-ramp {{
      height: 14px;
      margin: 7px 0;
      border: 1px solid rgba(17, 24, 39, 0.2);
      border-radius: 999px;
      background: linear-gradient(to right, #800000, #ff0000, #ff5a00, #ffc300, #ffff00, #aad400, #61b000, #007f00);
    }}
    .legend-values {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    @media (max-width: 760px) {{
      .toolbar {{
        left: 12px;
        right: 12px;
        top: auto;
        bottom: 12px;
        width: auto;
        max-height: 48%;
      }}
      .legend {{ bottom: calc(48% + 24px); }}
      .title-chip {{ max-width: calc(100% - 96px); }}
    }}
  </style>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
  <div id="map"></div>
  <div class="title-chip">{title_html}</div>
  <div class="toolbar">
    <div class="toolbar-title">Faults</div>
    <div class="summary"><span id="fault-count">0</span> visible of <span id="fault-total">0</span></div>
    <input id="fault-search" type="search" placeholder="Search faults" oninput="renderFaultControls()">
    <div class="toolbar-row">
      <button type="button" onclick="setAllFaults(true)">All</button>
      <button type="button" onclick="setAllFaults(false)">None</button>
    </div>
    <div id="fault-controls" class="option-list"></div>
    <div class="toolbar-section">
      <div class="toolbar-title">Injection Wells</div>
      <div class="summary"><span id="well-count">0</span> visible of <span id="well-total">0</span></div>
      <input id="well-search" type="search" placeholder="Search wells" oninput="renderWellControls()">
      <div class="toolbar-row">
        <button type="button" onclick="setAllWells(true)">All</button>
        <button type="button" onclick="setAllWells(false)">None</button>
      </div>
      <div id="well-controls" class="option-list"></div>
    </div>
  </div>
  <div id="legend" class="legend">
    <div>{legend_title_html}</div>
    <div class="legend-ramp"></div>
    <div class="legend-values"><span id="legend-min"></span><span id="legend-max"></span></div>
  </div>
  <script>
    const title = {title_json};
    const faults = {faults_json};
    const wells = {wells_json};
    const initialBounds = {bounds_json};
    const faultColor = {fault_color_json};
    const valueColumn = {value_column_json};
    const legendTitle = {legend_title_json};
    const valueMin = {value_min_json};
    const valueMax = {value_max_json};
    const colorScale = {color_scale_json};
    const fieldLabels = {field_labels_json};
    const faultLayers = [];
    const wellLayers = [];
    let selectedFaults = new Set();
    let selectedWells = new Set();

    const map = L.map('map', {{ preferCanvas: true }});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    function parseLineString(wkt) {{
      const match = /^LINESTRING\\s*\\((.*)\\)$/i.exec(wkt || '');
      if (!match) return [];
      return match[1].split(',').map(function(pair) {{
        const parts = pair.trim().split(/\\s+/).map(Number);
        return [parts[1], parts[0]];
      }}).filter(function(point) {{
        return Number.isFinite(point[0]) && Number.isFinite(point[1]);
      }});
    }}

    function escapeHtml(value) {{
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }}

    function popupHtml(properties) {{
      return Object.keys(properties || {{}}).map(function(key) {{
        const value = properties[key];
        return '<div><strong>' + escapeHtml(fieldLabels[key] || key) + ':</strong> ' + escapeHtml(value) + '</div>';
      }}).join('');
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

    function scaledColor(value) {{
      if (!valueColumn || valueMin === null || valueMax === null || valueMax <= valueMin) return faultColor;
      const raw = Number(value);
      if (!Number.isFinite(raw)) return faultColor;
      const normalized = Math.max(0, Math.min(1, (raw - valueMin) / (valueMax - valueMin)));
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

    function makeFaultLayer(fault, index) {{
      const line = fault.wkt ? parseLineString(fault.wkt) : [];
      const color = scaledColor(fault.properties && fault.properties[valueColumn]);
      const options = {{ color, weight: 3.5, opacity: 0.92 }};
      let layer = null;
      if (line.length >= 2) {{
        layer = L.polyline(line, options);
      }} else if (Number.isFinite(Number(fault.lat)) && Number.isFinite(Number(fault.lon))) {{
        layer = L.circleMarker([fault.lat, fault.lon], {{
          radius: 5,
          color,
          fillColor: color,
          fillOpacity: 0.78,
          weight: 1
        }});
      }}
      if (!layer) return null;
      layer.bindPopup(popupHtml(fault.properties));
      return {{ key: String(index), label: fault.label || ('Fault ' + (index + 1)), layer }};
    }}

    function makeWellLayer(well, index) {{
      if (!Number.isFinite(Number(well.lat)) || !Number.isFinite(Number(well.lon))) return null;
      const layer = L.circleMarker([well.lat, well.lon], {{
        radius: 6,
        color: '#1d4ed8',
        fillColor: '#3b82f6',
        fillOpacity: 0.9,
        weight: 1
      }});
      layer.bindPopup(popupHtml(well.properties));
      return {{ key: String(index), label: well.label || ('Well ' + (index + 1)), layer }};
    }}

    function applyVisibility(items, selectedSet) {{
      items.forEach(function(item) {{
        const enabled = selectedSet.has(item.key);
        if (enabled && !map.hasLayer(item.layer)) item.layer.addTo(map);
        if (!enabled && map.hasLayer(item.layer)) map.removeLayer(item.layer);
      }});
    }}

    function filteredItems(items, inputId) {{
      const query = String(document.getElementById(inputId).value || '').toLowerCase();
      if (!query) return items;
      return items.filter(function(item) {{ return String(item.label).toLowerCase().includes(query); }});
    }}

    function renderControls(containerId, countId, totalId, inputId, items, selectedSet, updateFn) {{
      const container = document.getElementById(containerId);
      const visibleItems = filteredItems(items, inputId);
      container.innerHTML = '';
      visibleItems.forEach(function(item) {{
        const label = document.createElement('label');
        label.className = 'map-option';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = selectedSet.has(item.key);
        input.onchange = function() {{
          if (input.checked) selectedSet.add(item.key);
          else selectedSet.delete(item.key);
          updateFn();
        }};
        label.appendChild(input);
        label.appendChild(document.createTextNode(item.label));
        container.appendChild(label);
      }});
      document.getElementById(countId).textContent = selectedSet.size;
      document.getElementById(totalId).textContent = items.length;
    }}

    function updateFaultLayers() {{
      applyVisibility(faultLayers, selectedFaults);
      renderFaultControls();
    }}

    function updateWellLayers() {{
      applyVisibility(wellLayers, selectedWells);
      renderWellControls();
    }}

    function renderFaultControls() {{
      renderControls('fault-controls', 'fault-count', 'fault-total', 'fault-search', faultLayers, selectedFaults, updateFaultLayers);
    }}

    function renderWellControls() {{
      renderControls('well-controls', 'well-count', 'well-total', 'well-search', wellLayers, selectedWells, updateWellLayers);
    }}

    function setAllFaults(enabled) {{
      selectedFaults = new Set(enabled ? faultLayers.map(function(item) {{ return item.key; }}) : []);
      updateFaultLayers();
    }}

    function setAllWells(enabled) {{
      selectedWells = new Set(enabled ? wellLayers.map(function(item) {{ return item.key; }}) : []);
      updateWellLayers();
    }}

    const layerBounds = [];
    faults.forEach(function(fault, index) {{
      const item = makeFaultLayer(fault, index);
      if (!item) return;
      faultLayers.push(item);
      const bounds = item.layer.getBounds ? item.layer.getBounds() : item.layer.getLatLng().toBounds(1000);
      if (bounds && bounds.isValid && bounds.isValid()) layerBounds.push(bounds);
    }});
    wells.forEach(function(well, index) {{
      const item = makeWellLayer(well, index);
      if (!item) return;
      wellLayers.push(item);
      layerBounds.push(item.layer.getLatLng().toBounds(1000));
    }});

    if (valueColumn && valueMin !== null && valueMax !== null) {{
      document.getElementById('legend').style.display = 'block';
      document.getElementById('legend-min').textContent = Number(valueMin).toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
      document.getElementById('legend-max').textContent = Number(valueMax).toLocaleString(undefined, {{ maximumFractionDigits: 2 }});
    }}

    setAllFaults(true);
    setAllWells(true);
    const initial = L.latLngBounds(initialBounds);
    if (initial.isValid()) layerBounds.push(initial);
    const combined = layerBounds.reduce(function(bounds, next) {{
      return bounds ? bounds.extend(next) : L.latLngBounds(next);
    }}, null);
    if (combined && combined.isValid()) map.fitBounds(combined.pad(0.06));
    else map.setView([31, -100], 6);
  </script>
</body>
</html>
"""


def save_deterministic_geomechanics_map_artifact(
    helper,
    step_index: int,
    fault_df: pd.DataFrame,
    well_df: Optional[pd.DataFrame],
    *,
    artifact_key: str,
    title: str,
    caption: str,
    display_order: int,
    fault_popup_fields: Iterable[str],
    fault_color: str,
    value_column: str,
    legend_title: str,
    well_id_column: str = "WellID",
    well_latitude_column: str = "Latitude(WGS84)",
    well_longitude_column: str = "Longitude(WGS84)",
    well_popup_fields: Optional[Iterable[str]] = None,
    field_labels: Optional[dict] = None,
) -> Optional[str]:
    prefix = f"{title} map was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)

        faults = fault_payload(fault_df, popup_fields=fault_popup_fields, value_column=value_column)
        if not faults:
            add_graph_warning(helper, step_index, f"{prefix}: fault results layer skipped; no valid fault geometry was found.")
            return None

        wells = well_payload(
            well_df,
            id_column=well_id_column,
            latitude_column=well_latitude_column,
            longitude_column=well_longitude_column,
            popup_fields=well_popup_fields or [well_id_column, well_latitude_column, well_longitude_column],
        )
        if well_df is not None and not well_df.empty and not wells:
            add_graph_warning(
                helper,
                step_index,
                f"{prefix}: injection wells layer skipped; missing columns: {well_latitude_column}, {well_longitude_column}",
            )

        fault_points = []
        for fault in faults:
            if "lat" in fault and "lon" in fault:
                fault_points.append([fault["lat"], fault["lon"]])
        well_points = [[well["lat"], well["lon"]] for well in wells]
        bounds = _merge_bounds([
            _bounds_from_points(fault_points),
            _bounds_from_points(well_points),
        ])

        value_min, value_max = _numeric_range(fault_df, value_column)
        html_text = _deterministic_geomechanics_leaflet_html(
            title=title,
            faults=faults,
            wells=wells,
            bounds=bounds,
            fault_color=fault_color,
            value_column=value_column if value_column in fault_df.columns else None,
            legend_title=legend_title,
            value_min=value_min,
            value_max=value_max,
            field_labels=field_labels or {},
        )

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
            preferredHeight=640,
        )
        return output_path
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def write_standalone_leaflet_artifact(
    helper,
    *,
    artifact_key: str,
    title: str,
    caption: str,
    display_order: int,
    preferred_height: int,
    fault_df: pd.DataFrame,
    fault_popup_fields: Iterable[str],
    fault_color: str,
    value_column: Optional[str] = None,
    legend_title: str = "Value",
    well_df: Optional[pd.DataFrame] = None,
    well_id_column: str = "WellID",
    well_latitude_column: str = "Latitude(WGS84)",
    well_longitude_column: str = "Longitude(WGS84)",
    well_popup_fields: Optional[Iterable[str]] = None,
    field_labels: Optional[dict] = None,
) -> Optional[str]:
    faults = fault_payload(fault_df, popup_fields=fault_popup_fields, value_column=value_column)
    wells = well_payload(
        well_df,
        id_column=well_id_column,
        latitude_column=well_latitude_column,
        longitude_column=well_longitude_column,
        popup_fields=well_popup_fields or [well_id_column, well_latitude_column, well_longitude_column],
    )

    if not faults and not wells:
        return None

    fault_points = []
    for fault in faults:
        if "lat" in fault and "lon" in fault:
            fault_points.append([fault["lat"], fault["lon"]])
    well_points = [[well["lat"], well["lon"]] for well in wells]
    bounds = _merge_bounds([
        _bounds_from_points(fault_points),
        _bounds_from_points(well_points),
    ])

    value_min, value_max = _numeric_range(fault_df, value_column)
    html_text = _standalone_leaflet_html(
        title=title,
        faults=faults,
        wells=wells,
        bounds=bounds,
        fault_color=fault_color,
        value_column=value_column if value_column in fault_df.columns else None,
        legend_title=legend_title,
        value_min=value_min,
        value_max=value_max,
        field_labels=field_labels or {},
    )

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
