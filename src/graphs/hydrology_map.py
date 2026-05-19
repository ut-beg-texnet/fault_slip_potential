"""Standalone Leaflet hydrology pressure map artifacts."""
import html
import json
import math
import os

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
    add_graph_warning,
    graph_artifacts_dir,
    has_columns,
    remove_graph_artifact,
    remove_step_messages,
)


PRESSURE_COLOR_SCALE = [
    [0.0, "#1f3bff"],
    [0.22, "#1db7ff"],
    [0.42, "#8ff0c0"],
    [0.62, "#ffff55"],
    [0.78, "#ffb32c"],
    [1.0, "#b51616"],
]


def _finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clean_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _with_fault_wkt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "wkt" in df.columns:
        return df
    required = ["Latitude(WGS84)", "Longitude(WGS84)", "Strike", "LengthKm"]
    if not has_columns(df, required):
        return df
    return latlon_to_wkt(df)


def _grid_payload(per_well_grid_df: pd.DataFrame):
    required = ["WellID", "Latitude", "Longitude", "Pressure_psi"]
    if not has_columns(per_well_grid_df, required):
        return None

    grid_df = per_well_grid_df[required].copy()
    for column in ["Latitude", "Longitude", "Pressure_psi"]:
        grid_df[column] = pd.to_numeric(grid_df[column], errors="coerce")
    grid_df = grid_df.dropna(subset=required)
    if grid_df.empty:
        return None

    latitudes = sorted(float(v) for v in grid_df["Latitude"].unique())
    longitudes = sorted(float(v) for v in grid_df["Longitude"].unique())
    if not latitudes or not longitudes:
        return None

    well_grids = {}
    for well_id, well_df in grid_df.groupby(grid_df["WellID"].astype(str), sort=True):
        pivot = (
            well_df.pivot_table(index="Latitude", columns="Longitude", values="Pressure_psi", aggfunc="mean")
            .reindex(index=latitudes, columns=longitudes)
            .fillna(0.0)
        )
        well_grids[str(well_id)] = np.maximum(pivot.values.astype(float), 0.0).tolist()

    total = None
    for values in well_grids.values():
        arr = np.asarray(values, dtype=float)
        total = arr if total is None else total + arr
    max_pressure = float(np.max(total)) if total is not None and total.size else 0.0

    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "wellGrids": well_grids,
        "bounds": [[latitudes[0], longitudes[0]], [latitudes[-1], longitudes[-1]]],
        "maxPressure": max_pressure,
    }


def _fault_payload(fault_df: pd.DataFrame):
    fault_df = _with_fault_wkt(fault_df)
    if fault_df is None or fault_df.empty or "wkt" not in fault_df.columns:
        return []

    fields = ["FaultID", "ID", "pressure_psi", "year", "Strike", "Dip", "LengthKm"]
    rows = []
    for _, row in fault_df.iterrows():
        wkt = str(row.get("wkt", "") or "")
        if not wkt.upper().startswith("LINESTRING"):
            continue
        props = {
            field: _clean_json_value(row.get(field))
            for field in fields
            if field in fault_df.columns
        }
        rows.append({"wkt": wkt, "properties": props})
    return rows


def _well_payload(well_df: pd.DataFrame):
    if well_df is None or well_df.empty or not has_columns(well_df, ["WellID", "Latitude", "Longitude"]):
        return []
    popup_fields = ["WellID", "Latitude", "Longitude", "StartDate", "EndDate", "MaxRate_bbl_day", "MeanRate_bbl_day"]
    rows = []
    for _, row in well_df.iterrows():
        lat = _finite_float(row.get("Latitude"), None)
        lon = _finite_float(row.get("Longitude"), None)
        if lat is None or lon is None:
            continue
        rows.append({
            "wellId": str(row.get("WellID", "")),
            "lat": lat,
            "lon": lon,
            "properties": {
                field: _clean_json_value(row.get(field))
                for field in popup_fields
                if field in well_df.columns
            },
        })
    return rows


def _hydrology_map_html(title: str, grid_payload: dict, faults: list, wells: list) -> str:
    title_json = json.dumps(title)
    payload_json = json.dumps(grid_payload)
    faults_json = json.dumps(faults)
    wells_json = json.dumps(wells)
    color_scale_json = json.dumps(PRESSURE_COLOR_SCALE)
    escaped_title = html.escape(title)
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
    .leaflet-container {{
      background: {MODERN_PAPER_BG};
    }}
    .leaflet-popup-content-wrapper {{
      border-radius: 8px;
      box-shadow: {MODERN_SHADOW};
    }}
    .leaflet-popup-content {{
      margin: 12px 14px;
      color: {MODERN_TEXT_COLOR};
      line-height: 1.45;
    }}
    .toolbar {{
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 1000;
      width: 286px;
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
      transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
    }}
    .toolbar button:hover {{
      background: #f8fafc;
      border-color: #94a3b8;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
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
    .well-option {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 4px 0;
      white-space: nowrap;
      color: {MODERN_MUTED_TEXT_COLOR};
    }}
    .well-option input {{
      accent-color: #2563eb;
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
      min-width: 210px;
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
      background: linear-gradient(to right, #1f3bff, #1db7ff, #8ff0c0, #ffff55, #ffb32c, #b51616);
    }}
    .legend-values {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }}
    .title-chip {{
      position: absolute;
      top: 12px;
      left: 54px;
      z-index: 1000;
      max-width: calc(100% - 380px);
      padding: 8px 12px;
      border: 1px solid {MODERN_BORDER_COLOR};
      border-radius: 8px;
      background: {MODERN_CONTROL_BG};
      font-weight: 700;
      font-size: 13px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
      backdrop-filter: blur(10px);
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
      .legend {{
        bottom: calc(48% + 24px);
      }}
      .title-chip {{
        max-width: calc(100% - 96px);
      }}
    }}
  </style>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
  <div id="map"></div>
  <div class="title-chip">{escaped_title}</div>
  <div class="toolbar">
    <div class="toolbar-title">Pressure Grid Wells</div>
    <input id="well-grid-search" type="search" placeholder="Search wells" oninput="populateWellControls()">
    <div class="toolbar-row">
      <button type="button" onclick="setAllWells(true)">All</button>
      <button type="button" onclick="setAllWells(false)">None</button>
    </div>
    <div id="well-controls"></div>
    <div class="toolbar-section">
      <div class="toolbar-title">Faults</div>
      <input id="fault-search" type="search" placeholder="Search faults" oninput="populateLayerControls('fault-controls', faultLayers, selectedFaults, updateFaultLayers, 'fault-search')">
      <div class="toolbar-row">
        <button type="button" onclick="setAllFaults(true)">All</button>
        <button type="button" onclick="setAllFaults(false)">None</button>
      </div>
      <div id="fault-controls"></div>
    </div>
    <div class="toolbar-section">
      <div class="toolbar-title">Injection Well Markers</div>
      <input id="well-marker-search" type="search" placeholder="Search wells" oninput="populateLayerControls('well-marker-controls', wellMarkerLayers, selectedWellMarkers, updateWellMarkerLayers, 'well-marker-search')">
      <div class="toolbar-row">
        <button type="button" onclick="setAllWellMarkers(true)">All</button>
        <button type="button" onclick="setAllWellMarkers(false)">None</button>
      </div>
      <div id="well-marker-controls"></div>
    </div>
    <div class="range-row">
      <label class="range-field">Min PSI
        <input id="pressure-min" type="number" step="any" oninput="updatePressureOverlay()">
      </label>
      <label class="range-field">Max PSI
        <input id="pressure-max" type="number" step="any" oninput="updatePressureOverlay()">
      </label>
    </div>
  </div>
  <div class="legend">
    <div>Pressure Front (PSI)</div>
    <div class="legend-ramp"></div>
    <div class="legend-values"><span id="legend-min">0</span><span id="legend-max">0</span></div>
  </div>
  <script>
    const title = {title_json};
    const gridPayload = {payload_json};
    const faults = {faults_json};
    const wells = {wells_json};
    const colorScale = {color_scale_json};
    let selectedWells = new Set(Object.keys(gridPayload.wellGrids || {{}}));
    let selectedFaults = new Set();
    let selectedWellMarkers = new Set();
    let pressureOverlay = null;
    const faultLayers = [];
    const wellMarkerLayers = [];

    const map = L.map('map', {{preferCanvas: true}});
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    function hexToRgb(hex) {{
      const cleaned = hex.replace('#', '');
      return [
        parseInt(cleaned.substring(0, 2), 16),
        parseInt(cleaned.substring(2, 4), 16),
        parseInt(cleaned.substring(4, 6), 16)
      ];
    }}

    function pressureColor(normalized) {{
      const n = Math.max(0, Math.min(1, normalized));
      for (let i = 1; i < colorScale.length; i++) {{
        const previous = colorScale[i - 1];
        const current = colorScale[i];
        if (n <= current[0]) {{
          const span = current[0] - previous[0] || 1;
          const ratio = (n - previous[0]) / span;
          const a = hexToRgb(previous[1]);
          const b = hexToRgb(current[1]);
          return [
            Math.round(a[0] + (b[0] - a[0]) * ratio),
            Math.round(a[1] + (b[1] - a[1]) * ratio),
            Math.round(a[2] + (b[2] - a[2]) * ratio)
          ];
        }}
      }}
      return hexToRgb(colorScale[colorScale.length - 1][1]);
    }}

    function sumSelectedGrid() {{
      const rows = gridPayload.latitudes.length;
      const cols = gridPayload.longitudes.length;
      const summed = Array.from({{length: rows}}, () => Array(cols).fill(0));
      selectedWells.forEach(function(wellId) {{
        const grid = gridPayload.wellGrids[wellId];
        if (!grid) return;
        for (let r = 0; r < rows; r++) {{
          for (let c = 0; c < cols; c++) {{
            summed[r][c] += Number(grid[r][c]) || 0;
          }}
        }}
      }});
      return summed;
    }}

    function defaultMaxPressure(autoMaxValue) {{
      return autoMaxValue > 1000 ? autoMaxValue : 1000;
    }}

    function selectedColorRange(autoMaxValue) {{
      const minInput = document.getElementById('pressure-min');
      const maxInput = document.getElementById('pressure-max');
      const minValue = Number(minInput.value);
      const maxValue = Number(maxInput.value);
      if (Number.isFinite(minValue) && Number.isFinite(maxValue) && maxValue > minValue) {{
        return {{minValue, maxValue, custom: true}};
      }}
      return {{minValue: 0, maxValue: defaultMaxPressure(autoMaxValue), custom: false}};
    }}

    function gridToDataUrl(grid) {{
      const rows = grid.length;
      const cols = rows ? grid[0].length : 0;
      const canvas = document.createElement('canvas');
      canvas.width = cols;
      canvas.height = rows;
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(cols, rows);
      let maxValue = 0;
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          maxValue = Math.max(maxValue, Number(grid[r][c]) || 0);
        }}
      }}
      const colorRange = selectedColorRange(maxValue);
      const colorSpan = colorRange.maxValue - colorRange.minValue || 1;
      const threshold = Math.max(maxValue * 0.01, 1e-9);
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const value = Number(grid[r][c]) || 0;
          const flippedRow = rows - 1 - r;
          const offset = (flippedRow * cols + c) * 4;
          if (value >= threshold && maxValue > 0) {{
            const normalized = Math.max(0, Math.min(1, (value - colorRange.minValue) / colorSpan));
            const rgb = pressureColor(normalized);
            const alpha = Math.round(45 + (205 - 45) * normalized);
            imageData.data[offset] = rgb[0];
            imageData.data[offset + 1] = rgb[1];
            imageData.data[offset + 2] = rgb[2];
            imageData.data[offset + 3] = alpha;
          }}
        }}
      }}
      ctx.putImageData(imageData, 0, 0);
      return {{url: canvas.toDataURL('image/png'), maxValue, colorRange}};
    }}

    function updatePressureOverlay() {{
      if (pressureOverlay) {{
        map.removeLayer(pressureOverlay);
        pressureOverlay = null;
      }}
      const grid = sumSelectedGrid();
      const rendered = gridToDataUrl(grid);
      const minInput = document.getElementById('pressure-min');
      const maxInput = document.getElementById('pressure-max');
      if (!rendered.colorRange.custom) {{
        minInput.value = '0';
        maxInput.value = rendered.colorRange.maxValue.toFixed(2);
      }}
      document.getElementById('legend-min').textContent = rendered.colorRange.minValue.toLocaleString(undefined, {{maximumFractionDigits: 2}});
      document.getElementById('legend-max').textContent = rendered.colorRange.maxValue.toLocaleString(undefined, {{maximumFractionDigits: 2}});
      if (rendered.maxValue <= 0 || selectedWells.size === 0) return;
      pressureOverlay = L.imageOverlay(rendered.url, gridPayload.bounds, {{opacity: 0.76}});
      pressureOverlay.addTo(map);
    }}

    function populateWellControls() {{
      const container = document.getElementById('well-controls');
      const query = String(document.getElementById('well-grid-search').value || '').toLowerCase();
      container.innerHTML = '';
      Object.keys(gridPayload.wellGrids || {{}}).forEach(function(wellId) {{
        if (query && !String(wellId).toLowerCase().includes(query)) return;
        const label = document.createElement('label');
        label.className = 'well-option';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = selectedWells.has(wellId);
        input.onchange = function() {{
          if (input.checked) selectedWells.add(wellId);
          else selectedWells.delete(wellId);
          updatePressureOverlay();
        }};
        label.appendChild(input);
        label.appendChild(document.createTextNode(wellId));
        container.appendChild(label);
      }});
    }}

    function setAllWells(enabled) {{
      selectedWells = new Set(enabled ? Object.keys(gridPayload.wellGrids || {{}}) : []);
      populateWellControls();
      updatePressureOverlay();
    }}

    function itemLabel(properties, candidates, fallback) {{
      for (let i = 0; i < candidates.length; i++) {{
        const value = properties && properties[candidates[i]];
        if (value !== undefined && value !== null && String(value) !== '') return String(value);
      }}
      return fallback;
    }}

    function populateLayerControls(containerId, items, selectedSet, onChange, searchInputId) {{
      const container = document.getElementById(containerId);
      const query = searchInputId ? String(document.getElementById(searchInputId).value || '').toLowerCase() : '';
      container.innerHTML = '';
      items.forEach(function(item) {{
        if (query && !String(item.label).toLowerCase().includes(query)) return;
        const label = document.createElement('label');
        label.className = 'well-option';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = selectedSet.has(item.key);
        input.onchange = function() {{
          if (input.checked) selectedSet.add(item.key);
          else selectedSet.delete(item.key);
          onChange();
        }};
        label.appendChild(input);
        label.appendChild(document.createTextNode(item.label));
        container.appendChild(label);
      }});
    }}

    function updateFaultLayers() {{
      faultLayers.forEach(function(item) {{
        const enabled = selectedFaults.has(item.key);
        if (enabled && !map.hasLayer(item.layer)) item.layer.addTo(map);
        if (!enabled && map.hasLayer(item.layer)) map.removeLayer(item.layer);
      }});
    }}

    function updateWellMarkerLayers() {{
      wellMarkerLayers.forEach(function(item) {{
        const enabled = selectedWellMarkers.has(item.key);
        if (enabled && !map.hasLayer(item.layer)) item.layer.addTo(map);
        if (!enabled && map.hasLayer(item.layer)) map.removeLayer(item.layer);
      }});
    }}

    function setAllFaults(enabled) {{
      selectedFaults = new Set(enabled ? faultLayers.map(function(item) {{ return item.key; }}) : []);
      populateLayerControls('fault-controls', faultLayers, selectedFaults, updateFaultLayers, 'fault-search');
      updateFaultLayers();
    }}

    function setAllWellMarkers(enabled) {{
      selectedWellMarkers = new Set(enabled ? wellMarkerLayers.map(function(item) {{ return item.key; }}) : []);
      populateLayerControls('well-marker-controls', wellMarkerLayers, selectedWellMarkers, updateWellMarkerLayers, 'well-marker-search');
      updateWellMarkerLayers();
    }}

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

    function popupHtml(properties) {{
      return Object.keys(properties || {{}}).map(function(key) {{
        const value = properties[key];
        return '<strong>' + key + '</strong>: ' + String(value == null ? '' : value);
      }}).join('<br>');
    }}

    const layerBounds = [];
    faults.forEach(function(fault, index) {{
      const line = parseLineString(fault.wkt);
      if (line.length < 2) return;
      const layer = L.polyline(line, {{color: '#0f172a', weight: 3.5, opacity: 0.95}});
      layer.bindPopup(popupHtml(fault.properties));
      layerBounds.push(layer.getBounds());
      const label = itemLabel(fault.properties, ['FaultID', 'ID', 'id'], 'Fault ' + (index + 1));
      faultLayers.push({{key: String(index), label: label, layer: layer}});
    }});

    wells.forEach(function(well, index) {{
      const marker = L.circleMarker([well.lat, well.lon], {{
        radius: 6,
        color: '#1d4ed8',
        fillColor: '#3b82f6',
        fillOpacity: 0.9,
        weight: 1
      }});
      marker.bindPopup(popupHtml(well.properties));
      layerBounds.push(marker.getLatLng().toBounds(1000));
      const key = String(well.wellId || index);
      const label = well.wellId || itemLabel(well.properties, ['WellID', 'ID'], 'Well ' + (index + 1));
      wellMarkerLayers.push({{key: key, label: String(label), layer: marker}});
    }});

    populateWellControls();
    setAllFaults(true);
    setAllWellMarkers(true);
    updatePressureOverlay();
    const gridBounds = L.latLngBounds(gridPayload.bounds);
    layerBounds.push(gridBounds);
    const combined = layerBounds.reduce(function(bounds, next) {{
      return bounds ? bounds.extend(next) : L.latLngBounds(next);
    }}, null);
    if (combined && combined.isValid()) map.fitBounds(combined.pad(0.05));
    else map.setView([31, -100], 6);
  </script>
</body>
</html>
"""


def save_direct_hydrology_pressure_map_artifact(
    helper,
    step_index: int,
    per_well_grid_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    well_df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    caption: str,
    display_order: int,
):
    prefix = f"{title} map was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)
        grid = _grid_payload(per_well_grid_df)
        if grid is None:
            add_graph_warning(helper, step_index, f"{prefix} because required per-well pressure grid columns are missing.")
            return None

        html_text = _hydrology_map_html(title, grid, _fault_payload(fault_df), _well_payload(well_df))
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
