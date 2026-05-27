"""Leaflet map artifact helpers for FSP portal output."""
import json
import os
import struct
import zlib
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from fsp.io.coords import latlon_to_wkt
from graphs.artifacts import (
    SLIP_PRESSURE_COLOR_SCALE,
    add_graph_warning,
    graph_artifacts_dir,
    has_columns,
    remove_graph_artifact,
    remove_step_messages,
)
from graphs.html_map import write_standalone_leaflet_artifact


DEFAULT_BASEMAP = {
    "urlTemplate": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "attribution": "&copy; OpenStreetMap contributors",
    "maxZoom": 19,
}

DEFAULT_LEAFLET_HEAT_ASSETS = {
    "scriptUrl": "https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js",
}

PRESSURE_COLOR_SCALE = [
    [0.0, "#1f3bff"],
    [0.22, "#1db7ff"],
    [0.42, "#8ff0c0"],
    [0.62, "#ffff55"],
    [0.78, "#ffb32c"],
    [1.0, "#b51616"],
]

DETERMINISTIC_GEOMECHANICS_FIELD_LABELS = {
    "slip_pressure": "Deterministic Pore Pressure to Slip",
    "coulomb_failure_function": "Coulomb Failure Function",
    "shear_capacity_utilization": "Shear Capacity Utilization",
}

INITIAL_FAULT_POPUP_FIELDS = [
    "FaultID",
    "ID",
    "Strike",
    "Dip",
    "Rake",
    "LengthKm",
    "Latitude(WGS84)",
    "Longitude(WGS84)",
]


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("._") or "layer"


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]):
    if df is None or df.empty:
        return None
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _filter_group(group_key: str, title: str, all_layer_key: str, item_layer_keys: list) -> Optional[dict]:
    item_layer_keys = [key for key in item_layer_keys if key]
    if not item_layer_keys:
        return None
    return {
        "key": group_key,
        "title": title,
        "mode": "all-or-selected",
        "allLayerKey": all_layer_key,
        "itemLayerKeys": item_layer_keys,
    }


def _add_filter_metadata(layer: Optional[dict], group_key: str, item_value=None, is_all: bool = False):
    if layer is None:
        return layer
    layer["filter"] = {
        "groupKey": group_key,
        "isAll": bool(is_all),
    }
    if item_value is not None:
        layer["filter"]["itemValue"] = str(item_value)
    return layer


def _suppress_layer_legend(layer: Optional[dict]):
    if layer is None:
        return layer
    style = dict(layer.get("style") or {})
    for key in ("legendTitle", "allowUserRange"):
        style.pop(key, None)
    layer["style"] = style
    return layer


def _fault_midpoint_style(style: dict) -> dict:
    midpoint_style = dict(style)
    midpoint_style.update({
        "radius": 5,
        "weight": 1,
        "opacity": 0.95,
        "fillOpacity": 0.82,
    })
    return midpoint_style


def _relative_to_scratch(helper, path: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(helper.scratchPath))


def _map_data_dir(helper) -> str:
    path = os.path.join(graph_artifacts_dir(helper), "map_data")
    os.makedirs(path, exist_ok=True)
    return path


def _numeric_bounds(df: pd.DataFrame, lat_col: str, lon_col: str):
    if not has_columns(df, [lat_col, lon_col]):
        return None
    lats = pd.to_numeric(df[lat_col], errors="coerce")
    lons = pd.to_numeric(df[lon_col], errors="coerce")
    valid = pd.DataFrame({"lat": lats, "lon": lons}).dropna()
    if valid.empty:
        return None
    return [
        [float(valid["lat"].min()), float(valid["lon"].min())],
        [float(valid["lat"].max()), float(valid["lon"].max())],
    ]


def _merge_bounds(bounds_list):
    valid = [bounds for bounds in bounds_list if bounds]
    if not valid:
        return None
    return [
        [min(bounds[0][0] for bounds in valid), min(bounds[0][1] for bounds in valid)],
        [max(bounds[1][0] for bounds in valid), max(bounds[1][1] for bounds in valid)],
    ]


def _write_layer_csv(helper, layer_key: str, df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> str:
    output_path = os.path.join(_map_data_dir(helper), f"{_safe_filename(layer_key)}.csv")
    if columns:
        keep = [column for column in columns if column in df.columns]
        layer_df = df[keep].copy()
    else:
        layer_df = df.copy()
    layer_df.to_csv(output_path, index=False)
    return _relative_to_scratch(helper, output_path)


def _write_layer_png(helper, layer_key: str, rgba: np.ndarray) -> str:
    output_path = os.path.join(_map_data_dir(helper), f"{_safe_filename(layer_key)}.png")
    _write_png_rgba(output_path, rgba)
    return _relative_to_scratch(helper, output_path)


def _missing_columns(df: Optional[pd.DataFrame], columns: Iterable[str]) -> list:
    if df is None or df.empty:
        return list(columns)
    return [column for column in columns if column not in df.columns]


def _numeric_range(df: pd.DataFrame, column: str):
    if not has_columns(df, [column]):
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


def _with_fault_wkt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "wkt" in df.columns:
        return df
    required = ["Latitude(WGS84)", "Longitude(WGS84)", "Strike", "LengthKm"]
    if not has_columns(df, required):
        return df
    return latlon_to_wkt(df)


def _style_with_value_scale(
    style: dict,
    df: pd.DataFrame,
    *,
    value_column: Optional[str],
    legend_title: str,
    value_min_default: Optional[float] = None,
):
    if not value_column or value_column not in df.columns:
        return style
    min_value, max_value = _numeric_range(df, value_column)
    if min_value is None or max_value is None:
        return style
    if value_min_default is not None:
        min_value = float(value_min_default)
        if max_value <= min_value:
            max_value = min_value + 1.0
    scaled_style = dict(style)
    scaled_style.update({
        "valueColumn": value_column,
        "colorScale": SLIP_PRESSURE_COLOR_SCALE,
        "legendTitle": legend_title,
        "minValue": min_value,
        "maxValue": max_value,
        "allowUserRange": True,
    })
    return scaled_style


def _hex_to_rgb(hex_color: str):
    color = str(hex_color).lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _pressure_color(normalized: float):
    stops = sorted((float(stop), _hex_to_rgb(color)) for stop, color in PRESSURE_COLOR_SCALE)
    if normalized <= stops[0][0]:
        return stops[0][1]
    for index in range(1, len(stops)):
        stop, color = stops[index]
        previous_stop, previous_color = stops[index - 1]
        if normalized <= stop:
            span = stop - previous_stop or 1.0
            ratio = (normalized - previous_stop) / span
            return tuple(int(round(previous_color[channel] + (color[channel] - previous_color[channel]) * ratio)) for channel in range(3))
    return stops[-1][1]


def _write_png_rgba(path: str, rgba: np.ndarray):
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("RGBA PNG data must have four channels.")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) +
            kind +
            data +
            struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        )

    scanlines = b"".join(b"\x00" + rgba[row].astype(np.uint8).tobytes() for row in range(height))
    contents = (
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(scanlines, 6)) +
        chunk(b"IEND", b"")
    )
    with open(path, "wb") as fh:
        fh.write(contents)


def _pressure_grid_to_rgba(pressure_grid, *, transparent_fraction=0.01, min_alpha=45, max_alpha=205):
    values = np.asarray(pressure_grid, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, None, None

    valid_values = values[finite]
    max_value = float(np.max(valid_values))
    min_value = float(np.min(valid_values))
    if max_value <= 0:
        return None, min_value, max_value

    threshold = max(max_value * transparent_fraction, 1e-9)
    normalized = np.clip(values / max_value, 0.0, 1.0)
    rgba = np.zeros((values.shape[0], values.shape[1], 4), dtype=np.uint8)
    visible = finite & (values >= threshold)

    for row, col in zip(*np.where(visible)):
        red, green, blue = _pressure_color(float(normalized[row, col]))
        alpha = int(round(min_alpha + (max_alpha - min_alpha) * float(normalized[row, col])))
        rgba[row, col] = [red, green, blue, alpha]

    return np.flipud(rgba), min_value, max_value


def _grid_from_heatmap_df(heatmap_df: pd.DataFrame):
    required = ["Latitude", "Longitude", "Pressure_psi"]
    if not has_columns(heatmap_df, required):
        return None, None, None

    df = heatmap_df[required].copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None, None, None

    grid_df = df.pivot_table(index="Latitude", columns="Longitude", values="Pressure_psi", aggfunc="mean").sort_index().sort_index(axis=1)
    if grid_df.empty:
        return None, None, None

    bounds = [
        [float(grid_df.index.min()), float(grid_df.columns.min())],
        [float(grid_df.index.max()), float(grid_df.columns.max())],
    ]
    return grid_df.values, bounds, df


def _image_overlay_layer(
    helper,
    *,
    key: str,
    title: str,
    pressure_grid,
    bounds,
    legend_title: str,
    visible: bool = True,
):
    rgba, min_value, max_value = _pressure_grid_to_rgba(pressure_grid)
    if rgba is None:
        return None, None
    path = _write_layer_png(helper, key, rgba)
    return {
        "key": key,
        "title": title,
        "type": "imageOverlay",
        "visible": visible,
        "style": {
            "legendTitle": legend_title,
            "colorScale": PRESSURE_COLOR_SCALE,
            "minValue": min_value if min_value is not None else 0.0,
            "maxValue": max_value if max_value is not None else 1.0,
            "opacity": 0.72,
        },
        "bounds": bounds,
        "source": {
            "type": "image",
            "path": path,
            "contentType": "image/png",
        },
    }, bounds


def _point_layer(
    helper,
    *,
    key: str,
    title: str,
    df: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    popup_fields: Iterable[str],
    style: dict,
    visible: bool = True,
    max_features: int = 5000,
    value_column: Optional[str] = None,
    legend_title: str = "Value",
    value_min_default: Optional[float] = None,
    field_labels: Optional[dict] = None,
):
    required = [latitude_column, longitude_column]
    if not has_columns(df, required):
        return None, None
    popup_fields = [field for field in popup_fields if field in df.columns]
    property_fields = list(dict.fromkeys(popup_fields + ([value_column] if value_column in df.columns else [])))
    columns = list(dict.fromkeys(required + property_fields))
    path = _write_layer_csv(helper, key, df, columns)
    style = _style_with_value_scale(
        style,
        df,
        value_column=value_column,
        legend_title=legend_title,
        value_min_default=value_min_default,
    )
    return {
        "key": key,
        "title": title,
        "type": "point",
        "visible": visible,
        "maxFeatures": max_features,
        "popupFields": popup_fields,
        "fieldLabels": field_labels or {},
        "propertyFields": property_fields,
        "style": style,
        "source": {
            "type": "csv",
            "path": path,
        },
        "geometry": {
            "type": "point",
            "latitudeColumn": latitude_column,
            "longitudeColumn": longitude_column,
        },
    }, _numeric_bounds(df, latitude_column, longitude_column)


def _wkt_layer(
    helper,
    *,
    key: str,
    title: str,
    df: pd.DataFrame,
    wkt_column: str,
    popup_fields: Iterable[str],
    style: dict,
    visible: bool = True,
    max_features: int = 5000,
    value_column: Optional[str] = None,
    legend_title: str = "Value",
    value_min_default: Optional[float] = None,
    field_labels: Optional[dict] = None,
):
    if not has_columns(df, [wkt_column]):
        return None
    popup_fields = [field for field in popup_fields if field in df.columns]
    property_fields = list(dict.fromkeys(popup_fields + ([value_column] if value_column in df.columns else [])))
    columns = list(dict.fromkeys([wkt_column] + property_fields))
    path = _write_layer_csv(helper, key, df, columns)
    style = _style_with_value_scale(
        style,
        df,
        value_column=value_column,
        legend_title=legend_title,
        value_min_default=value_min_default,
    )
    return {
        "key": key,
        "title": title,
        "type": "line",
        "visible": visible,
        "maxFeatures": max_features,
        "popupFields": popup_fields,
        "fieldLabels": field_labels or {},
        "propertyFields": property_fields,
        "style": style,
        "source": {
            "type": "csv",
            "path": path,
        },
        "geometry": {
            "type": "line",
            "wktColumn": wkt_column,
        },
    }


def _heatmap_layer(
    helper,
    *,
    key: str,
    title: str,
    df: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    value_column: str,
    legend_title: str,
    visible: bool = True,
    max_features: int = 5000,
):
    required = [latitude_column, longitude_column, value_column]
    if not has_columns(df, required):
        return None, None
    layer_df = df[required].copy()
    path = _write_layer_csv(helper, key, layer_df, required)
    min_value, max_value = _numeric_range(layer_df, value_column)
    style = {
        "valueColumn": value_column,
        "colorScale": SLIP_PRESSURE_COLOR_SCALE,
        "legendTitle": legend_title,
        "minValue": min_value if min_value is not None else 0.0,
        "maxValue": max_value if max_value is not None else 1.0,
        "allowUserRange": True,
        "radius": 25,
        "blur": 15,
        "minOpacity": 0.25,
    }
    return {
        "key": key,
        "title": title,
        "type": "heatmap",
        "visible": visible,
        "maxFeatures": max_features,
        "popupFields": [value_column],
        "fieldLabels": {value_column: legend_title},
        "propertyFields": [value_column],
        "style": style,
        "source": {
            "type": "csv",
            "path": path,
        },
        "geometry": {
            "type": "point",
            "latitudeColumn": latitude_column,
            "longitudeColumn": longitude_column,
        },
        "plugins": ["leaflet-heat"],
    }, _numeric_bounds(layer_df, latitude_column, longitude_column)


def write_leaflet_map_artifact(
    helper,
    *,
    artifact_key: str,
    title: str,
    caption: str,
    display_order: int,
    preferred_height: int,
    layers: list,
    bounds_list: Optional[list] = None,
    filter_groups: Optional[list] = None,
) -> Optional[str]:
    layers = [layer for layer in layers if layer]
    if not layers:
        return None

    manifest = {
        "version": 1,
        "basemap": DEFAULT_BASEMAP,
        "plugins": {
            "leafletHeat": DEFAULT_LEAFLET_HEAT_ASSETS,
        },
        "initialView": {
            "center": [31.0, -100.0],
            "zoom": 6,
        },
        "layers": layers,
    }

    filter_groups = [group for group in (filter_groups or []) if group]
    if filter_groups:
        manifest["filterControls"] = {
            "version": 1,
            "groups": filter_groups,
        }

    bounds = _merge_bounds(bounds_list or [])
    if bounds:
        manifest["initialView"]["bounds"] = bounds

    output_path = os.path.join(graph_artifacts_dir(helper), f"{artifact_key}.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    remove_graph_artifact(helper, artifact_key)
    helper.saveGraphArtifact(
        key=artifact_key,
        title=title,
        renderer="leaflet-map",
        path=output_path,
        contentType="application/json",
        caption=caption,
        displayOrder=display_order,
        preferredHeight=preferred_height,
    )
    return output_path


def save_model_inputs_map_artifact(helper, step_index: int, faults_df: pd.DataFrame, injection_df: pd.DataFrame):
    prefix = "Model inputs map was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)
        well_id_col = "WellID" if "WellID" in injection_df.columns else "UWI" if "UWI" in injection_df.columns else "API Number"
        well_lat_col = "Latitude(WGS84)" if "Latitude(WGS84)" in injection_df.columns else "Surface Latitude"
        well_lon_col = "Longitude(WGS84)" if "Longitude(WGS84)" in injection_df.columns else "Surface Longitude"
        well_df = injection_df.drop_duplicates(subset=[well_id_col]) if well_id_col in injection_df.columns else injection_df

        if not has_columns(faults_df, ["Latitude(WGS84)", "Longitude(WGS84)"]) and "wkt" not in faults_df.columns:
            missing = ", ".join(_missing_columns(faults_df, ["Latitude(WGS84)", "Longitude(WGS84)"]))
            add_graph_warning(helper, step_index, f"{prefix}: faults layer skipped; missing columns: {missing}")

        if not has_columns(well_df, [well_lat_col, well_lon_col]):
            missing = ", ".join(_missing_columns(injection_df, [well_lat_col, well_lon_col]))
            add_graph_warning(helper, step_index, f"{prefix}: injection wells layer skipped; missing columns: {missing}")

        return write_standalone_leaflet_artifact(
            helper,
            artifact_key="fsp-model-inputs-map",
            title="Model Inputs Map",
            caption="Leaflet map of FSP faults and injection wells.",
            display_order=11,
            preferred_height=560,
            fault_df=faults_df,
            fault_popup_fields=["FaultID", "Strike", "Dip", "LengthKm"],
            fault_color="#dc2626",
            well_df=well_df,
            well_id_column=well_id_col,
            well_latitude_column=well_lat_col,
            well_longitude_column=well_lon_col,
            well_popup_fields=[well_id_col, well_lat_col, well_lon_col],
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_fault_results_map_artifact(
    helper,
    step_index: int,
    df: pd.DataFrame,
    *,
    artifact_key: str,
    title: str,
    caption: str,
    display_order: int,
    result_fields: Iterable[str],
    color: str,
    value_column: Optional[str] = None,
    legend_title: str = "Value",
    value_min_default: Optional[float] = None,
    well_df: Optional[pd.DataFrame] = None,
    extra_layers: Optional[list] = None,
    field_labels: Optional[dict] = None,
):
    prefix = f"{title} map was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)
        df = _with_fault_wkt(df)
        popup_fields = list(dict.fromkeys(
            [field for field in INITIAL_FAULT_POPUP_FIELDS if field in df.columns] +
            [field for field in result_fields if field in df.columns]
        ))
        if not has_columns(df, ["Latitude(WGS84)", "Longitude(WGS84)"]) and "wkt" not in df.columns:
            missing = ", ".join(_missing_columns(df, ["Latitude(WGS84)", "Longitude(WGS84)"]))
            add_graph_warning(helper, step_index, f"{prefix}: fault results layer skipped; missing columns: {missing}")

        style = {
            "color": color,
            "fillColor": color,
            "weight": 3,
            "opacity": 0.92,
            "fillOpacity": 0.24,
            "radius": 6,
        }
        layer = None
        bounds = None
        fault_id_col = _first_existing_column(df, ["FaultID", "ID", "id"])
        if "wkt" in df.columns:
            layer = _wkt_layer(
                helper,
                key="fault-results",
                title=title,
                df=df,
                wkt_column="wkt",
                popup_fields=popup_fields,
                style=style,
                value_column=value_column,
                legend_title=legend_title,
                value_min_default=value_min_default,
                field_labels=field_labels,
            )
            bounds = _numeric_bounds(df, "Latitude(WGS84)", "Longitude(WGS84)")
        else:
            layer, bounds = _point_layer(
                helper,
                key="fault-results",
                title=title,
                df=df,
                latitude_column="Latitude(WGS84)",
                longitude_column="Longitude(WGS84)",
                popup_fields=popup_fields,
                style=style,
                value_column=value_column,
                legend_title=legend_title,
                value_min_default=value_min_default,
                field_labels=field_labels,
            )

        if layer is None:
            return None

        layer = _add_filter_metadata(layer, "fault-results", is_all=True)
        fault_layers = [layer]
        bounds_list = [bounds]
        fault_item_keys = []
        midpoint_layer = None
        if "wkt" in df.columns and has_columns(df, ["Latitude(WGS84)", "Longitude(WGS84)"]):
            midpoint_layer, midpoint_bounds = _point_layer(
                helper,
                key="fault-midpoints",
                title=f"{title} Midpoints",
                df=df,
                latitude_column="Latitude(WGS84)",
                longitude_column="Longitude(WGS84)",
                popup_fields=popup_fields,
                style=_fault_midpoint_style(style),
                value_column=value_column,
                legend_title=legend_title,
                value_min_default=value_min_default,
                field_labels=field_labels,
            )
            midpoint_layer = _suppress_layer_legend(midpoint_layer)
            midpoint_layer = _add_filter_metadata(midpoint_layer, "fault-results", is_all=True)
            if midpoint_layer is not None:
                fault_layers.append(midpoint_layer)
            if midpoint_bounds is not None:
                bounds_list.append(midpoint_bounds)
        if fault_id_col:
            for item_index, (_, fault_row) in enumerate(df.iterrows(), start=1):
                fault_id = str(fault_row.get(fault_id_col, f"Fault {item_index}"))
                single_fault_df = pd.DataFrame([fault_row])
                if "wkt" in df.columns:
                    single_layer = _wkt_layer(
                        helper,
                        key=f"fault-result-{item_index}",
                        title=fault_id,
                        df=single_fault_df,
                        wkt_column="wkt",
                        popup_fields=popup_fields,
                        style=style,
                        visible=False,
                        value_column=value_column,
                        legend_title=legend_title,
                        value_min_default=value_min_default,
                        field_labels=field_labels,
                    )
                else:
                    single_layer, _ = _point_layer(
                        helper,
                        key=f"fault-result-{item_index}",
                        title=fault_id,
                        df=single_fault_df,
                        latitude_column="Latitude(WGS84)",
                        longitude_column="Longitude(WGS84)",
                        popup_fields=popup_fields,
                        style=style,
                        visible=False,
                        value_column=value_column,
                        legend_title=legend_title,
                        value_min_default=value_min_default,
                        field_labels=field_labels,
                    )
                single_layer = _suppress_layer_legend(single_layer)
                single_layer = _add_filter_metadata(single_layer, "fault-results", item_index)
                fault_layers.append(single_layer)
                if single_layer:
                    fault_item_keys.append(single_layer["key"])
                if "wkt" in df.columns and has_columns(single_fault_df, ["Latitude(WGS84)", "Longitude(WGS84)"]):
                    single_midpoint_layer, _ = _point_layer(
                        helper,
                        key=f"fault-midpoint-{item_index}",
                        title=f"{fault_id} Midpoint",
                        df=single_fault_df,
                        latitude_column="Latitude(WGS84)",
                        longitude_column="Longitude(WGS84)",
                        popup_fields=popup_fields,
                        style=_fault_midpoint_style(style),
                        visible=False,
                        value_column=value_column,
                        legend_title=legend_title,
                        value_min_default=value_min_default,
                        field_labels=field_labels,
                    )
                    single_midpoint_layer = _suppress_layer_legend(single_midpoint_layer)
                    single_midpoint_layer = _add_filter_metadata(single_midpoint_layer, "fault-results", item_index)
                    if single_midpoint_layer is not None:
                        fault_layers.append(single_midpoint_layer)

        layers = fault_layers
        filter_groups = [_filter_group("fault-results", "Faults", "fault-results", fault_item_keys)]

        if well_df is not None and not well_df.empty:
            well_popup_fields = [field for field in ["WellID", "Latitude(WGS84)", "Longitude(WGS84)"] if field in well_df.columns]
            well_layer, well_bounds = _point_layer(
                helper,
                key="injection-wells",
                title="Injection Wells",
                df=well_df,
                latitude_column="Latitude(WGS84)",
                longitude_column="Longitude(WGS84)",
                popup_fields=well_popup_fields,
                style={"color": "#2563eb", "fillColor": "#3b82f6", "radius": 6, "weight": 1, "fillOpacity": 0.9},
            )
            if well_layer is None:
                missing = ", ".join(_missing_columns(well_df, ["Latitude(WGS84)", "Longitude(WGS84)"]))
                add_graph_warning(helper, step_index, f"{prefix}: injection wells layer skipped; missing columns: {missing}")
            else:
                well_layer = _add_filter_metadata(well_layer, "injection-wells", is_all=True)
                well_layers = [well_layer]
                well_item_keys = []
                for item_index, (_, well_row) in enumerate(well_df.iterrows(), start=1):
                    well_id = str(well_row.get("WellID", f"Well {item_index}"))
                    single_well_df = pd.DataFrame([well_row])
                    single_layer, single_bounds = _point_layer(
                        helper,
                        key=f"injection-well-{item_index}",
                        title=well_id,
                        df=single_well_df,
                        latitude_column="Latitude(WGS84)",
                        longitude_column="Longitude(WGS84)",
                        popup_fields=well_popup_fields,
                        style={"color": "#1d4ed8", "fillColor": "#3b82f6", "radius": 6, "weight": 1, "fillOpacity": 0.9},
                        visible=False,
                    )
                    single_layer = _add_filter_metadata(single_layer, "injection-wells", item_index)
                    well_layers.append(single_layer)
                    if single_bounds is not None:
                        bounds_list.append(single_bounds)
                    if single_layer:
                        well_item_keys.append(single_layer["key"])
                layers.extend(well_layers)
                if well_bounds is not None:
                    bounds_list.append(well_bounds)
                filter_groups.append(_filter_group("injection-wells", "Injection Wells", "injection-wells", well_item_keys))

        for extra_layer, extra_bounds in extra_layers or []:
            if extra_layer is not None:
                layers.append(extra_layer)
            if extra_bounds is not None:
                bounds_list.append(extra_bounds)

        return write_leaflet_map_artifact(
            helper,
            artifact_key=artifact_key,
            title=title,
            caption=caption,
            display_order=display_order,
            preferred_height=560,
            layers=layers,
            bounds_list=bounds_list,
            filter_groups=filter_groups,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None


def save_hydrology_heatmap_fault_map_artifact(
    helper,
    step_index: int,
    heatmap_df: pd.DataFrame,
    fault_df: pd.DataFrame,
    well_df: Optional[pd.DataFrame] = None,
    *,
    artifact_key: str,
    title: str,
    caption: str,
    display_order: int,
):
    prefix = f"{title} map was not generated"
    try:
        remove_step_messages(helper, step_index, prefix)
        fault_df = _with_fault_wkt(fault_df)
        pressure_grid, raster_bounds, _ = _grid_from_heatmap_df(heatmap_df)
        heat_layer, heat_bounds = (None, None)
        if pressure_grid is not None:
            heat_layer, heat_bounds = _image_overlay_layer(
                helper,
                key="pressure-grid",
                title="Pressure Grid",
                pressure_grid=pressure_grid,
                bounds=raster_bounds,
                legend_title="Pressure (PSI)",
            )
        if heat_layer is None:
            heat_layer, heat_bounds = _heatmap_layer(
                helper,
                key="pressure-grid",
                title="Pressure Grid",
                df=heatmap_df,
                latitude_column="Latitude",
                longitude_column="Longitude",
                value_column="Pressure_psi",
                legend_title="Pressure (PSI)",
                max_features=5000,
            )
        if heat_layer is None:
            missing = ", ".join(_missing_columns(heatmap_df, ["Latitude", "Longitude", "Pressure_psi"]))
            add_graph_warning(helper, step_index, f"{prefix}: pressure grid layer skipped; missing columns: {missing}")

        fault_popup_fields = ["pressure_psi", "year"]
        fault_layer = None
        fault_bounds = None
        filter_groups = []
        fault_id_col = _first_existing_column(fault_df, ["FaultID", "ID", "id"])
        fault_item_keys = []
        if "wkt" in fault_df.columns:
            fault_layer = _wkt_layer(
                helper,
                key="fault-results",
                title="Fault Pressures",
                df=fault_df,
                wkt_column="wkt",
                popup_fields=["FaultID", "ID"] + fault_popup_fields,
                style={"color": "#111827", "weight": 3, "opacity": 0.95},
            )
            fault_layer = _add_filter_metadata(fault_layer, "fault-results", is_all=True)
            fault_bounds = _numeric_bounds(fault_df, "Latitude(WGS84)", "Longitude(WGS84)")
        else:
            fault_layer, fault_bounds = _point_layer(
                helper,
                key="fault-results",
                title="Fault Pressures",
                df=fault_df,
                latitude_column="Latitude(WGS84)",
                longitude_column="Longitude(WGS84)",
                popup_fields=["FaultID", "ID"] + fault_popup_fields,
                style={"color": "#111827", "fillColor": "#111827", "radius": 5},
            )
            fault_layer = _add_filter_metadata(fault_layer, "fault-results", is_all=True)
            if fault_layer is None:
                missing = ", ".join(_missing_columns(fault_df, ["Latitude(WGS84)", "Longitude(WGS84)"]))
                add_graph_warning(helper, step_index, f"{prefix}: fault results layer skipped; missing columns: {missing}")

        fault_layers = [fault_layer]
        if fault_id_col:
            for item_index, (_, fault_row) in enumerate(fault_df.iterrows(), start=1):
                fault_id = str(fault_row.get(fault_id_col, "fault"))
                if "wkt" in fault_df.columns:
                    single_layer = _wkt_layer(
                        helper,
                        key=f"fault-result-{item_index}",
                        title=f"Fault {item_index}",
                        df=pd.DataFrame([fault_row]),
                        wkt_column="wkt",
                        popup_fields=["FaultID", "ID"] + fault_popup_fields,
                        style={"color": "#111827", "weight": 4, "opacity": 0.95},
                        visible=False,
                    )
                else:
                    single_layer, _ = _point_layer(
                        helper,
                        key=f"fault-result-{item_index}",
                        title=f"Fault {item_index}",
                        df=pd.DataFrame([fault_row]),
                        latitude_column="Latitude(WGS84)",
                        longitude_column="Longitude(WGS84)",
                        popup_fields=["FaultID", "ID"] + fault_popup_fields,
                        style={"color": "#111827", "fillColor": "#374151", "radius": 6},
                        visible=False,
                    )
                fault_layers.append(_add_filter_metadata(single_layer, "fault-results", item_index))
                if single_layer:
                    fault_item_keys.append(single_layer["key"])
        filter_groups.append(_filter_group("fault-results", "Faults", "fault-results", fault_item_keys))

        well_layers = []
        well_bounds = []
        if well_df is not None and not well_df.empty:
            well_layer, grouped_well_bounds = _point_layer(
                helper,
                key="injection-wells",
                title="Injection Wells",
                df=well_df,
                latitude_column="Latitude",
                longitude_column="Longitude",
                popup_fields=["WellID", "Latitude", "Longitude", "StartDate", "EndDate", "MaxRate_bbl_day", "MeanRate_bbl_day"],
                style={"color": "#2563eb", "fillColor": "#2563eb", "radius": 6, "weight": 1},
            )
            well_layers.append(_add_filter_metadata(well_layer, "injection-wells", is_all=True))
            well_bounds.append(grouped_well_bounds)

            if well_layer is None:
                missing = ", ".join(_missing_columns(well_df, ["Latitude", "Longitude"]))
                add_graph_warning(helper, step_index, f"{prefix}: injection wells layer skipped; missing columns: {missing}")
            else:
                well_item_keys = []
                for item_index, (_, well_row) in enumerate(well_df.iterrows(), start=1):
                    well_id = str(well_row.get("WellID", "well"))
                    single_well_df = pd.DataFrame([well_row])
                    single_layer, single_bounds = _point_layer(
                        helper,
                        key=f"injection-well-{item_index}",
                        title=f"Well {item_index}",
                        df=single_well_df,
                        latitude_column="Latitude",
                        longitude_column="Longitude",
                        popup_fields=["WellID", "Latitude", "Longitude", "StartDate", "EndDate", "MaxRate_bbl_day", "MeanRate_bbl_day"],
                        style={"color": "#1d4ed8", "fillColor": "#3b82f6", "radius": 6, "weight": 1},
                        visible=False,
                    )
                    well_layers.append(_add_filter_metadata(single_layer, "injection-wells", item_index))
                    well_bounds.append(single_bounds)
                    if single_layer:
                        well_item_keys.append(single_layer["key"])
                filter_groups.append(_filter_group("injection-wells", "Injection Wells", "injection-wells", well_item_keys))

        return write_leaflet_map_artifact(
            helper,
            artifact_key=artifact_key,
            title=title,
            caption=caption,
            display_order=display_order,
            preferred_height=560,
            layers=[heat_layer] + fault_layers + well_layers,
            bounds_list=[heat_bounds, fault_bounds] + well_bounds,
            filter_groups=filter_groups,
        )
    except Exception as exc:
        add_graph_warning(helper, step_index, f"{prefix}: {exc}")
        return None
