"""External deterministic pore-pressure model support.

The portal supplies a CSV snapshot grid instead of injection wells when users
choose the external hydrologic-model workflow.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, QhullError


EXTERNAL_HYDROLOGY_COLUMNS = (
    "Latitude (WGS84)",
    "Longitude (WGS84)",
    "Change in PSI",
    "Year",
)
NORMALIZED_COLUMNS = ("Latitude(WGS84)", "Longitude(WGS84)", "Pressure_psi", "Year")
_PORTAL_TO_INTERNAL = {
    "Latitude (WGS84)": "Latitude(WGS84)",
    "Longitude (WGS84)": "Longitude(WGS84)",
    "Change in PSI": "Pressure_psi",
}
_INTERNAL_TO_PORTAL = {internal: portal for portal, internal in _PORTAL_TO_INTERNAL.items()}
# Collapse projected coordinates onto grid axes. Regular lat/lon snapshots stay
# rectilinear after the equirectangular km projection.
_GRID_AXIS_RTOL = 1e-9


def is_truthy(value) -> bool:
    """Accept portal checkbox values before and after JSON serialization."""
    return value is True or value == 1 or str(value).strip().lower() in {"1", "true"}


def load_external_hydrology(path: str | Path) -> pd.DataFrame:
    """Read, normalize, and validate an external pressure snapshot CSV."""
    source = pd.read_csv(path)
    if set(EXTERNAL_HYDROLOGY_COLUMNS).issubset(source.columns):
        data = source.loc[:, EXTERNAL_HYDROLOGY_COLUMNS].rename(columns=_PORTAL_TO_INTERNAL).copy()
    elif set(NORMALIZED_COLUMNS).issubset(source.columns):
        # Older Step 1 outputs used these names; keep reading them.
        data = source.loc[:, NORMALIZED_COLUMNS].copy()
    else:
        missing = [column for column in EXTERNAL_HYDROLOGY_COLUMNS if column not in source.columns]
        raise ValueError("External hydrologic model CSV is missing required columns: " + ", ".join(missing))
    for column in NORMALIZED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.isna().any().any() or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("External hydrologic model CSV must contain finite numeric values in every row.")
    if not np.equal(data["Year"].to_numpy(), np.floor(data["Year"].to_numpy())).all():
        raise ValueError("External hydrologic model Year values must be whole calendar years.")
    data["Year"] = data["Year"].astype(int)
    if data.duplicated(["Latitude(WGS84)", "Longitude(WGS84)", "Year"]).any():
        raise ValueError("External hydrologic model CSV contains duplicate latitude/longitude/year rows.")
    if ((data["Latitude(WGS84)"] < -90) | (data["Latitude(WGS84)"] > 90) |
            (data["Longitude(WGS84)"] < -180) | (data["Longitude(WGS84)"] > 180)).any():
        raise ValueError("External hydrologic model coordinates must be WGS84 latitude/longitude values.")

    data = data.loc[:, NORMALIZED_COLUMNS].sort_values(["Year", "Latitude(WGS84)", "Longitude(WGS84)"]).reset_index(drop=True)
    for year, snapshot in data.groupby("Year", sort=False):
        _validate_snapshot_points(snapshot, int(year))
    return data


def to_portal_columns(model: pd.DataFrame) -> pd.DataFrame:
    """Return a copy using the uploaded CSV column names.

    Interpolation keeps the internal names (`Latitude(WGS84)`, `Pressure_psi`,
    …). Step 1 writes this portal-named copy so the output CSV matches the
    original upload.
    """
    return (
        model.rename(columns=_INTERNAL_TO_PORTAL)
        .loc[:, list(EXTERNAL_HYDROLOGY_COLUMNS)]
        .copy()
    )


def load_external_hydrology_from_step1(helper, step_index: int = 0) -> pd.DataFrame | None:
    """Return the Step 1 normalized model only when external mode is selected."""
    enabled = is_truthy(helper.getParameterValueWithStepIndexAndParamName(
        step_index, "use_external_hydrologic_model"
    ))
    if not enabled:
        return None
    path = helper.getOptionalDatasetFilePathWithStepIndexAndParamName(
        step_index, "external_hydrology_model_output"
    )
    if not path:
        raise ValueError("External hydrologic model mode was selected, but Step 1 did not produce a model output.")
    return load_external_hydrology(path)


def available_years(model: pd.DataFrame) -> list[int]:
    return sorted(int(year) for year in model["Year"].unique())


def resolve_external_year(model: pd.DataFrame, requested_year: int) -> int:
    """Mirror MATLAB's imported-model slider: choose the next available year."""
    years = available_years(model)
    if not years:
        raise ValueError("External hydrologic model has no snapshots.")
    for year in years:
        if year >= int(requested_year):
            return year
    return years[-1]


def external_year_message(requested_year: int, selected_year: int) -> str | None:
    if requested_year == selected_year:
        return None
    return f"External hydrologic model has no {requested_year} snapshot; using {selected_year}."


def interpolate_fault_pressures(model: pd.DataFrame, requested_year: int, latitudes, longitudes) -> tuple[int, np.ndarray]:
    """Linearly interpolate the selected snapshot and reject uncovered faults."""
    selected_year = resolve_external_year(model, requested_year)
    snapshot = model[model["Year"] == selected_year]
    values = _interpolate_snapshot(snapshot, latitudes, longitudes)
    uncovered = np.flatnonzero(~np.isfinite(values))
    if len(uncovered):
        positions = ", ".join(str(index + 1) for index in uncovered[:10])
        suffix = "" if len(uncovered) <= 10 else ", ..."
        raise ValueError(
            f"External hydrologic model does not cover fault row(s) {positions}{suffix} for {selected_year}."
        )
    return selected_year, values


def interpolated_grid(model: pd.DataFrame, selected_year: int, n: int = 100) -> pd.DataFrame:
    """Return finite cells of a linear pressure surface for the external-model map."""
    snapshot = model[model["Year"] == selected_year]
    latitudes = snapshot["Latitude(WGS84)"].to_numpy(dtype=float)
    longitudes = snapshot["Longitude(WGS84)"].to_numpy(dtype=float)
    lat_grid, lon_grid = np.meshgrid(
        np.linspace(latitudes.min(), latitudes.max(), n),
        np.linspace(longitudes.min(), longitudes.max(), n),
        indexing="ij",
    )
    pressure = _interpolate_snapshot(snapshot, lat_grid.ravel(), lon_grid.ravel()).reshape(lat_grid.shape)
    grid = pd.DataFrame({
        "Latitude": lat_grid.ravel(),
        "Longitude": lon_grid.ravel(),
        "Pressure_psi": pressure.ravel(),
    }).dropna(subset=["Pressure_psi"])
    grid.insert(0, "WellID", "External model")
    return grid.reset_index(drop=True)


def pressure_rows_for_faults(model: pd.DataFrame, faults: pd.DataFrame, years=None) -> pd.DataFrame:
    """Interpolate each supplied snapshot onto every fault for summary outputs."""
    if faults.empty:
        return pd.DataFrame(columns=["ID", "Pressure", "Year"])
    target_years = available_years(model) if years is None else [int(year) for year in years]
    rows = []
    latitudes = faults["Latitude(WGS84)"].to_numpy(dtype=float)
    longitudes = faults["Longitude(WGS84)"].to_numpy(dtype=float)
    ids = faults["FaultID"].astype(str).to_numpy()
    for year in target_years:
        selected_year, pressures = interpolate_fault_pressures(model, year, latitudes, longitudes)
        for fault_id, pressure in zip(ids, pressures):
            rows.append({"ID": fault_id, "Pressure": round(float(pressure), 2), "Year": selected_year})
    return pd.DataFrame(rows)


def _validate_snapshot_points(snapshot: pd.DataFrame, year: int) -> None:
    if len(snapshot) < 3:
        raise ValueError(f"External hydrologic model year {year} needs at least three pressure points.")
    points = _project_points(snapshot["Latitude(WGS84)"], snapshot["Longitude(WGS84)"])[0]
    try:
        Delaunay(points)
    except QhullError as exc:
        raise ValueError(f"External hydrologic model year {year} points must not be collinear.") from exc


def _interpolate_snapshot(snapshot: pd.DataFrame, latitudes, longitudes) -> np.ndarray:
    points, lat0, lon0 = _project_points(snapshot["Latitude(WGS84)"], snapshot["Longitude(WGS84)"])
    targets = _project_targets(latitudes, longitudes, lat0, lon0)
    values = snapshot["Pressure_psi"].to_numpy(dtype=float)
    rectilinear = _as_rectilinear_grid(points, values)
    if rectilinear is not None:
        # MATLAB R2012b griddata → TriScatteredInterp/DelaunayTri (CGAL) splits
        # each rectangular cell NW–SE. SciPy LinearNDInterpolator (Qhull) splits
        # SW–NE on the same lattice, which was the ~6% regression gap.
        x_axis, y_axis, value_grid = rectilinear
        return _interpolate_rectilinear_nw_se(x_axis, y_axis, value_grid, targets)
    interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
    return np.asarray(interpolator(targets), dtype=float)


def _unique_axis(values: np.ndarray) -> np.ndarray:
    """Merge coordinates that agree within a relative span tolerance."""
    ordered = np.sort(np.asarray(values, dtype=float))
    span = float(max(ordered[-1] - ordered[0], np.max(np.abs(ordered)), 1.0))
    bins = [ordered[0]]
    for value in ordered[1:]:
        if abs(value - bins[-1]) > _GRID_AXIS_RTOL * span:
            bins.append(value)
    return np.asarray(bins, dtype=float)


def _as_rectilinear_grid(
    points: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (x_axis, y_axis, value_grid) when points fill a complete x×y lattice."""
    xs = points[:, 0]
    ys = points[:, 1]
    x_axis = _unique_axis(xs)
    y_axis = _unique_axis(ys)
    nx = len(x_axis)
    ny = len(y_axis)
    if nx < 2 or ny < 2 or nx * ny != len(points):
        return None
    ix = np.argmin(np.abs(xs[:, None] - x_axis[None, :]), axis=1)
    iy = np.argmin(np.abs(ys[:, None] - y_axis[None, :]), axis=1)
    flat = iy * nx + ix
    if len(np.unique(flat)) != len(points):
        return None
    grid = np.full(ny * nx, np.nan)
    grid[flat] = values
    grid = grid.reshape(ny, nx)
    if not np.isfinite(grid).all():
        return None
    return x_axis, y_axis, grid


def _interpolate_rectilinear_nw_se(
    x_axis: np.ndarray, y_axis: np.ndarray, value_grid: np.ndarray, targets: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear interpolation with MATLAB's NW–SE diagonal on each cell."""
    xq = np.asarray(targets[:, 0], dtype=float)
    yq = np.asarray(targets[:, 1], dtype=float)
    result = np.full(xq.shape, np.nan)
    inside = (
        (xq >= x_axis[0]) & (xq <= x_axis[-1])
        & (yq >= y_axis[0]) & (yq <= y_axis[-1])
    )
    if not np.any(inside):
        return result
    xi = xq[inside]
    yi = yq[inside]
    ix = np.clip(np.searchsorted(x_axis, xi, side="right") - 1, 0, len(x_axis) - 2)
    iy = np.clip(np.searchsorted(y_axis, yi, side="right") - 1, 0, len(y_axis) - 2)
    x0 = x_axis[ix]
    x1 = x_axis[ix + 1]
    y0 = y_axis[iy]
    y1 = y_axis[iy + 1]
    fx = (xi - x0) / (x1 - x0)
    fy = (yi - y0) / (y1 - y0)
    v_sw = value_grid[iy, ix]
    v_se = value_grid[iy, ix + 1]
    v_nw = value_grid[iy + 1, ix]
    v_ne = value_grid[iy + 1, ix + 1]
    # fy >= 1 - fx → triangle NW–NE–SE; otherwise SW–SE–NW. Equal on the diagonal.
    upper = fy >= (1.0 - fx)
    result[inside] = np.where(
        upper,
        (1.0 - fx) * v_nw + (1.0 - fy) * v_se + (fx + fy - 1.0) * v_ne,
        (1.0 - fx - fy) * v_sw + fx * v_se + fy * v_nw,
    )
    return result


def _project_points(latitudes, longitudes) -> tuple[np.ndarray, float, float]:
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    lat0 = float(np.mean(latitudes))
    lon0 = float(np.mean(longitudes))
    return _project_targets(latitudes, longitudes, lat0, lon0), lat0, lon0


def _project_targets(latitudes, longitudes, lat0: float, lon0: float) -> np.ndarray:
    km_per_degree_lon = 111.0 * math.cos(math.radians(lat0))
    if abs(km_per_degree_lon) < 1e-9:
        raise ValueError("External hydrologic model cannot be projected near the poles.")
    return np.column_stack(((np.asarray(longitudes, dtype=float) - lon0) * km_per_degree_lon,
                            (np.asarray(latitudes, dtype=float) - lat0) * 111.0))
