"""
Coordinate utilities: haversine, WKT generation, spatial grid.
Port of FSP/core/utilities.jl latlon_to_wkt, create_spatial_grid_latlon, haversine_distance.
"""
import math
import numpy as np
import pandas as pd


_EARTH_R_KM = 6371.0
_KM_PER_DEG_LAT = 111.0
_DEG2RAD = math.pi / 180.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km.  Accepts scalars or numpy arrays."""
    lat1 = np.asarray(lat1, dtype=float) * _DEG2RAD
    lat2 = np.asarray(lat2, dtype=float) * _DEG2RAD
    lon1 = np.asarray(lon1, dtype=float) * _DEG2RAD
    lon2 = np.asarray(lon2, dtype=float) * _DEG2RAD

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return _EARTH_R_KM * c


def offset_km_to_latlon(lat0: float, lon0: float, dx_km: float, dy_km: float):
    """Convert local km offsets (E, N) to (lat, lon) via planar approximation."""
    dlat = dy_km / _KM_PER_DEG_LAT
    dlon = dx_km / (_KM_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def create_spatial_grid(lat_min, lat_max, lon_min, lon_max, n=50):
    """Create a lat/lon meshgrid.

    Returns (lat_grid, lon_grid) — matches Julia's 'flipped' convention
    used in pfieldcalc_all_rates where first arg is lat_grid.
    """
    lats = np.linspace(lat_min, lat_max, n)
    lons = np.linspace(lon_min, lon_max, n)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


def create_projected_spatial_grid(latitudes, longitudes, n=150, margin_fraction=0.3, min_margin_km=1.0):
    """Create an approximately uniform local ENU grid and return lat/lon nodes.

    The pressure calculations operate on distances, so a grid that is uniform in
    local kilometers gives a more faithful surface than equally spaced degrees.
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    valid = np.isfinite(lats) & np.isfinite(lons)
    if not np.any(valid):
        raise ValueError("At least one valid latitude/longitude is required.")

    lats = lats[valid]
    lons = lons[valid]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    km_per_deg_lon = _KM_PER_DEG_LAT * math.cos(math.radians(lat0))
    if abs(km_per_deg_lon) < 1e-9:
        raise ValueError("Cannot create projected grid near the poles.")

    x_km = (lons - lon0) * km_per_deg_lon
    y_km = (lats - lat0) * _KM_PER_DEG_LAT

    x_min = float(np.min(x_km))
    x_max = float(np.max(x_km))
    y_min = float(np.min(y_km))
    y_max = float(np.max(y_km))

    x_span = max(x_max - x_min, 0.0)
    y_span = max(y_max - y_min, 0.0)
    x_margin = max(x_span * margin_fraction, min_margin_km)
    y_margin = max(y_span * margin_fraction, min_margin_km)

    xs = np.linspace(x_min - x_margin, x_max + x_margin, n)
    ys = np.linspace(y_min - y_margin, y_max + y_margin, n)
    x_grid, y_grid = np.meshgrid(xs, ys)

    lat_grid = lat0 + y_grid / _KM_PER_DEG_LAT
    lon_grid = lon0 + x_grid / km_per_deg_lon
    bounds = [
        [float(lat_grid.min()), float(lon_grid.min())],
        [float(lat_grid.max()), float(lon_grid.max())],
    ]
    return lat_grid, lon_grid, bounds


def latlon_to_wkt(faults_df: pd.DataFrame,
                  lat_col: str = "Latitude(WGS84)",
                  lon_col: str = "Longitude(WGS84)",
                  strike_col: str = "Strike",
                  length_col: str = "LengthKm") -> pd.DataFrame:
    """Add a WKT LINESTRING column to faults_df (in-place, also returned).

    Port of Julia latlon_to_wkt using small-angle ENU approximation.
    """
    wkt_strings = []
    for _, row in faults_df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        strike_deg = float(row[strike_col])
        half_km = float(row[length_col]) / 2.0

        strike_rad = math.radians(strike_deg)
        dx = math.sin(strike_rad) * half_km   # km east
        dy = math.cos(strike_rad) * half_km   # km north

        start_lat, start_lon = offset_km_to_latlon(lat, lon, -dx, -dy)
        end_lat, end_lon = offset_km_to_latlon(lat, lon, dx, dy)

        wkt = f"LINESTRING ({start_lon} {start_lat},{end_lon} {end_lat})"
        wkt_strings.append(wkt)

    faults_df = faults_df.copy()
    faults_df["wkt"] = wkt_strings
    return faults_df


def reformat_pressure_grid_to_heatmap(lat_grid, lon_grid, pressure_grid) -> pd.DataFrame:
    """Flatten a 2-D pressure grid to a DataFrame with Latitude, Longitude, Pressure_psi."""
    rows = {
        "Latitude": lat_grid.ravel(),
        "Longitude": lon_grid.ravel(),
        "Pressure_psi": pressure_grid.ravel(),
    }
    return pd.DataFrame(rows)
