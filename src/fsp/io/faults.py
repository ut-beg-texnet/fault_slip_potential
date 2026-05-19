"""
Fault data loading: CSV, shapefile, and random generation.
Port of FSP/core/utilities.jl shapefile_to_fsp_csv, generate_randomized_faults_csv.
"""
import math
import numpy as np
import pandas as pd


def load_faults_csv(path: str) -> pd.DataFrame:
    """Load FSP-format fault CSV with FaultID as string."""
    return pd.read_csv(path, dtype={"FaultID": str})


def generate_randomized_faults(num: int,
                                 strike_min: float, strike_max: float,
                                 dip_min: float, dip_max: float,
                                 length_km: float = 5.0,
                                 center_lat: float = 31.023892,
                                 center_lon: float = -100.185739) -> pd.DataFrame:
    """Generate randomised synthetic faults.

    Port of Julia generate_randomized_faults_csv.
    Fault centroids are placed within ~15 km of (center_lat, center_lon).
    """
    strike_lo = max(strike_min, 0.0)
    strike_hi = min(strike_max, 360.0)
    dip_lo = max(dip_min, 0.0)
    dip_hi = min(dip_max, 90.0)

    if strike_lo > strike_hi:
        raise ValueError("Strike lower bound > upper bound")
    if dip_lo > dip_hi:
        raise ValueError("Dip lower bound > upper bound")

    strikes = np.random.uniform(strike_lo, strike_hi, num)
    dips = np.random.uniform(dip_lo, dip_hi, num)

    lats, lons = _random_points_within_radius(center_lat, center_lon, num, radius_km=15.0)

    from .coords import offset_km_to_latlon

    return pd.DataFrame({
        "FaultID": [f"Fault_{i+1}" for i in range(num)],
        "Latitude(WGS84)": lats,
        "Longitude(WGS84)": lons,
        "Strike": strikes,
        "Dip": dips,
        "LengthKm": np.full(num, length_km),
        "FrictionCoefficient": np.full(num, np.nan),
    })


def _random_points_within_radius(lat0, lon0, n, radius_km=15.0):
    from .coords import offset_km_to_latlon
    lats = np.empty(n)
    lons = np.empty(n)
    lats[0] = lat0
    lons[0] = lon0
    for i in range(1, n):
        r = radius_km * math.sqrt(np.random.random())
        theta = 2 * math.pi * np.random.random()
        dx = r * math.cos(theta)
        dy = r * math.sin(theta)
        lats[i], lons[i] = offset_km_to_latlon(lat0, lon0, dx, dy)
    return lats, lons


def load_faults_shapefile(path: str) -> pd.DataFrame:
    """Load a fault shapefile (CSV exported by portal containing MULTILINESTRING WKT).

    Port of Julia shapefile_to_fsp_csv.
    Expects columns: FID, dip, SHAPE (MULTILINESTRING WKT).
    Returns DataFrame matching FSP fault CSV format (without FrictionCoefficient).
    """
    import math
    shapefile_csv = pd.read_csv(path)
    rows = []

    for _, row in shapefile_csv.iterrows():
        fid = row["FID"]
        dip = float(row["dip"]) if not pd.isna(row["dip"]) else 0.0
        wkt = str(row["SHAPE"])
        points = _parse_multilinestring(wkt)

        seg = 1
        for i in range(len(points) - 1):
            start_lon, start_lat = points[i]
            end_lon, end_lat = points[i + 1]
            mid_lon = (start_lon + end_lon) / 2.0
            mid_lat = (start_lat + end_lat) / 2.0

            from .coords import haversine_distance
            length_km = haversine_distance(start_lat, start_lon, end_lat, end_lon)

            lat1_r = math.radians(start_lat)
            lat2_r = math.radians(end_lat)
            dlon_r = math.radians(end_lon - start_lon)
            x = math.cos(lat2_r) * math.sin(dlon_r)
            y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
            bearing = math.degrees(math.atan2(x, y))
            strike = bearing % 360.0

            rows.append({
                "FaultID": f"{fid}Segment-{seg}",
                "Longitude(WGS84)": mid_lon,
                "Latitude(WGS84)": mid_lat,
                "Strike": strike,
                "Dip": dip,
                "LengthKm": length_km,
            })
            seg += 1

    return pd.DataFrame(rows)


def _parse_multilinestring(wkt: str):
    """Parse a MULTILINESTRING WKT string into a list of (lon, lat) tuples."""
    wkt = wkt.strip().strip('"')
    if not wkt.upper().startswith("MULTILINESTRING"):
        raise ValueError("Expected MULTILINESTRING WKT")
    wkt = wkt.replace("MULTILINESTRING ((", "").replace("MULTILINESTRING((", "")
    wkt = wkt.rstrip(")").rstrip(")")
    points = []
    for part in wkt.split(","):
        part = part.strip()
        if part:
            nums = part.split()
            if len(nums) >= 2:
                points.append((float(nums[0]), float(nums[1])))
    return points
