"""Manual hydrology scale benchmark.

Run from the repository root:
    .\fsp_python_venv\Scripts\python.exe tests\benchmark_hydrology_scale.py
"""
import os
import sys
import time
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fsp.models.hydrology import HydrologyParams
from fsp.monte_carlo.hydrology_mc import run_hydrology_mc_time_series


def _synthetic_wells(n_wells=20):
    wells = []
    for i in range(n_wells):
        start_year = 2005 + (i % 4)
        days = np.arange(1.0, 365.0 * 12.0, 30.0)
        rates = 900.0 + 250.0 * np.sin(np.linspace(0.0, 5.0, len(days))) + i * 12.0
        wells.append(SimpleNamespace(
            well_id=f"well-{i + 1}",
            latitude=30.0 + (i % 5) * 0.08,
            longitude=-97.0 - (i // 5) * 0.08,
            start_date=date(start_year, 1, 1),
            start_year=start_year,
            days=days,
            rates=rates,
        ))
    return wells


def _synthetic_faults(n_faults=500):
    side = int(np.ceil(np.sqrt(n_faults)))
    lat_offsets, lon_offsets = np.meshgrid(
        np.linspace(-0.35, 0.35, side),
        np.linspace(-0.35, 0.35, side),
    )
    lats = 30.15 + lat_offsets.ravel()[:n_faults]
    lons = -97.15 + lon_offsets.ravel()[:n_faults]
    return pd.DataFrame({
        "FaultID": [f"fault-{i + 1}" for i in range(n_faults)],
        "Latitude(WGS84)": lats,
        "Longitude(WGS84)": lons,
    })


def main():
    params = HydrologyParams(
        aquifer_thickness=120.0,
        porosity=0.12,
        permeability=150.0,
        fluid_density=1000.0,
        dynamic_viscosity=8e-4,
        fluid_compressibility=3.6e-10,
        rock_compressibility=1.08e-9,
        plus_minus={
            "aquifer_thickness": 5.0,
            "porosity": 0.01,
            "permeability": 10.0,
            "fluid_density": 1.0,
            "dynamic_viscosity": 1e-5,
            "fluid_compressibility": 1e-12,
            "rock_compressibility": 1e-11,
        },
        n_iterations=100,
    )
    years = list(range(2006, 2026))
    start = time.perf_counter()
    result = run_hydrology_mc_time_series(
        params,
        _synthetic_wells(),
        _synthetic_faults(),
        years,
        result_mode="mean",
    )
    elapsed = time.perf_counter() - start
    print(f"rows={len(result):,} years={len(years)} elapsed_seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
