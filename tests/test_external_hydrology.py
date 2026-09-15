"""Tests for uploaded external hydrologic model normalization and interpolation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fsp.io.external_hydrology import (  # noqa: E402
    load_external_hydrology,
    interpolate_fault_pressures,
    pressure_rows_for_faults,
    resolve_external_year,
    to_portal_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = REPO_ROOT / "comparison_datasets"
# MATLAB R2012b griddata on comparison_datasets/external_hydrology_model_realistic_test_portal.csv
MATLAB_REGULAR_GRID_PRESSURES = {
    ("CMP_FAULT_01", 2020): 33.94299999999991,
    ("CMP_FAULT_02", 2020): 26.67939999999896,
    ("CMP_FAULT_03", 2020): 20.25380000000521,
    ("CMP_FAULT_01", 2023): 84.86349999999979,
    ("CMP_FAULT_02", 2023): 67.16893333333077,
    ("CMP_FAULT_03", 2023): 52.22166666667911,
    ("CMP_FAULT_01", 2026): 137.7734999999996,
    ("CMP_FAULT_02", 2026): 108.350333333329,
    ("CMP_FAULT_03", 2026): 83.01520000002083,
}


def _model_file(tmp_path, rows):
    path = tmp_path / "external_pressure.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _rows():
    rows = []
    for year, offset in [(2020, 0), (2022, 10)]:
        rows.extend([
            {"Latitude (WGS84)": 30, "Longitude (WGS84)": -97, "Change in PSI": offset, "Year": year},
            {"Latitude (WGS84)": 30, "Longitude (WGS84)": -96, "Change in PSI": 100 + offset, "Year": year},
            {"Latitude (WGS84)": 31, "Longitude (WGS84)": -97, "Change in PSI": 200 + offset, "Year": year},
        ])
    return rows


def test_normalizes_and_uses_legacy_next_year_rule(tmp_path):
    model = load_external_hydrology(_model_file(tmp_path, _rows()))

    assert list(model.columns) == ["Latitude(WGS84)", "Longitude(WGS84)", "Pressure_psi", "Year"]
    assert resolve_external_year(model, 2020) == 2020
    assert resolve_external_year(model, 2021) == 2022
    assert resolve_external_year(model, 2050) == 2022

    portal_path = tmp_path / "step1_output.csv"
    portal = to_portal_columns(model)
    assert list(portal.columns) == ["Latitude (WGS84)", "Longitude (WGS84)", "Change in PSI", "Year"]
    portal.to_csv(portal_path, index=False)
    assert load_external_hydrology(portal_path).equals(model)

    # Older Step 1 CSVs used the internal names; keep reading those too.
    internal_path = tmp_path / "step1_internal_output.csv"
    model.to_csv(internal_path, index=False)
    assert load_external_hydrology(internal_path).equals(model)


def test_interpolates_in_projected_coordinates(tmp_path):
    model = load_external_hydrology(_model_file(tmp_path, _rows()))

    year, pressure = interpolate_fault_pressures(model, 2020, [30.0], [-96.5])

    assert year == 2020
    assert pressure[0] == pytest.approx(50.0, abs=0.1)


def test_rejects_duplicate_coordinate_year_rows(tmp_path):
    rows = _rows()
    rows.append(rows[0].copy())

    with pytest.raises(ValueError, match="duplicate"):
        load_external_hydrology(_model_file(tmp_path, rows))


def test_uncovered_fault_is_nan_and_zero_in_summary(tmp_path):
    """Outside the hull stays NaN from interpolate; summary rows use 0 additional psi."""
    model = load_external_hydrology(_model_file(tmp_path, _rows()))

    year, pressures = interpolate_fault_pressures(
        model, 2020, [30.0, 40.0], [-96.5, -97.0]
    )

    assert year == 2020
    assert pressures[0] == pytest.approx(50.0, abs=0.1)
    assert np.isnan(pressures[1])

    faults = pd.DataFrame({
        "FaultID": ["inside", "outside"],
        "Latitude(WGS84)": [30.0, 40.0],
        "Longitude(WGS84)": [-96.5, -97.0],
    })
    rows = pressure_rows_for_faults(model, faults, years=[2020])
    by_id = rows.set_index("ID")["Pressure"]
    assert by_id["inside"] == pytest.approx(50.0, abs=0.1)
    assert by_id["outside"] == 0.0


def test_rectilinear_cell_uses_matlab_nw_se_diagonal(tmp_path):
    """A query above the NW–SE diagonal must match MATLAB griddata, not Qhull's SW–NE split."""
    rows = [
        {"Latitude (WGS84)": 30.0, "Longitude (WGS84)": -97.0, "Change in PSI": 10.0, "Year": 2020},
        {"Latitude (WGS84)": 30.0, "Longitude (WGS84)": -96.0, "Change in PSI": 20.0, "Year": 2020},
        {"Latitude (WGS84)": 31.0, "Longitude (WGS84)": -97.0, "Change in PSI": 40.0, "Year": 2020},
        {"Latitude (WGS84)": 31.0, "Longitude (WGS84)": -96.0, "Change in PSI": 80.0, "Year": 2020},
    ]
    model = load_external_hydrology(_model_file(tmp_path, rows))

    year, pressure = interpolate_fault_pressures(model, 2020, [30.8], [-96.7])

    assert year == 2020
    # fx=0.3, fy=0.8 → NW–SE barycentric 40 psi; SW–NE would be 46 psi.
    assert pressure[0] == pytest.approx(40.0, abs=1e-9)


@pytest.mark.skipif(
    not (COMPARISON_DIR / "python_portal_faults.csv").is_file()
    or not (COMPARISON_DIR / "external_hydrology_model_realistic_test_portal.csv").is_file(),
    reason="comparison_datasets CSVs are not present",
)
def test_regular_portal_grid_matches_matlab_griddata():
    faults = pd.read_csv(COMPARISON_DIR / "python_portal_faults.csv", dtype={"FaultID": str})
    model = load_external_hydrology(COMPARISON_DIR / "external_hydrology_model_realistic_test_portal.csv")
    latitudes = faults["Latitude(WGS84)"].to_numpy(dtype=float)
    longitudes = faults["Longitude(WGS84)"].to_numpy(dtype=float)

    for year in (2020, 2023, 2026):
        selected_year, pressures = interpolate_fault_pressures(model, year, latitudes, longitudes)
        assert selected_year == year
        for fault_id, pressure in zip(faults["FaultID"].astype(str), pressures):
            expected = MATLAB_REGULAR_GRID_PRESSURES[(fault_id, year)]
            assert pressure == pytest.approx(expected, rel=1e-12, abs=1e-9)
