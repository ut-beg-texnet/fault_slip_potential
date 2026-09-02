#!/usr/bin/env python
"""Run a MATLAB R2012b versus Python external-hydrology regression comparison.

Each side interpolates an uploaded pressure-snapshot CSV onto every fault at
every snapshot year. No wells, Theis, or Monte Carlo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fsp.io.external_hydrology import (
    _project_points,
    _project_targets,
    available_years,
    interpolate_fault_pressures,
    load_external_hydrology,
)

# Change these based on your setup
DEFAULT_FAULTS = ROOT / "examples" / "demo_texas_faults_fsp_100_variable_fsp.csv"
DEFAULT_MODEL = ROOT / "examples" / "demo_external_hydrology_model.csv"
DEFAULT_MATLAB = Path(r"C:\Program Files\MATLAB\R2012b\bin\matlab.exe")
DEFAULT_MATLAB_CODE = ROOT / "reference_old_code"
PSI_ABS_THRESHOLD = 1.0
PSI_ABS_LIMIT = 0.01
PSI_REL_LIMIT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", type=Path, default=DEFAULT_FAULTS)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Portal-format external hydrologic model CSV "
             "(Latitude (WGS84), Longitude (WGS84), Change in PSI, Year).",
    )
    parser.add_argument("--matlab-executable", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument(
        "--matlab-code",
        "--matlab_code",
        dest="matlab_code",
        type=Path,
        default=DEFAULT_MATLAB_CODE,
        help="Root folder of the legacy MATLAB FSP code (must contain 'support code').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "external_hydrology_regression_output",
    )
    return parser.parse_args()


def matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def resolve_matlab_support(matlab_code: Path) -> Path:
    """Return the MATLAB support-code folder that holds SpreadsheetStrings2HydrologyData."""
    support_code = matlab_code / "support code"
    converter = support_code / "SpreadsheetStrings2HydrologyData.m"
    if not converter.is_file():
        raise FileNotFoundError(
            f"MATLAB SpreadsheetStrings2HydrologyData.m was not found at {converter}. "
            "Pass --matlab-code pointing at the MATLAB FSP root "
            "(the folder that contains a 'support code' subdirectory)."
        )
    return support_code


def write_gui_stubs(output: Path) -> None:
    """Headless handle class so SpreadsheetStrings2HydrologyData can mutate hDV."""
    (output / "RegressionHDV.m").write_text(
        "classdef RegressionHDV < handle\n    properties\n        data\n        hfig\n    end\nend\n",
        encoding="utf-8",
    )


def _remove_stale_outputs(paths: list[Path]) -> None:
    """Delete leftover MATLAB CSVs so we never load a previous run."""
    for path in paths:
        if path.exists():
            path.unlink()


def _wait_for_fresh_outputs(paths: list[Path], started_at: float, timeout_s: float = 90.0) -> bool:
    """Wait until every path exists and was written after this MATLAB launch."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(path.exists() and path.stat().st_mtime >= started_at - 1.0 and path.stat().st_size > 0 for path in paths):
            return True
        time.sleep(0.25)
    return False


def python_fault_pressures(model: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    """Unrounded interpolate_fault_pressures at every snapshot year."""
    latitudes = faults["Latitude(WGS84)"].to_numpy(dtype=float)
    longitudes = faults["Longitude(WGS84)"].to_numpy(dtype=float)
    ids = faults["FaultID"].astype(str).to_numpy()
    rows = []
    for year in available_years(model):
        selected_year, pressures = interpolate_fault_pressures(model, year, latitudes, longitudes)
        for fault_index, (fault_id, pressure) in enumerate(zip(ids, pressures), start=1):
            rows.append({
                "year": int(selected_year),
                "fault_index": fault_index,
                "fault_id": str(fault_id),
                "python_pressure_psi": float(pressure),
            })
    return pd.DataFrame(rows)


def write_matlab_inputs(
    model: pd.DataFrame, faults: pd.DataFrame, model_csv: Path, faults_csv: Path,
) -> list[dict]:
    """Project each snapshot (and the faults) into that year's km frame for MATLAB griddata."""
    model_rows = []
    fault_rows = []
    origins = []
    fault_lats = faults["Latitude(WGS84)"].to_numpy(dtype=float)
    fault_lons = faults["Longitude(WGS84)"].to_numpy(dtype=float)
    for year in available_years(model):
        snapshot = model[model["Year"] == year]
        xy_km, lat0, lon0 = _project_points(snapshot["Latitude(WGS84)"], snapshot["Longitude(WGS84)"])
        origins.append({"year": int(year), "latitude": lat0, "longitude": lon0})
        for (east, north), pressure in zip(xy_km, snapshot["Pressure_psi"].to_numpy(dtype=float)):
            model_rows.append({
                "East km": float(east),
                "North km": float(north),
                "Change in PSI": float(pressure),
                "Year": int(year),
            })
        fault_xy = _project_targets(fault_lats, fault_lons, lat0, lon0)
        for x_km, y_km in fault_xy:
            fault_rows.append({"year": int(year), "x_km": float(x_km), "y_km": float(y_km)})
    pd.DataFrame(model_rows).to_csv(model_csv, index=False, float_format="%.16g")
    pd.DataFrame(fault_rows).to_csv(faults_csv, index=False, float_format="%.16g")
    return origins


def run_matlab(
    args: argparse.Namespace, model_csv: Path, faults_csv: Path,
) -> pd.DataFrame:
    """Generate and run an R2012b driver that parses the CSV then calls griddata."""
    if not args.matlab_executable.exists():
        raise FileNotFoundError(f"MATLAB R2012b was not found: {args.matlab_executable}")
    support_code = resolve_matlab_support(args.matlab_code)

    output = args.output_dir
    write_gui_stubs(output)
    pressure_file = output / "matlab_fault_pressure.csv"
    script_file = output / "matlab_external_hydrology_regression_driver.m"
    error_file = output / "matlab_error.txt"
    logfile = output / "matlab_run.log"
    required = [pressure_file]
    _remove_stale_outputs(required + [error_file, logfile])

    script = f"""try
addpath('{matlab_path(output)}');
addpath('{matlab_path(support_code)}');
fid_model = fopen('{matlab_path(model_csv)}');
readInData = textscan(fid_model,'%s%s%s%s','Delimiter',',','Headerlines',1,'multipledelimsasone',true);
fclose(fid_model);
hDV = RegressionHDV();
hDV.hfig = [];
hDV.data = struct();
hDV.data.reservoir = struct();
hDV.data.reservoir.stringsImportedHydrology = [readInData{{:,:}}];
SpreadsheetStrings2HydrologyData(hDV.data.reservoir.stringsImportedHydrology, hDV);
numbersImportedHydrology = hDV.data.reservoir.numbersImportedHydrology;
yearsRepresentedHydroImport = hDV.data.reservoir.yearsRepresentedHydroImport;
faults = csvread('{matlab_path(faults_csv)}',1,0);
fid_out = fopen('{matlab_path(pressure_file)}','w');
fprintf(fid_out,'year,fault_index,pressure_psi\\n');
for year_index = 1:length(yearsRepresentedHydroImport)
    ts = yearsRepresentedHydroImport(year_index);
    thisYearData = numbersImportedHydrology(numbersImportedHydrology(:,4)==ts,:);
    thisYearX = thisYearData(:,1);
    thisYearY = thisYearData(:,2);
    thisYearPSI = thisYearData(:,3);
    fault_rows = faults(faults(:,1)==ts,:);
    fault_x = fault_rows(:,2);
    fault_y = fault_rows(:,3);
    ppOnFault = griddata(thisYearX,thisYearY,thisYearPSI,fault_x,fault_y);
    for k = 1:numel(ppOnFault)
        fprintf(fid_out,'%.16g,%d,%.16g\\n',ts,k,ppOnFault(k));
    end
end
fclose(fid_out);
catch ME
fid = fopen('{matlab_path(error_file)}','w');
fprintf(fid,'%s\\n',ME.message);
fclose(fid);
end
exit
"""
    script_file.write_text(script, encoding="utf-8")
    command = [
        str(args.matlab_executable), "-wait", "-nojvm", "-nosplash", "-nodesktop",
        "-logfile", str(logfile),
        "-r", (
            f"try, run('{matlab_path(script_file)}'); "
            f"catch ME, fid=fopen('{matlab_path(error_file)}','w'); "
            f"fprintf(fid,'%s\\n',ME.message); fclose(fid); end; exit"
        ),
    ]
    started_at = time.time()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if not _wait_for_fresh_outputs(required, started_at):
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        error_text = error_file.read_text(encoding="utf-8", errors="replace") if error_file.exists() else ""
        raise RuntimeError(
            f"MATLAB R2012b did not produce all outputs. {detail}\n{error_text}".strip()
        )
    return pd.read_csv(pressure_file)


def compare_frame(python_df: pd.DataFrame, matlab_df: pd.DataFrame) -> pd.DataFrame:
    """Align Python and MATLAB rows by snapshot year and fault order."""
    matlab_df = matlab_df.rename(columns={"pressure_psi": "matlab_pressure_psi"})
    merged = python_df.merge(matlab_df, on=["year", "fault_index"], how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", ["year", "fault_index", "_merge"]]
        raise ValueError(
            "Python/MATLAB fault-pressure rows did not line up. "
            f"Unmatched rows:\n{missing.to_string(index=False)}"
        )
    python_values = merged["python_pressure_psi"].to_numpy(dtype=float)
    matlab_values = merged["matlab_pressure_psi"].to_numpy(dtype=float)
    absolute_error = np.abs(python_values - matlab_values)
    relative_error = absolute_error / np.maximum(np.abs(matlab_values), 1.0)
    return pd.DataFrame({
        "fault_id": merged["fault_id"].astype(str),
        "year": merged["year"].astype(int),
        "python_pressure_psi": python_values,
        "matlab_pressure_psi": matlab_values,
        "absolute_error_psi": absolute_error,
        "relative_error": relative_error,
    })


def write_plot(comparison: pd.DataFrame, output: Path) -> Path:
    figure = go.Figure()
    for year, group in comparison.groupby("year", sort=True):
        figure.add_trace(go.Scatter(
            x=group["python_pressure_psi"],
            y=group["matlab_pressure_psi"],
            mode="markers",
            name=str(int(year)),
            hovertext=group["fault_id"],
        ))
    finite = comparison[["python_pressure_psi", "matlab_pressure_psi"]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if len(finite):
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        figure.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
            line={"color": "#6b7280", "dash": "dash"},
        ))
    figure.update_xaxes(title_text="Python pressure [psi]")
    figure.update_yaxes(title_text="MATLAB R2012b pressure [psi]", scaleanchor="x", scaleratio=1)
    figure.update_layout(
        title="Imported hydrology fault pressure: Python vs MATLAB R2012b",
        template="plotly_white", width=800, height=700,
    )
    figure.write_html(output, include_plotlyjs=True)
    return output


def validate_pressure(comparison: pd.DataFrame) -> bool:
    if comparison.empty:
        return False
    if not np.isfinite(comparison[["python_pressure_psi", "matlab_pressure_psi"]].to_numpy(dtype=float)).all():
        return False
    significant = comparison["matlab_pressure_psi"].abs() >= PSI_ABS_THRESHOLD
    relative_pass = (comparison.loc[significant, "relative_error"] <= PSI_REL_LIMIT).all()
    absolute_pass = (comparison.loc[~significant, "absolute_error_psi"] <= PSI_ABS_LIMIT).all()
    return bool(relative_pass and absolute_pass)


def run() -> int:
    args = parse_args()
    if not args.faults.exists():
        raise FileNotFoundError(
            f"Faults CSV was not found: {args.faults}. "
            "examples/ is not in git; pass --faults explicitly."
        )
    if not args.model.exists():
        raise FileNotFoundError(
            f"External hydrologic model CSV was not found: {args.model}. "
            "examples/ is not in git; pass --model explicitly."
        )
    resolve_matlab_support(args.matlab_code)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    faults = pd.read_csv(args.faults, dtype={"FaultID": str})
    required_fault_columns = {"FaultID", "Latitude(WGS84)", "Longitude(WGS84)"}
    missing = required_fault_columns.difference(faults.columns)
    if missing:
        raise ValueError(
            "Faults CSV must include FaultID, Latitude(WGS84), Longitude(WGS84). "
            f"Missing: {', '.join(sorted(missing))}."
        )
    if faults.empty:
        raise ValueError("Faults CSV has no rows.")

    model = load_external_hydrology(args.model)
    years = available_years(model)
    python_df = python_fault_pressures(model, faults)
    python_path = args.output_dir / "python_fault_pressure.csv"
    python_df.to_csv(python_path, index=False)

    model_csv = args.output_dir / "matlab_model_input.csv"
    faults_csv = args.output_dir / "matlab_faults_xy.csv"
    origins = write_matlab_inputs(model, faults, model_csv, faults_csv)

    matlab_df = run_matlab(args, model_csv, faults_csv)
    comparison = compare_frame(python_df, matlab_df)
    comparison_path = args.output_dir / "fault_pressure_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    plot_path = write_plot(comparison, args.output_dir / "fault_pressure_parity.html")

    passed = validate_pressure(comparison)
    finite_relative = comparison["relative_error"].to_numpy(dtype=float)
    max_relative_error = float(np.nanmax(finite_relative)) if len(finite_relative) else float("nan")
    outputs = {
        "python_pressure_csv": str(python_path.resolve()),
        "matlab_pressure_csv": str((args.output_dir / "matlab_fault_pressure.csv").resolve()),
        "matlab_model_csv": str(model_csv.resolve()),
        "matlab_faults_xy_csv": str(faults_csv.resolve()),
        "fault_csv": str(comparison_path.resolve()),
        "plot_html": str(plot_path.resolve()),
    }
    summary = {
        "passed": passed,
        "max_relative_error": max_relative_error,
        "nan_count": int((~np.isfinite(comparison[["python_pressure_psi", "matlab_pressure_psi"]].to_numpy(dtype=float))).any(axis=1).sum()),
        "tolerance": {
            "relative_at_least_1_psi": PSI_REL_LIMIT,
            "absolute_below_1_psi": PSI_ABS_LIMIT,
        },
        "inputs": {
            "faults": str(args.faults.resolve()),
            "model": str(args.model.resolve()),
            "matlab_code": str(args.matlab_code.resolve()),
            "years": years,
            "n_faults": int(len(faults)),
            "projection_origin_wgs84_by_year": origins,
        },
        "outputs": outputs,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pass: {passed}")
    print(f"Years: {years}")
    print(f"Faults: {len(faults)}")
    print(f"Maximum relative pressure error: {max_relative_error:.6%}")
    print(f"Summary: {summary_path.resolve()}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
