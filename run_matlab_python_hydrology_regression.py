#!/usr/bin/env python
"""Run a MATLAB R2012b versus Python hydrology regression comparison."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fsp.hydrology.params import calcST
from fsp.hydrology.theis import pressureScenario_Rall

# Change these based on your setup
DEFAULT_FAULTS = ROOT / "examples" / "demo_texas_faults_fsp_100_variable_fsp.csv"
DEFAULT_WELLS = ROOT / "examples" / "demo_texas_injection_wells_monthly_fsp_20wells_variable_fsp.csv"
DEFAULT_MATLAB = Path(r"C:\Program Files\MATLAB\R2012b\bin\matlab.exe")
KM_PER_DEG_LAT = 111.0


@dataclass
class WellSeries:
    well_id: str
    latitude: float
    longitude: float
    x_km: float
    y_km: float
    start_date: date
    days: np.ndarray
    rates_bpd: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", type=Path, default=DEFAULT_FAULTS)
    parser.add_argument("--wells", type=Path, default=DEFAULT_WELLS)
    parser.add_argument("--matlab-executable", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hydrology_regression_output")
    parser.add_argument("--year", type=int, default=2031)
    parser.add_argument("--well-id", help="Well ID used for the radial plot; defaults to the first active well.")
    parser.add_argument("--porosity-fraction", type=float, default=0.10)
    parser.add_argument("--aquifer-thickness-ft", type=float, default=100.0)
    parser.add_argument("--permeability-md", type=float, default=200.0)
    parser.add_argument("--fluid-density", type=float, default=1000.0)
    parser.add_argument("--dynamic-viscosity", type=float, default=0.0008)
    parser.add_argument("--fluid-compressibility", type=float, default=3.6e-10)
    parser.add_argument("--rock-compressibility", type=float, default=1.08e-9)
    parser.add_argument("--radial-points", type=int, default=200)
    parser.add_argument("--mc-iterations", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--mc-uncertainty-json",
        default="{}",
        help="JSON object of plus/minus bounds in Python units, e.g. '{\"permeability_md\": 20}'.",
    )
    return parser.parse_args()


def month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def next_month(value: date) -> date:
    return date(value.year + 1, 1, 1) if value.month == 12 else date(value.year, value.month + 1, 1)


def days_in_month(year: int, month: int) -> int:
    return (next_month(month_start(year, month)) - month_start(year, month)).days


def local_xy_km(latitudes, longitudes, origin_lat: float, origin_lon: float):
    """Convert WGS84 locations to one shared local Cartesian grid."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    x_km = (longitudes - origin_lon) * KM_PER_DEG_LAT * np.cos(np.radians(origin_lat))
    y_km = (latitudes - origin_lat) * KM_PER_DEG_LAT
    return x_km, y_km


def normalize_monthly_wells(wells: pd.DataFrame, cutoff: date, origin_lat: float, origin_lon: float) -> list[WellSeries]:
    """Build MATLAB-compatible monthly step series without changing production code."""
    required = {"WellID", "Latitude(WGS84)", "Longitude(WGS84)", "Year", "Month", "InjectionRate(bbl/month)"}
    missing = required.difference(wells.columns)
    if missing:
        raise ValueError(f"Monthly wells CSV is missing: {', '.join(sorted(missing))}")

    series = []
    for well_id, group in wells.groupby(wells["WellID"].astype(str), sort=False):
        monthly_rates: dict[tuple[int, int], float] = {}
        for _, row in group.iterrows():
            year, month = int(row["Year"]), int(row["Month"])
            start = month_start(year, month)
            if start <= cutoff:
                monthly_rates[(year, month)] = float(row["InjectionRate(bbl/month)"]) / days_in_month(year, month)
        if not monthly_rates:
            continue

        first_year, first_month = min(monthly_rates)
        start_date = month_start(first_year, first_month)
        current = start_date
        previous_rate = None
        days, rates = [], []
        while current <= cutoff:
            rate = monthly_rates.get((current.year, current.month), 0.0)
            if previous_rate is None or rate != previous_rate:
                days.append(float((current - start_date).days + 1))
                rates.append(rate)
                previous_rate = rate
            current = next_month(current)

        latitude = float(group.iloc[0]["Latitude(WGS84)"])
        longitude = float(group.iloc[0]["Longitude(WGS84)"])
        x_km, y_km = local_xy_km([latitude], [longitude], origin_lat, origin_lon)
        series.append(WellSeries(
            str(well_id), latitude, longitude, float(x_km[0]), float(y_km[0]),
            start_date, np.asarray(days), np.asarray(rates),
        ))
    return series


def python_pressure(distances_m: np.ndarray, well: WellSeries, strho, year: int) -> np.ndarray:
    evaluation_days = float((date(year - 1, 12, 31) - well.start_date).days + 1)
    return pressureScenario_Rall(well.rates_bpd, well.days, distances_m, strho, evaluation_days)


def matlab_vector(values) -> str:
    return "[" + ";".join(f"{float(value):.16g}" for value in values) + "]"


def matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def matlab_events(well: WellSeries, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Map Python step changes to pfront's preceding-timestamp convention."""
    evaluation_days = float((date(year - 1, 12, 31) - well.start_date).days + 1)
    return np.append(well.days, evaluation_days - 1.0 / 86400.0), np.concatenate(([0.0], well.rates_bpd))


def matlab_well_block(well: WellSeries, year: int, index: int) -> str:
    event_days, event_rates = matlab_events(well, year)
    return f"""
event_days_{index} = {matlab_vector(event_days)};
event_rates_{index} = {matlab_vector(event_rates)};
base_{index} = datenum({well.start_date.year},{well.start_date.month},{well.start_date.day},0,0,0);
d_{index} = [base_{index} + event_days_{index}, event_rates_{index}];
r_fault_{index} = sqrt((fault_x - {well.x_km:.16g}).^2 + (fault_y - {well.y_km:.16g}).^2) .* 1000;
"""


def _remove_stale_outputs(paths: list[Path]) -> None:
    """Delete leftover MATLAB CSVs so we never load a previous run."""
    for path in paths:
        if path.exists():
            path.unlink()


def _wait_for_fresh_outputs(paths: list[Path], started_at: float, timeout_s: float = 45.0) -> bool:
    """Wait until every path exists and was written after this MATLAB launch."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(path.exists() and path.stat().st_mtime >= started_at - 1.0 and path.stat().st_size > 0 for path in paths):
            return True
        time.sleep(0.25)
    return False


def run_matlab(args: argparse.Namespace, faults: pd.DataFrame, wells: list[WellSeries], strho,
               selected_well: WellSeries, samples_percent: np.ndarray | None) -> dict[str, np.ndarray]:
    """Generate and run a self-contained R2012b pfront driver."""
    if not args.matlab_executable.exists():
        raise FileNotFoundError(f"MATLAB R2012b was not found: {args.matlab_executable}")

    output = args.output_dir
    radial_file = output / "matlab_radial_pressure.csv"
    fault_file = output / "matlab_fault_pressure.csv"
    mc_file = output / "matlab_mc_fault_pressure.csv"
    script_file = output / "matlab_hydrology_regression_driver.m"
    required = [radial_file, fault_file] + ([mc_file] if samples_percent is not None else [])
    _remove_stale_outputs(required)

    fault_x = faults["x_km"].to_numpy(float)
    fault_y = faults["y_km"].to_numpy(float)
    blocks = "\n".join(matlab_well_block(well, args.year, i + 1) for i, well in enumerate(wells))
    sum_blocks = "\n".join(
        f"fault_pressure = fault_pressure + pfront(r_fault_{i + 1},{args.year},d_{i + 1},S,T,rho,9.81) .* 14.5;"
        for i in range(len(wells))
    )
    selected_days, selected_rates = matlab_events(selected_well, args.year)
    radial_km = np.linspace(0.1, 20.0, args.radial_points)
    porosity_percent = args.porosity_fraction * 100.0
    mc_setup = ""
    mc_loop = ""
    if samples_percent is not None:
        sample_file = output / "mc_samples_matlab_percent.csv"
        np.savetxt(sample_file, samples_percent, delimiter=",")
        mc_setup = f"samples = csvread('{matlab_path(sample_file)}'); mc_pressure = zeros(size(samples,1), length(fault_x));"
        mc_loop = f"""
for sample_index = 1:size(samples,1)
    h = samples(sample_index,1) * 0.3048;
    phi = samples(sample_index,2) / 100;
    kap = samples(sample_index,3);
    rho = samples(sample_index,4);
    mu = samples(sample_index,5);
    beta = samples(sample_index,6);
    alpha = samples(sample_index,7);
    [S,T] = calcST([0,0,0,0,rho,9.81,mu,beta,alpha],h,phi,kap);
    fault_pressure = zeros(length(fault_x),1);
    {sum_blocks}
    mc_pressure(sample_index,:) = fault_pressure';
end
csvwrite('{matlab_path(mc_file)}',mc_pressure);
"""

    script = f"""addpath('{matlab_path(ROOT / 'reference_old_code' / 'technical code')}');
fault_x = {matlab_vector(fault_x)};
fault_y = {matlab_vector(fault_y)};
{blocks}
h = {args.aquifer_thickness_ft:.16g} * 0.3048;
porosity_percent = {porosity_percent:.16g};
phi = porosity_percent / 100;
kap = {args.permeability_md:.16g};
rho = {args.fluid_density:.16g};
mu = {args.dynamic_viscosity:.16g};
beta = {args.fluid_compressibility:.16g};
alpha = {args.rock_compressibility:.16g};
[S,T] = calcST([0,0,0,0,rho,9.81,mu,beta,alpha],h,phi,kap);
fault_pressure = zeros(length(fault_x),1);
{sum_blocks}
csvwrite('{matlab_path(fault_file)}',fault_pressure);
radial_days = {matlab_vector(selected_days)};
radial_rates = {matlab_vector(selected_rates)};
radial_base = datenum({selected_well.start_date.year},{selected_well.start_date.month},{selected_well.start_date.day},0,0,0);
radial_data = [radial_base + radial_days, radial_rates];
radial_km = {matlab_vector(radial_km)};
radial_pressure = pfront(radial_km .* 1000,{args.year},radial_data,S,T,rho,9.81) .* 14.5;
csvwrite('{matlab_path(radial_file)}',[radial_km,radial_pressure]);
{mc_setup}
{mc_loop}
exit
"""
    script_file.write_text(script, encoding="utf-8")
    # -wait keeps matlab.exe attached until the script's exit (R2012b Windows).
    command = [
        str(args.matlab_executable), "-wait", "-nojvm", "-nosplash", "-nodesktop",
        "-r", f"run('{matlab_path(script_file)}')",
    ]
    started_at = time.time()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if not _wait_for_fresh_outputs(required, started_at):
        detail = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(f"MATLAB R2012b did not produce all outputs. {detail}")

    result = {
        "radial": np.atleast_2d(np.loadtxt(radial_file, delimiter=",")),
        "fault": np.atleast_1d(np.loadtxt(fault_file, delimiter=",")),
    }
    if samples_percent is not None:
        result["mc"] = np.atleast_2d(np.loadtxt(mc_file, delimiter=","))
    return result


def make_samples(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    if args.mc_iterations <= 0:
        return None, None, {}
    uncertainty = json.loads(args.mc_uncertainty_json)
    names = [
        "aquifer_thickness_ft", "porosity_fraction", "permeability_md",
        "fluid_density", "dynamic_viscosity", "fluid_compressibility", "rock_compressibility",
    ]
    bases = np.array([
        args.aquifer_thickness_ft, args.porosity_fraction, args.permeability_md,
        args.fluid_density, args.dynamic_viscosity, args.fluid_compressibility, args.rock_compressibility,
    ])
    deltas = np.array([float(uncertainty.get(name, 0.0)) for name in names])
    lower = bases - deltas
    if lower[0] <= 0 or lower[1] <= 0 or lower[1] >= 1 or np.any(lower[2:] <= 0):
        raise ValueError("MC uncertainty produces nonphysical hydrology inputs.")
    rng = np.random.default_rng(args.random_seed)
    python_samples = rng.uniform(lower, bases + deltas, size=(args.mc_iterations, len(names)))
    matlab_samples = python_samples.copy()
    matlab_samples[:, 1] *= 100.0
    return python_samples, matlab_samples, dict(zip(names, deltas.tolist()))


def compare_frame(distance_or_id, python_values, matlab_values, key: str) -> pd.DataFrame:
    labels = np.asarray(distance_or_id).reshape(-1)
    python_values = np.asarray(python_values, dtype=float).reshape(-1)
    matlab_values = np.asarray(matlab_values, dtype=float).reshape(-1)
    if not (len(labels) == len(python_values) == len(matlab_values)):
        raise ValueError(
            f"{key} comparison length mismatch: {len(labels)} labels, "
            f"{len(python_values)} Python values, {len(matlab_values)} MATLAB values. "
            "Stale MATLAB CSVs from a previous run can cause this."
        )
    absolute_error = np.abs(python_values - matlab_values)
    relative_error = absolute_error / np.maximum(np.abs(matlab_values), 1.0)
    return pd.DataFrame({
        key: labels,
        "python_pressure_psi": python_values,
        "matlab_pressure_psi": matlab_values,
        "absolute_error_psi": absolute_error,
        "relative_error": relative_error,
    })


def write_plot(comparison: pd.DataFrame, output: Path, well_id: str, year: int) -> Path:
    figure = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                           subplot_titles=("Python", "MATLAB R2012b"))
    figure.add_trace(go.Scatter(x=comparison["distance_km"], y=comparison["python_pressure_psi"],
                                mode="lines", line={"color": "#2563eb"}), row=1, col=1)
    figure.add_trace(go.Scatter(x=comparison["distance_km"], y=comparison["matlab_pressure_psi"],
                                mode="lines", line={"color": "#dc2626"}), row=1, col=2)
    figure.update_xaxes(title_text="Distance [km]", range=[0, 20])
    figure.update_yaxes(title_text="Pressure [psi]", rangemode="tozero")
    figure.update_layout(title=f"Pressure vs Distance: {well_id}, Jan 1 {year}",
                         template="plotly_white", showlegend=False, width=1200, height=500)
    figure.write_html(output, include_plotlyjs=True)
    return output


def validate(comparison: pd.DataFrame) -> bool:
    significant = comparison["matlab_pressure_psi"].abs() >= 1.0
    relative_pass = (comparison.loc[significant, "relative_error"] <= 0.01).all()
    absolute_pass = (comparison.loc[~significant, "absolute_error_psi"] <= 0.01).all()
    return bool(relative_pass and absolute_pass)


def run() -> int:
    args = parse_args()
    if not 0.0 < args.porosity_fraction < 1.0:
        raise ValueError("--porosity-fraction must be greater than 0 and less than 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    faults = pd.read_csv(args.faults, dtype={"FaultID": str})
    wells_data = pd.read_csv(args.wells, dtype={"WellID": str})
    all_lats = np.concatenate([faults["Latitude(WGS84)"].to_numpy(float), wells_data["Latitude(WGS84)"].to_numpy(float)])
    all_lons = np.concatenate([faults["Longitude(WGS84)"].to_numpy(float), wells_data["Longitude(WGS84)"].to_numpy(float)])
    origin_lat, origin_lon = float(all_lats.mean()), float(all_lons.mean())
    faults["x_km"], faults["y_km"] = local_xy_km(
        faults["Latitude(WGS84)"], faults["Longitude(WGS84)"], origin_lat, origin_lon
    )
    wells = normalize_monthly_wells(wells_data, date(args.year - 1, 12, 31), origin_lat, origin_lon)
    if not wells:
        raise ValueError("No wells are active for the requested year.")
    selected_well = next((well for well in wells if well.well_id == args.well_id), wells[0])
    if args.well_id and selected_well.well_id != args.well_id:
        raise ValueError(f"Well {args.well_id!r} is not active for {args.year}.")

    strho = calcST(args.aquifer_thickness_ft, args.porosity_fraction, args.permeability_md,
                   args.fluid_density, args.dynamic_viscosity, 9.81,
                   args.fluid_compressibility, args.rock_compressibility)
    radial_km = np.linspace(0.1, 20.0, args.radial_points)
    python_radial = python_pressure(radial_km * 1000.0, selected_well, strho, args.year)

    fault_x = faults["x_km"].to_numpy(float)
    fault_y = faults["y_km"].to_numpy(float)
    python_fault = np.zeros(len(faults))
    for well in wells:
        distance_m = np.hypot(fault_x - well.x_km, fault_y - well.y_km) * 1000.0
        python_fault += python_pressure(distance_m, well, strho, args.year)

    python_samples, matlab_samples, uncertainty = make_samples(args)
    python_mc = None
    if python_samples is not None:
        python_mc = np.zeros((len(python_samples), len(faults)))
        for sample_index, sample in enumerate(python_samples):
            sample_strho = calcST(sample[0], sample[1], sample[2], sample[3], sample[4], 9.81, sample[5], sample[6])
            for well in wells:
                distance_m = np.hypot(fault_x - well.x_km, fault_y - well.y_km) * 1000.0
                python_mc[sample_index] += python_pressure(distance_m, well, sample_strho, args.year)

    matlab = run_matlab(args, faults, wells, strho, selected_well, matlab_samples)
    radial = compare_frame(radial_km, python_radial, matlab["radial"][:, 1], "distance_km")
    fault = compare_frame(faults["FaultID"].astype(str), python_fault, np.ravel(matlab["fault"]), "fault_id")
    radial_path = args.output_dir / "radial_pressure_comparison.csv"
    fault_path = args.output_dir / "fault_pressure_comparison.csv"
    plot_path = write_plot(radial, args.output_dir / "pressure_distance_side_by_side.html", selected_well.well_id, args.year)
    radial.to_csv(radial_path, index=False)
    fault.to_csv(fault_path, index=False)

    comparisons = [radial, fault]
    outputs = {"radial_csv": str(radial_path.resolve()), "fault_csv": str(fault_path.resolve()), "plot_html": str(plot_path.resolve())}
    if python_mc is not None:
        matlab_mc = matlab["mc"]
        rows = []
        cdf_rows = []
        for simulation_id in range(len(python_mc)):
            frame = compare_frame(faults["FaultID"].astype(str), python_mc[simulation_id], matlab_mc[simulation_id], "fault_id")
            frame.insert(0, "simulation_id", simulation_id + 1)
            rows.append(frame)
        mc_comparison = pd.concat(rows, ignore_index=True)
        mc_path = args.output_dir / "mc_fault_pressure_comparison.csv"
        mc_comparison.to_csv(mc_path, index=False)
        comparisons.append(mc_comparison)
        for fault_index, fault_id in enumerate(faults["FaultID"].astype(str)):
            python_sorted = np.sort(python_mc[:, fault_index])
            matlab_sorted = np.sort(matlab_mc[:, fault_index])
            ranks = np.arange(1, len(python_sorted) + 1)
            cdf_rows.append(pd.DataFrame({
                "fault_id": fault_id, "rank": ranks,
                "exceedance_probability": 1.0 - ranks / len(ranks),
                "python_pressure_psi": python_sorted,
                "matlab_pressure_psi": matlab_sorted,
            }))
        cdf_path = args.output_dir / "mc_cdf_comparison.csv"
        pd.concat(cdf_rows, ignore_index=True).to_csv(cdf_path, index=False)
        outputs.update({"mc_comparison_csv": str(mc_path.resolve()), "mc_cdf_csv": str(cdf_path.resolve())})

    passed = all(validate(frame) for frame in comparisons)
    max_relative_error = max(float(frame["relative_error"].max()) for frame in comparisons)
    summary = {
        "passed": passed,
        "max_relative_error": max_relative_error,
        "tolerance": {"relative_at_least_1_psi": 0.01, "absolute_below_1_psi": 0.01},
        "inputs": {
            "year": args.year, "well_id": selected_well.well_id,
            "porosity_fraction_python": args.porosity_fraction,
            "porosity_percent_matlab": args.porosity_fraction * 100.0,
            "projection_origin_wgs84": {"latitude": origin_lat, "longitude": origin_lon},
            "random_seed": args.random_seed, "mc_iterations": args.mc_iterations,
            "mc_uncertainty": uncertainty,
        },
        "outputs": outputs,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Pass: {passed}")
    print(f"Maximum relative error: {max_relative_error:.6%}")
    print(f"Summary: {summary_path.resolve()}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
 