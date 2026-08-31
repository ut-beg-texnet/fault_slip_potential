#!/usr/bin/env python
"""Run a MATLAB R2012b versus Python hydrology regression comparison.

Each side independently converts the injection CSV (monthly or annual) with
its production preprocessor, then compares daily rates and Theis pressure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
from fsp.io.wells import load_injection_wells, normalize_wells_to_well_data, preprocess_well_data

# Change these based on your setup
DEFAULT_FAULTS = ROOT / "examples" / "demo_texas_faults_fsp_100_variable_fsp.csv"
DEFAULT_WELLS = ROOT / "examples" / "demo_texas_injection_wells_monthly_fsp_20wells_variable_fsp.csv"
DEFAULT_MATLAB = Path(r"C:\Program Files\MATLAB\R2012b\bin\matlab.exe")
DEFAULT_MATLAB_CODE = ROOT / "reference_old_code"
KM_PER_DEG_LAT = 111.0
MONTHLY_VOLUME_COLUMNS = (
    "InjectionRate(bbl/month)",
    "Injection Rate (bbl/month)",
    "MonthlyInjectionRate",
)
RATE_ABS_LIMIT = 0.01
# MATLAB datenum(1,1,1) = 367 = Python date.toordinal() + 366 (year 0 is 366 days).
_MATLAB_ORDINAL_OFFSET = 366


@dataclass
class WellSeries:
    """Python-preprocessed well plus shared Cartesian location."""
    well_id: str
    latitude: float
    longitude: float
    x_km: float
    y_km: float
    start_date: date
    days: np.ndarray
    rates_bpd: np.ndarray
    csv_start_year: int = 0
    csv_end_year: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", type=Path, default=DEFAULT_FAULTS)
    parser.add_argument("--wells", type=Path, default=DEFAULT_WELLS)
    parser.add_argument(
        "--wells-format",
        choices=("auto", "monthly_fsp", "annual_fsp"),
        default="auto",
        help="Injection CSV format. auto detects annual vs monthly from columns.",
    )
    parser.add_argument(
        "--extrapolate-injection-rates",
        action="store_true",
        help="Monthly only: continue the last listed monthly rate to the cutoff "
             "(MATLAB 'Extrapolate Injection?' checkbox). Ignored for annual.",
    )
    parser.add_argument("--matlab-executable", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument(
        "--matlab-code",
        "--matlab_code",
        dest="matlab_code",
        type=Path,
        default=DEFAULT_MATLAB_CODE,
        help="Root folder of the legacy MATLAB FSP code (must contain 'technical code').",
    )
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


def monthly_volume_column(df: pd.DataFrame) -> str:
    """Return the monthly volume column name used by production well loading."""
    for col in MONTHLY_VOLUME_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        "Monthly wells CSV is missing a volume column "
        f"(tried {', '.join(MONTHLY_VOLUME_COLUMNS)})."
    )


def detect_wells_format(df: pd.DataFrame, explicit: str) -> str:
    """Resolve annual_fsp vs monthly_fsp from --wells-format or CSV columns."""
    if explicit != "auto":
        return explicit
    annual_cols = {"StartYear", "EndYear", "InjectionRate(bbl/day)"}
    if annual_cols.issubset(df.columns):
        return "annual_fsp"
    if {"Year", "Month"}.issubset(df.columns):
        monthly_volume_column(df)
        return "monthly_fsp"
    raise ValueError(
        "Could not detect wells format. Need annual columns "
        "StartYear, EndYear, InjectionRate(bbl/day) or monthly Year, Month, "
        "and a bbl/month volume column. Pass --wells-format."
    )


def local_xy_km(latitudes, longitudes, origin_lat: float, origin_lon: float):
    """Convert WGS84 locations to one shared local Cartesian grid."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    x_km = (longitudes - origin_lon) * KM_PER_DEG_LAT * np.cos(np.radians(origin_lat))
    y_km = (latitudes - origin_lat) * KM_PER_DEG_LAT
    return x_km, y_km


def csv_year_lookup(inj_df: pd.DataFrame, year_col: str) -> dict[str, int]:
    """First listed StartYear/EndYear per well (annual CSV, uncapped by cutoff)."""
    lookup = {}
    for well_id, group in inj_df.groupby(inj_df["WellID"].astype(str), sort=False):
        lookup[str(well_id)] = int(group.iloc[0][year_col])
    return lookup


def processed_to_series(
    processed, origin_lat: float, origin_lon: float,
    start_years: dict[str, int], end_years: dict[str, int],
) -> list[WellSeries]:
    """Attach Cartesian km to production ProcessedWellData."""
    series = []
    for wd in processed:
        x_km, y_km = local_xy_km([wd.latitude], [wd.longitude], origin_lat, origin_lon)
        series.append(WellSeries(
            well_id=wd.well_id,
            latitude=wd.latitude,
            longitude=wd.longitude,
            x_km=float(x_km[0]),
            y_km=float(y_km[0]),
            start_date=wd.start_date,
            days=np.asarray(wd.days, dtype=float),
            rates_bpd=np.asarray(wd.rates, dtype=float),
            csv_start_year=start_years.get(wd.well_id, wd.start_date.year),
            csv_end_year=end_years.get(wd.well_id, wd.end_date.year + 1),
        ))
    return series


def python_pressure(distances_m: np.ndarray, well: WellSeries, strho, year: int) -> np.ndarray:
    evaluation_days = float((date(year - 1, 12, 31) - well.start_date).days + 1)
    return pressureScenario_Rall(well.rates_bpd, well.days, distances_m, strho, evaluation_days)


def matlab_vector(values) -> str:
    return "[" + ";".join(f"{float(value):.16g}" for value in values) + "]"


def matlab_cellstr(values) -> str:
    return "{" + ";".join("'" + str(value).replace("'", "''") + "'" for value in values) + "}"


def matlab_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def resolve_matlab_folders(matlab_code: Path, wells_format: str) -> tuple[Path, Path | None]:
    """Return (technical code, support code or None). Support is required for monthly."""
    technical_code = matlab_code / "technical code"
    if not technical_code.is_dir():
        raise FileNotFoundError(
            f"MATLAB technical code was not found at {technical_code}. "
            "Pass --matlab-code pointing at the MATLAB FSP root "
            "(the folder that contains a 'technical code' subdirectory)."
        )
    missing = [name for name in ("pfront.m", "calcST.m") if not (technical_code / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MATLAB technical code at {technical_code} is missing: {', '.join(missing)}."
        )
    support_code = None
    if wells_format == "monthly_fsp":
        support_code = matlab_code / "support code"
        converter = support_code / "SpreadsheetStrings2WellData.m"
        if not converter.is_file():
            raise FileNotFoundError(
                f"MATLAB SpreadsheetStrings2WellData.m was not found at {converter}. "
                "Monthly preprocessing needs the MATLAB FSP 'support code' folder."
            )
    return technical_code, support_code


def write_gui_stubs(output: Path) -> None:
    """Headless stand-ins so SpreadsheetStrings2WellData can run under -nojvm.

    SpreadsheetStrings2WellData mutates hDV in place, so hDV must be a handle
    (the GUI passes a class instance). A plain struct would drop the results.
    """
    (output / "msgbox.m").write_text("function h = msgbox(varargin)\nh = 1;\nend\n", encoding="utf-8")
    (output / "errordlg.m").write_text("function h = errordlg(varargin)\nh = 1;\nend\n", encoding="utf-8")
    (output / "centerFigure.m").write_text("function centerFigure(varargin)\nend\n", encoding="utf-8")
    (output / "RegressionHDV.m").write_text(
        "classdef RegressionHDV < handle\n    properties\n        data\n        hfig\n    end\nend\n",
        encoding="utf-8",
    )


def write_matlab_monthly_csv(
    path: Path, inj_df: pd.DataFrame, origin_lat: float, origin_lon: float,
) -> None:
    """MATLAB monthly layout: UniqueID, Easting km, Northing km, Year, Month, bbl/month."""
    rate_col = monthly_volume_column(inj_df)
    x_km, y_km = local_xy_km(
        inj_df["Latitude(WGS84)"], inj_df["Longitude(WGS84)"], origin_lat, origin_lon,
    )
    pd.DataFrame({
        "UniqueID/Name": inj_df["WellID"].astype(str),
        "Easting (km)": x_km,
        "Northing (km)": y_km,
        "Year": inj_df["Year"].astype(int),
        "Month (1-12)": inj_df["Month"].astype(int),
        "InjectionVolume (bbl/month)": pd.to_numeric(inj_df[rate_col], errors="coerce"),
    }).to_csv(path, index=False)


def dump_python_well_rates(path: Path, wells: list[WellSeries]) -> None:
    """Raw production days/rates for debugging."""
    rows = []
    for well in wells:
        for day, rate in zip(well.days, well.rates_bpd):
            rows.append({
                "well_id": well.well_id,
                "start_date": well.start_date.isoformat(),
                "day": float(day),
                "rate_bbl_day": float(rate),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


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


def _matlab_well_prep(
    args: argparse.Namespace, wells: list[WellSeries], wells_format: str,
    wells_csv: Path | None,
) -> str:
    """MATLAB production conversion: SpreadsheetStrings2WellData or annual 2-row constructor."""
    extrapolate = 1 if (args.extrapolate_injection_rates and wells_format == "monthly_fsp") else 0
    if wells_format == "monthly_fsp":
        if wells_csv is None:
            raise ValueError("Monthly MATLAB prep requires matlab_wells_input.csv.")
        return f"""
fid_wells = fopen('{matlab_path(wells_csv)}');
readInData = textscan(fid_wells,'%s%s%s%s%s%s','Delimiter',',','Headerlines',1);
fclose(fid_wells);
hDV = RegressionHDV();
hDV.hfig = [];
hDV.data = struct();
hDV.data.nwells_max = 10000;
hDV.data.realWellData = struct();
hDV.data.realWellData.columnIsNumber = logical([0,1,1,1,1,1]);
hDV.data.realWellData.stringsWellDataAdvanced = [readInData{{:,:}}];
hDV.data.realWellData.extrapolateInjectionCheck = 0;
SpreadsheetStrings2WellData(hDV.data.realWellData.stringsWellDataAdvanced, hDV);
if {extrapolate}
    for k = 1:hDV.data.nwells
        hDV.data.realWellData.datenumBarrelsPerDay{{k}}(end,1) = datenum({args.year},1,1,0,1,0);
    end
end
nwells = hDV.data.nwells;
well_x = hDV.data.realWellData.XEasting;
well_y = hDV.data.realWellData.YNorthing;
well_names = hDV.data.realWellData.wellNames;
d = hDV.data.realWellData.datenumBarrelsPerDay;
"""

    ids = matlab_cellstr(well.well_id for well in wells)
    xs = matlab_vector(well.x_km for well in wells)
    ys = matlab_vector(well.y_km for well in wells)
    start_years = matlab_vector(well.csv_start_year for well in wells)
    end_years = matlab_vector(well.csv_end_year for well in wells)
    rates = matlab_vector(well.rates_bpd[0] if len(well.rates_bpd) else 0.0 for well in wells)
    return f"""
well_names = {ids};
well_x = {xs};
well_y = {ys};
start_years = {start_years};
end_years = {end_years};
well_rates = {rates};
nwells = length(well_x);
d = cell(nwells,1);
for k = 1:nwells
    % dataentryWells.m constant-rate constructor
    d{{k}} = [datenum(start_years(k),1,1,0,1,0), 0; datenum(end_years(k),1,1,0,1,0), well_rates(k)];
end
"""


def run_matlab(
    args: argparse.Namespace, faults: pd.DataFrame, wells: list[WellSeries],
    selected_well: WellSeries, samples_percent: np.ndarray | None,
    wells_format: str, wells_csv: Path | None,
) -> dict[str, np.ndarray]:
    """Generate and run an R2012b driver that converts wells then calls pfront."""
    if not args.matlab_executable.exists():
        raise FileNotFoundError(f"MATLAB R2012b was not found: {args.matlab_executable}")
    technical_code, support_code = resolve_matlab_folders(args.matlab_code, wells_format)

    output = args.output_dir
    write_gui_stubs(output)
    radial_file = output / "matlab_radial_pressure.csv"
    fault_file = output / "matlab_fault_pressure.csv"
    rates_file = output / "matlab_well_rates.csv"
    mc_file = output / "matlab_mc_fault_pressure.csv"
    script_file = output / "matlab_hydrology_regression_driver.m"
    error_file = output / "matlab_error.txt"
    logfile = output / "matlab_run.log"
    required = [radial_file, fault_file, rates_file] + ([mc_file] if samples_percent is not None else [])
    _remove_stale_outputs(required + [error_file, logfile])

    fault_x = faults["x_km"].to_numpy(float)
    fault_y = faults["y_km"].to_numpy(float)
    radial_km = np.linspace(0.1, 20.0, args.radial_points)
    porosity_percent = args.porosity_fraction * 100.0
    well_prep = _matlab_well_prep(args, wells, wells_format, wells_csv)

    addpath_support = ""
    if support_code is not None:
        addpath_support = f"addpath('{matlab_path(support_code)}');"

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
    for k = 1:nwells
        r_fault = sqrt((fault_x - well_x(k)).^2 + (fault_y - well_y(k)).^2) .* 1000;
        fault_pressure = fault_pressure + pfront(r_fault,{args.year},d{{k}},S,T,rho,9.81) .* 14.5;
    end
    mc_pressure(sample_index,:) = fault_pressure';
end
csvwrite('{matlab_path(mc_file)}',mc_pressure);
"""

    script = f"""try
addpath('{matlab_path(output)}');
{addpath_support}
addpath('{matlab_path(technical_code)}');
fault_x = {matlab_vector(fault_x)};
fault_y = {matlab_vector(fault_y)};
{well_prep}
selected_index = 1;
for k = 1:nwells
    if strcmp(char(well_names{{k}}), {matlab_string(selected_well.well_id)})
        selected_index = k;
        break;
    end
end
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
for k = 1:nwells
    r_fault = sqrt((fault_x - well_x(k)).^2 + (fault_y - well_y(k)).^2) .* 1000;
    fault_pressure = fault_pressure + pfront(r_fault,{args.year},d{{k}},S,T,rho,9.81) .* 14.5;
end
csvwrite('{matlab_path(fault_file)}',fault_pressure);
radial_km = {matlab_vector(radial_km)};
radial_pressure = pfront(radial_km .* 1000,{args.year},d{{selected_index}},S,T,rho,9.81) .* 14.5;
csvwrite('{matlab_path(radial_file)}',[radial_km,radial_pressure]);
fid_rates = fopen('{matlab_path(rates_file)}','w');
fprintf(fid_rates,'well_id,datenum,rate_bbl_day\\n');
for k = 1:nwells
    series = d{{k}};
    name = char(well_names{{k}});
    for r = 1:size(series,1)
        fprintf(fid_rates,'%s,%.16g,%.16g\\n',name,series(r,1),series(r,2));
    end
end
fclose(fid_rates);
{mc_setup}
{mc_loop}
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

    result = {
        "radial": np.atleast_2d(np.loadtxt(radial_file, delimiter=",")),
        "fault": np.atleast_1d(np.loadtxt(fault_file, delimiter=",")),
        "rates": pd.read_csv(rates_file, dtype={"well_id": str}),
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


def matlab_datenum(value: date, hour: int = 0, minute: int = 0, second: int = 0) -> float:
    """Python equivalent of MATLAB datenum(Y,M,D,H,MI,S)."""
    fraction = (hour * 3600 + minute * 60 + second) / 86400.0
    return float(datetime(value.year, value.month, value.day).toordinal() + _MATLAB_ORDINAL_OFFSET + fraction)


def python_rate_on_offset(
    days: np.ndarray, rates: np.ndarray, offset: float, annual_hold_last_day: bool,
) -> float:
    """Start-of-interval stair: rate[i] from days[i] until days[i+1].

    Annual production places a trailing 0 on the last day of injection (Theis
    endpoint). MATLAB shuts in at EndYear-01-01 00:01, so that last calendar
    day still injects. Shift the trailing 0 to the next day for comparison.
    """
    days = np.asarray(days, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if annual_hold_last_day and len(rates) >= 2 and rates[-1] == 0.0:
        days = days.copy()
        days[-1] = days[-1] + 1.0
    idx = int(np.searchsorted(days, offset, side="right") - 1)
    if idx < 0:
        return 0.0
    return float(rates[idx])


def matlab_rate_at_noon(datenums: np.ndarray, rates: np.ndarray, the_day: date) -> float:
    """End-of-interval stair: rate[i] applies until datenum[i]; sample at noon."""
    noon = matlab_datenum(the_day, 12, 0, 0)
    datenums = np.asarray(datenums, dtype=float)
    rates = np.asarray(rates, dtype=float)
    idx = int(np.searchsorted(datenums, noon, side="left"))
    if idx >= len(datenums):
        return 0.0
    return float(rates[idx])


def compare_daily_rates(
    wells: list[WellSeries], matlab_rates: pd.DataFrame, cutoff: date, wells_format: str,
) -> pd.DataFrame:
    """Sample both stairs onto each calendar day from well start through cutoff."""
    matlab_rates = matlab_rates.copy()
    matlab_rates["well_id"] = matlab_rates["well_id"].astype(str)
    grouped = {well_id: group for well_id, group in matlab_rates.groupby("well_id", sort=False)}
    annual_hold = wells_format == "annual_fsp"
    rows = []
    missing = []
    for well in wells:
        if well.well_id not in grouped:
            missing.append(well.well_id)
            continue
        group = grouped[well.well_id]
        datenums = group["datenum"].to_numpy(float)
        matlab_bpd = group["rate_bbl_day"].to_numpy(float)
        current = well.start_date
        while current <= cutoff:
            offset = float((current - well.start_date).days + 1)
            python_bpd = python_rate_on_offset(well.days, well.rates_bpd, offset, annual_hold)
            matlab_value = matlab_rate_at_noon(datenums, matlab_bpd, current)
            rows.append({
                "well_id": well.well_id,
                "date": current.isoformat(),
                "python_rate_bbl_day": python_bpd,
                "matlab_rate_bbl_day": matlab_value,
                "absolute_error_bbl_day": abs(python_bpd - matlab_value),
            })
            current += timedelta(days=1)
    if missing:
        raise ValueError(
            "MATLAB well conversion did not return series for: "
            + ", ".join(missing)
        )
    return pd.DataFrame(rows)


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


def validate_pressure(comparison: pd.DataFrame) -> bool:
    significant = comparison["matlab_pressure_psi"].abs() >= 1.0
    relative_pass = (comparison.loc[significant, "relative_error"] <= 0.01).all()
    absolute_pass = (comparison.loc[~significant, "absolute_error_psi"] <= 0.01).all()
    return bool(relative_pass and absolute_pass)


def validate_rates(comparison: pd.DataFrame) -> bool:
    if comparison.empty:
        return False
    return bool((comparison["absolute_error_bbl_day"] <= RATE_ABS_LIMIT).all())


def run() -> int:
    args = parse_args()
    if not 0.0 < args.porosity_fraction < 1.0:
        raise ValueError("--porosity-fraction must be greater than 0 and less than 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    faults = pd.read_csv(args.faults, dtype={"FaultID": str})
    peek = pd.read_csv(args.wells, dtype={"WellID": str}, nrows=5)
    wells_format = detect_wells_format(peek, args.wells_format)
    resolve_matlab_folders(args.matlab_code, wells_format)

    inj_df = load_injection_wells(str(args.wells), wells_format)
    if "Latitude(WGS84)" not in inj_df.columns or "Longitude(WGS84)" not in inj_df.columns:
        raise ValueError("Wells CSV must include Latitude(WGS84) and Longitude(WGS84).")

    cutoff = date(args.year - 1, 12, 31)
    all_lats = np.concatenate([faults["Latitude(WGS84)"].to_numpy(float), inj_df["Latitude(WGS84)"].to_numpy(float)])
    all_lons = np.concatenate([faults["Longitude(WGS84)"].to_numpy(float), inj_df["Longitude(WGS84)"].to_numpy(float)])
    origin_lat, origin_lon = float(all_lats.mean()), float(all_lons.mean())
    faults["x_km"], faults["y_km"] = local_xy_km(
        faults["Latitude(WGS84)"], faults["Longitude(WGS84)"], origin_lat, origin_lon
    )

    extrapolate = bool(args.extrapolate_injection_rates) if wells_format == "monthly_fsp" else False
    processed = normalize_wells_to_well_data(
        preprocess_well_data(inj_df, wells_format),
        wells_format, cutoff,
        extrapolate_injection_rates=extrapolate,
    )
    start_years = csv_year_lookup(inj_df, "StartYear") if wells_format == "annual_fsp" else {}
    end_years = csv_year_lookup(inj_df, "EndYear") if wells_format == "annual_fsp" else {}
    wells = processed_to_series(processed, origin_lat, origin_lon, start_years, end_years)
    if not wells:
        raise ValueError("No wells are active for the requested year.")
    selected_well = next((well for well in wells if well.well_id == args.well_id), wells[0])
    if args.well_id and selected_well.well_id != args.well_id:
        raise ValueError(f"Well {args.well_id!r} is not active for {args.year}.")

    python_rates_path = args.output_dir / "python_well_rates.csv"
    dump_python_well_rates(python_rates_path, wells)

    wells_csv = None
    if wells_format == "monthly_fsp":
        wells_csv = args.output_dir / "matlab_wells_input.csv"
        write_matlab_monthly_csv(wells_csv, inj_df, origin_lat, origin_lon)

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

    matlab = run_matlab(
        args, faults, wells, selected_well, matlab_samples, wells_format, wells_csv,
    )
    rates = compare_daily_rates(wells, matlab["rates"], cutoff, wells_format)
    radial = compare_frame(radial_km, python_radial, matlab["radial"][:, 1], "distance_km")
    fault = compare_frame(faults["FaultID"].astype(str), python_fault, np.ravel(matlab["fault"]), "fault_id")
    rates_path = args.output_dir / "well_rate_series_comparison.csv"
    radial_path = args.output_dir / "radial_pressure_comparison.csv"
    fault_path = args.output_dir / "fault_pressure_comparison.csv"
    plot_path = write_plot(radial, args.output_dir / "pressure_distance_side_by_side.html", selected_well.well_id, args.year)
    rates.to_csv(rates_path, index=False)
    radial.to_csv(radial_path, index=False)
    fault.to_csv(fault_path, index=False)

    comparisons = [radial, fault]
    outputs = {
        "rate_csv": str(rates_path.resolve()),
        "python_rates_csv": str(python_rates_path.resolve()),
        "matlab_rates_csv": str((args.output_dir / "matlab_well_rates.csv").resolve()),
        "radial_csv": str(radial_path.resolve()),
        "fault_csv": str(fault_path.resolve()),
        "plot_html": str(plot_path.resolve()),
    }
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

    rates_passed = validate_rates(rates)
    pressure_passed = all(validate_pressure(frame) for frame in comparisons)
    passed = rates_passed and pressure_passed
    max_relative_error = max(float(frame["relative_error"].max()) for frame in comparisons)
    max_rate_error = float(rates["absolute_error_bbl_day"].max()) if not rates.empty else float("nan")
    summary = {
        "passed": passed,
        "rates_passed": rates_passed,
        "pressure_passed": pressure_passed,
        "max_relative_error": max_relative_error,
        "max_rate_absolute_error_bbl_day": max_rate_error,
        "tolerance": {
            "relative_at_least_1_psi": 0.01,
            "absolute_below_1_psi": 0.01,
            "rate_absolute_bbl_day": RATE_ABS_LIMIT,
        },
        "inputs": {
            "year": args.year, "well_id": selected_well.well_id,
            "wells_format": wells_format,
            "extrapolate_injection_rates": extrapolate,
            "matlab_code": str(args.matlab_code.resolve()),
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
    print(f"Rates passed: {rates_passed}")
    print(f"Pressure passed: {pressure_passed}")
    print(f"Maximum rate absolute error: {max_rate_error:.6g} bbl/day")
    print(f"Maximum relative pressure error: {max_relative_error:.6%}")
    print(f"Summary: {summary_path.resolve()}")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
