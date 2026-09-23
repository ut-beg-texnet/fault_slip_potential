#!/usr/bin/env python
"""Run a MATLAB R2012b versus Python geomechanics regression comparison."""
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
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from fsp.geomechanics.mohr import mohr_diagram_data_to_d3_portal
from fsp.geomechanics.slip import (
    calculate_cff,
    calculate_fault_effective_stresses,
    calculate_scu,
    calculate_slip_pressure,
)
from fsp.geomechanics.stress import calculate_absolute_stresses, stress_regime_label
from fsp.monte_carlo.geomechanics_mc import run_geomechanics_mc

# Change these based on your setup
DEFAULT_FAULTS = ROOT / "examples" / "demo_texas_faults_fsp_100_variable_fsp.csv"
DEFAULT_MATLAB = Path(r"C:\Program Files\MATLAB\R2012b\bin\matlab.exe")
DEFAULT_MATLAB_CODE = ROOT / "reference_old_code"

# MATLAB aphi.use: 0 = gradients, 11 = standard A-Phi, 12 = modified A-Phi (Shmin given)
STRESS_MODE_TO_APHI_USE = {
    "gradients": 0,
    "aphi_no_min": 11,
    "aphi_min": 12,
}
MC_UNCERTAINTY_KEYS = (
    "vertical_stress_gradient_uncertainty",
    "min_horizontal_stress_uncertainty",
    "max_horizontal_stress_uncertainty",
    "initial_pore_pressure_gradient_uncertainty",
    "max_stress_azimuth_uncertainty",
    "strike_angles_uncertainty",
    "dip_angles_uncertainty",
    "friction_coefficient_uncertainty",
    "aphi_value_uncertainty",
)
PSI_ABS_THRESHOLD = 1.0
PSI_ABS_LIMIT = 0.01
PSI_REL_LIMIT = 0.01
SCU_ABS_THRESHOLD = 0.01
SCU_ABS_LIMIT = 1e-4
SCU_REL_LIMIT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", type=Path, default=DEFAULT_FAULTS)
    parser.add_argument("--matlab-executable", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument(
        "--matlab-code",
        "--matlab_code",
        dest="matlab_code",
        type=Path,
        default=DEFAULT_MATLAB_CODE,
        help="Root folder of the legacy MATLAB FSP code (must contain 'technical code').",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "geomechanics_regression_output")
    parser.add_argument(
        "--stress-mode",
        choices=tuple(STRESS_MODE_TO_APHI_USE),
        default="gradients",
        help="Stress field mode. MATLAB aphi.use is 0 / 11 / 12 for these names.",
    )
    parser.add_argument("--reference-depth", type=float, default=11000.0, help="Depth in feet.")
    parser.add_argument("--vertical-stress", type=float, default=1.1, help="Sv gradient, psi/ft.")
    parser.add_argument("--min-horizontal-stress", type=float, default=0.693, help="Shmin gradient, psi/ft.")
    parser.add_argument("--max-horizontal-stress", type=float, default=1.11, help="SHmax gradient, psi/ft.")
    parser.add_argument("--pore-pressure", type=float, default=0.43, help="Pore-pressure gradient, psi/ft.")
    parser.add_argument("--max-stress-azimuth", type=float, default=70.0, help="SHmax azimuth, degrees CW from N.")
    parser.add_argument("--aphi-value", type=float, default=1.22, help="A-Phi in [0, 3]; used only for A-Phi modes.")
    parser.add_argument("--friction-coefficient", type=float, default=0.58, help="Fault friction mu (one value for all faults).")
    parser.add_argument("--poissons-ratio", type=float, default=0.5, help="Poisson's ratio; MATLAB geo tab default is 0.5.")
    parser.add_argument("--mc-iterations", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument(
        "--mc-uncertainty-json",
        default="{}",
        help="JSON object of plus/minus half-widths in Python units, "
             "e.g. '{\"strike_angles_uncertainty\": 10, \"friction_coefficient_uncertainty\": 0.05}'.",
    )
    return parser.parse_args()


def matlab_vector(values) -> str:
    return "[" + ";".join(f"{float(value):.16g}" for value in np.atleast_1d(values)) + "]"


def matlab_row(values) -> str:
    """Row vector. mohrs_3D's Mohr kron() path requires Sig0 as 1x3, not 3x1."""
    return "[" + ",".join(f"{float(value):.16g}" for value in np.atleast_1d(values)) + "]"


def matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def resolve_matlab_folders(matlab_code: Path, stress_mode: str) -> tuple[Path, Path]:
    """Return the MATLAB technical-code and data-entry folders."""
    technical_code = matlab_code / "technical code"
    data_entry = matlab_code / "data entry functions"
    if not technical_code.is_dir():
        raise FileNotFoundError(
            f"MATLAB technical code was not found at {technical_code}. "
            "Pass --matlab-code pointing at the MATLAB FSP root "
            "(the folder that contains a 'technical code' subdirectory)."
        )
    if not (technical_code / "mohrs_3D.m").is_file():
        raise FileNotFoundError(f"MATLAB technical code at {technical_code} is missing mohrs_3D.m.")
    if stress_mode != "gradients":
        if not data_entry.is_dir() or not (data_entry / "getHorFromAPhi.m").is_file():
            raise FileNotFoundError(
                f"A-Phi mode requires getHorFromAPhi.m under {data_entry}."
            )
    return technical_code, data_entry


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


def load_faults(path: Path) -> pd.DataFrame:
    faults = pd.read_csv(path, dtype={"FaultID": str})
    required = {"FaultID", "Strike", "Dip"}
    missing = required.difference(faults.columns)
    if missing:
        raise ValueError(f"Fault CSV is missing: {', '.join(sorted(missing))}")
    if faults.empty:
        raise ValueError("Fault CSV has no rows.")
    return faults


def stress_inputs_from_args(args: argparse.Namespace) -> dict:
    """Build the dict expected by calculate_absolute_stresses."""
    inputs = {
        "reference_depth": args.reference_depth,
        "vertical_stress": args.vertical_stress,
        "min_horizontal_stress": args.min_horizontal_stress,
        "max_horizontal_stress": args.max_horizontal_stress,
        "pore_pressure": args.pore_pressure,
        "max_stress_azimuth": args.max_stress_azimuth,
        "aphi_value": args.aphi_value,
    }
    # A-Phi modes must not also supply the derived horizontal gradient(s).
    if args.stress_mode == "aphi_no_min":
        inputs["min_horizontal_stress"] = None
        inputs["max_horizontal_stress"] = None
    elif args.stress_mode == "aphi_min":
        inputs["max_horizontal_stress"] = None
    return inputs


def validate_stress_args(args: argparse.Namespace) -> None:
    if args.friction_coefficient <= 0:
        raise ValueError("--friction-coefficient must be greater than 0.")
    if not 0.0 < args.poissons_ratio < 1.0:
        raise ValueError("--poissons-ratio must be greater than 0 and less than 1.")
    if args.stress_mode != "gradients" and not (0.0 <= args.aphi_value <= 3.0):
        raise ValueError("--aphi-value must be in [0, 3] for A-Phi modes.")
    if args.mc_iterations < 0:
        raise ValueError("--mc-iterations must be >= 0.")


def python_step2_metrics(strike, dip, stress_state, p0, mu):
    """Deterministic metrics using the same calls as fsp_step2.py."""
    sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(
        strike, dip, stress_state, p0, 0.0
    )
    # Step 2 clamps PPF at 0, integer-rounds CFF, and uses calculate_scu (clamped to 1).
    ppf = np.maximum(
        calculate_slip_pressure(sig, tau, float(mu), p0, 1.0, 0.5, 0.0,
                                s11, s22, s33, s12, n1, n2),
        0.0,
    )
    cff = np.round(calculate_cff(sig, tau, float(mu)))
    scu = calculate_scu(sig, tau, float(mu))
    return sig, tau, ppf, cff, scu


def parse_mc_uncertainty(args: argparse.Namespace) -> dict:
    if args.mc_iterations <= 0:
        return {}
    uncertainty = json.loads(args.mc_uncertainty_json)
    if not isinstance(uncertainty, dict):
        raise ValueError("--mc-uncertainty-json must be a JSON object.")
    return {key: float(uncertainty.get(key, 0.0)) for key in MC_UNCERTAINTY_KEYS}


def matlab_samples_from_production(samples_df: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    """Convert run_geomechanics_mc sample inputs to mohrs_3D indatacell absolute units."""
    depth = args.reference_depth
    n_rows = len(samples_df)
    sv = samples_df["vertical_stress_gradient"].to_numpy(float) * depth
    p0 = samples_df["initial_pore_pressure_gradient"].to_numpy(float) * depth
    if "min_horizontal_stress_gradient" in samples_df.columns:
        sh = samples_df["min_horizontal_stress_gradient"].to_numpy(float) * depth
    else:
        sh = np.full(n_rows, args.min_horizontal_stress * depth)
    if "max_horizontal_stress_gradient" in samples_df.columns:
        sH = samples_df["max_horizontal_stress_gradient"].to_numpy(float) * depth
    else:
        sH = np.full(n_rows, args.max_horizontal_stress * depth)
    columns = [
        sv, sh, sH, p0,
        samples_df["strike_angle"].to_numpy(float),
        samples_df["dip_angle"].to_numpy(float),
        samples_df["max_stress_azimuth"].to_numpy(float),
        samples_df["friction_coefficient"].to_numpy(float),
    ]
    if "aphi_value" in samples_df.columns:
        columns.append(samples_df["aphi_value"].to_numpy(float))
    return np.column_stack(columns)


def run_matlab(args: argparse.Namespace, matlab_sig: np.ndarray, p0: float,
               strikes: np.ndarray, dips: np.ndarray, samples: np.ndarray | None) -> dict[str, np.ndarray]:
    """Generate and run a self-contained R2012b mohrs_3D driver."""
    if not args.matlab_executable.exists():
        raise FileNotFoundError(f"MATLAB R2012b was not found: {args.matlab_executable}")
    technical_code, data_entry = resolve_matlab_folders(args.matlab_code, args.stress_mode)

    output = args.output_dir
    metrics_file = output / "matlab_fault_metrics.csv"
    mohr_file = output / "matlab_mohr_arcs.csv"
    mc_file = output / "matlab_mc_slip_pressure.csv"
    script_file = output / "matlab_geomechanics_regression_driver.m"
    required = [metrics_file, mohr_file] + ([mc_file] if samples is not None else [])
    _remove_stale_outputs(required)

    aphi_use = STRESS_MODE_TO_APHI_USE[args.stress_mode]
    aphi_cell = ""
    if args.stress_mode != "gradients":
        aphi_cell = f", {args.aphi_value:.16g}"

    addpath_data_entry = ""
    if data_entry.is_dir():
        addpath_data_entry = f"addpath('{matlab_path(data_entry)}');"

    mc_setup = ""
    mc_loop = ""
    if samples is not None:
        sample_file = output / "mc_samples.csv"
        np.savetxt(sample_file, samples, delimiter=",", fmt="%.16g")
        # A-Phi: 10-cell kernel plus 11th sample; gradients must not pass cell 11.
        extra_cell = ",0" if args.stress_mode != "gradients" else ""
        aphi_mc = "inputCell{11} = samples(sample_index,9);" if args.stress_mode != "gradients" else ""
        mc_setup = f"samples = csvread('{matlab_path(sample_file)}'); mc_ppf = zeros(size(samples,1),1);"
        mc_loop = f"""
for sample_index = 1:size(samples,1)
    Sig0 = samples(sample_index,1:3)';
    p0_mc = samples(sample_index,4);
    strike_mc = samples(sample_index,5);
    dip_mc = samples(sample_index,6);
    SHdir_mc = samples(sample_index,7);
    mu_mc = samples(sample_index,8);
    inputCell = {{Sig0,0.00,p0_mc,strike_mc,dip_mc,SHdir_mc,0,mu_mc,biot,nu{extra_cell}}};
    {aphi_mc}
    mc_ppf(sample_index) = mohrs_3D(inputCell,hDV);
end
csvwrite('{matlab_path(mc_file)}',mc_ppf);
"""

    error_file = output / "matlab_error.txt"
    logfile = output / "matlab_run.log"
    _remove_stale_outputs([error_file, logfile])
    # try/catch so an error cannot leave R2012b sitting at the prompt until our timeout.
    script = f"""try
addpath('{matlab_path(technical_code)}');
{addpath_data_entry}
hDV = struct();
hDV.data.stress.aphi.use = {aphi_use};
hDV.hfig = [];
sig = {matlab_row(matlab_sig)};
pp0 = {p0:.16g};
strikes = {matlab_vector(strikes)};
dips = {matlab_vector(dips)};
SHdir = {args.max_stress_azimuth:.16g};
mu = {args.friction_coefficient:.16g};
biot = 1;
nu = {args.poissons_ratio:.16g};
inputCell = {{sig,0.00,pp0,strikes,dips,SHdir,0*strikes,mu,biot,nu{aphi_cell}}};
[failout,outs,C1,C2,C3,sig_fault,tau_fault] = mohrs_3D(inputCell,hDV);
csvwrite('{matlab_path(metrics_file)}',real([failout(:),outs.cff(:),outs.scu(:),sig_fault(:),tau_fault(:)]));
csvwrite('{matlab_path(mohr_file)}',[real(C1(1,:)).',imag(C1(1,:)).',real(C2(1,:)).',imag(C2(1,:)).',real(C3(1,:)).',imag(C3(1,:)).']);
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
    # -wait keeps matlab.exe attached until the script's exit (R2012b Windows).
    # The outer try/catch/exit covers a failure inside run() itself.
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
    timeout_s = 1800 if samples is not None else 300
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
        matlab_output = (completed.stdout + "\n" + completed.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        matlab_output = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        raise RuntimeError(f"MATLAB R2012b timed out after {timeout_s}s. {matlab_output}") from exc
    if error_file.exists():
        raise RuntimeError(f"MATLAB R2012b driver failed: {error_file.read_text(encoding='utf-8', errors='replace').strip()}")
    if not _wait_for_fresh_outputs(required, started_at):
        log_text = logfile.read_text(encoding="utf-8", errors="replace") if logfile.exists() else ""
        raise RuntimeError(f"MATLAB R2012b did not produce all outputs. {matlab_output}\n{log_text}")

    result = {
        "metrics": np.loadtxt(metrics_file, delimiter=",", ndmin=2),
        "mohr": np.loadtxt(mohr_file, delimiter=",", ndmin=2),
    }
    if samples is not None:
        result["mc"] = np.atleast_1d(np.loadtxt(mc_file, delimiter=","))
    return result


def series_errors(python_values, matlab_values, rel_floor: float) -> tuple[np.ndarray, np.ndarray]:
    python_values = np.asarray(python_values, dtype=float).reshape(-1)
    matlab_values = np.asarray(matlab_values, dtype=float).reshape(-1)
    absolute_error = np.abs(python_values - matlab_values)
    relative_error = absolute_error / np.maximum(np.abs(matlab_values), rel_floor)
    return absolute_error, relative_error


def validate_series(matlab_values, absolute_error, relative_error,
                    abs_threshold: float, abs_limit: float, rel_limit: float) -> bool:
    matlab_values = np.asarray(matlab_values, dtype=float).reshape(-1)
    absolute_error = np.asarray(absolute_error, dtype=float).reshape(-1)
    relative_error = np.asarray(relative_error, dtype=float).reshape(-1)
    significant = np.abs(matlab_values) >= abs_threshold
    relative_pass = bool((relative_error[significant] <= rel_limit).all()) if significant.any() else True
    absolute_pass = bool((absolute_error[~significant] <= abs_limit).all()) if (~significant).any() else True
    return relative_pass and absolute_pass


def ensure_length(label: str, *arrays) -> None:
    lengths = [len(np.asarray(array).reshape(-1)) for array in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"{label} comparison length mismatch: {lengths}. "
            "Stale MATLAB CSVs from a previous run can cause this."
        )


def write_mohr_plot(arcs: pd.DataFrame, points: pd.DataFrame, output: Path, mu: float) -> Path:
    figure = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
                           subplot_titles=("Python", "MATLAB R2012b"))
    circle_colors = {"circle1": "#1d4ed8", "circle2": "#2563eb", "circle3": "#93c5fd"}
    matlab_colors = {"circle1": "#b91c1c", "circle2": "#dc2626", "circle3": "#fca5a5"}
    for circle_id, color in circle_colors.items():
        subset = arcs[arcs["circle_id"] == circle_id]
        figure.add_trace(go.Scatter(x=subset["python_x_psi"], y=subset["python_y_psi"],
                                    mode="lines", line={"color": color}, name=circle_id,
                                    showlegend=False), row=1, col=1)
    for circle_id, color in matlab_colors.items():
        subset = arcs[arcs["circle_id"] == circle_id]
        figure.add_trace(go.Scatter(x=subset["matlab_x_psi"], y=subset["matlab_y_psi"],
                                    mode="lines", line={"color": color}, name=circle_id,
                                    showlegend=False), row=1, col=2)
    x_max = float(np.nanmax([arcs["python_x_psi"].max(), arcs["matlab_x_psi"].max(), 1.0]))
    line_x = np.array([0.0, x_max * 1.05])
    line_y = mu * line_x
    figure.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                                line={"color": "#111827", "dash": "dash"}, showlegend=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                                line={"color": "#111827", "dash": "dash"}, showlegend=False), row=1, col=2)
    figure.add_trace(go.Scatter(x=points["python_sigma_psi"], y=points["python_tau_psi"],
                                mode="markers", marker={"color": "#1d4ed8", "size": 7},
                                showlegend=False), row=1, col=1)
    figure.add_trace(go.Scatter(x=points["matlab_sigma_psi"], y=points["matlab_tau_psi"],
                                mode="markers", marker={"color": "#b91c1c", "size": 7},
                                showlegend=False), row=1, col=2)
    figure.update_xaxes(title_text="Effective normal stress [psi]")
    figure.update_yaxes(title_text="Shear stress [psi]", rangemode="tozero")
    figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
    figure.update_layout(title="Mohr diagram: Python vs MATLAB R2012b",
                         template="plotly_white", width=1200, height=560)
    figure.write_html(output, include_plotlyjs=True)
    return output


def run() -> int:
    args = parse_args()
    validate_stress_args(args)
    resolve_matlab_folders(args.matlab_code, args.stress_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    faults = load_faults(args.faults)
    strikes = faults["Strike"].to_numpy(float)
    dips = faults["Dip"].to_numpy(float)
    fault_ids = faults["FaultID"].astype(str)
    mu = args.friction_coefficient
    nu = args.poissons_ratio
    biot = 1.0

    stress_state, p0 = calculate_absolute_stresses(
        stress_inputs_from_args(args), mu, args.stress_mode
    )
    sV, sh, sH = (float(value) for value in stress_state.principal_stresses)
    sig, tau, ppf, cff, scu = python_step2_metrics(strikes, dips, stress_state, p0, mu)
    arcs_df, _, _ = mohr_diagram_data_to_d3_portal(
        sh, sH, sV, tau, sig, p0, biot, 0.5, 0.0, strikes, mu,
        stress_regime_label(
            args.vertical_stress,
            args.min_horizontal_stress,
            args.max_horizontal_stress,
            aphi=args.aphi_value if args.stress_mode != "gradients" else None,
        ),
        ppf, fault_ids.tolist(),
    )
    python_arcs = arcs_df[arcs_df["id"].isin(("circle1", "circle2", "circle3"))].copy()

    # MATLAB still receives a 3-element Sig0; A-Phi overwrites Sh/SH inside mohrs_3D.
    matlab_sig = np.array([
        sV,
        args.min_horizontal_stress * args.reference_depth if args.stress_mode != "gradients" else sh,
        args.max_horizontal_stress * args.reference_depth if args.stress_mode != "gradients" else sH,
    ], dtype=float)

    uncertainty = parse_mc_uncertainty(args)
    samples = None
    python_mc = None
    if args.mc_iterations > 0:
        mc_df, samples_df = run_geomechanics_mc(
            stress_inputs_from_args(args),
            faults,
            args.mc_iterations,
            uncertainty,
            args.stress_mode,
            mu,
            random_seed=args.random_seed,
            return_sample_inputs=True,
        )
        python_mc = mc_df["SlipPressure"].to_numpy(float)
        samples = matlab_samples_from_production(samples_df, args)

    matlab = run_matlab(args, matlab_sig, float(p0), strikes, dips, samples)
    matlab_metrics = matlab["metrics"]
    ensure_length("fault metrics", fault_ids, ppf, matlab_metrics[:, 0])
    matlab_ppf = matlab_metrics[:, 0]
    matlab_cff = matlab_metrics[:, 1]
    matlab_scu = matlab_metrics[:, 2]
    matlab_sig_fault = matlab_metrics[:, 3]
    matlab_tau_fault = matlab_metrics[:, 4]

    ppf_abs, ppf_rel = series_errors(ppf, matlab_ppf, PSI_ABS_THRESHOLD)
    cff_abs, cff_rel = series_errors(cff, matlab_cff, PSI_ABS_THRESHOLD)
    scu_abs, scu_rel = series_errors(scu, matlab_scu, SCU_ABS_THRESHOLD)
    sig_abs, sig_rel = series_errors(sig, matlab_sig_fault, PSI_ABS_THRESHOLD)
    tau_abs, tau_rel = series_errors(tau, matlab_tau_fault, PSI_ABS_THRESHOLD)

    metrics = pd.DataFrame({
        "fault_id": fault_ids,
        "python_ppf_psi": ppf, "matlab_ppf_psi": matlab_ppf,
        "ppf_absolute_error_psi": ppf_abs, "ppf_relative_error": ppf_rel,
        "python_cff_psi": cff, "matlab_cff_psi": matlab_cff,
        "cff_absolute_error_psi": cff_abs, "cff_relative_error": cff_rel,
        "python_scu": scu, "matlab_scu": matlab_scu,
        "scu_absolute_error": scu_abs, "scu_relative_error": scu_rel,
        "python_sigma_psi": sig, "matlab_sigma_psi": matlab_sig_fault,
        "sigma_absolute_error_psi": sig_abs, "sigma_relative_error": sig_rel,
        "python_tau_psi": tau, "matlab_tau_psi": matlab_tau_fault,
        "tau_absolute_error_psi": tau_abs, "tau_relative_error": tau_rel,
    })

    matlab_mohr = matlab["mohr"]
    circle_ids = np.repeat(["circle1", "circle2", "circle3"], len(matlab_mohr))
    python_x = np.concatenate([
        python_arcs.loc[python_arcs["id"] == cid, "x"].to_numpy(float)
        for cid in ("circle1", "circle2", "circle3")
    ])
    python_y = np.concatenate([
        python_arcs.loc[python_arcs["id"] == cid, "y"].to_numpy(float)
        for cid in ("circle1", "circle2", "circle3")
    ])
    matlab_x = np.concatenate([matlab_mohr[:, 0], matlab_mohr[:, 2], matlab_mohr[:, 4]])
    matlab_y = np.concatenate([matlab_mohr[:, 1], matlab_mohr[:, 3], matlab_mohr[:, 5]])
    ensure_length("mohr arcs", python_x, matlab_x, python_y, matlab_y)
    point_index = np.tile(np.arange(len(matlab_mohr)), 3)
    arc_x_abs, arc_x_rel = series_errors(python_x, matlab_x, PSI_ABS_THRESHOLD)
    arc_y_abs, arc_y_rel = series_errors(python_y, matlab_y, PSI_ABS_THRESHOLD)
    arcs = pd.DataFrame({
        "circle_id": circle_ids,
        "point_index": point_index,
        "python_x_psi": python_x, "matlab_x_psi": matlab_x,
        "x_absolute_error_psi": arc_x_abs, "x_relative_error": arc_x_rel,
        "python_y_psi": python_y, "matlab_y_psi": matlab_y,
        "y_absolute_error_psi": arc_y_abs, "y_relative_error": arc_y_rel,
    })

    point_x_abs, point_x_rel = series_errors(sig, matlab_sig_fault, PSI_ABS_THRESHOLD)
    point_y_abs, point_y_rel = series_errors(tau, matlab_tau_fault, PSI_ABS_THRESHOLD)
    points = pd.DataFrame({
        "fault_id": fault_ids,
        "python_sigma_psi": sig, "matlab_sigma_psi": matlab_sig_fault,
        "sigma_absolute_error_psi": point_x_abs, "sigma_relative_error": point_x_rel,
        "python_tau_psi": tau, "matlab_tau_psi": matlab_tau_fault,
        "tau_absolute_error_psi": point_y_abs, "tau_relative_error": point_y_rel,
    })

    metrics_path = args.output_dir / "fault_metrics_comparison.csv"
    arcs_path = args.output_dir / "mohr_arcs_comparison.csv"
    points_path = args.output_dir / "mohr_fault_points_comparison.csv"
    plot_path = write_mohr_plot(arcs, points, args.output_dir / "mohr_side_by_side.html", mu)
    metrics.to_csv(metrics_path, index=False)
    arcs.to_csv(arcs_path, index=False)
    points.to_csv(points_path, index=False)

    comparisons_passed = [
        validate_series(matlab_ppf, ppf_abs, ppf_rel, PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT),
        validate_series(matlab_cff, cff_abs, cff_rel, PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT),
        validate_series(matlab_scu, scu_abs, scu_rel, SCU_ABS_THRESHOLD, SCU_ABS_LIMIT, SCU_REL_LIMIT),
        validate_series(matlab_sig_fault, sig_abs, sig_rel, PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT),
        validate_series(matlab_tau_fault, tau_abs, tau_rel, PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT),
        validate_series(matlab_x, arc_x_abs, arc_x_rel, PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT),
        validate_series(matlab_y, arc_y_abs, arc_y_rel, PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT),
    ]
    relative_error_values = [
        float(ppf_rel.max()), float(cff_rel.max()), float(scu_rel.max()),
        float(sig_rel.max()), float(tau_rel.max()), float(arc_x_rel.max()), float(arc_y_rel.max()),
    ]
    outputs = {
        "fault_metrics_csv": str(metrics_path.resolve()),
        "mohr_arcs_csv": str(arcs_path.resolve()),
        "mohr_fault_points_csv": str(points_path.resolve()),
        "plot_html": str(plot_path.resolve()),
    }

    if python_mc is not None:
        matlab_mc = np.asarray(matlab["mc"], dtype=float).reshape(-1)
        ensure_length("mc slip pressure", python_mc, matlab_mc)
        n_faults = len(faults)
        n_sims = args.mc_iterations
        # run_geomechanics_mc flattens (n_faults, n_sims) in C order: all sims of fault 0, then fault 1, ...
        python_mc_grid = python_mc.reshape(n_faults, n_sims)
        matlab_mc_grid = matlab_mc.reshape(n_faults, n_sims)
        rows = []
        cdf_rows = []
        mc_rel_all = []
        for simulation_id in range(n_sims):
            abs_err, rel_err = series_errors(
                python_mc_grid[:, simulation_id], matlab_mc_grid[:, simulation_id], PSI_ABS_THRESHOLD
            )
            rows.append(pd.DataFrame({
                "simulation_id": simulation_id + 1,
                "fault_id": fault_ids,
                "python_ppf_psi": python_mc_grid[:, simulation_id],
                "matlab_ppf_psi": matlab_mc_grid[:, simulation_id],
                "absolute_error_psi": abs_err,
                "relative_error": rel_err,
            }))
            comparisons_passed.append(validate_series(
                matlab_mc_grid[:, simulation_id], abs_err, rel_err,
                PSI_ABS_THRESHOLD, PSI_ABS_LIMIT, PSI_REL_LIMIT,
            ))
            mc_rel_all.append(rel_err)
        mc_comparison = pd.concat(rows, ignore_index=True)
        mc_path = args.output_dir / "mc_slip_pressure_comparison.csv"
        mc_comparison.to_csv(mc_path, index=False)
        for fault_index, fault_id in enumerate(fault_ids):
            python_sorted = np.sort(python_mc_grid[fault_index])
            matlab_sorted = np.sort(matlab_mc_grid[fault_index])
            ranks = np.arange(1, len(python_sorted) + 1)
            cdf_rows.append(pd.DataFrame({
                "fault_id": fault_id,
                "rank": ranks,
                "cumulative_probability": ranks / len(ranks),
                "python_ppf_psi": python_sorted,
                "matlab_ppf_psi": matlab_sorted,
            }))
        cdf_path = args.output_dir / "mc_cdf_comparison.csv"
        pd.concat(cdf_rows, ignore_index=True).to_csv(cdf_path, index=False)
        outputs.update({
            "mc_comparison_csv": str(mc_path.resolve()),
            "mc_cdf_csv": str(cdf_path.resolve()),
        })
        relative_error_values.append(float(np.concatenate(mc_rel_all).max()))

    passed = all(comparisons_passed)
    max_relative_error = max(relative_error_values)
    summary = {
        "passed": passed,
        "max_relative_error": max_relative_error,
        "tolerance": {
            "psi": {"relative_at_least_1_psi": PSI_REL_LIMIT, "absolute_below_1_psi": PSI_ABS_LIMIT},
            "scu": {"relative_at_least_0_01": SCU_REL_LIMIT, "absolute_below_0_01": SCU_ABS_LIMIT},
        },
        "notes": {
            "python_deterministic": "fsp_step2.py: calculate_slip_pressure (PPF clamped at 0), round(CFF), calculate_scu.",
            "python_mc": "fsp_step3.py: run_geomechanics_mc / ComputeCriticalPorePressureForFailure.",
            "matlab": "mohrs_3D quadratic kernel for both deterministic and MC (legacy production).",
            "not_compared": ["Stereonet and tornado charts."],
            "friction_line_slope": mu,
        },
        "inputs": {
            "matlab_code": str(args.matlab_code.resolve()),
            "stress_mode": args.stress_mode,
            "aphi_use_matlab": STRESS_MODE_TO_APHI_USE[args.stress_mode],
            "reference_depth_ft": args.reference_depth,
            "vertical_stress_psi_ft": args.vertical_stress,
            "min_horizontal_stress_psi_ft": args.min_horizontal_stress,
            "max_horizontal_stress_psi_ft": args.max_horizontal_stress,
            "pore_pressure_psi_ft": args.pore_pressure,
            "max_stress_azimuth_deg": args.max_stress_azimuth,
            "aphi_value": args.aphi_value if args.stress_mode != "gradients" else None,
            "friction_coefficient": mu,
            "poissons_ratio": nu,
            "resolved_stress_psi": {"Sv": sV, "Shmin": sh, "SHmax": sH, "p0": float(p0)},
            "faults": str(args.faults.resolve()),
            "n_faults": int(len(faults)),
            "random_seed": args.random_seed,
            "mc_iterations": args.mc_iterations,
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
