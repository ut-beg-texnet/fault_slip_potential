#!/usr/bin/env python
"""
MATLAB-vs-Python regression for Fault Slip Potential (FSP).

Runs the Python production code in ``src/fsp`` on the webtool (WEB) input files in
``Input Data/`` and compares the results with each saved legacy MATLAB FSP session
(``FSP_*_MATLAB`` files under ``MATLAB Results/``, one folder per case).

What a saved MATLAB session contains (and therefore what can be compared):
  * every input (stress, faults, wells, hydrology, uncertainties);
  * the geomechanics Monte Carlo *input samples* (1000 per fault) - no slip pressures;
  * the hydrology Monte Carlo *input samples* (one row per simulation);
  * one MATLAB *output*: ``probHydrology.outsAllFaults``, pore pressure (psi) per
    simulation per fault at the hydrology evaluation year.
Optional MATLAB exports in the session's folder add more outputs: ``CDF_*.csv`` (pore
pressure to slip at each probability), ``FSPvalues_*.csv`` (displayed FSP per fault) and
``PP2SLIP_*.jpg`` (screenshot of the deterministic pore pressure to slip map).

Checks
  I   The WEB input files given to Python vs what the MATLAB session ran with.
  A   Injection schedule: Python's well loader vs MATLAB's saved rate schedule.
  B1  Theis kernel: MATLAB samples + MATLAB schedule + MATLAB distances -> Python
      Theis kernel, vs outsAllFaults. Also probes MATLAB's evaluation instant.
  B2  Python hydrology pipeline: MATLAB samples + MATLAB distances, but Python's own
      well loader and year-of-interest convention, vs outsAllFaults.
  C   Python hydrology pipeline with production geometry (WEB lat/lon -> haversine),
      vs outsAllFaults. Informational: isolates the coordinate effect.
  D   Python's own hydrology Monte Carlo vs MATLAB's distribution (statistical).
  E   Python's geomechanics Monte Carlo input sampling vs MATLAB's saved samples
      (statistical).
  F   Geomechanics: Python's slip pressure for each of MATLAB's saved draws vs the
      MATLAB CDF export, plus the deterministic pore pressure to slip (needs exports).
  G   FSP replay: Python's FSP calculation fed MATLAB's geomechanics draws and MATLAB's
      hydrology pressures, vs the MATLAB FSP export (exact, needs FSPvalues).
  H   Python's own end-to-end Monte Carlo FSP over several seeds vs the MATLAB FSP
      export (statistical: is MATLAB inside Python's run-to-run noise?).

A, B1, B2 and G gate the exit code. I, C, D, E, F and H are reported but never fail the run.

Usage (from the repository root):
    .\\fsp_python_14_venv\\Scripts\\python.exe regression_testing\\run_fsp_matlab_regression.py
See regression_testing/README.md for flags and how to read the output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fsp.geomechanics.slip import (  # noqa: E402
    ComputeCriticalPorePressureForFailure,
    calculate_fault_effective_stresses,
    calculate_slip_pressure,
    fold_dip_degrees,
    wrap_strike_degrees,
)
from fsp.geomechanics.stress import calculate_absolute_stresses  # noqa: E402
from fsp.geomechanics.validation import (  # noqa: E402
    validate_geomechanics_uncertainties,
    validate_stress_inputs,
)
from fsp.hydrology.params import calcST  # noqa: E402
from fsp.hydrology.pressure_field import (  # noqa: E402
    pfieldcalc_all_rates_for_distances,
    well_fault_distances_m,
)
from fsp.hydrology.theis import _NUMBA_AVAILABLE  # noqa: E402
from fsp.io.faults import load_faults_csv  # noqa: E402
from fsp.io.wells import (  # noqa: E402
    get_date_bounds,
    load_injection_wells,
    normalize_wells_to_well_data,
    preprocess_well_data,
)
from fsp.models.hydrology import HydrologyParams  # noqa: E402
from fsp.monte_carlo.geomechanics_mc import run_geomechanics_mc  # noqa: E402
from fsp.monte_carlo.hydrology_mc import (  # noqa: E402
    _run_all_sims_batched,
    _single_sim_pressure,
    run_hydrology_mc_time_series,
)
from fsp.probabilistic_fsp import (  # noqa: E402
    displayed_legacy_fsp,
    legacy_fsp,
    normalize_year_of_interest,
)

FT_TO_M = 0.3048
CASE_GLOB = "**/FSP_*_MATLAB"
# Session file names give the regime and injection type, e.g. FSP_NormalFault_AnnualInjection_MATLAB,
# FSP_Reverse_Monthly_MATLAB, FSP_StrikeSlip_4Well_AnnualInjection_MATLAB.
CASE_NAME_RE = re.compile(
    r"FSP_(?:[A-Za-z]+_)??(?P<regime>Normal|Reverse|StrikeSlip)(?:Fault)?_(?:\d+Wells?_)?"
    r"(?P<injection>Annual|Monthly)(?:Injection)?_MATLAB$",
    re.IGNORECASE,
)
REGIME_CANONICAL = {"normal": "Normal", "reverse": "Reverse", "strikeslip": "StrikeSlip"}
REGIME_LABELS = {"Normal": "Normal", "StrikeSlip": "Strike-Slip", "Reverse": "Reverse"}
KS_ALPHA = 0.01

# MATLAB datsviz.sigvals order (labels in datsviz.distributions.distirbutionTxt)
SIG_SV, SIG_SHMIN, SIG_SHMAX, _SIG_IGNORE, SIG_PP, SIG_STRIKE, SIG_DIP, SIG_AZ = range(8)
SIG_MU = 9
# MATLAB geomechanics sample columns (datsviz.distributionsinData{fault}, 1000 x 11)
GEO_STRESS, GEO_PP, GEO_STRIKE, GEO_DIP, GEO_AZ, GEO_MU, GEO_APHI = 0, 2, 3, 4, 5, 7, 10
# MATLAB hydrology sample columns (datsviz.probHydrology.distributionsinData, n x 15)
H_WX, H_WY, H_WID, H_FX, H_FY, H_G, H_YEAR = 0, 1, 2, 3, 4, 5, 6
H_THICK_M, H_PHI, H_PERM, H_RHO, H_VISC, H_BETA, H_ALPHA, H_SCHED = 7, 8, 9, 10, 11, 12, 13, 14

HYDRO_PARAMS = [
    # (python key, label, unit, MATLAB sample column)
    ("aquifer_thickness", "Aquifer thickness", "ft", H_THICK_M),
    ("porosity", "Porosity", "fraction", H_PHI),
    ("permeability", "Permeability", "mD", H_PERM),
    ("fluid_density", "Fluid density", "kg/m³", H_RHO),
    ("dynamic_viscosity", "Dynamic viscosity", "Pa·s", H_VISC),
    ("fluid_compressibility", "Fluid compressibility", "1/Pa", H_BETA),
    ("rock_compressibility", "Rock compressibility", "1/Pa", H_ALPHA),
]


# =============================================================================
# Loading the MATLAB session
# =============================================================================
@dataclass
class MatlabCase:
    name: str
    path: Path
    regime: str
    injection: str            # "annual" | "monthly"
    stress_model_type: str    # python name, e.g. "aphi_min"
    stress: dict              # gradients psi/ft, azimuth deg, depth ft
    aphi: float
    aphi_sigma: float
    sigmas: dict              # python uncertainty keys -> value
    friction: float
    faults: pd.DataFrame      # FaultID, x_km, y_km, Strike, Dip, LengthKm
    wells: pd.DataFrame       # WellID, x_km, y_km (+ annual/monthly columns)
    extrapolate: bool
    hydro_nominal: dict       # python HydrologyParams fields
    hydro_sigma: dict         # python plus_minus keys
    hydro_iterations: int
    gravity: float
    year: int
    pressures: np.ndarray     # (n_sims, n_faults) outsAllFaults, psi
    hydro_samples: pd.DataFrame
    hydro_well_xy: np.ndarray  # (n_wells, 2) km
    hydro_well_ids: list
    hydro_fault_xy: np.ndarray  # (n_faults, 2) km
    matlab_schedules: list     # per well (n_rows, 2) [datenum, bbl/day]
    geo_samples: dict          # name -> (n_faults, n_samples)
    geo_cells: object = None   # raw distributionsinData cells, for the slip-pressure replay
    exports: dict = field(default_factory=dict)  # MATLAB exports next to the session (CDF, FSP, screenshot)
    notes: list = field(default_factory=list)

    @property
    def label(self) -> str:
        wells = f"{len(self.wells['WellID'].unique())} well"
        wells += "s" if not wells.startswith("1 ") else ""
        return f"{REGIME_LABELS.get(self.regime, self.regime)} | {wells}, {self.injection} injection"


def _cellstr(values) -> list:
    return [str(v) for v in np.atleast_1d(values)]


def load_matlab_case(path: Path) -> MatlabCase:
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False, appendmat=False)
    if str(m.get("chkstring", "")) != "Saved FSP Session":
        raise ValueError(f"{path.name}: not a saved MATLAB FSP session")
    d = m["datsviz"]
    match = CASE_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"{path.name}: file name does not match {CASE_NAME_RE.pattern}")
    regime = REGIME_CANONICAL[match["regime"].lower()]
    injection = match["injection"].lower()
    notes: list[str] = []

    # ---- stress -----------------------------------------------------------
    sv, shmin, shmax, azimuth, depth, pp = (float(v) for v in d.stress.vals)
    aphi_use = int(d.stress.aphi.use)
    if aphi_use == 12:
        stress_model_type = "aphi_min"
    else:
        raise NotImplementedError(
            f"{path.name}: stress.aphi.use={aphi_use}; only the A-Phi with minimum horizontal "
            "stress model (use=12) is decoded by this script."
        )
    aphi = float(np.atleast_1d(d.stress.aphi.vals)[0])
    aphi_sigma = float(np.atleast_1d(d.stress.aphi.sigvals)[0])
    sig = np.asarray(d.sigvals, dtype=float)
    sigmas = {
        "vertical_stress_gradient_uncertainty": sig[SIG_SV],
        "initial_pore_pressure_gradient_uncertainty": sig[SIG_PP],
        "max_stress_azimuth_uncertainty": sig[SIG_AZ],
        "aphi_value_uncertainty": aphi_sigma,
        "min_horizontal_stress_uncertainty": sig[SIG_SHMIN],
        "strike_angles_uncertainty": sig[SIG_STRIKE],
        "dip_angles_uncertainty": sig[SIG_DIP],
        "friction_coefficient_uncertainty": sig[SIG_MU],
    }
    stress = {
        "reference_depth": depth,
        "vertical_stress": sv,
        "min_horizontal_stress": shmin,
        "max_horizontal_stress": None,  # computed from A-Phi in this mode
        "pore_pressure": pp,
        "max_stress_azimuth": azimuth,
        "aphi_value": aphi,
        "matlab_shmax_gradient": shmax,
    }

    # ---- faults -----------------------------------------------------------
    ff = np.asarray(d.fault.file, dtype=float)
    friction = float(d.fault.vals[1])
    muf = np.unique(np.atleast_1d(d.fault.muf).astype(float))
    if muf.size != 1 or not np.isclose(muf[0], friction):
        notes.append(f"Per-fault friction differs from the global value {friction}: {muf}")
    faults = pd.DataFrame({
        "FaultID": [str(i + 1) for i in range(len(ff))],
        "x_km": ff[:, 0], "y_km": ff[:, 1],
        "Strike": ff[:, 2], "Dip": ff[:, 3], "LengthKm": ff[:, 4],
    })

    # ---- wells ------------------------------------------------------------
    extrapolate = bool(int(d.realWellData.extrapolateInjectionCheck))
    if extrapolate:
        notes.append("The session has 'Extrapolate Injection?' ticked"
                     + ("; it only applies to monthly data, so it has no effect on this annual case."
                        if injection == "annual" else "."))
    if injection == "annual":
        n = int(d.nwells)
        inj = np.atleast_2d(np.asarray(d.inject.vals, dtype=float))[:n]
        wells = pd.DataFrame({
            "WellID": [str(i + 1) for i in range(n)],
            "x_km": inj[:, 0], "y_km": inj[:, 1],
            "InjectionRate(bbl/day)": inj[:, 2],
            "StartYear": inj[:, 3].astype(int), "EndYear": inj[:, 4].astype(int),
        })
    else:
        if int(d.realWellData.use) != 1:
            raise ValueError(f"{path.name}: monthly case but realWellData.use != 1")
        rows = np.atleast_2d(d.realWellData.stringsWellDataAdvanced)
        wells = pd.DataFrame({
            "WellID": [str(r[0]) for r in rows],
            "x_km": [float(r[1]) for r in rows], "y_km": [float(r[2]) for r in rows],
            "Year": [int(float(r[3])) for r in rows], "Month": [int(float(r[4])) for r in rows],
            "InjectionRate(bbl/month)": [float(r[5]) for r in rows],
        })

    # ---- hydrology --------------------------------------------------------
    res = np.asarray(d.reservoir.vals, dtype=float)
    adv = np.asarray(d.adv.vals, dtype=float)
    hs = np.asarray(d.probHydrology.sigvals, dtype=float)
    hydro_nominal = {
        "aquifer_thickness": res[0], "porosity": res[1] / 100.0, "permeability": res[2],
        "fluid_density": adv[4], "dynamic_viscosity": adv[6],
        "fluid_compressibility": adv[7], "rock_compressibility": adv[8],
    }
    hydro_sigma = {
        "aquifer_thickness": hs[0], "porosity": hs[1] / 100.0, "permeability": hs[2],
        "fluid_density": hs[3], "dynamic_viscosity": hs[4],
        "fluid_compressibility": hs[5], "rock_compressibility": hs[6],
    }
    if int(d.probHydrology.probabilistic1VsDeterministic2) != 1:
        raise ValueError(f"{path.name}: hydrology was not run probabilistically")
    H = d.probHydrology.distributionsinData
    years = np.unique([int(v) for v in H[:, H_YEAR]])
    if years.size != 1:
        raise ValueError(f"{path.name}: hydrology rows have several years {years}")
    year = int(years[0])
    hydro_samples = pd.DataFrame({
        key: np.asarray([float(v) for v in H[:, col]]) for key, _l, _u, col in HYDRO_PARAMS
    })
    hydro_samples["aquifer_thickness"] /= FT_TO_M  # MATLAB stores metres
    hydro_samples.insert(0, "SimulationID", np.arange(1, len(H) + 1))
    gravity = float(np.unique([float(v) for v in H[:, H_G]])[0])
    well_x = np.atleast_1d(H[0, H_WX]).astype(float)
    well_y = np.atleast_1d(H[0, H_WY]).astype(float)
    well_ids = [str(int(v)) for v in np.atleast_1d(H[0, H_WID])]
    fault_xy = np.column_stack([np.asarray(H[0, H_FX], float), np.asarray(H[0, H_FY], float)])
    sched = H[0, H_SCHED]
    schedules = list(sched) if isinstance(sched, np.ndarray) and sched.dtype == object else [sched]
    schedules = [np.atleast_2d(np.asarray(s, dtype=float)) for s in schedules]
    pressures = np.asarray(d.probHydrology.outsAllFaults, dtype=float)

    # ---- geomechanics samples --------------------------------------------
    cells = np.atleast_1d(d.distributionsinData)
    n_faults = len(faults)
    if len(cells) != n_faults:
        raise ValueError(f"{path.name}: {len(cells)} geomechanics cells for {n_faults} faults")

    def col(c, i):
        return np.asarray([float(v) for v in cells[i][:, c]])

    stress_abs = [np.vstack([np.asarray(v, float) for v in cells[i][:, GEO_STRESS]]) for i in range(n_faults)]
    geo_samples = {
        "vertical_stress_gradient": np.vstack([s[:, 0] for s in stress_abs]) / depth,
        "min_horizontal_stress_gradient": np.vstack([s[:, 1] for s in stress_abs]) / depth,
        "initial_pore_pressure_gradient": np.vstack([col(GEO_PP, i) for i in range(n_faults)]) / depth,
        "max_stress_azimuth": np.vstack([col(GEO_AZ, i) for i in range(n_faults)]),
        "friction_coefficient": np.vstack([col(GEO_MU, i) for i in range(n_faults)]),
        "aphi_value": np.vstack([col(GEO_APHI, i) for i in range(n_faults)]),
        "strike_angle": np.vstack([col(GEO_STRIKE, i) for i in range(n_faults)]),
        "dip_angle": np.vstack([col(GEO_DIP, i) for i in range(n_faults)]),
    }

    n_wells = wells["WellID"].nunique()
    case = MatlabCase(
        name=f"{regime}Fault_{n_wells}Well_{injection.capitalize()}Injection",
        path=path, regime=regime, injection=injection, stress_model_type=stress_model_type,
        stress=stress, aphi=aphi, aphi_sigma=aphi_sigma, sigmas=sigmas, friction=friction,
        faults=faults, wells=wells, extrapolate=extrapolate, hydro_nominal=hydro_nominal,
        hydro_sigma=hydro_sigma, hydro_iterations=int(hs[7]), gravity=gravity, year=year,
        pressures=pressures, hydro_samples=hydro_samples,
        hydro_well_xy=np.column_stack([well_x, well_y]), hydro_well_ids=well_ids,
        hydro_fault_xy=fault_xy, matlab_schedules=schedules, geo_samples=geo_samples, geo_cells=cells,
        notes=notes,
    )
    _sanity_check_case(case)
    case.exports = load_matlab_exports(path, regime, n_faults)
    return case


# ---- MATLAB exports saved next to a session ----------------------------------
def _export_file(folder: Path, patterns, regime: str) -> Path | None:
    """The one export in the session's folder matching a pattern (disambiguated by regime)."""
    hits = sorted({p for pat in patterns for p in folder.glob(pat) if p.is_file()})
    if len(hits) > 1:
        keys = REGIME_FILE_KEYS.get(regime, (regime.lower(),))
        hits = [p for p in hits if any(k in p.stem.lower() for k in keys)]
    return hits[0] if len(hits) == 1 else None


def load_matlab_exports(path: Path, regime: str, n_faults: int) -> dict:
    """CDF export, displayed FSP values and the deterministic screenshot, when present.

    CDF_*.csv: MATLAB "export CSV of Probability of Fault Slip given a pore pressure increase".
    Row 2 holds the probabilities (0:0.001:1) from column 8 on; from row 4 on, one fault per
    row: number, centre x/y, strike, dip, length, mu, then the pressure (psi) at each probability.
    FSPvalues_*.csv: one line per fault, "Fault #k, 0.36 FSP" (the UI's %0.2f display).
    """
    folder = path.parent
    out = {}
    cdf_path = _export_file(folder, ["CDF*.csv"], regime)
    if cdf_path:
        lines = cdf_path.read_text(encoding="latin-1").splitlines()
        probs = np.array([float(v) for v in lines[1].split(",")[7:] if v.strip()])
        rows = [ln.split(",") for ln in lines[3:] if ln.strip() and not ln.startswith("#")]
        ids = [str(int(float(r[0]))) for r in rows]
        values = np.array([[float(v) for v in r[7:7 + len(probs)]] for r in rows])
        if values.shape != (n_faults, len(probs)):
            raise ValueError(f"{cdf_path.name}: CDF export shape {values.shape} is unexpected")
        out["cdf"] = {"path": cdf_path, "probs": probs, "values": values, "fault_ids": ids}
    fsp_path = _export_file(folder, ["FSPvalues*.csv", "*FSP*values*.csv"], regime)
    if fsp_path:
        found = re.findall(r"Fault\s*#\s*(\d+)\s*,\s*([-0-9.]+)\s*FSP", fsp_path.read_text(encoding="latin-1"))
        fsp = {str(int(k)): float(v) for k, v in found}
        if len(fsp) != n_faults:
            raise ValueError(f"{fsp_path.name}: {len(fsp)} FSP values for {n_faults} faults")
        out["fsp"] = {"path": fsp_path, "values": fsp}
    img = _export_file(folder, ["PP2SLIP*.jpg", "PP2SLIP*.png", "*PP*2*SLIP*.jpg", "*PP*2*SLIP*.png"], regime)
    if img:
        out["deterministic_image"] = img
    return out


def _sanity_check_case(c: MatlabCase) -> None:
    """Fail loudly if the decoded layout is not what this script assumes."""
    def within(values, nominal, sigma, what):
        lo, hi = nominal - sigma, nominal + sigma
        tol = 1e-9 * max(1.0, abs(nominal))
        if np.nanmin(values) < lo - tol or np.nanmax(values) > hi + tol:
            raise ValueError(
                f"{c.name}: {what} samples [{np.nanmin(values):.6g}, {np.nanmax(values):.6g}] "
                f"fall outside nominal ± uncertainty [{lo:.6g}, {hi:.6g}] - column mapping is wrong"
            )

    for key, label, _u, _col in HYDRO_PARAMS:
        within(c.hydro_samples[key], c.hydro_nominal[key], c.hydro_sigma[key], f"hydrology {label}")
    g = c.geo_samples
    s = c.stress
    within(g["vertical_stress_gradient"], s["vertical_stress"], c.sigmas["vertical_stress_gradient_uncertainty"], "Sv")
    within(g["min_horizontal_stress_gradient"], s["min_horizontal_stress"], c.sigmas["min_horizontal_stress_uncertainty"], "Shmin")
    within(g["initial_pore_pressure_gradient"], s["pore_pressure"], c.sigmas["initial_pore_pressure_gradient_uncertainty"], "pore pressure")
    within(g["max_stress_azimuth"], s["max_stress_azimuth"], c.sigmas["max_stress_azimuth_uncertainty"], "SHmax azimuth")
    within(g["friction_coefficient"], c.friction, c.sigmas["friction_coefficient_uncertainty"], "friction")
    within(g["aphi_value"], c.aphi, c.aphi_sigma, "A-Phi")
    for i in range(len(c.faults)):
        within(g["strike_angle"][i], c.faults["Strike"].iat[i], c.sigmas["strike_angles_uncertainty"], f"fault {i + 1} strike")
        within(g["dip_angle"][i], c.faults["Dip"].iat[i], c.sigmas["dip_angles_uncertainty"], f"fault {i + 1} dip")
    if c.pressures.shape != (len(c.hydro_samples), len(c.faults)):
        raise ValueError(f"{c.name}: outsAllFaults shape {c.pressures.shape} is unexpected")
    if not np.allclose(c.hydro_fault_xy, c.faults[["x_km", "y_km"]].to_numpy()):
        raise ValueError(f"{c.name}: hydrology fault coordinates differ from the fault table")
    if len(c.matlab_schedules) != len(c.hydro_well_ids):
        raise ValueError(f"{c.name}: {len(c.matlab_schedules)} schedules for {len(c.hydro_well_ids)} wells")


def discover_cases(root: Path) -> list[Path]:
    """Every saved MATLAB session under root."""
    return sorted(p for p in root.glob(CASE_GLOB) if p.is_file() and CASE_NAME_RE.match(p.name))


def select_cases(cases: list, selectors: list[str]) -> list:
    """Apply --cases substrings (case name or session file name) and order the cases.

    Each case (regime, number of wells, annual/monthly) must have exactly one session.
    """
    by_name = {}
    for c in cases:
        if c.name in by_name:
            raise ValueError(f"Two MATLAB sessions for {c.name}: {by_name[c.name].path} and {c.path}. "
                             "Keep one session per case.")
        by_name[c.name] = c
    run = list(by_name.values())
    if selectors and selectors != ["all"]:
        run = [c for c in run
               if any(s.lower() in c.name.lower() or s.lower() in c.path.name.lower() for s in selectors)]
    order = {"annual": 0, "monthly": 1}
    regimes = {"Normal": 0, "Reverse": 1, "StrikeSlip": 2}
    return sorted(run, key=lambda c: (order.get(c.injection, 9), regimes.get(c.regime, 9)))


# =============================================================================
# Python inputs
# =============================================================================
class Projector:
    """WGS84 lat/lon -> MATLAB's km grid (the projected CRS given by --epsg), for check I."""

    def __init__(self, epsg: int):
        from pyproj import Transformer

        self.epsg = epsg
        self._to_km = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    def to_km(self, lat, lon):
        x, y = self._to_km.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0


# ---- input files ("Input Data/{Fault Data,Injection Data}/{WEB,MATLAB}") -------
REGIME_FILE_KEYS = {"Normal": ("normal",), "StrikeSlip": ("strikeslip",), "Reverse": ("reverse",)}


def _find_one(folder: Path, pattern: str, keys=()) -> Path | None:
    hits = [p for p in folder.glob(pattern)
            if not p.name.startswith("~$") and (not keys or any(k in p.stem.lower() for k in keys))]
    if len(hits) > 1:
        raise ValueError(f"Ambiguous input files in {folder}: {[h.name for h in hits]}")
    return hits[0] if hits else None


def find_input_files(inputs_dir: Path, c: MatlabCase) -> dict:
    """The WEB (lat/lon) input CSVs Python runs on, and the MATLAB-format (km) fault CSVs.

    Every MATLAB-format fault file is returned so check I can say which one the session ran on.
    """
    keys = REGIME_FILE_KEYS.get(c.regime, (c.regime.lower(),))
    inj = "Annual*" if c.injection == "annual" else "Monthly*"
    return {
        "web_faults": _find_one(inputs_dir / "Fault Data" / "WEB", "*.csv", keys),
        "web_wells": _find_one(inputs_dir / "Injection Data" / "WEB", f"{inj}/*.csv"),
        "matlab_faults": sorted((inputs_dir / "Fault Data" / "MATLAB").glob("*.csv")),
    }


@dataclass
class PythonInputs:
    faults_csv: Path
    wells_csv: Path
    data_type: str
    faults: pd.DataFrame           # as loaded by production load_faults_csv
    wells_df: pd.DataFrame         # as loaded by production load_injection_wells
    wells: list                    # normalized ProcessedWellData
    year: int                      # after normalize_year_of_interest
    cutoff: date
    date_bounds: tuple
    consistency: dict = field(default_factory=dict)
    epsg: int = 0


def _web_input_frames(c: MatlabCase, files: dict):
    web = pd.read_csv(files["web_faults"], dtype={"FaultID": str})
    faults_out = web[["FaultID", "Latitude(WGS84)", "Longitude(WGS84)", "Strike", "Dip", "LengthKm"]].copy()
    faults_out["FrictionCoefficient"] = web["FrictionCoefficient"] if "FrictionCoefficient" in web else c.friction
    wells_out = pd.read_csv(files["web_wells"], dtype={"WellID": str})
    return faults_out, wells_out


def check_inputs(c: MatlabCase, proj: Projector, faults_out: pd.DataFrame, wells_out: pd.DataFrame,
                 files: dict | None) -> dict:
    """Compare what Python is given against what the MATLAB session actually ran with."""
    issues = []
    out = {"issues": issues}
    # Which MATLAB-format fault file did the session use?
    if files and files.get("matlab_faults"):
        sess = c.faults[["x_km", "y_km", "Strike", "Dip", "LengthKm"]].to_numpy(float)
        matched = [p.name for p in files["matlab_faults"]
                   if (arr := pd.read_csv(p, header=None).to_numpy(float)).shape == sess.shape
                   and np.allclose(arr, sess, rtol=0, atol=1e-6)]
        out["session_fault_file"] = matched[0] if matched else "none of the MATLAB fault files"
        keys = REGIME_FILE_KEYS.get(c.regime, (c.regime.lower(),))
        if not matched or not any(k in matched[0].lower() for k in keys):
            issues.append(f"MATLAB session fault table matches {out['session_fault_file']}, not the "
                          f"{REGIME_LABELS.get(c.regime, c.regime)} fault file in Input Data/Fault Data/MATLAB")
    # Fault table given to Python vs session
    f = faults_out.set_index(faults_out["FaultID"].astype(str))
    s = c.faults.set_index("FaultID")
    common = [i for i in s.index if i in f.index]
    if len(common) != len(s):
        issues.append(f"Python fault table has {len(f)} faults, session has {len(s)}")
    for col in ("Strike", "Dip", "LengthKm"):
        diff = np.abs(f.loc[common, col].to_numpy(float) - s.loc[common, col].to_numpy(float))
        out[f"fault_{col.lower()}_max_abs_diff"] = float(diff.max())
        if diff.max() > 1e-6:
            issues.append(f"fault {col} differs from the MATLAB session on {int((diff > 1e-6).sum())} faults "
                          f"(max {diff.max():.4g})")
    fx, fy = proj.to_km(f.loc[common, "Latitude(WGS84)"], f.loc[common, "Longitude(WGS84)"])
    off = np.hypot(fx - s.loc[common, "x_km"].to_numpy(), fy - s.loc[common, "y_km"].to_numpy()) * 1000.0
    out["fault_position_offsets_m"] = pd.DataFrame({
        "FaultID": common,
        "dx_m": (fx - s.loc[common, "x_km"].to_numpy()) * 1000.0,
        "dy_m": (fy - s.loc[common, "y_km"].to_numpy()) * 1000.0,
        "offset_m": off,
    })
    out["fault_position_max_offset_m"] = float(off.max())
    if off.max() > 1.0:
        issues.append(f"{int((off > 1.0).sum())} fault locations differ from the MATLAB session by > 1 m "
                      f"(max {off.max():.1f} m, fault {common[int(np.argmax(off))]})")
    # Wells
    wg = wells_out.groupby(wells_out["WellID"].astype(str), sort=False).first()
    sg = c.wells.groupby("WellID", sort=False).first()
    wid = [i for i in sg.index if i in wg.index]
    if len(wid) != len(sg):
        issues.append(f"Python wells {list(wg.index)} vs session wells {list(sg.index)}")
    wx, wy = proj.to_km(wg.loc[wid, "Latitude(WGS84)"], wg.loc[wid, "Longitude(WGS84)"])
    woff = np.hypot(wx - sg.loc[wid, "x_km"].to_numpy(), wy - sg.loc[wid, "y_km"].to_numpy()) * 1000.0
    out["well_position_max_offset_m"] = float(woff.max()) if len(woff) else float("nan")
    if len(woff) and woff.max() > 1.0:
        issues.append(f"well locations differ from the MATLAB session by up to {woff.max():.1f} m")
    rate_cols = [col for col in c.wells.columns if col.startswith("InjectionRate")]
    for col in rate_cols + [k for k in ("StartYear", "EndYear", "Year", "Month") if k in c.wells.columns]:
        a = wells_out[col].to_numpy(float)
        b = c.wells[col].to_numpy(float)
        if a.shape != b.shape or not np.allclose(a, b):
            issues.append(f"well column {col} differs from the MATLAB session")
    out["status"] = "PASS" if not issues else "WARN"
    return out


def build_python_inputs(c: MatlabCase, proj: Projector, out_dir: Path, inputs_dir: Path) -> PythonInputs:
    """Python's inputs for this case: the WEB fault and injection CSVs, as the webtool takes them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_input_files(inputs_dir, c)
    if not (files["web_faults"] and files["web_wells"]):
        raise FileNotFoundError(
            f"{c.name}: need a WEB fault CSV (Fault Data/WEB, name containing "
            f"{'/'.join(REGIME_FILE_KEYS.get(c.regime, (c.regime.lower(),)))}) and a WEB injection CSV "
            f"(Injection Data/WEB/{'Annual' if c.injection == 'annual' else 'Monthly'}*) under {inputs_dir}")
    faults_out, wells_out = _web_input_frames(c, files)

    data_type = "annual_fsp" if c.injection == "annual" else "monthly_fsp"
    faults_csv = out_dir / "faults_fsp.csv"
    wells_csv = out_dir / f"injection_wells_{c.injection}_fsp.csv"
    faults_out.to_csv(faults_csv, index=False)
    wells_out.to_csv(wells_csv, index=False)
    consistency = check_inputs(c, proj, faults_out, wells_out, files)
    consistency["web_faults_file"] = str(files["web_faults"])
    consistency["web_wells_file"] = str(files["web_wells"])

    # Load everything back through production code, as the portal steps do.
    faults = load_faults_csv(str(faults_csv))
    wells_df = load_injection_wells(str(wells_csv), data_type)
    bounds = get_date_bounds(wells_df)
    year, _start_year, _end_year = normalize_year_of_interest(c.year, bounds[0], bounds[1])
    if year != c.year:
        last_row = datenum_to_date(max(float(s[-1, 0]) for s in c.matlab_schedules))
        c.notes.append(
            f"Python clamps year of interest {c.year} -> {year} (last injection data year {bounds[1].year} + 3), "
            f"while MATLAB evaluated {c.year}"
            + (f" with injection extrapolated to {last_row}" if c.extrapolate else "")
            + ". B2, C, D and H therefore compare Python at a different year than MATLAB; B1 and G use "
            "MATLAB's own year and are unaffected.")
    cutoff = date(year - 1, 12, 31)
    wells = normalize_wells_to_well_data(
        preprocess_well_data(wells_df, data_type), data_type, cutoff,
        extrapolate_injection_rates=c.extrapolate,
    )
    return PythonInputs(faults_csv, wells_csv, data_type, faults, wells_df, wells, year, cutoff, bounds,
                        consistency, proj.epsg)


# =============================================================================
# Time helpers
# =============================================================================
def datenum_to_date(dn: float) -> date:
    """MATLAB datenum -> calendar date, rounded to the nearest whole day.

    MATLAB stores schedule times a few seconds/minutes off midnight; the schedules are
    daily-resolution, so rounding recovers the intended day.
    """
    return date.fromordinal(int(round(dn)) - 366)


def matlab_step_schedule(sched: np.ndarray):
    """MATLAB [t_i, rate over (t_{i-1}, t_i]] rows -> (step dates, rates) step function.

    Rate rates[k] applies from dates[k] until dates[k+1]. MATLAB's pfront.m always appends
    a zero-rate point after the last saved row (when the year of interest is past it), so
    the well shuts in there. The "Extrapolate Injection?" checkbox does not change this:
    any extrapolation is already written into the saved rows.
    """
    dates = [datenum_to_date(v) for v in sched[:, 0]]
    rates = list(sched[1:, 1].astype(float))
    step_dates = dates[:-1] + [dates[-1]]
    rates.append(0.0)
    return step_dates, np.asarray(rates)


def python_step_schedule(wd):
    """ProcessedWellData days (day 1 = start_date) -> (step dates, rates)."""
    return [wd.start_date + timedelta(days=int(round(d)) - 1) for d in wd.days], np.asarray(wd.rates, float)


def daily_rates(step_dates, rates, first: date, last: date) -> np.ndarray:
    n = (last - first).days + 1
    out = np.zeros(n)
    for k, d0 in enumerate(step_dates):
        i0 = max((d0 - first).days, 0)
        i1 = (step_dates[k + 1] - first).days if k + 1 < len(step_dates) else n
        if i1 > i0:
            out[i0:min(i1, n)] = rates[k]
    return out


# =============================================================================
# Checks
# =============================================================================
def _matlab_well_index(c: MatlabCase, py_wells) -> list[int]:
    """Python well order -> row index into MATLAB's hydrology well arrays."""
    if c.injection == "annual":
        ids = list(c.hydro_well_ids)
        return [ids.index(str(w.well_id)) for w in py_wells]
    if len(c.hydro_well_ids) != 1 or len(py_wells) != 1:
        raise NotImplementedError("Multi-well monthly sessions are not decoded by this script")
    return [0]


def check_a_schedule(c: MatlabCase, py: PythonInputs, max_days_per_well: int) -> dict:
    """Python well loader vs MATLAB's saved per-well schedule, day by day.

    Passes when no well differs on more than ``max_days_per_well`` days (default 1: the
    tools put an annual well's last injection day one day apart).
    """
    idx = _matlab_well_index(c, py.wells)
    eval_matlab = date(c.year, 1, 1)
    rows, series = [], []
    for wi, wd in enumerate(py.wells):
        m_dates, m_rates = matlab_step_schedule(c.matlab_schedules[idx[wi]])
        p_dates, p_rates = python_step_schedule(wd)
        first = min(m_dates[0], p_dates[0])
        last = py.cutoff
        m_daily = daily_rates(m_dates, m_rates, first, last)
        p_daily = daily_rates(p_dates, p_rates, first, last)
        diff = p_daily - m_daily
        bad = np.flatnonzero(np.abs(diff) > 1e-6 * max(1.0, np.abs(m_daily).max()))
        m_vol = m_daily.sum()
        p_vol = p_daily.sum()
        rows.append({
            "well": wd.well_id,
            "matlab_first_injection": str(m_dates[0]),
            "python_first_injection": str(p_dates[0]),
            "matlab_shut_in": str(m_dates[-1]) if m_rates[-1] == 0 else "none",
            "python_shut_in": str(p_dates[-1]) if p_rates[-1] == 0 else "none",
            "max_abs_rate_diff_bbl_day": float(np.abs(diff).max()),
            "days_differing": int(bad.size),
            "first_differing_day": str(first + timedelta(days=int(bad[0]))) if bad.size else "",
            "last_differing_day": str(first + timedelta(days=int(bad[-1]))) if bad.size else "",
            "matlab_volume_bbl": float(m_vol),
            "python_volume_bbl": float(p_vol),
            "volume_rel_diff": float((p_vol - m_vol) / m_vol) if m_vol else 0.0,
        })
        series.append({"well": wd.well_id, "first": first, "matlab": m_daily, "python": p_daily})
    days_bad = sum(r["days_differing"] for r in rows)
    # Evaluation instant, measured in days after first injection. Python's Theis evaluates
    # at eval_days = (cutoff - start).days + 1 with day 1 = start, i.e. (cutoff - start)
    # days after injection began. MATLAB's instant is the one Check B1 infers.
    start = py.wells[0].start_date
    worst_well = max((r["days_differing"] for r in rows), default=0)
    return {
        "status": "PASS" if worst_well <= max_days_per_well else "FAIL",
        "wells": rows,
        "days_differing": days_bad,
        "max_days_differing_per_well": worst_well,
        "tolerance_days_per_well": max_days_per_well,
        "python_eval_days_after_start": (py.cutoff - start).days,
        "matlab_eval_days_after_start_if_jan1": (eval_matlab - start).days,
        "_series": series,
    }


def _strho(samples: pd.DataFrame):
    return [
        calcST(r.aquifer_thickness, r.porosity, r.permeability, r.fluid_density,
               r.dynamic_viscosity, 9.81, r.fluid_compressibility, r.rock_compressibility)
        for r in samples.itertuples(index=False)
    ]


def _replay_stats(python: np.ndarray, matlab: np.ndarray, rtol: float, atol: float) -> dict:
    diff = python - matlab
    rel = np.abs(diff) / np.maximum(np.abs(matlab), 1e-12)
    ok = np.abs(diff) <= atol + rtol * np.abs(matlab)
    worst = np.unravel_index(np.argmax(rel), rel.shape)
    return {
        "max_abs_psi": float(np.abs(diff).max()),
        "median_abs_psi": float(np.median(np.abs(diff))),
        "max_rel": float(rel.max()),
        "median_rel": float(np.median(rel)),
        "mean_signed_rel": float(np.mean(diff / np.maximum(np.abs(matlab), 1e-12))),
        "points_outside_tolerance": int((~ok).sum()),
        "points": int(ok.size),
        "worst_sim": int(worst[0]) + 1,
        "worst_fault": int(worst[1]) + 1,
        "rtol": rtol, "atol": atol,
        "status": "PASS" if ok.all() else "FAIL",
    }


def matlab_distance_matrix_m(c: MatlabCase) -> np.ndarray:
    w, f = c.hydro_well_xy, c.hydro_fault_xy
    return np.hypot(w[:, None, 0] - f[None, :, 0], w[:, None, 1] - f[None, :, 1]) * 1000.0


def check_b1_kernel(c: MatlabCase, strho, D_matlab: np.ndarray, rtol, atol) -> dict:
    """Python Theis kernel driven entirely by MATLAB's schedule, distances and samples."""
    origin = date(1900, 1, 1)
    scheds = [matlab_step_schedule(s) for s in c.matlab_schedules]
    candidates = {
        f"{c.year - 1}-12-31": date(c.year - 1, 12, 31),
        f"{c.year}-01-01": date(c.year, 1, 1),
    }
    results = {}
    for label, instant in candidates.items():
        t_eval = float((instant - origin).days)
        out = np.zeros_like(c.pressures)
        for s, st in enumerate(strho):
            for w, (dates, rates) in enumerate(scheds):
                days = np.asarray([(d - origin).days for d in dates], float)
                out[s] += pfieldcalc_all_rates_for_distances(D_matlab[w], st, days, rates, t_eval)
        results[label] = (out, _replay_stats(out, c.pressures, rtol, atol))
    # The right instant leaves only MATLAB's constant unit factor (×14.5 psi/bar), i.e. the
    # same Python/MATLAB ratio everywhere. Picking by max error instead can be fooled when
    # the 1-day shift happens to offset that factor (wells still injecting at the instant).
    def ratio_spread(k):
        r = results[k][0] / c.pressures
        return float(np.nanmax(r) - np.nanmin(r))
    best = min(results, key=ratio_spread)
    stats_ = dict(results[best][1])
    stats_["matlab_eval_instant"] = best
    stats_["instant_probe_max_rel"] = {k: v[1]["max_rel"] for k, v in results.items()}
    # A residual that is the same ratio for every sim and fault (independent of distance,
    # i.e. of the Theis argument u) points at a unit constant, not at the physics.
    ratio = results[best][0] / c.pressures
    stats_["python_over_matlab_ratio_median"] = float(np.median(ratio))
    stats_["python_over_matlab_ratio_spread"] = float(ratio.max() - ratio.min())
    return {**stats_, "_pressures": results[best][0]}


def _pipeline_pressures(py: PythonInputs, strho, D: np.ndarray, use_numba: bool) -> np.ndarray:
    if use_numba:
        return _run_all_sims_batched(py.wells, strho, D, [py.year])[py.year]
    return np.vstack([_single_sim_pressure(py.wells, st, D, [py.year])[py.year] for st in strho])


def check_b2_pipeline(c, py, strho, D_matlab, rtol, atol) -> dict:
    """Python wells + Python year convention, MATLAB distances and samples."""
    idx = _matlab_well_index(c, py.wells)
    D = D_matlab[idx]
    result = {}
    p_scipy = _pipeline_pressures(py, strho, D, use_numba=False)
    result["scipy"] = _replay_stats(p_scipy, c.pressures, rtol, atol)
    if _NUMBA_AVAILABLE:
        p_numba = _pipeline_pressures(py, strho, D, use_numba=True)
        result["numba"] = _replay_stats(p_numba, c.pressures, rtol, atol)
        result["numba_vs_scipy_max_rel"] = float(
            (np.abs(p_numba - p_scipy) / np.maximum(np.abs(p_scipy), 1e-12)).max()
        )
    else:
        p_numba = None
        result["numba"] = None
    statuses = [result["scipy"]["status"]] + ([result["numba"]["status"]] if p_numba is not None else [])
    result["status"] = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    result["_pressures"] = p_numba if p_numba is not None else p_scipy
    result["_pressures_scipy"] = p_scipy
    return result


def check_c_geometry(c, py, strho, D_matlab, rtol) -> dict:
    """Production distances: haversine on the WEB lat/lon inputs."""
    lats = py.faults["Latitude(WGS84)"].to_numpy(float)
    lons = py.faults["Longitude(WGS84)"].to_numpy(float)
    D_py = well_fault_distances_m(py.wells, lats, lons)
    p = _pipeline_pressures(py, strho, D_py, use_numba=_NUMBA_AVAILABLE)
    stats_ = _replay_stats(p, c.pressures, rtol, 0.0)
    stats_["status"] = "PASS" if stats_["status"] == "PASS" else "WARN"
    D_m = D_matlab[_matlab_well_index(c, py.wells)]
    dist = pd.DataFrame({
        "well": np.repeat([w.well_id for w in py.wells], D_py.shape[1]),
        "fault": np.tile(py.faults["FaultID"].to_numpy(), D_py.shape[0]),
        "matlab_distance_m": D_m.ravel(),
        "python_haversine_m": D_py.ravel(),
    })
    dist["diff_m"] = dist["python_haversine_m"] - dist["matlab_distance_m"]
    dist["rel_diff"] = dist["diff_m"] / dist["matlab_distance_m"]
    stats_["distance_max_rel_diff"] = float(dist["rel_diff"].abs().max())
    stats_["distance_mean_rel_diff"] = float(dist["rel_diff"].mean())
    return {**stats_, "_pressures": p, "_distances": dist}


def _ks(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if np.ptp(a) == 0 and np.ptp(b) == 0:
        return 1.0 if np.isclose(a[0], b[0]) else 0.0
    return float(stats.ks_2samp(a, b).pvalue)


def check_d_hydro_mc(c: MatlabCase, py: PythonInputs, seed: int, n_iter: int) -> dict:
    hp = HydrologyParams(**c.hydro_nominal, plus_minus=dict(c.hydro_sigma), n_iterations=n_iter)
    hp.validate()
    np.random.seed(seed)  # Step 5 itself is unseeded; seed here for a reproducible report
    res, samples = run_hydrology_mc_time_series(
        hp, py.wells, py.faults, [py.year], return_sample_inputs=True, result_mode="raw",
    )
    res = res[res["Year"] == py.year]
    fault_ids = py.faults["FaultID"].astype(str).to_list()
    py_mat = (res.pivot(index="SimulationID", columns="ID", values="Pressure")[fault_ids]).to_numpy()
    rows = []
    for j, fid in enumerate(fault_ids):
        m, p = c.pressures[:, j], py_mat[:, j]
        rows.append({
            "fault": fid,
            "matlab_median": np.median(m), "python_median": np.median(p),
            "matlab_p10": np.percentile(m, 10), "python_p10": np.percentile(p, 10),
            "matlab_p90": np.percentile(m, 90), "python_p90": np.percentile(p, 90),
            "matlab_mean": m.mean(), "python_mean": p.mean(),
            "median_rel_diff": (np.median(p) - np.median(m)) / np.median(m),
            "mean_rel_diff": (p.mean() - m.mean()) / m.mean(),
            "ks_pvalue": _ks(m, p),
        })
    per_fault = pd.DataFrame(rows)
    params = []
    for key, label, unit, _col in HYDRO_PARAMS:
        m, p = c.hydro_samples[key].to_numpy(), samples[key].to_numpy()
        params.append({
            "parameter": key, "label": label, "unit": unit,
            "nominal": c.hydro_nominal[key], "uncertainty": c.hydro_sigma[key],
            "matlab_min": m.min(), "matlab_max": m.max(), "python_min": p.min(), "python_max": p.max(),
            "matlab_mean": m.mean(), "python_mean": p.mean(),
            "ks_pvalue": _ks(m, p),
        })
    params = pd.DataFrame(params)
    n_low = int((per_fault["ks_pvalue"] < KS_ALPHA).sum())
    return {
        "status": "INFO",
        "iterations_python": n_iter, "iterations_matlab": len(c.pressures),
        "faults_ks_below_alpha": n_low, "faults": len(per_fault),
        "params_ks_below_alpha": int((params["ks_pvalue"] < KS_ALPHA).sum()),
        "median_rel_diff_abs_max": float(per_fault["median_rel_diff"].abs().max()),
        "median_rel_diff_abs_median": float(per_fault["median_rel_diff"].abs().median()),
        "_per_fault": per_fault, "_params": params, "_py_pressures": py_mat, "_py_samples": samples,
    }


GEO_PARAMS = [
    # key, label, unit
    ("vertical_stress_gradient", "Sv gradient", "psi/ft"),
    ("min_horizontal_stress_gradient", "Shmin gradient", "psi/ft"),
    ("initial_pore_pressure_gradient", "Pore pressure gradient", "psi/ft"),
    ("max_stress_azimuth", "SHmax azimuth", "deg"),
    ("friction_coefficient", "Friction coefficient", "-"),
    ("aphi_value", "A-Phi", "-"),
    ("strike_deviation", "Strike − nominal", "deg"),
    ("dip_deviation", "Dip − nominal (folded)", "deg"),
]


def check_e_geomech_sampling(c: MatlabCase, py: PythonInputs, seed: int, n_iter: int) -> dict:
    stress_inputs = {k: v for k, v in c.stress.items() if k != "matlab_shmax_gradient"}
    stress_inputs["friction_coefficient"] = c.friction
    validate_stress_inputs(stress_inputs, c.friction, c.stress_model_type)
    validate_geomechanics_uncertainties(stress_inputs, c.friction, c.sigmas, {})
    faults_df = py.faults.drop(columns=["FrictionCoefficient"], errors="ignore")
    _res, samples = run_geomechanics_mc(
        stress_inputs, faults_df, n_iter, dict(c.sigmas), c.stress_model_type, c.friction,
        random_seed=seed, return_sample_inputs=True,
    )
    fault_ids = faults_df["FaultID"].astype(str).to_numpy()
    nominal_strike = dict(zip(fault_ids, faults_df["Strike"].astype(float)))
    nominal_dip = dict(zip(fault_ids, faults_df["Dip"].astype(float)))
    first = samples[samples["FaultID"] == fault_ids[0]]  # stress draws are shared by all faults

    g = c.geo_samples
    m_strike_dev = (wrap_strike_degrees(g["strike_angle"]) - c.faults["Strike"].to_numpy()[:, None] + 180) % 360 - 180
    m_dip_raw_dev = g["dip_angle"] - c.faults["Dip"].to_numpy()[:, None]
    m_dip_fold_dev = fold_dip_degrees(g["dip_angle"]) - c.faults["Dip"].to_numpy()[:, None]
    p_strike_dev = (samples["strike_angle"].to_numpy() - samples["FaultID"].map(nominal_strike).to_numpy() + 180) % 360 - 180
    p_dip_dev = samples["dip_angle"].to_numpy() - samples["FaultID"].map(nominal_dip).to_numpy()

    matlab = {k: g[k].ravel() for k, _l, _u in GEO_PARAMS[:6]}
    python = {k: first[k].to_numpy() for k, _l, _u in GEO_PARAMS[:6]}
    matlab["strike_deviation"], python["strike_deviation"] = m_strike_dev.ravel(), p_strike_dev
    matlab["dip_deviation"], python["dip_deviation"] = m_dip_fold_dev.ravel(), p_dip_dev
    nominal = {
        "vertical_stress_gradient": (c.stress["vertical_stress"], c.sigmas["vertical_stress_gradient_uncertainty"]),
        "min_horizontal_stress_gradient": (c.stress["min_horizontal_stress"], c.sigmas["min_horizontal_stress_uncertainty"]),
        "initial_pore_pressure_gradient": (c.stress["pore_pressure"], c.sigmas["initial_pore_pressure_gradient_uncertainty"]),
        "max_stress_azimuth": (c.stress["max_stress_azimuth"], c.sigmas["max_stress_azimuth_uncertainty"]),
        "friction_coefficient": (c.friction, c.sigmas["friction_coefficient_uncertainty"]),
        "aphi_value": (c.aphi, c.aphi_sigma),
        "strike_deviation": (0.0, c.sigmas["strike_angles_uncertainty"]),
        "dip_deviation": (0.0, c.sigmas["dip_angles_uncertainty"]),
    }
    rows = []
    for key, label, unit in GEO_PARAMS:
        m, p = matlab[key], python[key]
        nom, sig = nominal[key]
        rows.append({
            "parameter": key, "label": label, "unit": unit, "nominal": nom, "uncertainty": sig,
            "matlab_n": m.size, "python_n": p.size,
            "matlab_min": m.min(), "matlab_max": m.max(), "python_min": p.min(), "python_max": p.max(),
            "matlab_mean": m.mean(), "python_mean": p.mean(), "matlab_std": m.std(), "python_std": p.std(),
            "ks_pvalue": _ks(m, p),
        })
    table = pd.DataFrame(rows)
    dips_folded = int((g["dip_angle"] > 90).sum())
    return {
        "status": "INFO",
        "stress_model_type": c.stress_model_type,
        "iterations_python": n_iter, "samples_matlab_per_fault": g["dip_angle"].shape[1],
        "params_ks_below_alpha": int((table["ks_pvalue"] < KS_ALPHA).sum()),
        "matlab_dip_samples_above_90": dips_folded,
        "_table": table, "_matlab": matlab, "_python": python,
        "_matlab_dip_raw_dev": m_dip_raw_dev.ravel(),
    }


# =============================================================================
# Geomechanics outputs and FSP (need MATLAB exports next to the session)
# =============================================================================
FSP_DISPLAY_HALF_STEP = 0.005  # MATLAB shows FSP as %0.2f


def _stress_inputs(c: MatlabCase) -> dict:
    s = {k: v for k, v in c.stress.items() if k != "matlab_shmax_gradient"}
    s["friction_coefficient"] = c.friction
    return s


def replay_geomech_draws(c: MatlabCase) -> np.ndarray:
    """Python slip pressure (psi) for every MATLAB geomechanics draw: (n_faults, n_draws).

    Uses the same production calls as run_geomechanics_mc's A-Phi loop
    (calculate_absolute_stresses -> calculate_fault_effective_stresses ->
    ComputeCriticalPorePressureForFailure), but with MATLAB's own per-fault draws.
    Strike is wrapped and dip folded exactly as run_geomechanics_mc does.
    """
    depth = c.stress["reference_depth"]
    out = np.zeros((len(c.geo_cells), c.geo_cells[0].shape[0]))
    for i, cell in enumerate(c.geo_cells):
        for j in range(cell.shape[0]):
            s_abs = np.asarray(cell[j, GEO_STRESS], float)
            mu = float(cell[j, GEO_MU])
            draw = {
                "reference_depth": depth,
                "vertical_stress": s_abs[0] / depth,
                "min_horizontal_stress": s_abs[1] / depth,
                "pore_pressure": float(cell[j, GEO_PP]) / depth,
                "max_stress_azimuth": float(cell[j, GEO_AZ]),
                "aphi_value": float(cell[j, GEO_APHI]),
            }
            state, p0 = calculate_absolute_stresses(draw, mu, c.stress_model_type)
            sig, tau, *_ = calculate_fault_effective_stresses(
                wrap_strike_degrees(np.array([float(cell[j, GEO_STRIKE])])),
                fold_dip_degrees(np.array([float(cell[j, GEO_DIP])])),
                state, p0, np.zeros(1),
            )
            out[i, j] = ComputeCriticalPorePressureForFailure(sig, tau, mu, p0)[0]
    return out


def python_deterministic(c: MatlabCase) -> pd.DataFrame:
    """Deterministic pore pressure to slip from the session's nominal inputs, as fsp_step2 computes it."""
    stress = _stress_inputs(c)
    state, p0 = calculate_absolute_stresses(stress, c.friction, c.stress_model_type)
    strikes = c.faults["Strike"].to_numpy(float)
    dips = c.faults["Dip"].to_numpy(float)
    sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(strikes, dips, state, p0, 0.0)
    slip = np.maximum(calculate_slip_pressure(sig, tau, c.friction, p0, 1.0, 0.5, 0.0,
                                              s11, s22, s33, s12, n1, n2), 0.0)
    df = c.faults[["FaultID", "x_km", "y_km", "Strike", "Dip", "LengthKm"]].copy()
    df["python_pp_to_slip_psi"] = np.round(slip, 3)
    df["python_label"] = np.round(slip).astype(int)  # MATLAB labels the map with whole psi
    return df


def check_f_geomech(c: MatlabCase, geo_replay: np.ndarray) -> dict:
    """Python slip pressures from MATLAB's draws vs MATLAB's CDF export."""
    cdf = c.exports.get("cdf")
    out = {"deterministic_image": str(c.exports.get("deterministic_image", "")) or None}
    if cdf is None:
        return {**out, "status": "n/a", "reason": "no CDF export next to the session"}
    values = cdf["values"]
    # MATLAB's ecdf repeats the minimum at p = 0, so columns 1.. are the 1000 sorted draws.
    export_draws = values[:, 1:]
    replay_sorted = np.sort(geo_replay, axis=1)
    replay_cdf = np.concatenate([replay_sorted[:, :1], replay_sorted], axis=1)
    if replay_cdf.shape == values.shape:
        rel = np.abs(replay_cdf - values) / np.maximum(np.abs(values), 1e-9)
        max_rel = float(rel.max())
    else:
        max_rel = float("nan")
    # The export is written with ~5 significant digits, so "same draws" means within ~1e-4.
    same_draws = bool(np.isfinite(max_rel) and max_rel <= 1e-4)
    rows = []
    for i, fid in enumerate(c.faults["FaultID"]):
        m, p = export_draws[i], geo_replay[i]
        rows.append({
            "fault": fid,
            "matlab_export_median_psi": float(np.median(m)), "python_replay_median_psi": float(np.median(p)),
            "matlab_export_p10_psi": float(np.percentile(m, 10)), "python_replay_p10_psi": float(np.percentile(p, 10)),
            "matlab_export_p90_psi": float(np.percentile(m, 90)), "python_replay_p90_psi": float(np.percentile(p, 90)),
            "median_rel_diff": float((np.median(p) - np.median(m)) / np.median(m)) if np.median(m) else float("nan"),
            "ks_pvalue": _ks(m, p),
        })
    per_fault = pd.DataFrame(rows)
    res = {
        **out,
        "status": "PASS" if same_draws else "INFO",
        "export_file": str(cdf["path"]),
        "same_draws_as_session": same_draws,
        "max_rel_vs_export": max_rel,
        "faults_ks_below_alpha": int((per_fault["ks_pvalue"] < KS_ALPHA).sum()),
        "faults": len(per_fault),
        "median_rel_diff_abs_max": float(per_fault["median_rel_diff"].abs().max()),
        "_per_fault": per_fault, "_export_draws": export_draws,
    }
    fsp = c.exports.get("fsp")
    if fsp is not None and not same_draws:
        # How much MATLAB's own FSP moves between its two geomechanics Monte Carlo runs.
        m_fsp = np.array([fsp["values"][fid] for fid in c.faults["FaultID"]])
        alt = np.array([displayed_legacy_fsp(export_draws[i], c.pressures[:, i]) for i in range(len(m_fsp))])
        res["export_run_fsp_faults_differing"] = int((np.abs(alt - m_fsp) > 1e-9).sum())
        res["export_run_fsp_max_abs_diff"] = float(np.abs(alt - m_fsp).max())
        res["_export_run_fsp"] = alt
    return res


def check_g_fsp_replay(c: MatlabCase, geo_replay: np.ndarray, hydro: dict, fsp_tol: float) -> dict:
    """Python FSP (probabilistic_fsp.py) fed MATLAB's draws, vs the MATLAB FSP export.

    ``hydro`` maps a variant name to a (n_sims, n_faults) pressure array. The gate is the
    ``matlab_hydro`` variant (MATLAB's own outsAllFaults); the others show how far the
    hydrology differences of checks B2 and C move the displayed FSP. A fault fails when
    its displayed FSP differs by more than ``fsp_tol`` (default one display step, 0.01,
    so a value sitting on a rounding boundary can't fail the run).
    """
    fsp = c.exports.get("fsp")
    if fsp is None:
        return {"status": "n/a", "reason": "no FSPvalues export next to the session"}
    fids = c.faults["FaultID"].to_list()
    m_fsp = np.array([fsp["values"][fid] for fid in fids])
    variants, table = {}, pd.DataFrame({"fault": fids, "matlab_fsp": m_fsp})
    for name, pressures in hydro.items():
        raw = np.array([legacy_fsp(geo_replay[i], pressures[:, i]) for i in range(len(fids))])
        shown = np.array([displayed_legacy_fsp(geo_replay[i], pressures[:, i]) for i in range(len(fids))])
        diff = shown - m_fsp
        variants[name] = {
            "faults_differing": int((np.abs(diff) > fsp_tol + 1e-9).sum()),  # outside tolerance
            "faults_not_identical": int((np.abs(diff) > 1e-9).sum()),
            "max_abs_diff": float(np.abs(diff).max()),
        }
        table[f"python_fsp_{name}"] = shown
        table[f"python_fsp_{name}_unrounded"] = raw
    gate = variants["matlab_hydro"]
    return {
        "status": "PASS" if gate["faults_differing"] == 0 else "FAIL",
        "export_file": str(fsp["path"]),
        "faults": len(fids),
        "faults_intermediate": int(((m_fsp > 0.0) & (m_fsp < 1.0)).sum()),
        "faults_differing": gate["faults_differing"], "faults_not_identical": gate["faults_not_identical"],
        "max_abs_diff": gate["max_abs_diff"], "tolerance": fsp_tol,
        "variants": variants,
        "_table": table,
    }


def check_h_fsp_mc(c: MatlabCase, py: PythonInputs, seed: int, n_seeds: int, geo_iter: int,
                   hydro_iter: int) -> dict:
    """Python's own Monte Carlo, end to end, over several seeds vs MATLAB's displayed FSP."""
    fsp = c.exports.get("fsp")
    if fsp is None or n_seeds <= 0:
        return {"status": "n/a", "reason": "no FSPvalues export next to the session" if fsp is None
                else "--fsp-seeds 0"}
    stress = _stress_inputs(c)
    faults_df = py.faults.drop(columns=["FrictionCoefficient"], errors="ignore")
    fids = faults_df["FaultID"].astype(str).to_list()
    hp = HydrologyParams(**c.hydro_nominal, plus_minus=dict(c.hydro_sigma), n_iterations=hydro_iter)
    hp.validate()
    runs = np.zeros((n_seeds, len(fids)))
    for k in range(n_seeds):
        s = seed + k
        geo = run_geomechanics_mc(stress, faults_df, geo_iter, dict(c.sigmas), c.stress_model_type,
                                  c.friction, random_seed=s)
        geo_p = geo["SlipPressure"].to_numpy().reshape(len(fids), geo_iter)  # fault-major
        np.random.seed(s)
        res = run_hydrology_mc_time_series(hp, py.wells, py.faults, [py.year], result_mode="raw")
        res = res[res["Year"] == py.year]
        hyd = res.pivot(index="SimulationID", columns="ID", values="Pressure")[fids].to_numpy()
        runs[k] = [legacy_fsp(geo_p[i], hyd[:, i]) for i in range(len(fids))]
    m_fsp = np.array([fsp["values"][fid] for fid in fids])
    mean, std = runs.mean(axis=0), runs.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros(len(fids))
    lo, hi = runs.min(axis=0), runs.max(axis=0)
    # MATLAB is one more Monte Carlo realization, shown rounded to 0.01.
    allowed = 3.0 * std + FSP_DISPLAY_HALF_STEP
    flagged = np.abs(m_fsp - mean) > allowed
    table = pd.DataFrame({
        "fault": fids, "matlab_fsp": m_fsp, "python_mean": mean, "python_std": std,
        "python_min": lo, "python_max": hi, "diff_mean_minus_matlab": mean - m_fsp, "flagged": flagged,
    })
    return {
        "status": "WARN" if flagged.any() else "INFO",
        "seeds": n_seeds, "first_seed": seed, "geo_iterations": geo_iter, "hydro_iterations": hydro_iter,
        "faults": len(fids), "faults_flagged": int(flagged.sum()),
        "max_abs_diff_mean": float(np.abs(mean - m_fsp).max()),
        "python_std_max": float(std.max()),
        "_table": table, "_runs": runs,
    }


# =============================================================================
# Plotting
# =============================================================================
ALT_COLOR = "#1baf7a"
GEOM_COLOR = "#6250d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# Categorical palette; the combined graphs use it for the fault types.
WELL_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
FIGSIZE = (13.333, 7.5)
DPI = 200


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 12, "axes.labelcolor": INK_2, "axes.edgecolor": "#b5b4ae",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.frameon": False, "legend.fontsize": 11,
        "lines.linewidth": 2.0, "text.color": INK,
    })
    return plt


def _figure(plt, title, subtitle, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE, **kw)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=18, fontweight="bold", color=INK)
    import textwrap

    fig.text(0.012, 0.93, "\n".join(textwrap.wrap(subtitle, 140)), ha="left", va="top",
             fontsize=12, color=INK_2, linespacing=1.4)
    return fig, axes


def _save(fig, path: Path):
    fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)


# =============================================================================
# Presentation plots (plain language, for users and stakeholders)
# =============================================================================
MATLAB_NAME = "MATLAB (reference tool)"
PYTHON_NAME = "Python (web tool)"
REGIME_PUBLIC = {"Normal": "Normal faulting", "Reverse": "Reverse faulting", "StrikeSlip": "Strike-slip faulting"}
OK_FILL, BAD_FILL, WARN_FILL, NA_FILL = "#e3f4e3", "#fbe1e1", "#fdf0cc", "#efeeea"
OK_INK, BAD_INK, WARN_INK = "#1d6b1d", "#a12a2a", "#7a5a00"


def _pct(x: float) -> str:
    p = abs(x) * 100.0
    if p < 0.01:
        return "less than 0.01%"
    return f"{p:.2f}%" if p < 1 else f"{p:.0f}%"


def _callout(ax, text, ok: bool | None):
    fill, ink = (NA_FILL, INK_2) if ok is None else ((OK_FILL, OK_INK) if ok else (BAD_FILL, BAD_INK))
    ax.text(0.02, 0.96, text, transform=ax.transAxes, va="top", ha="left", fontsize=14, fontweight="bold",
            color=ink, bbox={"boxstyle": "round,pad=0.5", "facecolor": fill, "edgecolor": "none"}, zorder=10)


# One color and marker per fault type in the combined graphs.
REGIME_STYLE = {"Normal": (GEOM_COLOR, "o"), "Reverse": (WELL_COLORS[1], "s"), "StrikeSlip": (ALT_COLOR, "^")}
COMBINED_GRID = {"left": 0.08, "right": 0.62, "top": 0.8, "bottom": 0.1}


def _injection_group(results: list, injection: str) -> list:
    return [r for r in results if r["injection"] == injection and r.get("_plot") is not None]


def _group_subtitle(group: list) -> str:
    wells = sorted({r["inputs"]["n_wells"] for r in group})
    years = sorted({r["year"] for r in group})
    return (f"{' / '.join(map(str, wells))} injection well{'s' if max(wells) > 1 else ''}, "
            f"{group[0]['injection']} injection data · year {', '.join(map(str, years))}")


def _year_notes(group: list) -> str:
    notes = [f"{REGIME_PUBLIC.get(r['regime'], r['regime'])}: Python limits this case to year {r['python_year']}; "
             f"MATLAB's result is for {r['year']}." for r in group if r["python_year"] != r["year"]]
    return "".join(f"   ·   Note: {n}" for n in notes)


def _side_legend(ax, handles=None):
    kw = {"loc": "center left", "bbox_to_anchor": (1.04, 0.5), "fontsize": 11.5, "markerscale": 1.6,
          "borderaxespad": 0.0}
    if handles is not None:
        kw["handles"] = handles
    ax.legend(**kw)


def pub_fsp_combined(plt, results: list, injection: str, path: Path, fsp_tol: float) -> bool:
    """MATLAB vs Python FSP for every fault of every fault type in one injection group."""
    from matplotlib.lines import Line2D

    group = _injection_group(results, injection)
    with_fsp = [r for r in group if r["_plot"]["fsp"] is not None]
    if not with_fsp:
        return False
    missing = [REGIME_PUBLIC.get(r["regime"], r["regime"]) for r in group if r["_plot"]["fsp"] is None]
    has_mc = any(r["_plot"]["fsp_mc"] is not None for r in with_fsp)
    sub = (f"{_group_subtitle(group)}   ·   fault slip potential is the probability (0 to 1) that a fault slips. "
           "Here Python reuses MATLAB's random samples, so every fault should sit on the dashed line.")
    if has_mc:
        sub += "   ·   Vertical bars: range of Python's results when it draws its own random samples."
    if missing:
        sub += f"   ·   No MATLAB fault slip potential values were saved for: {', '.join(missing)}."
    sub += _year_notes(with_fsp)
    fig, ax = _figure(plt, f"Fault slip potential: Python vs MATLAB, {injection} injection", sub,
                      gridspec_kw=COMBINED_GRID)
    ax.plot([-0.04, 1.04], [-0.04, 1.04], color=INK_2, lw=1.2, ls="--", zorder=1)
    handles = [Line2D([], [], color=INK_2, lw=1.2, ls="--", label="Same result")]
    n_faults = n_differ = 0
    seeds = 0
    for r in with_fsp:
        color, marker = REGIME_STYLE.get(r["regime"], (INK_2, "o"))
        t = r["_plot"]["fsp"]
        m = t["matlab_fsp"].to_numpy()
        p = t["python_fsp_matlab_hydro"].to_numpy()
        mc = r["_plot"]["fsp_mc"]
        if mc is not None:
            ht = mc.set_index("fault").loc[t["fault"]]
            ax.vlines(m, ht["python_min"].to_numpy(), ht["python_max"].to_numpy(), color=color, lw=5,
                      alpha=0.22, zorder=2)
            seeds = max(seeds, r["summary"]["H"]["seeds"])
        ax.scatter(m, p, s=60, marker=marker, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        g = r["summary"]["G"]
        n_faults += g["faults"]
        n_differ += g["faults_differing"]
        handles.append(Line2D([], [], ls="none", marker=marker, ms=8, color=color, markeredgecolor=SURFACE,
                              label=f"{REGIME_PUBLIC.get(r['regime'], r['regime'])} ({g['faults']} faults)"))
    if has_mc:
        handles.append(Line2D([], [], color=INK_2, lw=5, alpha=0.22,
                              label=f"Python's own random samples\n(range of {seeds} runs)"))
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Fault slip potential, {MATLAB_NAME}")
    ax.set_ylabel(f"Fault slip potential, {PYTHON_NAME}")
    _side_legend(ax, handles)
    _callout(ax, f"{n_faults - n_differ} of {n_faults} faults match", n_differ == 0)
    _save(fig, path)
    return True


def pub_pressure_combined(plt, results: list, injection: str, path: Path, rtol: float) -> bool:
    """MATLAB vs Python pore-pressure increase (every sample × fault) for one injection group."""
    from matplotlib.lines import Line2D

    group = _injection_group(results, injection)
    if not group:
        return False
    sub = (f"{_group_subtitle(group)}   ·   pore-pressure increase at each fault, for each of MATLAB's random "
           "hydrology samples. Points on the dashed line mean both tools give the same pressure."
           + _year_notes(group))
    fig, ax = _figure(plt, f"Pressure from injection: Python vs MATLAB, {injection} injection", sub,
                      gridspec_kw=COMBINED_GRID)
    handles, max_rel, positive = [], 0.0, []
    for r in group:
        color, marker = REGIME_STYLE.get(r["regime"], (INK_2, "o"))
        m = np.asarray(r["_plot"]["matlab_pressures"], dtype=float).ravel()
        p = np.asarray(r["_plot"]["python_pressures"], dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(p - m) / np.abs(m)
        rel = rel[np.isfinite(rel)]
        if rel.size:
            max_rel = max(max_rel, float(rel.max()))
        keep = (m > 0) & (p > 0)  # log axes
        positive.append(m[keep])
        ax.scatter(m[keep], p[keep], s=12, marker=marker, color=color, alpha=0.35, linewidth=0, zorder=2)
        n_sims, n_faults = np.shape(r["_plot"]["matlab_pressures"])
        handles.append(Line2D([], [], ls="none", marker=marker, ms=7, color=color,
                              label=f"{REGIME_PUBLIC.get(r['regime'], r['regime'])}\n"
                                    f"({n_faults} faults × {n_sims} samples)"))
    vals = np.concatenate(positive) if positive else np.array([])
    lo = max(1e-2, float(vals.min()) * 0.7) if vals.size else 1e-2
    hi = float(vals.max()) * 1.4 if vals.size else 1.0
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1.2, ls="--", zorder=1)
    handles.insert(0, Line2D([], [], color=INK_2, lw=1.2, ls="--", label="Same result"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Pressure increase (psi), {MATLAB_NAME}")
    ax.set_ylabel(f"Pressure increase (psi), {PYTHON_NAME}")
    plain = plt.FuncFormatter(lambda v, _p: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(plt.NullFormatter())
    _side_legend(ax, handles)
    _callout(ax, f"Largest difference: {_pct(max_rel)}", max_rel <= rtol)
    _save(fig, path)
    return True


def _local_km(lat, lon, lat0: float, lon0: float):
    """Small-area equirectangular projection, km east/north of (lat0, lon0)."""
    x = (np.asarray(lon, float) - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (np.asarray(lat, float) - lat0) * 110.57
    return x, y


def pub_fsp_maps(plt, results: list, path: Path) -> bool:
    """Fault maps colored by FSP: MATLAB (top row) vs Python (bottom row), one column per case."""
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from graphs.artifacts import FSP_COLOR_SCALE

    cases = [r for r in results if r.get("_plot") is not None and r["_plot"]["fsp"] is not None]
    if not cases:
        return False
    cmap = LinearSegmentedColormap.from_list("fsp", [(stop, color) for stop, color in FSP_COLOR_SCALE])
    norm = Normalize(0.0, 1.0)
    all_lat = np.concatenate([np.r_[r["_plot"]["faults"]["Latitude(WGS84)"], r["_plot"]["wells"]["Latitude(WGS84)"]]
                              for r in cases])
    all_lon = np.concatenate([np.r_[r["_plot"]["faults"]["Longitude(WGS84)"],
                                    r["_plot"]["wells"]["Longitude(WGS84)"]] for r in cases])
    lat0, lon0 = float(all_lat.mean()), float(all_lon.mean())

    n = len(cases)
    fig, axes = _figure(
        plt, "Fault slip potential map: Python vs MATLAB",
        "Each line is a fault at its real position and orientation, colored by its fault slip potential; triangles "
        "are injection wells. Python reuses MATLAB's random samples, so the top and bottom maps should be identical.",
        2, n, squeeze=False,
        gridspec_kw={"left": 0.1, "right": 0.9, "top": 0.75, "bottom": 0.1, "wspace": 0.08, "hspace": 0.12},
    )
    label_size = 12 if n <= 3 else 11
    extent = []
    for j, r in enumerate(cases):
        f = r["_plot"]["faults"].copy()
        f["FaultID"] = f["FaultID"].astype(str)
        t = r["_plot"]["fsp"].assign(fault=lambda d: d["fault"].astype(str)).set_index("fault")
        f = f[f["FaultID"].isin(t.index)]
        x, y = _local_km(f["Latitude(WGS84)"], f["Longitude(WGS84)"], lat0, lon0)
        strike = np.radians(f["Strike"].to_numpy(float))
        half = f["LengthKm"].to_numpy(float) / 2.0
        dx, dy = np.sin(strike) * half, np.cos(strike) * half
        segs = np.stack([np.c_[x - dx, y - dy], np.c_[x + dx, y + dy]], axis=1)
        extent.append(segs.reshape(-1, 2))
        wx, wy = _local_km(r["_plot"]["wells"]["Latitude(WGS84)"], r["_plot"]["wells"]["Longitude(WGS84)"],
                           lat0, lon0)
        extent.append(np.c_[wx, wy])
        values = {"matlab": t.loc[f["FaultID"], "matlab_fsp"].to_numpy(),
                  "python": t.loc[f["FaultID"], "python_fsp_matlab_hydro"].to_numpy()}
        for i, key in enumerate(("matlab", "python")):
            ax = axes[i, j]
            lc = LineCollection(segs, colors=cmap(norm(values[key])), linewidths=3.2, capstyle="round", zorder=2,
                                path_effects=[pe.Stroke(linewidth=4.6, foreground="#3a3936"), pe.Normal()])
            ax.add_collection(lc)
            ax.scatter(wx, wy, marker="^", s=90, color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(True)
        g = r["summary"]["G"]
        n_ok = g["faults"] - g["faults_differing"]
        axes[0, j].set_title(f"{REGIME_PUBLIC.get(r['regime'], r['regime'])}\n{r['injection']} data\nyear {r['year']}",
                             loc="center", fontsize=label_size)
        ok = g["faults_differing"] == 0
        axes[1, j].set_xlabel(f"{'✓' if ok else '✗'} {n_ok} of {g['faults']}\nfaults match", fontsize=label_size,
                              fontweight="bold", color=OK_INK if ok else BAD_INK, labelpad=8)
    pts = np.concatenate(extent)
    pad = 0.08 * float(np.ptp(pts, axis=0).max())
    (xmin, ymin), (xmax, ymax) = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    for ax in axes.ravel():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    axes[0, 0].set_ylabel(MATLAB_NAME, fontsize=12.5, fontweight="bold", color=INK, labelpad=10)
    axes[1, 0].set_ylabel(PYTHON_NAME, fontsize=12.5, fontweight="bold", color=INK, labelpad=10)
    # 5 km scale bar in the first map.
    ax = axes[0, 0]
    sx, sy = xmin + 0.06 * (xmax - xmin), ymin + 0.06 * (ymax - ymin)
    ax.plot([sx, sx + 5], [sy, sy], color=INK, lw=2.5, solid_capstyle="butt", zorder=4)
    ax.text(sx + 2.5, sy + 0.025 * (ymax - ymin), "5 km", ha="center", va="bottom", fontsize=10, color=INK)
    cax = fig.add_axes([0.925, 0.2, 0.014, 0.5])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Fault slip potential", fontsize=12)
    cb.outline.set_visible(False)
    _save(fig, path)
    return True


def _overview_cells(r: dict, rtol: float) -> list:
    """(text, state) for each overview column; state is ok / bad / warn / na."""
    s = r["summary"]
    a, b2, g, h = s["A"], s["B2"], s["G"], s["H"]
    cells = []
    if a["status"] == "PASS":
        cells.append(("Same", "ok"))
    else:
        cells.append((f"Differs on up to\n{a['max_days_differing_per_well']} days", "bad"))
    if b2["status"] == "PASS":
        cells.append((f"Same\n(within {_pct(b2['max_rel'])})", "ok"))
    else:
        cells.append((f"Differs\n(up to {_pct(b2['max_rel'])})", "bad"))
    if g["status"] == "n/a":
        cells.append(("No MATLAB values", "na"))
    elif g["status"] == "PASS":
        cells.append((f"Same for all\n{g['faults']} faults", "ok"))
    else:
        cells.append((f"Differs on {g['faults_differing']}\nof {g['faults']} faults", "bad"))
    if h["status"] == "n/a":
        cells.append(("No MATLAB values", "na"))
    elif h["faults_flagged"] == 0:
        cells.append(("Within normal\nrandom variation", "ok"))
    else:
        cells.append((f"{h['faults_flagged']} of {h['faults']} faults outside\nnormal random variation", "warn"))
    return cells


def pub_overview(plt, results: list, path: Path, rtol: float, fsp_tol: float):
    from matplotlib.patches import FancyBboxPatch

    cols = ["Injection schedule", "Pressure from injection\nat each fault", "Fault slip potential\n(shared samples)",
            "Fault slip potential\n(independent runs)"]
    n = len(results)
    fig = plt.figure(figsize=FIGSIZE)
    fig.suptitle("Does the Python web tool reproduce the reference MATLAB tool?", x=0.012, y=0.985, ha="left",
                 fontsize=20, fontweight="bold")
    fig.text(0.012, 0.925, f"{n} test cases: three fault types, with annual and with monthly injection data. "
             "Each case was run in both tools with the same inputs.", fontsize=12.5, color=INK_2)
    ax = fig.add_axes([0.01, 0.2, 0.98, 0.69])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    x0, w0 = 0.0, 0.25
    cw = (1.0 - w0) / len(cols)
    for j, col in enumerate(cols):
        ax.text(w0 + (j + 0.5) * cw, n + 0.5, col, ha="center", va="center", fontsize=12, fontweight="bold")
    fills = {"ok": (OK_FILL, OK_INK, "✓ "), "bad": (BAD_FILL, BAD_INK, "✗ "), "warn": (WARN_FILL, WARN_INK, "! "),
             "na": (NA_FILL, INK_2, "")}
    footnotes = []
    for i, r in enumerate(results):
        y = n - 1 - i
        wells = r["inputs"]["n_wells"]
        ax.text(x0 + 0.005, y + 0.5, f"{REGIME_PUBLIC.get(r['regime'], r['regime'])}\n{wells} well"
                f"{'s' if wells > 1 else ''}, {r['injection']} data, year {r['year']}",
                ha="left", va="center", fontsize=11.5)
        for j, (text, state) in enumerate(_overview_cells(r, rtol)):
            fill, ink, icon = fills[state]
            ax.add_patch(FancyBboxPatch((w0 + j * cw + 0.006, y + 0.08), cw - 0.012, 0.84,
                                        boxstyle="round,pad=0,rounding_size=0.012", facecolor=fill,
                                        edgecolor="none", mutation_aspect=n / 2))
            ax.text(w0 + (j + 0.5) * cw, y + 0.5, icon + text, ha="center", va="center", fontsize=11, color=ink,
                    fontweight="bold" if state != "na" else "normal", linespacing=1.3)
        if r["python_year"] != r["year"]:
            footnotes.append(f"{REGIME_PUBLIC.get(r['regime'], r['regime'])}, {r['injection']} data: MATLAB's result "
                             f"is for {r['year']}; the Python tool limits this case to {r['python_year']}.")
    notes = ["“Same” allows for tiny numerical differences between the two programming languages "
             f"(pressure within {rtol * 100:g}%, fault slip potential within {fsp_tol:g}).",
             "Shared samples: Python reuses MATLAB's random samples, so results should be identical. Independent "
             "runs: Python draws its own samples; results should agree within normal random variation."] + footnotes
    import textwrap

    wrapped = [line for note in notes for line in textwrap.wrap(note, 175)]
    fig.text(0.012, 0.025, "\n".join(wrapped), fontsize=10.5, color=INK_2, va="bottom", linespacing=1.5)
    _save(fig, path)


# =============================================================================
# Reporting
# =============================================================================
def _public(d: dict) -> dict:
    """Drop private (underscore) entries holding arrays/DataFrames."""
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        out[k] = _public(v) if isinstance(v, dict) else v
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (date, datetime, Path)):
        return str(o)
    raise TypeError(type(o))


INPUT_NOTES = [
    "MATLAB draws the stress parameters independently for every fault; Python draws one value per simulation "
    "and shares it across faults. Per-fault distributions are the same; the cross-fault correlation is not.",
    "MATLAB samples dip unbounded (e.g. 85–95° around 90°). Python folds dip into [0, 90]; with a vertical "
    "principal stress the slip pressure is identical, so Check E compares the folded values.",
    "MATLAB stores hydrology gravity 9.8066 m/s²; Python uses 9.81. g cancels in the Theis solution "
    "(S and T both scale with ρg), so it does not affect the pressures.",
]


def _fgh_cells(s: dict) -> list:
    f, g, h = s["F"], s["G"], s["H"]
    f_txt = "n/a" if f["status"] == "n/a" else (
        "same draws, exact" if f["same_draws_as_session"] else f"{f['faults_ks_below_alpha']}/{f['faults']}")
    g_txt = "n/a" if g["status"] == "n/a" else f"{g['status']} {g['faults_differing']}/{g['faults']}"
    h_txt = "n/a" if h["status"] == "n/a" else f"{h['faults_flagged']}/{h['faults']}"
    return [f_txt, g_txt, h_txt]


def write_summary_md(results: list, args, path: Path, overall: str):
    lines = [
        "# FSP regression: Python vs legacy MATLAB",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} · overall **{overall}** · numba "
        f"{'available' if _NUMBA_AVAILABLE else 'NOT available'} · EPSG:{args.epsg} · seed {args.seed}",
        "",
        "| Case | Year | I inputs | A schedule | B1 kernel max rel | B2 pipeline max rel | C geometry max rel | "
        "D faults KS p<0.01 | E params KS p<0.01 | F CDF faults KS p<0.01 | G FSP replay faults differing | "
        "H faults outside Python MC 3σ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["summary"]
        lines.append(
            f"| {r['label'].replace(' | ', ' · ')} | {r['year']} | {s['I']['status']} | "
            f"{s['A']['status']} ({s['A']['days_differing']} d) | "
            f"{s['B1']['status']} {s['B1']['max_rel']:.2e} | {s['B2']['status']} {s['B2']['max_rel']:.2e} | "
            f"{s['C']['status']} {s['C']['max_rel']:.2e} | {s['D']['faults_ks_below_alpha']}/{s['D']['faults']} | "
            f"{s['E']['params_ks_below_alpha']}/8 | " + " | ".join(_fgh_cells(s)) + " |"
        )
    lines += [
        "",
        f"Tolerances: A passes when no well's daily rate differs on more than {args.schedule_days} day(s). "
        f"B1/B2 pass when |Δ| ≤ {args.kernel_atol:g} psi + {args.kernel_rtol * 100:g}% of MATLAB's pressure for "
        f"every simulation and fault. C warns above {args.geometry_rtol * 100:g}%. G passes when every displayed "
        f"FSP is within {args.fsp_tol:g} of MATLAB's. n/a = the session has no MATLAB export for that check.",
        "",
        "## Sessions",
        "",
    ]
    for r in results:
        ex = r.get("exports", {})
        lines.append(f"- **{r['label'].replace(' | ', ' · ')}**: `{r['file']}`"
                     + (f"; exports: {', '.join(sorted(ex))}" if ex else "; no exports"))
    lines += [
        "",
        "## Input consistency (I)",
        "",
        f"The WEB input CSVs Python runs on (`{args.inputs_dir}`) vs what each MATLAB session ran with. Positions "
        f"are compared after projecting the WEB lat/lon to EPSG:{args.epsg}; the session's fault table is matched "
        "against the MATLAB-format fault CSVs.",
        "",
    ]
    for r in results:
        i = r["summary"]["I"]
        lines.append(
            f"- **{r['label'].replace(' | ', ' · ')}**: session fault file "
            f"`{i.get('session_fault_file', 'n/a')}`; fault offset max {i['fault_position_max_offset_m']:.1f} m; "
            f"well offset max {i['well_position_max_offset_m']:.1f} m"
        )
        for issue in i["issues"]:
            lines.append(f"  - {issue}")
    lines += [
        "",
        "## What each check means",
        "",
        "- **I** - Input files given to Python vs the MATLAB session (fault table used, positions, strike/dip/length, "
        "well rates). Informational.",
        "- **A** - Python's well loader (`normalize_wells_to_well_data`) vs MATLAB's saved rate schedule, day by day.",
        "- **B1** - Python Theis kernel fed MATLAB's own schedule, distances and Monte Carlo samples. It also "
        "probes which evaluation instant MATLAB used.",
        "- **B2** - Python's hydrology pipeline (its well schedule and year-of-interest convention) with MATLAB's "
        "distances and samples. Both the scipy and the numba Theis paths are checked.",
        "- **C** - B2 with production geometry: the Python lat/lon inputs → haversine distance.",
        "- **D** - Python's own hydrology Monte Carlo vs MATLAB's pressure distribution (independent random streams).",
        "- **E** - Python's geomechanics Monte Carlo input sampling vs MATLAB's saved samples.",
        "- **F** - Python's slip pressure for each of MATLAB's saved geomechanics draws vs MATLAB's CDF export "
        "(exact if the export came from the same Monte Carlo run as the session, otherwise a KS test per fault). "
        "Also draws the deterministic pore pressure to slip next to MATLAB's screenshot.",
        "- **G** - Python's FSP code (`probabilistic_fsp.py`) fed MATLAB's geomechanics draws and MATLAB's "
        "hydrology pressures, vs MATLAB's displayed FSP. Also shown with Python's hydrology (B2, C).",
        "- **H** - Python's own Monte Carlo, end to end (its sampling, its hydrology pipeline and geometry), "
        f"repeated over {args.fsp_seeds} seeds. A fault is flagged when MATLAB's FSP is outside Python's mean "
        f"± 3σ (+ {FSP_DISPLAY_HALF_STEP} display rounding).",
        "",
        "## Kernel scale factor (B1)",
        "",
        "Python / MATLAB pressure ratio over every simulation and fault, with MATLAB's own schedule and distances. "
        "A spread near zero means the whole residual is one constant factor, independent of distance and time. "
        "Legacy MATLAB converts Pa → bar with ×1e-5 (`pfront.m`) and bar → psi with ×14.5 (`hydro1D.m`), i.e. "
        "×1.45e-4; Python uses 1/6894.76 = 1.450377e-4. That gives a ratio of "
        f"{(1 / 6894.76) / 1.45e-4:.7f}. MATLAB also evaluates at Jan 1 of the year of interest "
        "(`pfront.m`), Python at Dec 31 of the year before.",
        "",
    ]
    for r in results:
        b1 = r["summary"]["B1"]
        lines.append(
            f"- {r['label'].replace(' | ', ' · ')}: ratio median {b1['python_over_matlab_ratio_median']:.7f}, "
            f"spread {b1['python_over_matlab_ratio_spread']:.1e}"
        )
    lines += ["", "## Evaluation instant and schedule details", ""]
    for r in results:
        s = r["summary"]
        b1 = s["B1"]
        probe = ", ".join(f"{k}: {v:.2e}" for k, v in b1["instant_probe_max_rel"].items())
        lines.append(
            f"- **{r['label'].replace(' | ', ' · ')}** - MATLAB evaluation instant (best fit) {b1['matlab_eval_instant']} "
            f"(max rel per candidate: {probe}); Python evaluates {s['A']['python_eval_days_after_start']} days after "
            f"first injection, MATLAB {s['A']['matlab_eval_days_after_start_if_jan1']}."
        )
        for w in s["A"]["wells"]:
            if w["days_differing"]:
                lines.append(
                    f"  - well {w['well']}: {w['days_differing']} day(s) differ "
                    f"({w['first_differing_day']} → {w['last_differing_day']}); MATLAB shut-in {w['matlab_shut_in']}, "
                    f"Python shut-in {w['python_shut_in']}; volume rel diff {w['volume_rel_diff']:+.2e}"
                )
        for note in r["notes"]:
            lines.append(f"  - {note}")
    lines += ["", "## Geomechanics and FSP (F, G, H)", ""]
    for r in results:
        s = r["summary"]
        f, g, h = s["F"], s["G"], s["H"]
        lines.append(f"- **{r['label'].replace(' | ', ' · ')}**")
        if f["status"] == "n/a":
            lines.append(f"  - F: {f['reason']}")
        else:
            lines.append(
                f"  - F: CDF export {'is' if f['same_draws_as_session'] else 'is NOT'} from the session's own draws "
                f"(max rel {f['max_rel_vs_export']:.2e} vs Python's slip pressures); {f['faults_ks_below_alpha']}/"
                f"{f['faults']} faults KS p < {KS_ALPHA}; median differs by up to {f['median_rel_diff_abs_max']:.1%}")
            if "export_run_fsp_max_abs_diff" in f:
                lines.append(
                    f"  - F: FSP computed from the export's draws instead of the session's differs from MATLAB's "
                    f"displayed FSP on {f['export_run_fsp_faults_differing']} faults (max "
                    f"{f['export_run_fsp_max_abs_diff']:.2f}) - MATLAB's own run-to-run noise")
        if g["status"] == "n/a":
            lines.append(f"  - G: {g['reason']}")
        else:
            v = g["variants"]
            lines.append(
                f"  - G: {g['status']}; with MATLAB's hydrology {v['matlab_hydro']['faults_differing']}/{g['faults']} "
                f"faults differ ({g['faults_intermediate']} faults have 0 < FSP < 1); with Python's B2 hydrology "
                f"{v['python_b2']['faults_differing']} differ (max {v['python_b2']['max_abs_diff']:.2f}); with "
                f"Python's C geometry {v['python_c']['faults_differing']} differ (max {v['python_c']['max_abs_diff']:.2f})")
        if h["status"] == "n/a":
            lines.append(f"  - H: {h['reason']}")
        else:
            lines.append(
                f"  - H: {h['faults_flagged']}/{h['faults']} faults outside Python's mean ± 3σ over {h['seeds']} seeds; "
                f"max |Python mean − MATLAB| {h['max_abs_diff_mean']:.3f}; Python run-to-run σ up to "
                f"{h['python_std_max']:.3f}")
    lines += ["", "## Input notes", ""] + [f"- {n}" for n in INPUT_NOTES] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Driver
# =============================================================================
def run_case(c: MatlabCase, args, proj: Projector) -> dict:
    case_dir = args.out / c.name
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {c.name}  ({c.label}, year {c.year})")
    py = build_python_inputs(c, proj, case_dir / "inputs", args.inputs_dir)
    inp = py.consistency
    print(f"  I  inputs           {inp['status']:4s}  "
          f"fault offset max {inp['fault_position_max_offset_m']:.1f} m, "
          f"well offset max {inp['well_position_max_offset_m']:.1f} m")
    for issue in inp["issues"]:
        print(f"       - {issue}")

    a = check_a_schedule(c, py, args.schedule_days)
    print(f"  A  schedule         {a['status']:4s}  {a['days_differing']} day(s) differ")

    strho = _strho(c.hydro_samples)
    D_matlab = matlab_distance_matrix_m(c)
    b1 = check_b1_kernel(c, strho, D_matlab, args.kernel_rtol, args.kernel_atol)
    print(f"  B1 Theis kernel     {b1['status']:4s}  max rel {b1['max_rel']:.2e}  "
          f"max |Δ| {b1['max_abs_psi']:.3g} psi  (MATLAB instant {b1['matlab_eval_instant']})")
    b2 = check_b2_pipeline(c, py, strho, D_matlab, args.kernel_rtol, args.kernel_atol)
    b2_main = b2["numba"] or b2["scipy"]
    b2_flat = {**b2_main, "status": b2["status"], "scipy": b2["scipy"], "numba": b2["numba"],
               "numba_vs_scipy_max_rel": b2.get("numba_vs_scipy_max_rel")}
    print(f"  B2 Python pipeline  {b2['status']:4s}  max rel {b2_main['max_rel']:.2e}  "
          f"max |Δ| {b2_main['max_abs_psi']:.3g} psi"
          + (f"  (numba vs scipy {b2['numba_vs_scipy_max_rel']:.1e})" if b2.get("numba") else "  (scipy only)"))
    cc = check_c_geometry(c, py, strho, D_matlab, args.geometry_rtol)
    print(f"  C  lat/lon geometry {cc['status']:4s}  max rel {cc['max_rel']:.2e}  "
          f"(distance diff up to {cc['distance_max_rel_diff'] * 100:.3f}%)")
    d = check_d_hydro_mc(c, py, args.seed, args.hydro_iterations or c.hydro_iterations)
    print(f"  D  hydrology MC     INFO  {d['faults_ks_below_alpha']}/{d['faults']} faults KS p<{KS_ALPHA}, "
          f"|median rel diff| median {d['median_rel_diff_abs_median']:.1%} max {d['median_rel_diff_abs_max']:.1%}")
    e = check_e_geomech_sampling(c, py, args.seed, args.geo_iterations)
    print(f"  E  geomech sampling INFO  {e['params_ks_below_alpha']}/8 params KS p<{KS_ALPHA}")

    geo_replay = replay_geomech_draws(c)
    det = python_deterministic(c)
    f = check_f_geomech(c, geo_replay)
    if f["status"] == "n/a":
        print(f"  F  geomech CDF      n/a   {f['reason']}")
    else:
        print(f"  F  geomech CDF      {f['status']:4s}  export {'=' if f['same_draws_as_session'] else '!='} session "
              f"draws (max rel {f['max_rel_vs_export']:.1e}); {f['faults_ks_below_alpha']}/{f['faults']} faults KS "
              f"p<{KS_ALPHA}"
              + (f"; FSP from export draws differs on {f['export_run_fsp_faults_differing']} faults "
                 f"(max {f['export_run_fsp_max_abs_diff']:.2f})" if "export_run_fsp_max_abs_diff" in f else ""))
    g = check_g_fsp_replay(c, geo_replay, {
        "matlab_hydro": c.pressures, "python_b2": b2["_pressures"], "python_c": cc["_pressures"],
    }, args.fsp_tol)
    if g["status"] == "n/a":
        print(f"  G  FSP replay       n/a   {g['reason']}")
    else:
        v = g["variants"]
        print(f"  G  FSP replay       {g['status']:4s}  {g['faults_differing']}/{g['faults']} faults differ "
              f"({g['faults_intermediate']} with 0<FSP<1); with Python hydrology B2: "
              f"{v['python_b2']['faults_differing']} differ, C: {v['python_c']['faults_differing']} differ "
              f"(max {v['python_c']['max_abs_diff']:.2f})")
    h = check_h_fsp_mc(c, py, args.seed, args.fsp_seeds, args.geo_iterations,
                       args.hydro_iterations or c.hydro_iterations)
    if h["status"] == "n/a":
        print(f"  H  FSP Monte Carlo  n/a   {h['reason']}")
    else:
        print(f"  H  FSP Monte Carlo  {h['status']:4s}  {h['faults_flagged']}/{h['faults']} faults outside Python "
              f"mean ± 3σ ({h['seeds']} seeds); max |mean − MATLAB| {h['max_abs_diff_mean']:.3f}")

    # ---- CSV artifacts -----------------------------------------------------
    fault_ids = c.faults["FaultID"].to_numpy()

    def long(p, name):
        sims, faults = np.meshgrid(np.arange(1, p.shape[0] + 1), fault_ids, indexing="ij")
        df = pd.DataFrame({"SimulationID": sims.ravel(), "FaultID": faults.ravel(),
                           "matlab_psi": c.pressures.ravel(), "python_psi": p.ravel()})
        df["diff_psi"] = df["python_psi"] - df["matlab_psi"]
        df["rel_diff"] = df["diff_psi"] / df["matlab_psi"]
        df.to_csv(case_dir / name, index=False)

    long(b1["_pressures"], "hydro_replay_B1_kernel.csv")
    long(b2["_pressures"], "hydro_replay_B2_pipeline.csv")
    long(cc["_pressures"], "hydro_replay_C_python_geometry.csv")
    pd.DataFrame(a["wells"]).to_csv(case_dir / "injection_schedule_A.csv", index=False)
    cc["_distances"].to_csv(case_dir / "distances.csv", index=False)
    d["_per_fault"].to_csv(case_dir / "hydro_mc_D_per_fault.csv", index=False)
    d["_params"].to_csv(case_dir / "hydro_mc_D_parameters.csv", index=False)
    e["_table"].to_csv(case_dir / "geomech_sampling_E.csv", index=False)
    inp["fault_position_offsets_m"].to_csv(case_dir / "input_fault_offsets_I.csv", index=False)
    det.to_csv(case_dir / "geomech_deterministic_F.csv", index=False)
    pd.DataFrame(geo_replay.T, columns=fault_ids).to_csv(
        case_dir / "geomech_replay_F_python_slip_pressure_per_matlab_draw.csv", index_label="draw")
    if f["status"] != "n/a":
        f["_per_fault"].to_csv(case_dir / "geomech_cdf_F_per_fault.csv", index=False)
    if g["status"] != "n/a":
        g["_table"].to_csv(case_dir / "fsp_replay_G.csv", index=False)
    if h["status"] != "n/a":
        h["_table"].to_csv(case_dir / "fsp_monte_carlo_H.csv", index=False)

    inp_public = {k: v for k, v in inp.items() if not isinstance(v, pd.DataFrame)}
    summary = {
        "I": inp_public, "A": _public(a), "B1": _public(b1), "B2": _public(b2_flat), "C": _public(cc),
        "D": _public(d), "E": _public(e), "F": _public(f), "G": _public(g), "H": _public(h),
    }
    exports = {k: str(v["path"] if isinstance(v, dict) else v) for k, v in c.exports.items()}
    return {
        "case": c.name, "label": c.label, "file": str(c.path), "year": c.year, "exports": exports,
        "python_year": py.year, "regime": c.regime, "injection": c.injection,
        "inputs": {
            "stress_model_type": c.stress_model_type,
            "stress": {k: v for k, v in c.stress.items()}, "aphi": c.aphi, "aphi_uncertainty": c.aphi_sigma,
            "geomechanics_uncertainties": c.sigmas, "friction": c.friction,
            "fault_dip_range": [float(c.faults["Dip"].min()), float(c.faults["Dip"].max())],
            "n_wells": int(c.wells["WellID"].nunique()),
            "hydrology": c.hydro_nominal, "hydrology_uncertainties": c.hydro_sigma,
            "faults_csv": str(py.faults_csv), "wells_csv": str(py.wells_csv),
        },
        "notes": list(c.notes),
        "summary": summary,
        # Data for the combined presentation graphs; dropped before summary.json is written.
        "_plot": {
            "matlab_pressures": c.pressures, "python_pressures": b2["_pressures"],
            "fsp": g["_table"] if g["status"] != "n/a" else None,
            "fsp_mc": h["_table"] if h["status"] != "n/a" else None,
            "faults": py.faults[["FaultID", "Latitude(WGS84)", "Longitude(WGS84)", "Strike", "LengthKm"]],
            "wells": pd.read_csv(py.wells_csv)[["Latitude(WGS84)", "Longitude(WGS84)"]].drop_duplicates(),
        },
        "passed": all(summary[k]["status"] == "PASS" for k in ("A", "B1", "B2"))
        and summary["G"]["status"] in ("PASS", "n/a"),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", default=["all"],
                   help="'all' or substrings of case or session file names (e.g. Reverse Monthly)")
    p.add_argument("--reference-dir", type=Path, default=HERE / "MATLAB Results",
                   help="folder searched recursively for FSP_*_MATLAB session files (one per case), "
                        "with any MATLAB exports (CDF_*, FSPvalues_*, PP2SLIP_*) next to each session")
    p.add_argument("--fsp-seeds", type=int, default=30,
                   help="Python Monte Carlo repetitions for check H (0 to skip)")
    p.add_argument("--out", type=Path, default=HERE / "output",
                   help="base folder; each run writes into a new regression_results_DDMMYYYY_HHMMSS folder in it")
    p.add_argument("--epsg", type=int, default=32613,
                   help="projected CRS of MATLAB's km coordinates, used by check I (default UTM 13N)")
    p.add_argument("--inputs-dir", type=Path, default=HERE / "Input Data",
                   help="input CSVs: Fault Data/{WEB,MATLAB} and Injection Data/{WEB,MATLAB}/{Annual*,Monthly*}. "
                        "Python runs on the WEB files.")
    p.add_argument("--seed", type=int, default=200, help="seed for Python's Monte Carlo runs (Checks D, E)")
    p.add_argument("--geo-iterations", type=int, default=1000)
    p.add_argument("--hydro-iterations", type=int, default=None,
                   help="Python hydrology MC iterations (default: MATLAB's count from the session)")
    p.add_argument("--kernel-rtol", type=float, default=0.002,
                   help="B1/B2 relative pressure tolerance (0.002 = 0.2%%)")
    p.add_argument("--kernel-atol", type=float, default=0.01, help="B1/B2 absolute pressure tolerance, psi")
    p.add_argument("--geometry-rtol", type=float, default=0.05,
                   help="C warns above this relative pressure difference (0.05 = 5%%)")
    p.add_argument("--schedule-days", type=int, default=1,
                   help="A passes when no well's daily injection rate differs on more than this many days")
    p.add_argument("--fsp-tol", type=float, default=0.01,
                   help="G passes when every displayed FSP is within this of MATLAB's")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    args.out = args.out / f"regression_results_{datetime.now():%d%m%Y_%H%M%S}"
    results, errors = [], []
    loaded = []
    for path in discover_cases(args.reference_dir):
        try:
            loaded.append(load_matlab_case(path))
        except Exception as exc:  # a session that can't be decoded fails the run
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR loading {path.name}: {exc}", file=sys.stderr)
    try:
        cases = select_cases(loaded, args.cases)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not cases:
        print(f"No MATLAB sessions matching {args.cases} under {args.reference_dir}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    plt = None if args.no_plots else _setup_matplotlib()
    proj = Projector(args.epsg)
    print(f"FSP MATLAB regression - {len(cases)} case(s), numba {'on' if _NUMBA_AVAILABLE else 'OFF'}, "
          f"output {args.out}")

    for case in cases:
        path = case.path
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                results.append(run_case(case, args, proj))
        except Exception as exc:  # report and keep going; a crashed case fails the run
            import traceback

            traceback.print_exc()
            errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR {path.name}: {exc}", file=sys.stderr)

    overall = "PASS" if results and not errors and all(r["passed"] for r in results) else "FAIL"
    summary_dir = args.out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "overall": overall, "numba_available": _NUMBA_AVAILABLE,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "input_notes": INPUT_NOTES, "errors": errors,
        "cases": [{k: v for k, v in r.items() if k != "_plot"} for r in results],
    }
    (summary_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    if results:
        write_summary_md(results, args, summary_dir / "summary.md", overall)
        if plt is not None:
            pres = args.out / "presentation"
            pres.mkdir(parents=True, exist_ok=True)
            pub_overview(plt, results, pres / "00_overview.png", args.kernel_rtol, args.fsp_tol)
            for k, injection in enumerate(("annual", "monthly")):
                pub_fsp_combined(plt, results, injection, pres / f"{1 + k:02d}_fsp_{injection}.png", args.fsp_tol)
                pub_pressure_combined(plt, results, injection, pres / f"{3 + k:02d}_pressure_{injection}.png",
                                      args.kernel_rtol)
            pub_fsp_maps(plt, results, pres / "05_fsp_map.png")
    print(f"\nOverall: {overall}   ({sum(r['passed'] for r in results)}/{len(results)} cases passed exact checks"
          f"{', ' + str(len(errors)) + ' errored' if errors else ''})")
    print(f"Results: {args.out}")
    print(f"Report: {summary_dir / 'summary.md'}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
