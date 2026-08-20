"""
Monte Carlo hydrology simulation.

When numba is available (the expected case — see requirements.txt), all
n_iterations simulations for a given (well, year) are evaluated in a single
threaded numba kernel call (`theis.theis_accumulate_all_sims`); see
`_run_all_sims_batched` below. This replaces a previous joblib/loky process
pool over one Python-level simulation per task, which — measured on this
codebase's own `tests/benchmark_hydrology_scale.py` geometry — spent 98% of
its time inside `scipy.special.exp1` while achieving only ~13% parallel
efficiency on a 22-core machine (312.8s for 750 sims). The batched kernel
reproduces the same values (max relative difference ~1e-7, identical after
the pipeline's own 2-decimal rounding) in ~33s for the same case.

If numba is unavailable, `_single_sim_pressure` + joblib below remain the
fallback path, preserving the original behaviour exactly.
"""
import numpy as np
import pandas as pd
from datetime import date
from joblib import Parallel, delayed

from ..hydrology.params import calcST
from ..hydrology.pressure_field import (
    pfieldcalc_all_rates_for_distances,
    well_fault_distances_m,
)
from ..hydrology.theis import _NUMBA_AVAILABLE, theis_accumulate_all_sims

if _NUMBA_AVAILABLE:
    import numba


def _single_sim_pressure(well_data_list, STRho, distance_matrix_m,
                          years_to_analyze):
    """Compute total pressure per (fault, year) for one set of ST parameters.

    Fallback path used only when numba is unavailable; see module docstring.
    """
    results = {}  # year -> array[n_faults]
    n_faults = distance_matrix_m.shape[1]

    for analysis_year in years_to_analyze:
        cutoff_date = date(analysis_year - 1, 12, 31)
        total = np.zeros(n_faults, dtype=float)

        for wi, wd in enumerate(well_data_list):
            if wd.start_date > cutoff_date:
                continue
            if wd.start_year > analysis_year:
                continue
            if len(wd.days) == 0:
                continue
            eval_days = float((cutoff_date - wd.start_date).days + 1)

            total += pfieldcalc_all_rates_for_distances(
                distance_matrix_m[wi], STRho, wd.days, wd.rates, eval_days
            )

        results[analysis_year] = np.maximum(total, 0.0)
    return results


def _run_all_sims_batched(well_data_list, STRho_list, distance_matrix_m,
                           years_to_analyze, n_jobs=-1):
    """Compute total pressure per (fault, year, sim) for every simulation at once.

    Threaded numba replacement for looping `_single_sim_pressure` per
    simulation through joblib. Returns {year: (n_sims, n_faults) array}.
    """
    n_sims = len(STRho_list)
    n_faults = distance_matrix_m.shape[1]

    if n_jobs and n_jobs > 0:
        numba.set_num_threads(min(n_jobs, numba.config.NUMBA_NUM_THREADS))

    a_s = np.array([S / (4.0 * T) for S, T, _rho in STRho_list], dtype=float)
    b_s = np.array(
        [(rho * 9.81 / 6894.76) / (4.0 * np.pi * T) for S, T, rho in STRho_list],
        dtype=float,
    )

    results = {}
    for analysis_year in years_to_analyze:
        cutoff_date = date(analysis_year - 1, 12, 31)
        out = np.zeros((n_sims, n_faults), dtype=float)

        for wi, wd in enumerate(well_data_list):
            if wd.start_date > cutoff_date:
                continue
            if wd.start_year > analysis_year:
                continue
            if len(wd.days) == 0:
                continue
            eval_days = float((cutoff_date - wd.start_date).days + 1)

            theis_accumulate_all_sims(
                distance_matrix_m[wi], wd.days, wd.rates, eval_days,
                a_s, b_s, out,
            )

        results[analysis_year] = out
    return results


def run_hydrology_mc_time_series(hydro_params, well_data_list,
                                  fault_df: pd.DataFrame,
                                  years_to_analyze,
                                  n_jobs: int = -1,
                                  return_sample_inputs: bool = False,
                                  result_mode: str = "raw",
                                  sample_year=None):
    """Monte Carlo hydrology time series.

    Parameters
    ----------
    hydro_params : HydrologyParams
    well_data_list : list of WellData (already pre-processed, filtered)
    fault_df : DataFrame with Latitude(WGS84), Longitude(WGS84), FaultID
    years_to_analyze : iterable of int

    result_mode:
        "raw" returns one row per SimulationID/Fault/Year.
        "mean" returns mean pressure per Fault/Year without materializing raw rows.
        "year_samples" returns raw simulation rows for sample_year only.

    If return_sample_inputs is true, also returns sampled hydrology inputs by SimulationID.
    """
    n_sims = hydro_params.n_iterations
    pm = hydro_params.plus_minus
    years = list(years_to_analyze)

    def _unc(key):
        return float(pm.get(key, 0.0))

    def _samp(base, delta, lo=None, hi=None):
        lo_v = max(base - delta, lo) if lo is not None else base - delta
        hi_v = min(base + delta, hi) if hi is not None else base + delta
        if lo_v >= hi_v:
            return np.full(n_sims, base)
        return np.random.uniform(lo_v, hi_v, n_sims)

    h_samples = _samp(hydro_params.aquifer_thickness, _unc("aquifer_thickness"), lo=0.001)
    phi_samples = _samp(hydro_params.porosity, _unc("porosity"), lo=0.001, hi=0.999)
    kap_samples = _samp(hydro_params.permeability, _unc("permeability"), lo=1e-6)
    rho_samples = _samp(hydro_params.fluid_density, _unc("fluid_density"), lo=1.0)
    mu_samples = _samp(hydro_params.dynamic_viscosity, _unc("dynamic_viscosity"), lo=1e-6)
    beta_samples = _samp(hydro_params.fluid_compressibility, _unc("fluid_compressibility"), lo=0.0)
    alphav_samples = _samp(hydro_params.rock_compressibility, _unc("rock_compressibility"), lo=0.0)

    fault_lats = fault_df["Latitude(WGS84)"].values.astype(float)
    fault_lons = fault_df["Longitude(WGS84)"].values.astype(float)
    fault_ids = fault_df["FaultID"].astype(str).values
    distance_matrix_m = well_fault_distances_m(well_data_list, fault_lats, fault_lons)

    # Pre-compute STRho for all simulations
    STRho_list = [
        calcST(h_samples[i], phi_samples[i], kap_samples[i],
               rho_samples[i], mu_samples[i], 9.81,
               beta_samples[i], alphav_samples[i])
        for i in range(n_sims)
    ]

    n_faults = len(fault_ids)
    mode = str(result_mode or "raw").lower()

    years_to_emit = years
    if mode == "year_samples":
        selected_year = int(sample_year) if sample_year is not None else years[-1]
        years_to_emit = [selected_year] if selected_year in years else years[-1:]
    years_needed = years if mode == "mean" else years_to_emit

    # year -> (n_sims, n_faults) pressure array, one row per simulation
    if _NUMBA_AVAILABLE:
        year_matrix = _run_all_sims_batched(
            well_data_list, STRho_list, distance_matrix_m, years_needed, n_jobs=n_jobs
        )
    else:
        sim_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_single_sim_pressure)(well_data_list, STRho_list[i],
                                           distance_matrix_m, years_needed)
            for i in range(n_sims)
        )
        year_matrix = {}
        for yr in years_needed:
            mat = np.zeros((n_sims, n_faults), dtype=float)
            for sim_i, sim_result in enumerate(sim_results):
                pressures = sim_result.get(yr)
                if pressures is not None:
                    mat[sim_i, :] = pressures
            year_matrix[yr] = mat

    if mode == "mean":
        # Accumulate per-year mean pressures across all simulations
        year_pressures = {
            yr: year_matrix[yr].mean(axis=0)
            for yr in years if yr in year_matrix
        }

        valid_years = sorted(year_pressures.keys())
        n_years = len(valid_years)
        if n_years > 0:
            pressure_2d = np.column_stack([year_pressures[yr] for yr in valid_years])  # (n_faults, n_years)
            results_df = pd.DataFrame({
                "ID": np.tile(fault_ids, n_years),
                "Pressure": pressure_2d.T.ravel(),
                "Year": np.repeat(valid_years, n_faults),
            })
        else:
            results_df = pd.DataFrame(columns=["ID", "Pressure", "Year"])

    else:
        n_years = len(years_to_emit)
        # Stack into a 3D array: (n_sims, n_years, n_faults)
        pressure_3d = np.zeros((n_sims, n_years, n_faults), dtype=float)
        for yi, yr in enumerate(years_to_emit):
            mat = year_matrix.get(yr)
            if mat is not None:
                pressure_3d[:, yi, :] = mat

        # Build column arrays via broadcasting
        results_df = pd.DataFrame({
            "SimulationID": np.repeat(np.arange(1, n_sims + 1), n_years * n_faults),
            "ID": np.tile(np.tile(fault_ids, n_years), n_sims),
            "Pressure": pressure_3d.ravel(),
            "Year": np.tile(np.repeat(years_to_emit, n_faults), n_sims),
        })

    if return_sample_inputs:
        sample_inputs_df = pd.DataFrame({
            "SimulationID": np.arange(1, n_sims + 1),
            "aquifer_thickness": h_samples,
            "porosity": phi_samples,
            "permeability": kap_samples,
            "fluid_density": rho_samples,
            "dynamic_viscosity": mu_samples,
            "fluid_compressibility": beta_samples,
            "rock_compressibility": alphav_samples,
        })
        return results_df, sample_inputs_df
    return results_df
