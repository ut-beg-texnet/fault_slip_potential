"""
Monte Carlo hydrology simulation — joblib-parallelised.
Port of FSP/summary_process.jl run_mc_hydrology_time_series.
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


def _single_sim_pressure(well_data_list, STRho, distance_matrix_m,
                          years_to_analyze):
    """Compute total pressure per (fault, year) for one set of ST parameters."""
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

    # Parallel MC
    sim_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_sim_pressure)(well_data_list, STRho_list[i],
                                       distance_matrix_m, years)
        for i in range(n_sims)
    )

    mode = str(result_mode or "raw").lower()
    if mode == "mean":
        rows = []
        for yr in years:
            total = np.zeros(len(fault_ids), dtype=float)
            count = 0
            for sim_result in sim_results:
                pressures = sim_result.get(yr)
                if pressures is None:
                    continue
                total += pressures
                count += 1
            if count == 0:
                continue
            mean_pressure = total / float(count)
            rows.extend({
                "ID": fid,
                "Pressure": float(mean_pressure[fi]),
                "Year": int(yr),
            } for fi, fid in enumerate(fault_ids))
        results_df = pd.DataFrame(rows)
    else:
        years_to_emit = years
        if mode == "year_samples":
            selected_year = int(sample_year) if sample_year is not None else years[-1]
            years_to_emit = [selected_year] if selected_year in years else years[-1:]

        rows = []
        for sim_i, sim_result in enumerate(sim_results):
            for yr in years_to_emit:
                pressures = sim_result.get(yr)
                if pressures is None:
                    continue
                rows.extend({
                    "SimulationID": sim_i + 1,
                    "ID": fid,
                    "Pressure": float(pressures[fi]),
                    "Year": int(yr),
                } for fi, fid in enumerate(fault_ids))
        results_df = pd.DataFrame(rows)

    if return_sample_inputs:
        sample_rows = []
        for sim_i in range(n_sims):
            sample_rows.append({
                "SimulationID": sim_i + 1,
                "aquifer_thickness": float(h_samples[sim_i]),
                "porosity": float(phi_samples[sim_i]),
                "permeability": float(kap_samples[sim_i]),
                "fluid_density": float(rho_samples[sim_i]),
                "dynamic_viscosity": float(mu_samples[sim_i]),
                "fluid_compressibility": float(beta_samples[sim_i]),
                "rock_compressibility": float(alphav_samples[sim_i]),
            })
        return results_df, pd.DataFrame(sample_rows)
    return results_df
