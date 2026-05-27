"""
FSP Step 5 — Probabilistic Hydrology (Monte Carlo)
Samples aquifer parameter uncertainties to produce probabilistic pore pressure per fault.

Portal invocation: python fsp_step5.py <scratch_path>
Step index (0-based): 4
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import date
from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp.models.hydrology import HydrologyParams
from fsp.io.wells import load_injection_wells, preprocess_well_data, get_date_bounds
from fsp.monte_carlo.hydrology_mc import run_hydrology_mc_time_series
from graphs.leaflet_map import save_fault_results_map_artifact
from graphs.scientific import (
    save_hydrology_input_distribution_histograms_artifact,
    save_probabilistic_hydrology_cdf_artifact,
)

STEP = 4   # 0-based index for Step 5
STEP_HYDRO = 3  # Step 4 (deterministic hydrology)


def _get_injection_path(helper):
    for param, dtype in [
        ("injection_wells_annual_prob_hydro", "annual_fsp"),
        ("injection_wells_monthly_prob_hydro", "monthly_fsp"),
        ("injection_tool_data_prob_hydro", "injection_tool_data"),
    ]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, param)
        if path is not None:
            return path, dtype
    # Fall back to step 4 injection data
    for param, dtype in [
        ("injection_wells_annual_hydrology", "annual_fsp"),
        ("injection_wells_monthly_hydrology", "monthly_fsp"),
        ("injection_tool_data_hydrology", "injection_tool_data"),
        ("injection_wells_annual", "annual_fsp"),
        ("injection_wells_monthly", "monthly_fsp"),
        ("injection_tool_data", "injection_tool_data"),
    ]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP_HYDRO, param)
        if path is not None:
            return path, dtype
    raise ValueError("No injection wells dataset provided.")


def _prob_hydrology_cdf(fault_pressures: pd.Series) -> pd.DataFrame:
    """Build exceedance CDF from Monte Carlo pressure samples."""
    sorted_p = np.sort(fault_pressures.values)
    n = len(sorted_p)
    exceedance = 1.0 - np.arange(1, n + 1) / n
    return pd.DataFrame({"slip_pressure": sorted_p, "probability": exceedance})


def _geomechanics_pressure_samples_by_fault(geo_cdf_df: pd.DataFrame) -> dict:
    if geo_cdf_df is None or geo_cdf_df.empty or "ID" not in geo_cdf_df.columns:
        return {}

    clean = geo_cdf_df.assign(
        ID=geo_cdf_df["ID"].astype(str),
        slip_pressure=pd.to_numeric(geo_cdf_df["slip_pressure"], errors="coerce"),
    ).dropna(subset=["slip_pressure"])

    return {
        str(fid): np.sort(group["slip_pressure"].to_numpy(dtype=float))
        for fid, group in clean.groupby("ID", sort=False)
    }


def _empirical_geomechanics_probabilities(geo_pressures, hydro_pressures) -> np.ndarray:
    """Evaluate P(geomechanics slip pressure <= hydrology pressure)."""
    geo = np.asarray(geo_pressures, dtype=float)
    hydro = np.asarray(hydro_pressures, dtype=float)
    geo = np.sort(geo[np.isfinite(geo)])
    hydro = hydro[np.isfinite(hydro)]
    if len(hydro) == 0:
        return np.array([], dtype=float)
    if len(geo) == 0:
        return np.zeros(len(hydro), dtype=float)
    return np.searchsorted(geo, hydro, side="right").astype(float) / float(len(geo))


def _combined_slip_potential_rows(fault_ids, pressure_groups: dict, geo_groups: dict, year_of_interest: int):
    rows = []
    probabilities = {}
    for fid in fault_ids:
        fid = str(fid)
        fp = np.asarray(pressure_groups.get(fid, pd.Series(dtype=float)), dtype=float)
        fp = fp[np.isfinite(fp)]
        geo_pressures = np.asarray(geo_groups.get(fid, np.array([], dtype=float)), dtype=float)
        fsp_values = _empirical_geomechanics_probabilities(geo_pressures, fp)
        probability = float(fsp_values.mean()) if len(fsp_values) else 0.0
        mean_pressure = float(fp.mean()) if len(fp) else 0.0
        representative_slip_pressure = (
            float(np.mean(geo_pressures[np.isfinite(geo_pressures)]))
            if np.any(np.isfinite(geo_pressures))
            else mean_pressure
        )
        probabilities[fid] = probability
        rows.append({
            "ID": fid,
            "slip_pressure": representative_slip_pressure,
            "probability": probability,
            "Pressure": mean_pressure,
            "Year": year_of_interest,
        })
    return rows, probabilities


def main():
    scratch_path = sys.argv[1]
    helper = TexNetWebToolLaunchHelper(scratch_path)

    try:
        def _p(step, name):
            return helper.getParameterValueWithStepIndexAndParamName(step, name)

        # ---- Hydrology base parameters (from Step 4) ----
        h_ft = float(_p(STEP_HYDRO, "aquifer_thickness_ft"))
        porosity = float(_p(STEP_HYDRO, "porosity"))
        kap_md = float(_p(STEP_HYDRO, "permeability_md"))
        fluid_density = float(_p(STEP_HYDRO, "fluid_density"))
        dyn_visc = float(_p(STEP_HYDRO, "dynamic_viscosity"))
        fluid_comp = float(_p(STEP_HYDRO, "fluid_compressibility"))
        rock_comp = float(_p(STEP_HYDRO, "rock_compressibility"))

        # ---- Uncertainty parameters (from Step 5) ----
        def _unc(name):
            v = _p(STEP, name)
            return float(v) if v is not None else 0.0

        pm = {
            "aquifer_thickness": _unc("aquifer_thickness_uncertainty"),
            "porosity": _unc("porosity_uncertainty"),
            "permeability": _unc("permeability_uncertainty"),
            "fluid_density": _unc("fluid_density_uncertainty"),
            "dynamic_viscosity": _unc("dynamic_viscosity_uncertainty"),
            "fluid_compressibility": _unc("fluid_compressibility_uncertainty"),
            "rock_compressibility": _unc("rock_compressibility_uncertainty"),
        }

        n_iters = int(_p(STEP, "hydro_mc_iterations") or 750)
        year_of_interest = int(_p(STEP, "year_of_interest") or _p(STEP_HYDRO, "year_of_interest") or date.today().year)
        hydro_model_type = str(_p(STEP, "hydro_model_type") or "probabilistic").lower()
        model_run = 0 if "det" in hydro_model_type else 1
        helper.setParamValueWithStepIndexAndParamName(STEP, "model_run", model_run)

        hydro_params = HydrologyParams(
            aquifer_thickness=h_ft,
            porosity=porosity,
            permeability=kap_md,
            fluid_density=fluid_density,
            dynamic_viscosity=dyn_visc,
            fluid_compressibility=fluid_comp,
            rock_compressibility=rock_comp,
            plus_minus=pm,
            n_iterations=n_iters,
        )

        # ---- Load faults ----
        faults_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, "faults_model_inputs_output")
        if faults_path is None:
            faults_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP_HYDRO, "faults")
        if faults_path is None:
            raise ValueError("No fault dataset provided.")
        fault_df = pd.read_csv(faults_path, dtype={"FaultID": str})

        # ---- Load injection wells ----
        inj_path, inj_type = _get_injection_path(helper)
        inj_df = load_injection_wells(inj_path, inj_type)

        inj_start_date, inj_end_date = get_date_bounds(inj_df)
        start_year = inj_start_date.year
        end_year = min(inj_end_date.year + 3, year_of_interest)
        years_to_analyze = list(range(start_year, end_year + 1))

        # Pre-process wells (keep raw data for per-year cutoff)
        well_info = preprocess_well_data(inj_df, inj_type)

        # Build a simple list of ProcessedWellData with max date for general use
        from fsp.io.wells import normalize_wells_to_well_data
        cutoff_date = date(year_of_interest - 1, 12, 31)
        well_data_list = normalize_wells_to_well_data(well_info, inj_type, cutoff_date)

        # ---- Run MC ----
        raw_hydro_results_enabled = (
            helper.getParameterStateWithStepIndexAndParamName(
                STEP, "prob_hydrology_results"
            ) is not None
        )
        mc_results, hydro_sample_inputs = run_hydrology_mc_time_series(
            hydro_params, well_data_list, fault_df, years_to_analyze,
            return_sample_inputs=True,
            result_mode="raw" if raw_hydro_results_enabled else "year_samples",
            sample_year=year_of_interest,
        )

        # Portal CSV not needed; graph artifact covers this output.
        # if raw_hydro_results_enabled:
        #     helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_hydrology_results", mc_results)
        # if helper.getParameterStateWithStepIndexAndParamName(STEP, "prob_hydrology_sample_inputs") is not None:
        #     helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_hydrology_sample_inputs", hydro_sample_inputs)

        geo_cdf_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, "prob_geomechanics_cdf_graph_data_prob_hydro")
        geo_cdf_df = pd.read_csv(geo_cdf_path, dtype={"ID": str}) if geo_cdf_path else pd.DataFrame()

        # ---- Build per-fault CDF data for year of interest ----
        yr_data = mc_results[mc_results["Year"] == year_of_interest] if year_of_interest in mc_results["Year"].values else mc_results
        cdf_rows = []
        for fid, fault_pressures in yr_data.groupby(yr_data["ID"].astype(str), sort=False)["Pressure"]:
            fp = pd.to_numeric(fault_pressures, errors="coerce").dropna()
            cdf = _prob_hydrology_cdf(fp)
            cdf.insert(0, "ID", str(fid))
            cdf_rows.append(cdf)

        if cdf_rows:
            cdf_df = pd.concat(cdf_rows, ignore_index=True)
            # Portal CSV not needed; graph artifact covers this output.
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_hydrology_cdf_graph_data", cdf_df)
            save_probabilistic_hydrology_cdf_artifact(
                helper,
                STEP,
                cdf_df,
                geo_cdf_df,
                artifact_key="fsp-probabilistic-hydrology-cdf",
                title="Probability of Pressure Exceedance",
                display_order=50,
            )

        save_hydrology_input_distribution_histograms_artifact(
            helper,
            STEP,
            hydro_sample_inputs,
            artifact_key="fsp-probabilistic-hydrology-histogram",
            title="Probabilistic Hydrology Histogram",
            display_order=51,
        )

        faults_with_fsp = fault_df.copy()
        faults_with_fsp["prob_hydro_fsp"] = 0.0
        pressure_groups = {
            str(fid): pd.to_numeric(group["Pressure"], errors="coerce").dropna().astype(float)
            for fid, group in yr_data.groupby(yr_data["ID"].astype(str), sort=False)
        }
        geo_groups = _geomechanics_pressure_samples_by_fault(geo_cdf_df)
        _, probabilities = _combined_slip_potential_rows(
            faults_with_fsp["FaultID"].astype(str),
            pressure_groups,
            geo_groups,
            year_of_interest,
        )
        faults_with_fsp["prob_hydro_fsp"] = (
            faults_with_fsp["FaultID"].astype(str).map(probabilities).fillna(0.0)
        )

        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faults_with_prob_hydro_fsp", faults_with_fsp)
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "slip_potential_results", pd.DataFrame(slip_rows))
        save_fault_results_map_artifact(
            helper,
            STEP,
            faults_with_fsp,
            artifact_key="fsp-probabilistic-hydrology-map",
            title="Probabilistic Hydrology FSP Map",
            caption="Leaflet map of probabilistic hydrology fault slip probability results.",
            display_order=54,
            result_fields=["prob_hydro_fsp"],
            color="#be123c",
        )

        helper.setSuccessForStepIndex(STEP, True)

    except Exception as e:
        helper.addMessageWithStepIndex(STEP, str(e), 2)
        helper.setSuccessForStepIndex(STEP, False)
        helper.writeResultsFile()
        sys.exit(1)

    helper.writeResultsFile()


if __name__ == "__main__":
    main()
