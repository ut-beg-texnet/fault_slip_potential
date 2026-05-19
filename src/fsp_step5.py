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
from graphs.scientific import save_cdf_artifact, save_hydrology_input_distribution_histograms_artifact

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


def _mean_slip_pressure_by_fault(geo_cdf_df: pd.DataFrame) -> dict:
    if geo_cdf_df is None or geo_cdf_df.empty:
        return {}
    return (
        geo_cdf_df.assign(
            ID=geo_cdf_df["ID"].astype(str),
            slip_pressure=pd.to_numeric(geo_cdf_df["slip_pressure"], errors="coerce"),
        )
        .dropna(subset=["slip_pressure"])
        .groupby("ID")["slip_pressure"]
        .mean()
        .astype(float)
        .to_dict()
    )


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

        if raw_hydro_results_enabled:
            helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_hydrology_results", mc_results)
        if helper.getParameterStateWithStepIndexAndParamName(STEP, "prob_hydrology_sample_inputs") is not None:
            helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_hydrology_sample_inputs", hydro_sample_inputs)

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
            helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_hydrology_cdf_graph_data", cdf_df)
            save_cdf_artifact(
                helper,
                STEP,
                cdf_df,
                artifact_key="fsp-probabilistic-hydrology-cdf",
                title="Probabilistic Hydrology CDF",
                pressure_label="Pore Pressure Change (psi)",
                probability_label="Exceedance Probability",
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

        slip_pressure_by_fault = _mean_slip_pressure_by_fault(geo_cdf_df)
        slip_rows = []
        faults_with_fsp = fault_df.copy()
        faults_with_fsp["prob_hydro_fsp"] = 0.0
        pressure_groups = {
            str(fid): pd.to_numeric(group["Pressure"], errors="coerce").dropna().astype(float)
            for fid, group in yr_data.groupby(yr_data["ID"].astype(str), sort=False)
        }
        probabilities = {}
        for fid in faults_with_fsp["FaultID"].astype(str):
            fp = pressure_groups.get(fid, pd.Series(dtype=float))
            threshold = slip_pressure_by_fault.get(fid, float(fp.mean()) if len(fp) else 0.0)
            probability = float((fp >= threshold).mean()) if len(fp) else 0.0
            probabilities[fid] = probability
            slip_rows.append({
                "ID": fid,
                "slip_pressure": threshold,
                "probability": probability,
                "Pressure": float(fp.mean()) if len(fp) else 0.0,
                "Year": year_of_interest,
            })
        faults_with_fsp["prob_hydro_fsp"] = (
            faults_with_fsp["FaultID"].astype(str).map(probabilities).fillna(0.0)
        )

        helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faults_with_prob_hydro_fsp", faults_with_fsp)
        slip_potential_df = pd.DataFrame(slip_rows)
        helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "slip_potential_results", slip_potential_df)
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
