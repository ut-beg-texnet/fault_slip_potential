"""
FSP Step 6 — Summary / FSP
Aggregates geomechanics CDF + hydrology time series → final fault slip probability.

Portal invocation: python fsp_step6.py <scratch_path>
Step index (0-based): 5
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import date
from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp.hydrology.params import calcST
from fsp.io.wells import load_injection_wells, preprocess_well_data, normalize_wells_to_well_data, get_date_bounds
from fsp.monte_carlo.hydrology_mc import run_hydrology_mc_time_series
from fsp.models.hydrology import HydrologyParams
from fsp.hydrology.pressure_field import (
    pfieldcalc_all_rates_for_distances,
    well_fault_distances_m,
)
from graphs.artifacts import FSP_COLOR_SCALE, SLIP_PRESSURE_COLOR_SCALE
from graphs.leaflet_map import save_fault_results_map_artifact
from graphs.scientific import save_summary_artifacts
from progress import report_progress

STEP = 5       # 0-based index for Step 6
STEP_HYDRO = 3  # Step 4
STEP_PROB_HYDRO = 4  # Step 5


def _has_geomechanics_cdf(geo_cdf_df: pd.DataFrame) -> bool:
    return (
        geo_cdf_df is not None
        and not geo_cdf_df.empty
        and {"ID", "slip_pressure", "probability"}.issubset(geo_cdf_df.columns)
    )


def _get_injection_path(helper):
    """Look for injection data in step 6 parameters (summary variants)."""
    for param, dtype in [
        ("injection_wells_annual_summary", "annual_fsp"),
        ("injection_wells_monthly_summary", "monthly_fsp"),
        ("injection_tool_data_summary", "injection_tool_data"),
    ]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, param)
        if path:
            return path, dtype
    raise ValueError("No injection wells dataset provided for summary step.")


def _interpolate_cdf(pressures, probs, target_p: float) -> float:
    """Linear interpolation on a CDF at target_p."""
    if len(pressures) == 0:
        return 0.0
    if target_p <= pressures[0]:
        return float(probs[0])
    if target_p >= pressures[-1]:
        return float(probs[-1])
    i = np.searchsorted(pressures, target_p)
    p0, p1 = pressures[i - 1], pressures[i]
    q0, q1 = probs[i - 1], probs[i]
    if p1 == p0:
        return float(q0)
    t = (target_p - p0) / (p1 - p0)
    return float(q0 + t * (q1 - q0))


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


def _calculate_fsp(geo_cdf_df: pd.DataFrame,
                   hydro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute FSP per fault per year.

    geo_cdf_df: columns ID, slip_pressure, probability (cumulative CDF of Δp to slip)
    hydro_df: columns ID, Pressure, Year (MC sample or deterministic pressures)

    Returns DataFrame with columns: ID, Year, FSP, epoch_time
    """
    rows = []
    if not _has_geomechanics_cdf(geo_cdf_df):
        return pd.DataFrame(columns=["ID", "Year", "FSP", "epoch_time"])

    geo_groups = {
        str(fid): group.sort_values("slip_pressure")
        for fid, group in geo_cdf_df.assign(ID=geo_cdf_df["ID"].astype(str)).groupby("ID", sort=False)
    }
    years = sorted(hydro_df["Year"].unique())

    for yr in years:
        yr_data = hydro_df[hydro_df["Year"] == yr]
        epoch = float((date(yr, 1, 1) - date(1970, 1, 1)).days * 86400.0 * 1000.0)

        for fid, fault_pressures_df in yr_data.groupby(yr_data["ID"].astype(str), sort=False):
            fault_geo = geo_groups.get(str(fid))
            if fault_geo is None or fault_geo.empty:
                continue

            fault_pressures = fault_pressures_df["Pressure"].values
            if len(fault_pressures) == 0:
                continue
            mean_p = float(np.mean(fault_pressures))

            geo_p = fault_geo["slip_pressure"].values.astype(float)
            if "SimulationID" in fault_pressures_df.columns:
                fsp_values = _empirical_geomechanics_probabilities(geo_p, fault_pressures)
                fsp = float(fsp_values.mean()) if len(fsp_values) else 0.0
            else:
                geo_prob = fault_geo["probability"].values.astype(float)
                fsp = _interpolate_cdf(geo_p, geo_prob, mean_p)
            rows.append({"ID": str(fid), "Year": int(yr), "FSP": round(fsp, 2), "epoch_time": epoch})

    return pd.DataFrame(rows)


def _fault_summary_for_year(
    fault_df: pd.DataFrame,
    fsp_df: pd.DataFrame,
    pressure_df: pd.DataFrame,
    year_of_interest: int,
    *,
    include_fsp: bool,
) -> pd.DataFrame:
    fault_summary = fault_df.copy()
    fault_summary["summary_fsp"] = None
    fault_summary["summary_pressure"] = 0.0

    pressure_yr = pressure_df[pressure_df["Year"] == year_of_interest]
    pressure_lookup = (
        pressure_yr.assign(ID=pressure_yr["ID"].astype(str))
        .set_index("ID")["Pressure"]
        .to_dict()
    )

    fault_ids = fault_summary["FaultID"].astype(str)
    if include_fsp:
        fsp_yr = fsp_df[fsp_df["Year"] == year_of_interest]
        fsp_lookup = fsp_yr.assign(ID=fsp_yr["ID"].astype(str)).set_index("ID")["FSP"].to_dict()
        fault_summary["summary_fsp"] = fault_ids.map(fsp_lookup).fillna(0.0).astype(float)
    fault_summary["summary_pressure"] = fault_ids.map(pressure_lookup).fillna(0.0).astype(float)
    return fault_summary


def _summary_map_configuration(has_geomechanics_cdf: bool) -> dict:
    if has_geomechanics_cdf:
        return {
            "result_fields": ["summary_fsp", "summary_pressure"],
            "title": "Summary FSP Map",
            "caption": "Leaflet map of summary FSP and pressure results by fault.",
            "value_column": "summary_fsp",
            "legend_title": "Summary FSP",
            "color_scale": FSP_COLOR_SCALE,
            "value_min_default": 0.0,
            "value_max_default": 1.0,
        }
    return {
        "result_fields": ["summary_pressure"],
        "title": "Summary Pressure Map",
        "caption": (
            "Leaflet map of summary pressure results by fault. "
            "FSP was not computed because geomechanics was skipped."
        ),
        "value_column": "summary_pressure",
        "legend_title": "Summary Pressure",
        "color_scale": SLIP_PRESSURE_COLOR_SCALE,
        "value_min_default": None,
        "value_max_default": None,
    }


def _run_deterministic_hydro_time_series(STRho, well_data_list, fault_df, years_to_analyze):
    """Re-run deterministic hydrology for all years (used when model_run=0)."""
    fault_lats = fault_df["Latitude(WGS84)"].values.astype(float)
    fault_lons = fault_df["Longitude(WGS84)"].values.astype(float)
    fault_ids = fault_df["FaultID"].astype(str).values
    distance_matrix_m = well_fault_distances_m(well_data_list, fault_lats, fault_lons)
    rows = []

    for yr in years_to_analyze:
        cutoff = date(yr - 1, 12, 31)
        dp = np.zeros(len(fault_ids))
        for wi, wd in enumerate(well_data_list):
            if wd.start_date > cutoff or wd.start_year > yr:
                continue
            if len(wd.days) == 0:
                continue
            eval_days = float((cutoff - wd.start_date).days + 1)
            dp += pfieldcalc_all_rates_for_distances(
                distance_matrix_m[wi], STRho, wd.days, wd.rates, eval_days
            )
        dp = np.maximum(dp, 0.0)
        for fi, fid in enumerate(fault_ids):
            rows.append({"ID": fid, "Pressure": round(float(dp[fi]), 2), "Year": yr})

    return pd.DataFrame(rows)


def main():
    scratch_path = sys.argv[1]
    helper = TexNetWebToolLaunchHelper(scratch_path)

    try:
        def _p(step, name):
            return helper.getParameterValueWithStepIndexAndParamName(step, name)

        year_of_interest = int(_p(STEP, "year_of_interest_summary") or date.today().year)
        model_run = _p(STEP, "model_run_summary")
        if model_run is None:
            model_run = 1
        model_run = int(model_run)

        # ---- Load injection wells ----
        inj_path, inj_type = _get_injection_path(helper)
        inj_df_raw = pd.read_csv(inj_path, dtype={"WellID": str, "API Number": str})
        inj_df = inj_df_raw

        inj_start_date, inj_end_date = get_date_bounds(inj_df)
        start_year = inj_start_date.year
        end_year = inj_end_date.year + 3   # extra years for pressure diffusion
        years_to_analyze = list(range(start_year, end_year + 1))

        if year_of_interest < start_year:
            year_of_interest = start_year
        elif year_of_interest > end_year:
            year_of_interest = end_year

        cutoff_date = date(year_of_interest - 1, 12, 31)
        well_info = preprocess_well_data(inj_df, inj_type)
        well_data_list = normalize_wells_to_well_data(well_info, inj_type, cutoff_date)

        # ---- Load fault data ----
        fault_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, "faults")
        if not fault_path:
            fault_df = pd.DataFrame(columns=["FaultID", "Latitude(WGS84)", "Longitude(WGS84)"])
        else:
            fault_df = pd.read_csv(fault_path, dtype={"FaultID": str})
        has_faults = not fault_df.empty

        # ---- Load optional probabilistic geomechanics CDF ----
        geo_cdf_path = helper.getOptionalDatasetFilePathWithStepIndexAndParamName(STEP, "prob_geomechanics_cdf_graph_data_summary")
        geo_cdf_df = pd.read_csv(geo_cdf_path, dtype={"ID": str}) if geo_cdf_path else pd.DataFrame()
        has_geomechanics_cdf = _has_geomechanics_cdf(geo_cdf_df)

        if not has_faults:
            helper.addMessageWithStepIndex(
                STEP,
                "No fault dataset was provided, so summary fault pressure and FSP outputs were skipped.",
                1,
            )
            helper.setSuccessForStepIndex(STEP, True)
            helper.writeResultsFile()
            return

        # ---- Hydrology parameters ----
        h_ft = float(_p(STEP_HYDRO, "aquifer_thickness_ft"))
        porosity = float(_p(STEP_HYDRO, "porosity"))
        kap_md = float(_p(STEP_HYDRO, "permeability_md"))
        fluid_density = float(_p(STEP_HYDRO, "fluid_density"))
        dyn_visc = float(_p(STEP_HYDRO, "dynamic_viscosity"))
        fluid_comp = float(_p(STEP_HYDRO, "fluid_compressibility"))
        rock_comp = float(_p(STEP_HYDRO, "rock_compressibility"))

        S, T, rho = calcST(h_ft, porosity, kap_md, fluid_density, dyn_visc, 9.81, fluid_comp, rock_comp)
        STRho = (S, T, rho)

        def _unc(name):
            v = _p(STEP_PROB_HYDRO, name)
            return float(v) if v is not None else 0.0

        n_iters = int(_p(STEP_PROB_HYDRO, "hydro_mc_iterations") or 750)

        # ---- Run pressure time series ----
        report_progress("Calculating pressure over time")
        if model_run == 0:
            pressure_df = _run_deterministic_hydro_time_series(
                STRho, well_data_list, fault_df, years_to_analyze
            )
        else:
            pm = {
                "aquifer_thickness": _unc("aquifer_thickness_uncertainty"),
                "porosity": _unc("porosity_uncertainty"),
                "permeability": _unc("permeability_uncertainty"),
                "fluid_density": _unc("fluid_density_uncertainty"),
                "dynamic_viscosity": _unc("dynamic_viscosity_uncertainty"),
                "fluid_compressibility": _unc("fluid_compressibility_uncertainty"),
                "rock_compressibility": _unc("rock_compressibility_uncertainty"),
            }
            hydro_params = HydrologyParams(
                aquifer_thickness=h_ft, porosity=porosity, permeability=kap_md,
                fluid_density=fluid_density, dynamic_viscosity=dyn_visc,
                fluid_compressibility=fluid_comp, rock_compressibility=rock_comp,
                plus_minus=pm, n_iterations=n_iters,
            )
            pressure_samples_df = run_hydrology_mc_time_series(
                hydro_params,
                well_data_list,
                fault_df,
                years_to_analyze,
                result_mode="raw",
            )
            pressure_df = (
                pressure_samples_df
                .groupby(["ID", "Year"], sort=False)["Pressure"]
                .mean()
                .reset_index()
            )
            pressure_df["Pressure"] = pressure_df["Pressure"].round(2)

        pressure_df["epoch_time"] = pressure_df["Year"].apply(
            lambda yr: float((date(int(yr), 1, 1) - date(1970, 1, 1)).days * 86400.0 * 1000.0)
        )

        # ---- Calculate optional FSP per fault per year ----
        if has_geomechanics_cdf:
            report_progress("Calculating fault slip potential")
            fsp_source_df = pressure_df if model_run == 0 else pressure_samples_df
            fsp_df = _calculate_fsp(geo_cdf_df, fsp_source_df)
            fsp_df["FSP"] = fsp_df["FSP"].round(2)
        else:
            helper.addMessageWithStepIndex(
                STEP,
                "Geomechanics steps were skipped for this run, so FSP/slip-probability outputs were not generated. Pressure results are still available.",
                1,
            )
            fsp_df = pd.DataFrame(columns=["ID", "Year", "FSP", "epoch_time"])

        # ---- Populate fault summary at year of interest ----
        fault_summary = _fault_summary_for_year(
            fault_df,
            fsp_df,
            pressure_df,
            year_of_interest,
            include_fsp=has_geomechanics_cdf,
        )

        # ---- Save outputs ----
        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faults_with_summary_fsp", fault_summary)
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "pressure_through_time_results", pressure_df)
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "fsp_through_time_results", fsp_df)
        report_progress("Generating summary maps")
        save_summary_artifacts(
            helper,
            STEP,
            fsp_df,
            pressure_df,
            year_of_interest=year_of_interest,
            include_fsp=has_geomechanics_cdf,
        )
        map_config = _summary_map_configuration(has_geomechanics_cdf)
        save_fault_results_map_artifact(
            helper,
            STEP,
            fault_summary,
            artifact_key="fsp-summary-map",
            title=map_config["title"],
            caption=map_config["caption"],
            display_order=62,
            result_fields=map_config["result_fields"],
            color="#059669",
            value_column=map_config["value_column"],
            legend_title=map_config["legend_title"],
            color_scale=map_config["color_scale"],
            value_min_default=map_config["value_min_default"],
            value_max_default=map_config["value_max_default"],
        )

        # Portal CSV not needed; graph artifact covers this output.
        # if helper.getParameterStateWithStepIndexAndParamName(STEP, "year_of_interest_line") is not None:
        #     epoch = float((date(year_of_interest, 1, 1) - date(1970, 1, 1)).days * 86400.0 * 1000.0)
        #     year_line = pd.DataFrame([{"Year": year_of_interest, "epoch_time": epoch}])
        #     helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "year_of_interest_line", year_line)

        helper.setSuccessForStepIndex(STEP, True)

    except Exception as e:
        helper.addMessageWithStepIndex(STEP, str(e), 2)
        helper.setSuccessForStepIndex(STEP, False)
        helper.writeResultsFile()
        sys.exit(1)

    helper.writeResultsFile()


if __name__ == "__main__":
    main()
