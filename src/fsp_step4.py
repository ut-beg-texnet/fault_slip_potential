"""
FSP Step 4 — Deterministic Hydrology
Calculates pore pressure field and fault pressures using Theis radial flow.

Portal invocation: python fsp_step4.py <scratch_path>
Step index (0-based): 3
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import date
from joblib import Parallel, delayed
from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp.hydrology.params import calcST
from fsp.hydrology.pressure_field import (
    pfieldcalc_all_rates_for_distances,
    well_fault_distances_m,
)
from fsp.hydrology.theis import pressureScenario_Rall
from fsp.io.coords import create_projected_spatial_grid, haversine_distance, reformat_pressure_grid_to_heatmap
from fsp.io.wells import (
    load_injection_wells, preprocess_well_data, normalize_wells_to_well_data, get_date_bounds
)
from fsp.geomechanics.stress import calculate_absolute_stresses
from fsp.geomechanics.slip import analyze_fault_hydro
from fsp.geomechanics.mohr import mohr_diagram_hydro_data_to_d3_portal
from graphs.hydrology_map import save_direct_hydrology_pressure_map_artifact
from graphs.mohr_diagram import save_mohr_diagram_graph_artifact
from graphs.scientific import save_radial_curves_artifact
from progress import report_progress

STEP = 3   # 0-based index for Step 4
STEP_GEO = 1  # Step 2
STEP_PROB_GEO = 2  # Step 3
SKIPPED_GEOMECHANICS_TERMINAL_MESSAGE = (
    "Geomechanics steps have been skipped, so this is the last step of FSP for this run"
)


def _has_required_geomechanics_inputs(stress_inputs: dict, stress_model_type: str) -> bool:
    required = ["reference_depth", "vertical_stress", "pore_pressure", "max_stress_azimuth", "friction_coefficient"]
    if any(stress_inputs.get(key) is None for key in required):
        return False
    if stress_model_type in ("gradients", "all_gradients"):
        return stress_inputs.get("min_horizontal_stress") is not None and stress_inputs.get("max_horizontal_stress") is not None
    return stress_inputs.get("aphi_value") is not None


def _geomechanics_steps_were_skipped(helper) -> bool:
    return helper.isStepSkipped(STEP_GEO) and helper.isStepSkipped(STEP_PROB_GEO)


def _add_hydrology_run_message(helper, has_faults: bool, geomechanics_steps_skipped: bool) -> None:
    if geomechanics_steps_skipped:
        helper.addMessageWithStepIndex(
            STEP,
            SKIPPED_GEOMECHANICS_TERMINAL_MESSAGE,
            1,
        )
    elif not has_faults:
        helper.addMessageWithStepIndex(
            STEP,
            "No fault dataset was provided, so deterministic hydrology fault pressure outputs were skipped. Pressure grid and well-based outputs are still available.",
            1,
        )


def _compute_well_grid_p(wd, grid_lats_flat, grid_lons_flat, grid_shape, STRho, cutoff_date):
    """Compute the pressure grid contribution for a single well.

    Defined at module level so joblib (loky backend) can serialise it via cloudpickle.
    References module-level imports ``haversine_distance`` and
    ``pfieldcalc_all_rates_for_distances`` which cloudpickle captures automatically.
    """
    eval_days = float((cutoff_date - wd.start_date).days + 1)
    dist_m = haversine_distance(grid_lats_flat, grid_lons_flat, wd.latitude, wd.longitude) * 1000.0
    return pfieldcalc_all_rates_for_distances(dist_m, STRho, wd.days, wd.rates, eval_days).reshape(grid_shape)


def _get_injection_path(helper):
    for param, dtype in [
        ("injection_wells_annual_hydrology", "annual_fsp"),
        ("injection_wells_monthly_hydrology", "monthly_fsp"),
        ("injection_tool_data_hydrology", "injection_tool_data"),
        ("injection_wells_annual", "annual_fsp"),
        ("injection_wells_monthly", "monthly_fsp"),
        ("injection_tool_data", "injection_tool_data"),
    ]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, param)
        if path:
            return path, dtype
    raise ValueError("No injection wells dataset provided.")


def main():
    scratch_path = sys.argv[1]
    helper = TexNetWebToolLaunchHelper(scratch_path)

    try:
        # ---- Aquifer parameters ----
        def _p(name):
            return helper.getParameterValueWithStepIndexAndParamName(STEP, name)

        h_ft = float(_p("aquifer_thickness_ft"))
        porosity = float(_p("porosity"))
        kap_md = float(_p("permeability_md"))
        fluid_density = float(_p("fluid_density"))
        dynamic_viscosity = float(_p("dynamic_viscosity"))
        fluid_comp = float(_p("fluid_compressibility"))
        rock_comp = float(_p("rock_compressibility"))
        year_of_interest = int(_p("year_of_interest") or date.today().year)

        S, T, rho = calcST(h_ft, porosity, kap_md, fluid_density,
                           dynamic_viscosity, 9.81, fluid_comp, rock_comp)
        STRho = (S, T, rho)

        # ---- Load faults ----
        faults_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, "faults")
        if not faults_path:
            fault_df = pd.DataFrame(columns=["FaultID", "Latitude(WGS84)", "Longitude(WGS84)", "Strike", "Dip"])
        else:
            fault_df = pd.read_csv(faults_path, dtype={"FaultID": str})
        has_faults = not fault_df.empty
        geomechanics_steps_skipped = _geomechanics_steps_were_skipped(helper)
        _add_hydrology_run_message(helper, has_faults, geomechanics_steps_skipped)

        # ---- Load injection wells ----
        report_progress("Loading wells and faults")
        inj_path, inj_type = _get_injection_path(helper)
        inj_df = load_injection_wells(inj_path, inj_type)

        inj_start_date, inj_end_date = get_date_bounds(inj_df)
        cutoff_date = date(year_of_interest - 1, 12, 31)

        well_info = preprocess_well_data(inj_df, inj_type)
        well_data_list = normalize_wells_to_well_data(well_info, inj_type, cutoff_date)

        # ---- Compute pressure at each fault ----
        if has_faults:
            fault_lats = fault_df["Latitude(WGS84)"].values.astype(float)
            fault_lons = fault_df["Longitude(WGS84)"].values.astype(float)
            fault_ids = fault_df["FaultID"].astype(str).values
        else:
            fault_lats = np.array([], dtype=float)
            fault_lons = np.array([], dtype=float)
            fault_ids = np.array([], dtype=str)

        if has_faults:
            report_progress("Calculating pressure at faults")
            fault_distance_m = well_fault_distances_m(well_data_list, fault_lats, fault_lons)
            active_fault_wells = [(wi, wd) for wi, wd in enumerate(well_data_list) if len(wd.days) > 0]
            per_well_fault_p = Parallel(n_jobs=-1, backend="loky")(
                delayed(pfieldcalc_all_rates_for_distances)(
                    fault_distance_m[wi], STRho, wd.days, wd.rates,
                    float((cutoff_date - wd.start_date).days + 1)
                )
                for wi, wd in active_fault_wells
            )
            dp_faults = np.maximum(
                np.sum(per_well_fault_p, axis=0) if per_well_fault_p else np.zeros(len(fault_df), dtype=float),
                0.0,
            )
        else:
            dp_faults = np.array([], dtype=float)

        # ---- Pressure grid (raster overlay) ----
        report_progress("Building pressure grid")
        well_lats = np.array([wd.latitude for wd in well_data_list], dtype=float)
        well_lons = np.array([wd.longitude for wd in well_data_list], dtype=float)
        grid_lats = np.concatenate([fault_lats, well_lats]) if len(well_lats) else fault_lats
        grid_lons = np.concatenate([fault_lons, well_lons]) if len(well_lons) else fault_lons
        lat_grid, lon_grid, _ = create_projected_spatial_grid(
            grid_lats,
            grid_lons,
            n=150,
            margin_fraction=0.3,
            min_margin_km=1.0,
        )

        per_well_grid_rows = []
        per_well_grids_enabled = (
            helper.getParameterStateWithStepIndexAndParamName(
                STEP, "hydrology_pressure_grids_by_well"
            ) is not None
        )
        grid_shape = lat_grid.shape
        grid_lats_flat = lat_grid.ravel()
        grid_lons_flat = lon_grid.ravel()
        active_grid_wells = [wd for wd in well_data_list if len(wd.days) > 0]
        per_well_grid_pressures = Parallel(n_jobs=-1, backend="loky")(
            delayed(_compute_well_grid_p)(wd, grid_lats_flat, grid_lons_flat, grid_shape, STRho, cutoff_date)
            for wd in active_grid_wells
        )
        # Sum each well's clipped contribution (matches original per-well np.maximum before accumulation)
        total_grid = np.maximum(
            np.sum([np.maximum(gp, 0.0) for gp in per_well_grid_pressures], axis=0)
            if per_well_grid_pressures else np.zeros_like(lat_grid),
            0.0,
        )
        if per_well_grids_enabled:
            for wd, grid_p in zip(active_grid_wells, per_well_grid_pressures):
                well_grid_df = reformat_pressure_grid_to_heatmap(lat_grid, lon_grid, grid_p)
                well_grid_df.insert(0, "WellID", wd.well_id)
                per_well_grid_rows.append(well_grid_df)

        heatmap_df = reformat_pressure_grid_to_heatmap(lat_grid, lon_grid, total_grid)
        if per_well_grids_enabled:
            per_well_grid_df = (
                pd.concat(per_well_grid_rows, ignore_index=True)
                if per_well_grid_rows else pd.DataFrame()
            )
            # Portal CSV not needed; graph artifact covers this output.
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "hydrology_pressure_grids_by_well", per_well_grid_df)
        else:
            per_well_grid_df = heatmap_df.copy()
            per_well_grid_df.insert(0, "WellID", "Total")

        well_rows = []
        for wd in well_data_list:
            if len(wd.days) == 0:
                continue
            well_rows.append({
                "WellID": wd.well_id,
                "Latitude": float(wd.latitude),
                "Longitude": float(wd.longitude),
                "StartDate": wd.start_date.isoformat(),
                "EndDate": wd.end_date.isoformat(),
                "MaxRate_bbl_day": float(np.max(wd.rates)) if len(wd.rates) else 0.0,
                "MeanRate_bbl_day": float(np.mean(wd.rates)) if len(wd.rates) else 0.0,
            })
        well_summary_df = pd.DataFrame(well_rows)

        # ---- Radial curves (pressure vs distance for each well) ----
        report_progress("Generating maps and diagrams")
        r_km = np.linspace(0.1, 50.0, 200)
        r_m = r_km * 1000.0
        radial_dfs = []
        for wd in well_data_list:
            if len(wd.days) == 0:
                continue
            eval_days = float((cutoff_date - wd.start_date).days + 1)
            p_radial = pressureScenario_Rall(wd.rates, wd.days, r_m, STRho, eval_days).astype(float)
            radial_dfs.append(pd.DataFrame({
                "ID": wd.well_id,
                "WellID": wd.well_id,
                "Distance_km": r_km,
                "Pressure_psi": p_radial,
                "distance_km": r_km,
                "pressure_psi": p_radial,
            }))
        radial_df = pd.concat(radial_dfs, ignore_index=True) if radial_dfs else pd.DataFrame()
        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "radial_curves_data", radial_df)
        save_radial_curves_artifact(helper, STEP, radial_df)

        # ---- Deterministic hydrology results per fault ----
        hydro_result_df = fault_df.copy()
        hydro_result_df["pressure_psi"] = dp_faults
        hydro_result_df["year"] = year_of_interest
        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "deterministic_hydrology_results", hydro_result_df)
        save_direct_hydrology_pressure_map_artifact(
            helper,
            STEP,
            per_well_grid_df,
            hydro_result_df,
            well_summary_df,
            artifact_key="fsp-deterministic-hydrology-map",
            title="Hydrology Pressure Map",
            caption="Interactive hydrology pressure map with selected-well pressure grid summation.",
            display_order=41,
        )

        # ---- Optional updated Mohr diagram with hydro pressure ----
        stress_inputs = {
            "reference_depth": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "reference_depth"),
            "vertical_stress": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "vertical_stress"),
            "min_horizontal_stress": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "min_horizontal_stress"),
            "max_horizontal_stress": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "max_horizontal_stress"),
            "pore_pressure": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "pore_pressure"),
            "max_stress_azimuth": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "max_stress_azimuth"),
            "aphi_value": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "aphi_value"),
            "friction_coefficient": helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "friction_coefficient"),
        }
        stress_model_type = helper.getParameterValueWithStepIndexAndParamName(STEP_GEO, "stress_model_type") or "gradients"
        if has_faults and _has_required_geomechanics_inputs(stress_inputs, stress_model_type):
            friction = float(stress_inputs["friction_coefficient"])
            stress_state, p0 = calculate_absolute_stresses(stress_inputs, friction, stress_model_type)
            sV, sh, sH = stress_state.principal_stresses

            hydro_res_list = []
            for i, row in fault_df.iterrows():
                res = analyze_fault_hydro(
                    float(row["Strike"]), float(row["Dip"]), friction,
                    stress_state, p0, float(dp_faults[i])
                )
                hydro_res_list.append(res)

            tau_eff = [r["shear_stress"] for r in hydro_res_list]
            sigma_eff = [r["normal_stress"] for r in hydro_res_list]
            strikes = list(fault_df["Strike"].astype(float))

            if abs(sV) >= abs(sH) and abs(sH) >= abs(sh):
                hydro_regime = "Normal"
            elif abs(sH) >= abs(sh) and abs(sh) >= abs(sV):
                hydro_regime = "Reverse"
            else:
                hydro_regime = "Strike-Slip"

            arcs_df, slip_df, fault_df_mohr = mohr_diagram_hydro_data_to_d3_portal(
                float(sh), float(sH), float(sV),
                tau_eff, sigma_eff,
                p0, dp_faults.tolist(), strikes, friction,
                list(fault_ids),
            )
            # Portal CSV not needed; graph artifact covers this output.
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "arcsDF_hydro", arcs_df)
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "slipDF_hydro", slip_df)
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faultsDF_hydro", fault_df_mohr)
            save_mohr_diagram_graph_artifact(
                helper,
                arcs_df,
                slip_df,
                fault_df_mohr,
                step_index=STEP,
                artifact_key="fsp-deterministic-hydrology-mohr-diagram",
                title="Hydrology Mohr Diagram",
                display_order=42,
                stress_regime=hydro_regime,
            )

            faults_with_pp = fault_df.copy()
            faults_with_pp["det_hydro_pressure_psi"] = dp_faults
            faults_with_pp["pressure_psi"] = dp_faults
            faults_with_pp["pore_pressure_slip_det_hydro"] = [
                round(float(r.get("slip_pressure", 0.0)), 3) for r in hydro_res_list
            ]
            faults_with_pp["coulomb_failure_function_det_hydro"] = [
                round(float(r.get("coulomb_failure_function", 0.0)), 3) for r in hydro_res_list
            ]
            faults_with_pp["shear_capacity_utilization_det_hydro"] = [
                round(float(r.get("shear_capacity_utilization", 0.0)), 3) for r in hydro_res_list
            ]
            # Portal CSV not needed; graph artifact covers this output.
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faults_with_det_hydro_pp", faults_with_pp)
        elif not has_faults:
            pass
        elif not geomechanics_steps_skipped:
            helper.addMessageWithStepIndex(
                STEP,
                "Geomechanics steps were skipped for this run, so hydrology Mohr/slip overlays were not generated. Pressure results are still available.",
                1,
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
