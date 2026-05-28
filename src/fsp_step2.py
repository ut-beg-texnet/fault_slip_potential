"""
FSP Step 2 — Deterministic Geomechanics
Computes stress tensors, slip pressure, CFF, SCU for each fault.

Portal invocation: python fsp_step2.py <scratch_path>
Step index (0-based): 1
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp.geomechanics.stress import calculate_absolute_stresses
from fsp.geomechanics.slip import (
    calculate_fault_effective_stresses,
    calculate_slip_pressure,
    calculate_cff,
    calculate_scu,
)
from fsp.geomechanics.mohr import mohr_diagram_data_to_d3_portal
from graphs.leaflet_map import (
    DETERMINISTIC_GEOMECHANICS_FIELD_LABELS,
    save_fault_results_map_artifact,
)
from graphs.mohr_diagram import save_mohr_diagram_graph_artifact
from graphs.stereonet import save_stereonet_graph_artifact
from progress import report_progress

STEP = 1   # 0-based index for Step 2


def _get_stress_model_type(helper):
    """Determine stress model type from portal parameters."""
    mode = helper.getParameterValueWithStepIndexAndParamName(STEP, "stress_field_mode")
    if mode:
        return str(mode)
    # Legacy fallback
    aphi = helper.getParameterValueWithStepIndexAndParamName(STEP, "aphi_value")
    sh = helper.getParameterValueWithStepIndexAndParamName(STEP, "min_horizontal_stress")
    if aphi is not None:
        return "aphi_min" if sh is not None else "aphi_no_min"
    return "gradients"


def _load_map_ready_injection_wells(helper):
    for param_name, source_id_column, source_lat_column, source_lon_column in [
        ("injection_tool_data_filtered_map_layer", "UWI", "Latitude(WGS84)", "Longitude(WGS84)"),
        ("injection_wells_annual_output", "WellID", "Latitude(WGS84)", "Longitude(WGS84)"),
        ("injection_wells_monthly_output", "WellID", "Latitude(WGS84)", "Longitude(WGS84)"),
    ]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, param_name)
        if path is None:
            continue
        wells_df = pd.read_csv(path, dtype={source_id_column: str})
        normalized_df = wells_df.copy()
        if source_id_column in normalized_df.columns:
            normalized_df["WellID"] = normalized_df[source_id_column].astype(str)
        else:
            normalized_df["WellID"] = [f"Well {index + 1}" for index in range(len(normalized_df))]
        if source_lat_column in normalized_df.columns and source_lat_column != "Latitude(WGS84)":
            normalized_df["Latitude(WGS84)"] = normalized_df[source_lat_column]
        if source_lon_column in normalized_df.columns and source_lon_column != "Longitude(WGS84)":
            normalized_df["Longitude(WGS84)"] = normalized_df[source_lon_column]
        return normalized_df
    return None


def main():
    scratch_path = sys.argv[1]
    helper = TexNetWebToolLaunchHelper(scratch_path)

    try:
        # ---- Read stress parameters ----
        stress_inputs = {
            "reference_depth": helper.getParameterValueWithStepIndexAndParamName(STEP, "reference_depth"),
            "vertical_stress": helper.getParameterValueWithStepIndexAndParamName(STEP, "vertical_stress"),
            "min_horizontal_stress": helper.getParameterValueWithStepIndexAndParamName(STEP, "min_horizontal_stress"),
            "max_horizontal_stress": helper.getParameterValueWithStepIndexAndParamName(STEP, "max_horizontal_stress"),
            "pore_pressure": helper.getParameterValueWithStepIndexAndParamName(STEP, "pore_pressure"),
            "max_stress_azimuth": helper.getParameterValueWithStepIndexAndParamName(STEP, "max_stress_azimuth"),
            "aphi_value": helper.getParameterValueWithStepIndexAndParamName(STEP, "aphi_value"),
            "stress_field_mode": helper.getParameterValueWithStepIndexAndParamName(STEP, "stress_field_mode"),
            "friction_coefficient": helper.getParameterValueWithStepIndexAndParamName(STEP, "friction_coefficient"),
        }

        friction = float(stress_inputs["friction_coefficient"])
        stress_model_type = _get_stress_model_type(helper)

        # Validation
        if stress_inputs["aphi_value"] is not None and stress_inputs["max_horizontal_stress"] is not None:
            raise ValueError("Aphi value and Max Horizontal Stress Gradient cannot both be provided")

        # Store model type as output parameter
        helper.setParamValueWithStepIndexAndParamName(STEP, "stress_model_type", stress_model_type)

        # ---- Load faults ----
        faults_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, "faults_model_inputs_output")
        if faults_path is None:
            raise ValueError("No faults dataset provided.")
        faults_df = pd.read_csv(faults_path)

        # ---- Calculate stresses ----
        report_progress("Calculating stress field")
        stress_state, p0 = calculate_absolute_stresses(stress_inputs, friction, stress_model_type)

        sV, sh, sH = stress_state.principal_stresses

        # ---- Analyse all faults — vectorised over the full fault array ----
        strikes = faults_df["Strike"].to_numpy(dtype=float)
        dips = faults_df["Dip"].to_numpy(dtype=float)

        sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(
            strikes, dips, stress_state, p0, 0.0
        )
        slip_p = np.maximum(
            calculate_slip_pressure(sig, tau, friction, p0, 1.0, 0.5, 0.0,
                                    s11, s22, s33, s12, n1, n2),
            0.0,
        )
        cff = np.round(calculate_cff(sig, tau, friction))      # integer-valued (matches original)
        scu = calculate_scu(sig, tau, friction)

        # ---- Build output DataFrame ----
        step2_df = faults_df.copy()
        step2_df["slip_pressure"] = np.round(slip_p, 3)
        step2_df["coulomb_failure_function"] = np.round(cff, 3)
        step2_df["shear_capacity_utilization"] = np.round(scu, 3)
        step2_df["normal_stress"] = np.round(sig, 3)
        step2_df["shear_stress"] = np.round(tau, 3)

        helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "det_geomechanics_results", step2_df)
        report_progress("Generating maps and diagrams")
        wells_df = _load_map_ready_injection_wells(helper)

        save_fault_results_map_artifact(
            helper,
            STEP,
            step2_df,
            artifact_key="fsp-deterministic-geomechanics-map",
            title="Deterministic Geomechanics Map",
            caption="Leaflet map of deterministic geomechanics fault results.",
            display_order=21,
            result_fields=["slip_pressure", "coulomb_failure_function", "shear_capacity_utilization"],
            color="#7c3aed",
            value_column="slip_pressure",
            legend_title="Deterministic Pore Pressure to Slip",
            value_min_default=0.0,
            well_df=wells_df,
            field_labels=DETERMINISTIC_GEOMECHANICS_FIELD_LABELS,
        )
        save_stereonet_graph_artifact(
            helper,
            step2_df,
            stress_state,
            p0,
            friction,
            stress_inputs["max_stress_azimuth"],
        )

        # ---- Mohr diagram D3 data ----
        tau_eff = tau.tolist()
        sigma_eff = sig.tolist()
        slip_pressures = slip_p.tolist()
        fault_ids = faults_df["FaultID"].astype(str).tolist()
        strikes = list(faults_df["Strike"].astype(float))

        # Determine stress regime label
        if abs(sV) >= abs(sH) and abs(sH) >= abs(sh):
            regime = "Normal"
        elif abs(sH) >= abs(sh) and abs(sh) >= abs(sV):
            regime = "Reverse"
        else:
            regime = "Strike-Slip"

        arcs_df, slip_df, fault_df = mohr_diagram_data_to_d3_portal(
            float(sh), float(sH), float(sV),
            tau_eff, sigma_eff,
            p0, 1.0, 0.5, 0.0,
            strikes, friction,
            regime, slip_pressures, fault_ids,
        )

        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "arcsDF", arcs_df)
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "slipDF", slip_df)
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faultDF", fault_df)
        save_mohr_diagram_graph_artifact(helper, arcs_df, slip_df, fault_df, stress_regime=regime)

        helper.setSuccessForStepIndex(STEP, True)

    except Exception as e:
        helper.addMessageWithStepIndex(STEP, str(e), 2)
        helper.setSuccessForStepIndex(STEP, False)
        helper.writeResultsFile()
        sys.exit(1)

    helper.writeResultsFile()


if __name__ == "__main__":
    main()
