"""
FSP Step 1 — Model Inputs
Reads faults (CSV or shapefile) and injection wells (3 formats),
validates data, and prepares outputs for downstream steps.

Portal invocation: python fsp_step1.py <scratch_path>
Step index (0-based): 0
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp.io.faults import load_faults_csv, load_faults_shapefile, generate_randomized_faults
from fsp.io.coords import latlon_to_wkt
from fsp.io.wells import load_injection_wells, injection_rate_data_to_d3_bbl_day
from graphs.injection_rate import save_injection_rate_graph_artifact

STEP = 0   # 0-based index for Step 1


def _get_injection_path(helper):
    """Return (path, data_type) for the first injection dataset found."""
    for param, dtype in [
        ("injection_wells_annual", "annual_fsp"),
        ("injection_wells_monthly", "monthly_fsp"),
        ("injection_tool_data", "injection_tool_data"),
    ]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, param)
        if path is not None:
            return path, dtype
    raise ValueError("No injection wells dataset provided.")


def _get_fault_path(helper, randomize: bool):
    """Return (path_or_df, fault_type) for faults."""
    if randomize:
        return None, "fsp_native"
    for param, ftype in [("faults", "fsp_native"), ("FaultDataShapefile", "shapefile")]:
        path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, param)
        if path is not None:
            return path, ftype
    raise ValueError("No fault dataset provided.")


def main():
    scratch_path = sys.argv[1]
    helper = TexNetWebToolLaunchHelper(scratch_path)

    try:
        # ---- Faults ----
        randomize = helper.getParameterValueWithStepIndexAndParamName(STEP, "randomize_faults")
        if randomize is None:
            randomize = False

        if randomize:
            num_random = helper.getParameterValueWithStepIndexAndParamName(STEP, "num_random_faults") or 10
            strike_range = helper.getParameterValueWithStepIndexAndParamName(STEP, "random_strike_range") or {}
            dip_range = helper.getParameterValueWithStepIndexAndParamName(STEP, "random_dip_range") or {}
            strike_min = float(strike_range.get("min") or 240.0)
            strike_max = float(strike_range.get("max") or 330.0)
            dip_min = float(dip_range.get("min") or 45.0)
            dip_max = float(dip_range.get("max") or 90.0)
            faults_df = generate_randomized_faults(int(num_random), strike_min, strike_max,
                                                    dip_min, dip_max)
        else:
            fault_path, fault_type = _get_fault_path(helper, randomize)
            if fault_type == "fsp_native":
                faults_df = load_faults_csv(fault_path)
            else:
                faults_df = load_faults_shapefile(fault_path)

        # Validate fault count
        n_unique = faults_df["FaultID"].nunique()
        if n_unique > 1000:
            msg = "Number of faults provided is greater than 1000. Please provide a smaller number of faults."
            helper.addMessageWithStepIndex(STEP, msg, 2)
            raise ValueError(msg)

        # Add WKT column
        if "LengthKm" in faults_df.columns:
            faults_df = latlon_to_wkt(faults_df)

        # Add placeholder columns for downstream steps
        for col in ["slip_pressure", "coulomb_failure_function", "summary_end_year",
                     "summary_fsp", "summary_pressure", "prob_hydro_fsp"]:
            if col not in faults_df.columns:
                faults_df[col] = None

        helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faults_model_inputs_output", faults_df)

        # ---- Injection wells ----
        inj_path, inj_type = _get_injection_path(helper)

        if inj_type == "injection_tool_data":
            inj_df = pd.read_csv(inj_path, dtype={"API Number": str})
        else:
            inj_df = pd.read_csv(inj_path, dtype={"WellID": str})

        # Validate well count
        id_col = "API Number" if inj_type == "injection_tool_data" else "WellID"
        n_wells = inj_df[id_col].nunique()
        if n_wells > 200:
            msg = "Number of wells provided is greater than 200. Please provide a smaller number of wells."
            helper.addMessageWithStepIndex(STEP, msg, 2)
            raise ValueError(msg)

        # D3 injection rate data
        inj_rate_d3 = injection_rate_data_to_d3_bbl_day(inj_df, inj_type)
        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "injection_rate_d3_data", inj_rate_d3)
        save_injection_rate_graph_artifact(helper, inj_rate_d3)

        # Save raw injection wells
        if inj_type == "annual_fsp":
            helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "injection_wells_annual_output", inj_df)
        elif inj_type == "monthly_fsp":
            helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "injection_wells_monthly_output", inj_df)
        elif inj_type == "injection_tool_data":
            # Filtered map layer
            filtered = inj_df.drop_duplicates(subset=["API Number"])[["API Number", "Surface Latitude", "Surface Longitude"]].copy()
            filtered.rename(columns={"API Number": "UWI",
                                      "Surface Latitude": "Latitude(WGS84)",
                                      "Surface Longitude": "Longitude(WGS84)"}, inplace=True)
            helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "injection_tool_data_filtered_map_layer", filtered)
            # Portal CSV not needed; not read by downstream steps.
            # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "injection_tool_data_output", inj_df)

        helper.setSuccessForStepIndex(STEP, True)

    except Exception as e:
        helper.addMessageWithStepIndex(STEP, str(e), 2)
        helper.setSuccessForStepIndex(STEP, False)
        helper.writeResultsFile()
        sys.exit(1)

    helper.writeResultsFile()


if __name__ == "__main__":
    main()
