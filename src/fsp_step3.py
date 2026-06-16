"""
FSP Step 3 — Probabilistic Geomechanics (Monte Carlo)
Samples stress and fault uncertainties to produce a CDF of slip pressure per fault.

Portal invocation: python fsp_step3.py <scratch_path>
Step index (0-based): 2
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from TexNetWebToolGPWrappers import TexNetWebToolLaunchHelper
from fsp.monte_carlo.geomechanics_mc import run_geomechanics_mc
from fsp.geomechanics.stress import calculate_absolute_stresses
from fsp.geomechanics.slip import (
    calculate_fault_effective_stresses,
    ComputeCriticalPorePressureForFailure,
)
from graphs.scientific import (
    save_cdf_artifact,
    save_fault_sensitivity_artifact,
    save_input_distribution_histograms_artifact,
    save_uncertainty_tornado_artifact,
)
from graphs.leaflet_map import save_fault_results_map_artifact
from progress import report_progress

STEP = 2   # 0-based index for Step 3
STEP_PREV = 1  # Step 2 (deterministic geomechanics)


SENSITIVITY_PARAMETERS = {
    "vertical_stress_gradient_uncertainty": ("vertical_stress_gradient", "Vert Stress Grad"),
    "initial_pore_pressure_gradient_uncertainty": ("initial_pore_pressure_gradient", "Pore Press Grad"),
    "max_stress_azimuth_uncertainty": ("max_stress_azimuth", "SHmax Azimuth"),
    "max_horizontal_stress_uncertainty": ("max_horizontal_stress_gradient", "SHmax Gradient"),
    "min_horizontal_stress_uncertainty": ("min_horizontal_stress_gradient", "SHmin Gradient"),
    "aphi_value_uncertainty": ("aphi_value", "APhi Value"),
    "strike_angles_uncertainty": ("strike_angle", "Strike of fault"),
    "dip_angles_uncertainty": ("dip_angle", "Dip of fault"),
    "friction_coefficient_uncertainty": ("friction_coefficient", "Friction Coeff"),
}

DETERMINISTIC_GEOMECHANICS_FIELDS = [
    "slip_pressure",
    "coulomb_failure_function",
    "shear_capacity_utilization",
]

PROBABILISTIC_GEOMECHANICS_RESULT_FIELDS = [
    "Mean",
    "StdDev",
    "Median",
    "Min",
    "Max",
    "det_slip_pressure",
    "det_coulomb_failure_function",
    "det_shear_capacity_utilization",
]

PROBABILISTIC_GEOMECHANICS_FIELD_LABELS = {
    "Mean": "Mean Probabilistic Pore Pressure to Slip",
    "StdDev": "Probabilistic Pore Pressure to Slip Std Dev",
    "Median": "Median Probabilistic Pore Pressure to Slip",
    "Min": "Minimum Probabilistic Pore Pressure to Slip",
    "Max": "Maximum Probabilistic Pore Pressure to Slip",
    "det_slip_pressure": "Deterministic Pore Pressure to Slip",
    "det_coulomb_failure_function": "Deterministic Coulomb Failure Function",
    "det_shear_capacity_utilization": "Deterministic Shear Capacity Utilization",
}


def _prob_geomechanics_cdf(mc_df: pd.DataFrame, det_df: pd.DataFrame) -> pd.DataFrame:
    """Build CDF data per fault from MC results.

    Returns DataFrame with columns: ID, slip_pressure, probability, cumulative_probability.
    """
    det_lookup = {}
    if det_df is not None and not det_df.empty and "slip_pressure" in det_df.columns:
        det_fault_column = "FaultID" if "FaultID" in det_df.columns else "ID" if "ID" in det_df.columns else None
        if det_fault_column:
            det_faults = det_df[[det_fault_column, "slip_pressure"]].copy()
            det_faults[det_fault_column] = det_faults[det_fault_column].astype(str)
            det_faults["slip_pressure"] = pd.to_numeric(det_faults["slip_pressure"], errors="coerce")
            det_faults = det_faults.dropna(subset=["slip_pressure"]).drop_duplicates(subset=[det_fault_column], keep="first")
            det_lookup = det_faults.set_index(det_fault_column)["slip_pressure"].to_dict()

    frames = []
    for fid, fault_df in mc_df.groupby("FaultID", sort=False):
        samples = fault_df["SlipPressure"].values
        sorted_sp = np.sort(samples)
        n = len(sorted_sp)
        probs = np.arange(1, n + 1) / n
        fault_id = str(fid)

        frames.append(pd.DataFrame({
            "ID": fault_id,
            "slip_pressure": sorted_sp.astype(float),
            "probability": probs.astype(float),
            "cumulative_probability": probs.astype(float),
            "det_slip_pressure": float(det_lookup[fault_id]) if fault_id in det_lookup else np.nan,
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _active_uncertainty_parameters(uncertainties: dict):
    for uncertainty_key, (sample_column, label) in SENSITIVITY_PARAMETERS.items():
        delta = uncertainties.get(uncertainty_key)
        try:
            delta_f = float(delta)
        except (TypeError, ValueError):
            continue
        if delta_f > 0.0:
            yield uncertainty_key, sample_column, label


def _numeric(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _stress_parameter_mapping(stress_model_type: str, stress_inputs: dict):
    if stress_model_type in ("gradients", "all_gradients"):
        return {
            "vertical_stress_gradient_uncertainty": ("Vert Stress Grad", "vertical_stress"),
            "initial_pore_pressure_gradient_uncertainty": ("Pore Press Grad", "pore_pressure"),
            "max_stress_azimuth_uncertainty": ("SHmax Azimuth", "max_stress_azimuth"),
            "max_horizontal_stress_uncertainty": ("SHmax Gradient", "max_horizontal_stress"),
            "min_horizontal_stress_uncertainty": ("SHmin Gradient", "min_horizontal_stress"),
        }
    if stress_model_type == "aphi_min" or (
        stress_model_type == "aphi_model" and stress_inputs.get("min_horizontal_stress") is not None
    ):
        return {
            "vertical_stress_gradient_uncertainty": ("Vert Stress Grad", "vertical_stress"),
            "initial_pore_pressure_gradient_uncertainty": ("Pore Press Grad", "pore_pressure"),
            "max_stress_azimuth_uncertainty": ("SHmax Azimuth", "max_stress_azimuth"),
            "aphi_value_uncertainty": ("APhi Value", "aphi_value"),
            "min_horizontal_stress_uncertainty": ("SHmin Gradient", "min_horizontal_stress"),
        }
    return {
        "vertical_stress_gradient_uncertainty": ("Vert Stress Grad", "vertical_stress"),
        "initial_pore_pressure_gradient_uncertainty": ("Pore Press Grad", "pore_pressure"),
        "max_stress_azimuth_uncertainty": ("SHmax Azimuth", "max_stress_azimuth"),
        "aphi_value_uncertainty": ("APhi Value", "aphi_value"),
    }


def _fault_parameter_mapping():
    return {
        "strike_angles_uncertainty": ("Strike of fault", "Strike"),
        "dip_angles_uncertainty": ("Dip of fault", "Dip"),
        "friction_coefficient_uncertainty": ("Friction Coeff", "FrictionCoefficient"),
    }


def _uncertainty_variability_data(
    uncertainties: dict,
    stress_model_type: str,
    stress_inputs: dict,
    fault_inputs: pd.DataFrame,
) -> pd.DataFrame:
    stress_parameter_mapping = _stress_parameter_mapping(stress_model_type, stress_inputs)
    fault_parameter_mapping = _fault_parameter_mapping()

    rows = []
    next_id = 1

    for uncertainty_key, (label, stress_key) in stress_parameter_mapping.items():
        uncertainty_value = _numeric(uncertainties.get(uncertainty_key))
        base_value = _numeric(stress_inputs.get(stress_key))
        if uncertainty_value is None or uncertainty_value <= 0.0 or base_value in (None, 0.0):
            continue

        # Match the legacy MATLAB chart: azimuth uses a fixed 180-degree reference.
        reference_value = 180.0 if uncertainty_key == "max_stress_azimuth_uncertainty" else abs(base_value)
        percent_deviation = min((uncertainty_value / reference_value) * 100.0, 100.0)
        rows.append({
            "id": next_id,
            "label": label,
            "min": round(-percent_deviation, 2),
            "max": round(percent_deviation, 2),
        })
        next_id += 1

    for uncertainty_key, (label, fault_column) in fault_parameter_mapping.items():
        uncertainty_value = _numeric(uncertainties.get(uncertainty_key))
        if uncertainty_value is None or uncertainty_value <= 0.0 or fault_column not in fault_inputs.columns:
            continue

        fault_values = pd.to_numeric(fault_inputs[fault_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        base_value = _numeric(fault_values.mean())
        if base_value in (None, 0.0):
            continue

        # Match the legacy MATLAB chart: strike uses 180 degrees and dip uses 90 degrees.
        if uncertainty_key == "strike_angles_uncertainty":
            reference_value = 180.0
        elif uncertainty_key == "dip_angles_uncertainty":
            reference_value = 90.0
        else:
            reference_value = abs(base_value)

        percent_deviation = min((uncertainty_value / reference_value) * 100.0, 100.0)
        rows.append({
            "id": next_id,
            "label": label,
            "min": round(-percent_deviation, 2),
            "max": round(percent_deviation, 2),
        })
        next_id += 1

    if not rows:
        return pd.DataFrame(columns=["id", "label", "min", "max"])

    result = pd.DataFrame(rows)
    result["span"] = (result["max"] - result["min"]).abs()
    result = result.sort_values("span", ascending=False).drop(columns=["span"]).reset_index(drop=True)
    return result


def _bounded_parameter_value(parameter_key: str, base_value: float, delta: float) -> float:
    updated_value = base_value + delta

    if parameter_key == "max_stress_azimuth":
        return updated_value % 360.0
    if parameter_key == "aphi_value":
        return float(np.clip(updated_value, 0.0, 3.0))
    if parameter_key == "Strike":
        return updated_value % 360.0
    if parameter_key == "Dip":
        return float(np.clip(updated_value, 0.0, 90.0))
    if parameter_key == "FrictionCoefficient":
        return max(updated_value, 0.0)
    return max(updated_value, 0.0)


def _fault_slip_pressure(stress_inputs: dict, fault_row: pd.Series, stress_model_type: str) -> float:
    friction = _numeric(fault_row.get("FrictionCoefficient"))
    if friction is None:
        friction = _numeric(stress_inputs.get("friction_coefficient"))
    if friction is None:
        raise ValueError("Missing friction coefficient for deterministic fault sensitivity calculation.")

    stress_state_obj, p0_abs = calculate_absolute_stresses(stress_inputs, friction, stress_model_type)
    sig_normal, tau_normal, *_ = calculate_fault_effective_stresses(
        _numeric(fault_row["Strike"]),
        _numeric(fault_row["Dip"]),
        stress_state_obj,
        p0_abs,
        0.0,
    )
    slip_pressure = ComputeCriticalPorePressureForFailure(sig_normal, tau_normal, friction, p0_abs)
    return float(np.asarray(slip_pressure).item())


_TORNADO_METHOD = "+/- uncertainty one-at-a-time deterministic"
_TORNADO_COLUMNS = [
    "id",
    "FaultID",
    "parameter",
    "label",
    "low_slip_pressure",
    "high_slip_pressure",
    "low_delta",
    "high_delta",
    "impact",
    "method",
    "baseline_slip_pressure",
]


def _resolve_fault_frictions(fault_inputs: pd.DataFrame, stress_inputs: dict) -> np.ndarray:
    """Per-fault friction array, mirroring _fault_slip_pressure's resolution order.

    Uses the per-fault ``FrictionCoefficient`` where finite, otherwise the global
    ``friction_coefficient`` from stress_inputs. Raises if neither is available
    (same error as the scalar path).
    """
    n = len(fault_inputs)
    global_friction = _numeric(stress_inputs.get("friction_coefficient"))
    if "FrictionCoefficient" in fault_inputs.columns:
        per_fault = pd.to_numeric(fault_inputs["FrictionCoefficient"], errors="coerce").to_numpy(dtype=float)
    else:
        per_fault = np.full(n, np.nan)
    fallback = global_friction if global_friction is not None else np.nan
    frictions = np.where(np.isfinite(per_fault), per_fault, fallback)
    if not np.all(np.isfinite(frictions)):
        raise ValueError("Missing friction coefficient for deterministic fault sensitivity calculation.")
    return frictions


def _vectorized_fault_slip_pressures(stress_inputs: dict, strikes, dips, frictions,
                                     stress_model_type: str) -> np.ndarray:
    """Vectorized equivalent of _fault_slip_pressure over arrays of faults.

    The absolute stress state depends only on (stress_inputs, friction, model), so we
    group faults by their friction value and compute the stress state once per group —
    eliminating the per-fault recomputation in the original scalar loop. Each fault's
    result is identical to _fault_slip_pressure (same functions, same inputs).
    """
    strikes = np.asarray(strikes, dtype=float)
    dips = np.asarray(dips, dtype=float)
    frictions = np.asarray(frictions, dtype=float)
    out = np.empty(strikes.shape[0], dtype=float)
    for mu in np.unique(frictions):
        mask = frictions == mu
        stress_state_obj, p0_abs = calculate_absolute_stresses(stress_inputs, float(mu), stress_model_type)
        sig_normal, tau_normal, *_ = calculate_fault_effective_stresses(
            strikes[mask], dips[mask], stress_state_obj, p0_abs, 0.0,
        )
        slip = ComputeCriticalPorePressureForFailure(sig_normal, tau_normal, float(mu), p0_abs)
        out[mask] = np.asarray(slip, dtype=float)
    return out


def _bounded_fault_array(parameter_key: str, base_array: np.ndarray, delta: float) -> np.ndarray:
    """Vectorized form of _bounded_parameter_value for fault-geometry columns."""
    updated = base_array + delta
    if parameter_key == "Strike":
        return np.mod(updated, 360.0)
    if parameter_key == "Dip":
        return np.clip(updated, 0.0, 90.0)
    return np.maximum(updated, 0.0)  # FrictionCoefficient (and any other)


def _fault_sensitivity_tornado_data(
    stress_inputs: dict,
    fault_inputs: pd.DataFrame,
    uncertainties: dict,
    stress_model_type: str,
) -> pd.DataFrame:
    stress_parameter_mapping = _stress_parameter_mapping(stress_model_type, stress_inputs)
    fault_parameter_mapping = _fault_parameter_mapping()

    n_faults = len(fault_inputs)
    if n_faults == 0:
        return pd.DataFrame(columns=_TORNADO_COLUMNS)

    fault_ids = fault_inputs["FaultID"].astype(str).to_numpy()
    strikes = pd.to_numeric(fault_inputs["Strike"], errors="coerce").to_numpy(dtype=float)
    dips = pd.to_numeric(fault_inputs["Dip"], errors="coerce").to_numpy(dtype=float)
    frictions = _resolve_fault_frictions(fault_inputs, stress_inputs)

    # Baseline slip pressure per fault (stress state cached per unique friction).
    baseline = _vectorized_fault_slip_pressures(stress_inputs, strikes, dips, frictions, stress_model_type)

    # --- Stress-parameter perturbations: one perturbed stress field per param/direction,
    #     evaluated vectorized across all faults. ---
    stress_param_arrays = {}  # uncertainty_key -> (label, low_arr, high_arr)
    for uncertainty_key, (label, stress_key) in stress_parameter_mapping.items():
        uncertainty_value = _numeric(uncertainties.get(uncertainty_key))
        base_value = _numeric(stress_inputs.get(stress_key))
        if uncertainty_value is None or uncertainty_value <= 0.0 or base_value is None:
            continue

        low_inputs = dict(stress_inputs)
        high_inputs = dict(stress_inputs)
        low_inputs[stress_key] = _bounded_parameter_value(stress_key, base_value, -uncertainty_value)
        high_inputs[stress_key] = _bounded_parameter_value(stress_key, base_value, uncertainty_value)

        low_arr = _vectorized_fault_slip_pressures(low_inputs, strikes, dips, frictions, stress_model_type)
        high_arr = _vectorized_fault_slip_pressures(high_inputs, strikes, dips, frictions, stress_model_type)
        stress_param_arrays[uncertainty_key] = (label, low_arr, high_arr)

    # --- Fault-geometry perturbations: perturb the relevant fault array, base stress field. ---
    fault_param_arrays = {}  # uncertainty_key -> (label, low_arr, high_arr, valid_mask)
    for uncertainty_key, (label, fault_key) in fault_parameter_mapping.items():
        uncertainty_value = _numeric(uncertainties.get(uncertainty_key))
        if uncertainty_value is None or uncertainty_value <= 0.0:
            continue

        # Raw per-fault base values (no global fallback — mirrors fault_row.get(fault_key)).
        if fault_key == "Strike":
            raw_base = strikes
        elif fault_key == "Dip":
            raw_base = dips
        elif fault_key in fault_inputs.columns:
            raw_base = pd.to_numeric(fault_inputs[fault_key], errors="coerce").to_numpy(dtype=float)
        else:
            continue  # base_value is None for every fault -> param skipped entirely

        valid_mask = np.isfinite(raw_base)
        if not np.any(valid_mask):
            continue

        low_base = _bounded_fault_array(fault_key, raw_base, -uncertainty_value)
        high_base = _bounded_fault_array(fault_key, raw_base, uncertainty_value)

        if fault_key == "Strike":
            low_arr = _vectorized_fault_slip_pressures(stress_inputs, low_base, dips, frictions, stress_model_type)
            high_arr = _vectorized_fault_slip_pressures(stress_inputs, high_base, dips, frictions, stress_model_type)
        elif fault_key == "Dip":
            low_arr = _vectorized_fault_slip_pressures(stress_inputs, strikes, low_base, frictions, stress_model_type)
            high_arr = _vectorized_fault_slip_pressures(stress_inputs, strikes, high_base, frictions, stress_model_type)
        else:  # FrictionCoefficient — perturbs both the stress state and the slip mu.
            # Invalid faults are excluded at assembly; fill them with a finite value so the
            # grouped stress-state computation never sees NaN.
            low_friction = np.where(valid_mask, low_base, frictions)
            high_friction = np.where(valid_mask, high_base, frictions)
            low_arr = _vectorized_fault_slip_pressures(stress_inputs, strikes, dips, low_friction, stress_model_type)
            high_arr = _vectorized_fault_slip_pressures(stress_inputs, strikes, dips, high_friction, stress_model_type)

        fault_param_arrays[uncertainty_key] = (label, low_arr, high_arr, valid_mask)

    # --- Assemble rows in the same fault-major / param order as the original scalar loop
    #     so the (stable-input) post-sort output is identical. ---
    rows = []
    for i in range(n_faults):
        fault_id = str(fault_ids[i])
        baseline_slip_pressure = float(baseline[i])

        for uncertainty_key, (label, _stress_key) in stress_parameter_mapping.items():
            entry = stress_param_arrays.get(uncertainty_key)
            if entry is None:
                continue
            _label, low_arr, high_arr = entry
            low_slip_pressure = float(low_arr[i])
            high_slip_pressure = float(high_arr[i])
            rows.append({
                "id": f"{fault_id}:{uncertainty_key}",
                "FaultID": fault_id,
                "parameter": uncertainty_key,
                "label": label,
                "low_slip_pressure": low_slip_pressure,
                "high_slip_pressure": high_slip_pressure,
                "low_delta": low_slip_pressure - baseline_slip_pressure,
                "high_delta": high_slip_pressure - baseline_slip_pressure,
                "impact": high_slip_pressure - low_slip_pressure,
                "method": _TORNADO_METHOD,
                "baseline_slip_pressure": baseline_slip_pressure,
            })

        for uncertainty_key, (label, _fault_key) in fault_parameter_mapping.items():
            entry = fault_param_arrays.get(uncertainty_key)
            if entry is None:
                continue
            _label, low_arr, high_arr, valid_mask = entry
            if not valid_mask[i]:
                continue
            low_slip_pressure = float(low_arr[i])
            high_slip_pressure = float(high_arr[i])
            rows.append({
                "id": f"{fault_id}:{uncertainty_key}",
                "FaultID": fault_id,
                "parameter": uncertainty_key,
                "label": label,
                "low_slip_pressure": low_slip_pressure,
                "high_slip_pressure": high_slip_pressure,
                "low_delta": low_slip_pressure - baseline_slip_pressure,
                "high_delta": high_slip_pressure - baseline_slip_pressure,
                "impact": high_slip_pressure - low_slip_pressure,
                "method": _TORNADO_METHOD,
                "baseline_slip_pressure": baseline_slip_pressure,
            })

    if not rows:
        return pd.DataFrame(columns=_TORNADO_COLUMNS)

    result = pd.DataFrame(rows)
    result["abs_impact"] = result["impact"].abs()
    result = result.sort_values(["FaultID", "abs_impact"], ascending=[True, False]).drop(columns=["abs_impact"])
    return result.reset_index(drop=True)


def _mc_uncertainty_sensitivity_data(
    mc_df: pd.DataFrame,
    sample_inputs_df: pd.DataFrame,
    uncertainties: dict,
    quantile: float = 0.10,
) -> pd.DataFrame:
    """Estimate per-fault parameter impact from existing MC samples.

    Compares median slip pressure in the low and high sample quantile bands for each
    varied input. This reuses the original MC run and avoids a second OAT pass.
    """
    required = {"SimulationID", "FaultID"}
    if mc_df is None or sample_inputs_df is None or not required.issubset(mc_df.columns) or not required.issubset(sample_inputs_df.columns):
        return pd.DataFrame()

    merged = pd.merge(
        mc_df[["SimulationID", "FaultID", "SlipPressure"]].copy(),
        sample_inputs_df.copy(),
        on=["SimulationID", "FaultID"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    rows = []
    merged["FaultID"] = merged["FaultID"].astype(str)
    merged["SlipPressure"] = pd.to_numeric(merged["SlipPressure"], errors="coerce")

    for fault_id, fault_df in merged.groupby("FaultID", sort=True):
        fault_df = fault_df.dropna(subset=["SlipPressure"])
        if fault_df.empty:
            continue

        for parameter, sample_column, label in _active_uncertainty_parameters(uncertainties):
            if sample_column not in fault_df.columns:
                continue

            param_df = fault_df[["SlipPressure", sample_column]].copy()
            param_df[sample_column] = pd.to_numeric(param_df[sample_column], errors="coerce")
            param_df = param_df.dropna(subset=[sample_column])
            if param_df.empty or param_df[sample_column].nunique() <= 1:
                continue

            low_cutoff = param_df[sample_column].quantile(quantile)
            high_cutoff = param_df[sample_column].quantile(1.0 - quantile)
            low_band = param_df[param_df[sample_column] <= low_cutoff]
            high_band = param_df[param_df[sample_column] >= high_cutoff]
            if low_band.empty or high_band.empty:
                continue

            low_slip = float(low_band["SlipPressure"].median())
            high_slip = float(high_band["SlipPressure"].median())
            impact = high_slip - low_slip
            rows.append({
                "FaultID": str(fault_id),
                "parameter": parameter,
                "sample_column": sample_column,
                "label": label,
                "low_slip_pressure": low_slip,
                "high_slip_pressure": high_slip,
                "impact": float(impact),
                "low_delta": float(min(0.0, impact)),
                "high_delta": float(max(0.0, impact)),
                "method": f"P{int(quantile * 100)} vs P{int((1.0 - quantile) * 100)} sample-band median",
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["abs_impact"] = result["impact"].abs()
    result = result.sort_values(["FaultID", "abs_impact"], ascending=[True, False])
    result = result.drop(columns=["abs_impact"])
    result["id"] = result["FaultID"].astype(str) + ":" + result["parameter"].astype(str)
    return result


def _fault_sensitivity_data(mc_df: pd.DataFrame, det_df: pd.DataFrame) -> pd.DataFrame:
    det_slip_by_fault = {}
    if det_df is not None and not det_df.empty and "FaultID" in det_df.columns and "slip_pressure" in det_df.columns:
        det_slip_by_fault = {
            str(row["FaultID"]): float(row["slip_pressure"])
            for _, row in det_df[["FaultID", "slip_pressure"]].dropna().iterrows()
        }

    rows = []
    mc_clean = mc_df.assign(
        FaultID=mc_df["FaultID"].astype(str),
        SlipPressure=pd.to_numeric(mc_df["SlipPressure"], errors="coerce"),
    ).dropna(subset=["SlipPressure"])
    for fid, fault_df in mc_clean.groupby("FaultID", sort=False):
        samples = fault_df["SlipPressure"].values
        if len(samples) == 0:
            continue
        p10, p90 = np.percentile(samples, [10, 90])
        det_slip = det_slip_by_fault.get(fid, float(np.mean(samples)))
        rows.append({
            "id": fid,
            "FaultID": fid,
            "label": fid,
            "slip_pressure": float(p90 - p10),
            "probability": float(np.mean(samples <= det_slip)),
            "det_slip_pressure": det_slip,
        })
    return pd.DataFrame(rows)


def _probabilistic_geomechanics_fault_map_data(
    fault_inputs: pd.DataFrame,
    stats_df: pd.DataFrame,
    det_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge fault geometry, probabilistic stats, and prior deterministic answers."""
    map_df = fault_inputs.copy()
    if "FaultID" not in map_df.columns:
        return map_df

    map_df["FaultID"] = map_df["FaultID"].astype(str)

    if stats_df is not None and not stats_df.empty and "FaultID" in stats_df.columns:
        stats_columns = ["FaultID"] + [field for field in ["Mean", "StdDev", "Median", "Min", "Max"] if field in stats_df.columns]
        stats_subset = stats_df[stats_columns].copy()
        stats_subset["FaultID"] = stats_subset["FaultID"].astype(str)
        map_df = map_df.merge(stats_subset, on="FaultID", how="left")

    if det_df is not None and not det_df.empty and "FaultID" in det_df.columns:
        det_columns = ["FaultID"] + [field for field in DETERMINISTIC_GEOMECHANICS_FIELDS if field in det_df.columns]
        det_subset = det_df[det_columns].copy()
        det_subset["FaultID"] = det_subset["FaultID"].astype(str)
        det_subset = det_subset.rename(columns={
            "slip_pressure": "det_slip_pressure",
            "coulomb_failure_function": "det_coulomb_failure_function",
            "shear_capacity_utilization": "det_shear_capacity_utilization",
        })
        map_df = map_df.merge(det_subset, on="FaultID", how="left")

    return map_df


def main():
    scratch_path = sys.argv[1]
    helper = TexNetWebToolLaunchHelper(scratch_path)

    try:
        random_seed = helper.getParameterValueWithStepIndexAndParamName(STEP, "random_seed")

        # ---- Stress inputs from Step 2 ----
        stress_inputs = {
            "reference_depth": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "reference_depth"),
            "vertical_stress": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "vertical_stress"),
            "min_horizontal_stress": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "min_horizontal_stress"),
            "max_horizontal_stress": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "max_horizontal_stress"),
            "pore_pressure": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "pore_pressure"),
            "max_stress_azimuth": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "max_stress_azimuth"),
            "aphi_value": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "aphi_value"),
            "friction_coefficient": helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "friction_coefficient"),
        }
        friction = float(stress_inputs["friction_coefficient"])
        stress_model_type = helper.getParameterValueWithStepIndexAndParamName(STEP_PREV, "stress_model_type") or "gradients"

        # ---- Uncertainties ----
        def _p(name):
            return helper.getParameterValueWithStepIndexAndParamName(STEP, name)

        if stress_model_type in ("gradients", "all_gradients"):
            uncertainties = {
                "vertical_stress_gradient_uncertainty": _p("vertical_stress_gradient_uncertainty"),
                "initial_pore_pressure_gradient_uncertainty": _p("initial_pore_pressure_gradient_uncertainty"),
                "max_stress_azimuth_uncertainty": _p("max_stress_azimuth_uncertainty"),
                "max_horizontal_stress_uncertainty": _p("max_horizontal_stress_gradient_uncertainty"),
                "min_horizontal_stress_uncertainty": _p("min_horizontal_stress_gradient_uncertainty"),
                "strike_angles_uncertainty": _p("strike_angles_uncertainty"),
                "dip_angles_uncertainty": _p("dip_angles_uncertainty"),
                "friction_coefficient_uncertainty": _p("friction_coefficient_uncertainty"),
            }
        elif stress_model_type == "aphi_min" or (
            stress_model_type == "aphi_model" and stress_inputs["min_horizontal_stress"] is not None
        ):
            stress_model_type = "aphi_min"
            uncertainties = {
                "vertical_stress_gradient_uncertainty": _p("vertical_stress_gradient_uncertainty"),
                "initial_pore_pressure_gradient_uncertainty": _p("initial_pore_pressure_gradient_uncertainty"),
                "max_stress_azimuth_uncertainty": _p("max_stress_azimuth_uncertainty"),
                "aphi_value_uncertainty": _p("aphi_value_uncertainty"),
                "min_horizontal_stress_uncertainty": _p("min_horizontal_stress_gradient_uncertainty"),
                "strike_angles_uncertainty": _p("strike_angles_uncertainty"),
                "dip_angles_uncertainty": _p("dip_angles_uncertainty"),
                "friction_coefficient_uncertainty": _p("friction_coefficient_uncertainty"),
            }
        else:
            stress_model_type = "aphi_no_min"
            uncertainties = {
                "vertical_stress_gradient_uncertainty": _p("vertical_stress_gradient_uncertainty"),
                "initial_pore_pressure_gradient_uncertainty": _p("initial_pore_pressure_gradient_uncertainty"),
                "max_stress_azimuth_uncertainty": _p("max_stress_azimuth_uncertainty"),
                "aphi_value_uncertainty": _p("aphi_value_uncertainty"),
                "strike_angles_uncertainty": _p("strike_angles_uncertainty"),
                "dip_angles_uncertainty": _p("dip_angles_uncertainty"),
                "friction_coefficient_uncertainty": _p("friction_coefficient_uncertainty"),
            }

        # ---- Load faults ----
        report_progress("Loading fault data")
        faults_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP, "faults")
        if faults_path is None:
            raise ValueError("No fault dataset provided.")
        fault_inputs = pd.read_csv(faults_path, dtype={"FaultID": str})
        if fault_inputs.empty:
            raise ValueError("A fault dataset is required to run probabilistic geomechanics. Skip this step for injection-pressure-only runs.")

        # Populate FrictionCoefficient column
        if "FrictionCoefficient" not in fault_inputs.columns:
            fault_inputs["FrictionCoefficient"] = friction
        else:
            fault_inputs["FrictionCoefficient"] = friction

        n_sims = int(helper.getParameterValueWithStepIndexAndParamName(STEP, "mc_iterations") or 1000)

        # ---- Run MC ----
        report_progress("Running Monte Carlo simulations")
        mc_results, sample_inputs_df = run_geomechanics_mc(
            stress_inputs, fault_inputs, n_sims, uncertainties,
            stress_model_type, friction,
            random_seed=int(random_seed) if random_seed is not None else None,
            return_sample_inputs=True,
        )

        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_geomechanics_results", mc_results)

        # ---- Load deterministic results for CDF colours ----
        det_path = helper.getDatasetFilePathWithStepIndexAndParamName(STEP_PREV, "det_geomechanics_results")
        det_df = pd.read_csv(det_path) if det_path else pd.DataFrame()

        report_progress("Building probability curves")
        cdf_df = _prob_geomechanics_cdf(mc_results, det_df)
        helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_geomechanics_cdf_graph_data", cdf_df)
        save_cdf_artifact(
            helper,
            STEP,
            cdf_df,
            artifact_key="fsp-probabilistic-geomechanics-cdf",
            title="Probabilistic Geomechanics CDF",
            pressure_label="Delta Pore Pressure to Slip (psi)",
            probability_label="Cumulative Probability",
            display_order=30,
        )

        save_input_distribution_histograms_artifact(
            helper,
            STEP,
            sample_inputs_df,
            mc_results,
            artifact_key="fsp-probabilistic-geomechanics-histogram",
            title="Probabilistic Geomechanics Histogram",
            display_order=31,
        )

        # ---- Statistics ----
        stats_df = (
            mc_results.assign(SlipPressure=pd.to_numeric(mc_results["SlipPressure"], errors="coerce"))
            .dropna(subset=["SlipPressure"])
            .groupby("FaultID")["SlipPressure"]
            .agg(
                Mean="mean",
                StdDev=lambda values: float(np.std(values)),
                Median="median",
                Min="min",
                Max="max",
            )
            .reset_index()
        )
        for column in ["Mean", "StdDev", "Median", "Min", "Max"]:
            stats_df[column] = stats_df[column].round(2)
        stats_df["FaultID"] = stats_df["FaultID"].astype(str)
        #helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "prob_geomechanics_stats", stats_df)

        report_progress("Generating fault map and sensitivity analysis")
        fault_map_df = _probabilistic_geomechanics_fault_map_data(fault_inputs, stats_df, det_df)
        # Portal CSV not needed; graph artifact covers this output.
        # if helper.getParameterStateWithStepIndexAndParamName(STEP, "faults_with_prob_geomechanics_results") is not None:
        #     helper.saveDataFrameAsParameterWithStepIndexAndParamName(STEP, "faults_with_prob_geomechanics_results", fault_map_df)
        save_fault_results_map_artifact(
            helper,
            STEP,
            fault_map_df,
            artifact_key="fsp-probabilistic-geomechanics-map",
            title="Probabilistic Geomechanics Fault Map",
            caption="Leaflet map of probabilistic geomechanics fault results.",
            display_order=34,
            result_fields=PROBABILISTIC_GEOMECHANICS_RESULT_FIELDS,
            color="#2563eb",
            value_column="Mean",
            legend_title="Mean Probabilistic Pore Pressure to Slip",
            field_labels=PROBABILISTIC_GEOMECHANICS_FIELD_LABELS,
        )

        variability_df = _uncertainty_variability_data(uncertainties, stress_model_type, stress_inputs, fault_inputs)
        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(
        #     STEP,
        #     "uncertainty_variability_tornado_chart_data",
        #     variability_df,
        # )
        save_uncertainty_tornado_artifact(
            helper,
            STEP,
            variability_df,
            artifact_key="fsp-geomechanics-input-variability-tornado",
            title="Variability in Inputs",
            x_label="Percent Deviation [%]",
            display_order=32,
        )

        uncertainty_df = _fault_sensitivity_tornado_data(stress_inputs, fault_inputs, uncertainties, stress_model_type)
        # Portal CSV not needed; graph artifact covers this output.
        # helper.saveDataFrameAsParameterWithStepIndexAndParamName(
        #     STEP,
        #     "prob_geomechanics_fault_sensitivity_tornado_chart_data",
        #     uncertainty_df,
        # )
        save_uncertainty_tornado_artifact(
            helper,
            STEP,
            uncertainty_df,
            artifact_key="fsp-geomechanics-uncertainty-variability-tornado",
            title="Fault Sensitivity Analysis",
            x_label="Delta Pore Pressure to Slip [psi]",
            display_order=33,
        )

        #sensitivity_df = _fault_sensitivity_data(mc_results, det_df)
        '''
        save_fault_sensitivity_artifact(
            helper,
            STEP,
            sensitivity_df,
            artifact_key="fsp-geomechanics-fault-sensitivity",
            title="Geomechanics Fault Sensitivity",
            display_order=33,
        )
        '''

        helper.setSuccessForStepIndex(STEP, True)

    except Exception as e:
        helper.addMessageWithStepIndex(STEP, str(e), 2)
        helper.setSuccessForStepIndex(STEP, False)
        helper.writeResultsFile()
        sys.exit(1)

    helper.writeResultsFile()


if __name__ == "__main__":
    main()
