"""
Vectorised Monte Carlo for geomechanics uncertainties.
Port of FSP/probabilistic_geomechanics_process.jl run_monte_carlo.

Strategy:
  - Pre-generate ALL n_sims parameter samples as numpy arrays (no Python loop).
  - Vectorise calculate_fault_effective_stresses over all sims simultaneously.
  - Use ComputeCriticalPorePressureForFailure for the MC slip pressure.
"""
import numpy as np
import pandas as pd
from ..models.stress import StressState
from ..geomechanics.stress import calculate_absolute_stresses
from ..geomechanics.slip import (
    calculate_fault_effective_stresses,
    ComputeCriticalPorePressureForFailure,
)


def _sample_uniform(base, delta, n, lo=None, hi=None):
    """Sample n values from Uniform(base-delta, base+delta), optionally clamped."""
    lo_v = base - delta if lo is None else max(base - delta, lo)
    hi_v = base + delta if hi is None else min(base + delta, hi)
    if lo_v >= hi_v:
        return np.full(n, base)
    return np.random.uniform(lo_v, hi_v, n)


def _sample_azimuth(base, delta, n):
    """Sample n azimuth values with 0/360 wrap-around."""
    if delta <= 0.0:
        return np.full(n, base)
    lo = (base - delta) % 360.0
    hi = (base + delta) % 360.0
    if lo <= hi:
        return np.random.uniform(lo, hi, n)
    # Crosses 0/360 boundary: sample proportionally from two intervals
    span_lo = 360.0 - lo
    span_hi = hi
    frac = span_lo / (span_lo + span_hi)
    mask = np.random.random(n) < frac
    result = np.where(mask,
                      np.random.uniform(lo, 360.0, n),
                      np.random.uniform(0.0, hi, n))
    return result


def run_geomechanics_mc(stress_inputs: dict, faults_df: pd.DataFrame,
                        n_sims: int, uncertainties: dict,
                        stress_model_type: str,
                        friction_coefficient: float,
                        random_seed=None,
                        return_sample_inputs: bool = False):
    """Run Monte Carlo geomechanics simulation.

    Returns DataFrame with columns: SimulationID, FaultID, SlipPressure.
    If return_sample_inputs is true, also returns sampled inputs by SimulationID and FaultID.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_faults = len(faults_df)

    # ---- Sample stress parameters (n_sims each) ----
    def _unc(key, default=0.0):
        v = uncertainties.get(key)
        return float(v) if v is not None else default

    # Vertical stress gradient
    sv_base = float(stress_inputs["vertical_stress"])
    sv_unc = _unc("vertical_stress_gradient_uncertainty")
    sv_samples = _sample_uniform(sv_base, sv_unc, n_sims, lo=0.0)

    # Pore pressure gradient
    pp_base = float(stress_inputs["pore_pressure"])
    pp_unc = _unc("initial_pore_pressure_gradient_uncertainty")
    pp_samples = _sample_uniform(pp_base, pp_unc, n_sims, lo=0.0)

    # Azimuth
    az_base = float(stress_inputs["max_stress_azimuth"])
    az_unc = _unc("max_stress_azimuth_uncertainty")
    az_samples = _sample_azimuth(az_base, az_unc, n_sims)

    # Horizontal stress or aphi
    if stress_model_type in ("gradients", "all_gradients"):
        sh_grad_base = float(stress_inputs["min_horizontal_stress"])
        sh_unc = _unc("min_horizontal_stress_uncertainty")
        sh_samples = _sample_uniform(sh_grad_base, sh_unc, n_sims, lo=0.0)

        sH_grad_base = float(stress_inputs["max_horizontal_stress"])
        sH_unc = _unc("max_horizontal_stress_uncertainty")
        sH_grad_samples = _sample_uniform(sH_grad_base, sH_unc, n_sims, lo=0.0)
        aphi_samples = None

    elif stress_model_type in ("aphi_min",):
        sh_grad_base = float(stress_inputs["min_horizontal_stress"])
        sh_unc = _unc("min_horizontal_stress_uncertainty")
        sh_samples = _sample_uniform(sh_grad_base, sh_unc, n_sims, lo=0.0)

        aphi_base = float(stress_inputs["aphi_value"])
        aphi_unc = _unc("aphi_value_uncertainty")
        aphi_samples = _sample_uniform(aphi_base, aphi_unc, n_sims, lo=0.0, hi=3.0)
        sH_grad_samples = None

    else:  # aphi_no_min
        aphi_base = float(stress_inputs["aphi_value"])
        aphi_unc = _unc("aphi_value_uncertainty")
        aphi_samples = _sample_uniform(aphi_base, aphi_unc, n_sims, lo=0.0, hi=3.0)
        sh_samples = None
        sH_grad_samples = None

    # Fault geometry samples — per fault × per sim
    strike_base = faults_df["Strike"].values.astype(float)          # (n_faults,)
    dip_base = faults_df["Dip"].values.astype(float)                # (n_faults,)

    strike_unc = _unc("strike_angles_uncertainty")
    dip_unc = _unc("dip_angles_uncertainty")
    mu_unc = _unc("friction_coefficient_uncertainty")

    # shape: (n_faults, n_sims)
    if strike_unc > 0:
        strike_lo = (strike_base[:, None] - strike_unc) % 360.0
        strike_hi = (strike_base[:, None] + strike_unc) % 360.0
        # simplified: if no wrap needed just uniform
        strikes_mc = np.random.uniform(
            np.maximum(strike_base[:, None] - strike_unc, 0.0),
            np.minimum(strike_base[:, None] + strike_unc, 360.0),
            (n_faults, n_sims)
        )
    else:
        strikes_mc = np.tile(strike_base[:, None], (1, n_sims))

    if dip_unc > 0:
        dips_mc = np.random.uniform(
            np.maximum(dip_base[:, None] - dip_unc, 0.0),
            np.minimum(dip_base[:, None] + dip_unc, 90.0),
            (n_faults, n_sims)
        )
    else:
        dips_mc = np.tile(dip_base[:, None], (1, n_sims))

    if mu_unc > 0:
        mu_mc = np.random.uniform(
            max(friction_coefficient - mu_unc, 0.0),
            friction_coefficient + mu_unc,
            n_sims
        )
    else:
        mu_mc = np.full(n_sims, friction_coefficient)

    # ---- Run MC: vectorised over simulations for each fault ----
    pps_to_slip = np.zeros((n_faults, n_sims), dtype=float)
    ref_depth = float(stress_inputs["reference_depth"])

    for sim_i in range(n_sims):
        sim_stress = dict(stress_inputs)
        sim_stress["vertical_stress"] = sv_samples[sim_i]
        sim_stress["pore_pressure"] = pp_samples[sim_i]
        sim_stress["max_stress_azimuth"] = az_samples[sim_i]

        if stress_model_type in ("gradients", "all_gradients"):
            sim_stress["min_horizontal_stress"] = sh_samples[sim_i]
            sim_stress["max_horizontal_stress"] = sH_grad_samples[sim_i]
        elif stress_model_type == "aphi_min":
            sim_stress["min_horizontal_stress"] = sh_samples[sim_i]
            sim_stress["aphi_value"] = aphi_samples[sim_i]
        else:
            sim_stress["aphi_value"] = aphi_samples[sim_i]
            sim_stress.pop("min_horizontal_stress", None)

        stress_state_obj, p0_abs = calculate_absolute_stresses(
            sim_stress, friction_coefficient, stress_model_type
        )

        # All faults at once for this simulation
        s_vec = strikes_mc[:, sim_i]
        d_vec = dips_mc[:, sim_i]
        dp_vec = np.zeros(n_faults)

        sig, tau, *_ = calculate_fault_effective_stresses(
            s_vec, d_vec, stress_state_obj, p0_abs, dp_vec
        )

        mu_sim = mu_mc[sim_i]
        pp = ComputeCriticalPorePressureForFailure(sig, tau, mu_sim, p0_abs)
        pps_to_slip[:, sim_i] = pp

    # ---- Build result DataFrame ----
    fault_ids = faults_df["FaultID"].astype(str).values
    rows = []
    sample_rows = [] if return_sample_inputs else None
    for fi in range(n_faults):
        for si in range(n_sims):
            rows.append({
                "SimulationID": si + 1,
                "FaultID": fault_ids[fi],
                "SlipPressure": float(pps_to_slip[fi, si]),
            })
            if return_sample_inputs:
                sample_row = {
                    "SimulationID": si + 1,
                    "FaultID": fault_ids[fi],
                    "vertical_stress_gradient": float(sv_samples[si]),
                    "initial_pore_pressure_gradient": float(pp_samples[si]),
                    "max_stress_azimuth": float(az_samples[si]),
                    "friction_coefficient": float(mu_mc[si]),
                    "strike_angle": float(strikes_mc[fi, si]),
                    "dip_angle": float(dips_mc[fi, si]),
                }
                if sH_grad_samples is not None:
                    sample_row["max_horizontal_stress_gradient"] = float(sH_grad_samples[si])
                if sh_samples is not None:
                    sample_row["min_horizontal_stress_gradient"] = float(sh_samples[si])
                if aphi_samples is not None:
                    sample_row["aphi_value"] = float(aphi_samples[si])
                sample_rows.append(sample_row)

    results_df = pd.DataFrame(rows)
    if return_sample_inputs:
        return results_df, pd.DataFrame(sample_rows)
    return results_df
