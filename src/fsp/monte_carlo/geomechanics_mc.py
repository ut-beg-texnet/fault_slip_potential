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

    # ---- Run MC ----
    ref_depth = float(stress_inputs["reference_depth"])
    _DEG2RAD = np.pi / 180.0

    if stress_model_type in ("gradients", "all_gradients"):
        # Fully vectorized path: compute all (n_faults × n_sims) at once
        # Stress state arrays — shape (1, n_sims) for broadcasting with (n_faults, n_sims) fault arrays
        sV = np.round(sv_samples * ref_depth, 4)[np.newaxis, :]
        p0 = np.round(pp_samples * ref_depth, 4)[np.newaxis, :]
        sH = np.round(sH_grad_samples * ref_depth, 2)[np.newaxis, :]
        sh = np.round(sh_samples * ref_depth, 2)[np.newaxis, :]
        az_rad = az_samples[np.newaxis, :] * _DEG2RAD

        # Fault geometry — shape (n_faults, n_sims)
        str_rad = strikes_mc * _DEG2RAD
        dip_rad = dips_mc * _DEG2RAD

        cos_az = np.cos(az_rad)
        sin_az = np.sin(az_rad)
        sd = np.sin(dip_rad)
        cs = np.cos(str_rad)
        ss = np.sin(str_rad)

        s11 = sh * cos_az**2 + sH * sin_az**2 - p0
        s22 = sh * sin_az**2 + sH * cos_az**2 - p0
        s33 = sV - p0
        s12 = (sH - sh) * cos_az * sin_az

        n1 = sd * cs
        n2 = -sd * ss
        n1_sq = n1**2
        n2_sq = n2**2
        n1_n2 = n1 * n2
        n1_cubed = n1**3
        s11_s33 = s11 - s33
        s22_s33 = s22 - s33

        sqrt_arg = (
            n2_sq * (s12**2 - (-1.0 + n2_sq) * s22_s33**2)
            - n1_sq * n1_sq * s11_s33**2
            + 4.0 * n1_cubed * n2 * s12 * (-s11_s33)
            + 2.0 * n1_n2 * s12 * (s11 + s22 - 2.0 * n2_sq * s22 + 2.0 * (-1.0 + n2_sq) * s33)
            + n1_sq * (
                s11**2
                + (1.0 - 4.0 * n2_sq) * s12**2
                - 2.0 * s11 * (n2_sq * s22_s33 + s33)
                + s33 * (2.0 * n2_sq * s22_s33 + s33)
            )
        )
        sig = np.maximum(2.0 * n1_n2 * s12 + n1_sq * s11_s33 + n2_sq * s22_s33 + s33, 0.0)
        tau = np.sqrt(np.maximum(sqrt_arg, 0.0))

        # ComputeCriticalPorePressureForFailure — vectorized (n_faults, n_sims)
        mu_2d = mu_mc[np.newaxis, :]
        mobmu = np.where(sig > 0.0, np.abs(tau / np.maximum(sig, 1e-30)), mu_2d)
        pps_to_slip = np.where(mobmu < mu_2d, ((mu_2d - mobmu) / mu_2d) * np.abs(sig), 0.0)

    else:
        # APhi models: per-simulation loop (vectorized over faults within each sim)
        pps_to_slip = np.zeros((n_faults, n_sims), dtype=float)
        for sim_i in range(n_sims):
            sim_stress = dict(stress_inputs)
            sim_stress["vertical_stress"] = sv_samples[sim_i]
            sim_stress["pore_pressure"] = pp_samples[sim_i]
            sim_stress["max_stress_azimuth"] = az_samples[sim_i]

            if stress_model_type == "aphi_min":
                sim_stress["min_horizontal_stress"] = sh_samples[sim_i]
                sim_stress["aphi_value"] = aphi_samples[sim_i]
            else:
                sim_stress["aphi_value"] = aphi_samples[sim_i]
                sim_stress.pop("min_horizontal_stress", None)

            stress_state_obj, p0_abs = calculate_absolute_stresses(
                sim_stress, friction_coefficient, stress_model_type
            )

            sig, tau, *_ = calculate_fault_effective_stresses(
                strikes_mc[:, sim_i], dips_mc[:, sim_i],
                stress_state_obj, p0_abs, np.zeros(n_faults),
            )
            pps_to_slip[:, sim_i] = ComputeCriticalPorePressureForFailure(
                sig, tau, mu_mc[sim_i], p0_abs
            )

    # ---- Build result DataFrame ----
    fault_ids = faults_df["FaultID"].astype(str).values

    # pps_to_slip is (n_faults, n_sims) — flatten in fault-major order
    sim_ids = np.tile(np.arange(1, n_sims + 1), n_faults)
    fid_col = np.repeat(fault_ids, n_sims)
    results_df = pd.DataFrame({
        "SimulationID": sim_ids,
        "FaultID": fid_col,
        "SlipPressure": pps_to_slip.ravel(),
    })

    if not return_sample_inputs:
        return results_df

    # Sample inputs: per-sim stress values tiled across faults, per-fault geometry values repeated
    sample_data = {
        "SimulationID": sim_ids,
        "FaultID": fid_col,
        "vertical_stress_gradient": np.tile(sv_samples, n_faults),
        "initial_pore_pressure_gradient": np.tile(pp_samples, n_faults),
        "max_stress_azimuth": np.tile(az_samples, n_faults),
        "friction_coefficient": np.tile(mu_mc, n_faults),
        "strike_angle": strikes_mc.ravel(),
        "dip_angle": dips_mc.ravel(),
    }
    if sH_grad_samples is not None:
        sample_data["max_horizontal_stress_gradient"] = np.tile(sH_grad_samples, n_faults)
    if sh_samples is not None:
        sample_data["min_horizontal_stress_gradient"] = np.tile(sh_samples, n_faults)
    if aphi_samples is not None:
        sample_data["aphi_value"] = np.tile(aphi_samples, n_faults)

    return results_df, pd.DataFrame(sample_data)
