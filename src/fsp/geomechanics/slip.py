"""
Fault slip calculations: effective stresses, slip pressure, CFF, SCU.
Direct port of FSP/core/geomechanics_model.jl.

All functions accept numpy scalars or arrays — vectorised over all faults in one call.
"""
import numpy as np
from ..models.stress import StressState

DEG2RAD = np.pi / 180.0


def calculate_fault_effective_stresses(strike, dip, stress_state: StressState, p0, dp):
    """Calculate normal and shear stresses on fault plane.

    Exact port of MATLAB mohrs_3D.m / Julia calculate_fault_effective_stresses.
    Accepts scalars or numpy arrays for strike, dip, p0, dp.

    Returns
    -------
    (sig_normal, tau_normal, s11, s22, s33, s12, n1, n2)
    """
    az = stress_state.sH_azimuth
    Svert = stress_state.principal_stresses[0]
    shmin = stress_state.principal_stresses[1]
    sHmax = stress_state.principal_stresses[2]

    az_rad = az * DEG2RAD
    str_rad = np.asarray(strike, dtype=float) * DEG2RAD
    dip_rad = np.asarray(dip, dtype=float) * DEG2RAD

    cos_az = np.cos(az_rad)
    sin_az = np.sin(az_rad)
    cs = np.cos(str_rad)
    ss = np.sin(str_rad)
    sd = np.sin(dip_rad)

    # Biot=1, nu=0.5 → f = nu/(1-nu) = 1.0
    f = 1.0

    dp = np.asarray(dp, dtype=float)
    p0 = np.asarray(p0, dtype=float)

    s11 = shmin * cos_az**2 + sHmax * sin_az**2 - p0 - f * dp
    s22 = shmin * sin_az**2 + sHmax * cos_az**2 - p0 - f * dp
    s33 = Svert - p0 - dp
    s12 = (sHmax - shmin) * cos_az * sin_az

    n1 = sd * cs
    n2 = -sd * ss

    n1_sq = n1 ** 2
    n2_sq = n2 ** 2
    s11_s33 = s11 - s33
    s22_s33 = s22 - s33
    n1_n2 = n1 * n2
    n1_cubed = n1 ** 3

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

    sqrt_arg = np.maximum(sqrt_arg, 0.0)
    tau_normal = np.sqrt(sqrt_arg)
    sig_normal = 2.0 * n1_n2 * s12 + n1_sq * s11_s33 + n2_sq * s22_s33 + s33

    sig_normal = np.maximum(sig_normal, 0.0)
    tau_normal = np.maximum(tau_normal, 0.0)

    return sig_normal, tau_normal, s11, s22, s33, s12, n1, n2


def calculate_slip_pressure(sig_fault, tau_fault, mu, p0, biot=1.0, nu=0.5, dp=0.0,
                             s11=0.0, s22=0.0, s33=0.0, s12=0.0, n1=0.0, n2=0.0):
    """Calculate pore pressure increment required to bring fault to Mohr-Coulomb failure.

    Solves A·Δp²+B·Δp+C = 0 (vectorised).
    Returns the physically meaningful root.
    Port of Julia calculate_slip_pressure in geomechanics_model.jl.
    """
    sig = np.asarray(sig_fault, dtype=float)
    tau = np.asarray(tau_fault, dtype=float)
    mu = float(mu)

    mobmu = np.where(sig > 0.0, tau / np.maximum(sig, 1e-30), mu)

    f = float(biot) * float(nu) / (1.0 - float(nu))

    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    s11 = np.asarray(s11, dtype=float)
    s22 = np.asarray(s22, dtype=float)
    s33 = np.asarray(s33, dtype=float)
    s12 = np.asarray(s12, dtype=float)

    n1_sq = n1 ** 2
    n2_sq = n2 ** 2
    n1_n2 = n1 * n2
    n1_cubed = n1 ** 3
    s11_s33 = s11 - s33
    s22_s33 = s22 - s33

    C = (
        -4.0 * (1.0 + mu**2) * n1_cubed * n2 * s12 * s11_s33
        - (1.0 + mu**2) * n1_sq**2 * s11_s33**2
        - (1.0 + mu**2) * n2_sq**2 * s22_s33**2
        - mu**2 * s33**2
        + 2.0 * n1_n2 * s12 * (s11 + (1.0 - 2.0*(1.0+mu**2)*n2_sq)*s22 + 2.0*(1.0+mu**2)*(-1.0+n2_sq)*s33)
        + n2_sq * (s12**2 + s22**2 - 2.0*(1.0+mu**2)*s22*s33 + (1.0+2.0*mu**2)*s33**2)
        + n1_sq * (
            s11**2
            + (1.0 - 4.0*(1.0+mu**2)*n2_sq) * s12**2
            - 2.0*(1.0+mu**2)*s11*(n2_sq*s22_s33 + s33)
            + s33*(2.0*(1.0+mu**2)*n2_sq*s22_s33 + s33 + 2.0*mu**2*s33)
        )
    )

    B = 2.0 * (
        2.0*(-1.0+f)*(1.0+mu**2)*n1_cubed*n2*s12
        + 2.0*n1_n2*(-(1.0+mu**2)*(-1.0+n2_sq) + f*(-1.0+(1.0+mu**2)*n2_sq))*s12
        + (-1.0+f)*(1.0+mu**2)*n1_sq**2*s11_s33
        + (-1.0+f)*(1.0+mu**2)*n2_sq**2*s22_s33
        + mu**2 * s33
        + n2_sq*((1.0-f+mu**2)*s22 + (-1.0+f-2.0*mu**2+f*mu**2)*s33)
        + n1_sq*(
            (-(1.0+mu**2)*(-1.0+n2_sq) + f*(-1.0+(1.0+mu**2)*n2_sq))*s11
            + (-1.0+f)*(1.0+mu**2)*n2_sq*(s22-2.0*s33)
            + (-1.0+f-2.0*mu**2+f*mu**2)*s33
        )
    )

    A = (
        -mu**2 * (1.0 + (-1.0+f)*n1_sq + (-1.0+f)*n2_sq)**2
        - (-1.0+f)**2 * (n1_sq**2 + n2_sq*(-1.0+n2_sq) + n1_sq*(-1.0+2.0*n2_sq))
    )

    disc = B**2 - 4.0*A*C

    horiz_dist = sig - tau / mu

    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    ppfail1 = np.where(disc >= 0, (-B - sqrt_disc) / (2.0*A), -horiz_dist)
    ppfail2 = np.where(disc >= 0, (-B + sqrt_disc) / (2.0*A), horiz_dist)

    # Root selection — mirrors Julia logic
    both_pos = (ppfail1 > 0) & (ppfail2 > 0)
    both_neg = (ppfail1 < 0) & (ppfail2 < 0)
    mix = ~both_pos & ~both_neg

    # below failure: want smallest positive root
    below = np.where(both_pos, np.minimum(ppfail1, ppfail2),
             np.where(mix, np.where(ppfail1 > 0, ppfail1, ppfail2),
             horiz_dist))

    # above failure: want least-magnitude negative root
    above = np.where(both_neg,
             np.where(np.abs(ppfail1) < np.abs(ppfail2), ppfail1, ppfail2),
             np.where(mix, np.where(ppfail1 < 0, ppfail1, ppfail2),
             horiz_dist))

    ppfail = np.where(mobmu < mu, below,
             np.where(mobmu > mu, above, 0.0))

    return ppfail


def ComputeCriticalPorePressureForFailure(sig_fault, tau_fault, mu, p0=0.0,
                                          biot=1.0, nu=0.5, dp=1.0):
    """Simplified Josimar formula for critical pore pressure increment.

    Used in probabilistic geomechanics MC (dp=0, scalar path).
    """
    sig = np.asarray(sig_fault, dtype=float)
    tau = np.asarray(tau_fault, dtype=float)
    mu = float(mu)

    mobmu = np.where(sig > 0.0, np.abs(tau / np.maximum(sig, 1e-30)), mu)
    return np.where(mobmu < mu, ((mu - mobmu) / mu) * np.abs(sig), 0.0)


def calculate_slip_tendency(sig_fault, tau_fault):
    sig = np.asarray(sig_fault, dtype=float)
    tau = np.asarray(tau_fault, dtype=float)
    return np.where(sig <= 0.0, 1.0, tau / np.maximum(sig, 1e-30))


def calculate_scu(sig_fault, tau_fault, mu):
    sig = np.asarray(sig_fault, dtype=float)
    tau = np.asarray(tau_fault, dtype=float)
    scu = tau / (float(mu) * np.maximum(sig, 1e-30))
    return np.minimum(np.where(np.isfinite(scu), scu, 1.0), 1.0)


def calculate_cff(sig_fault, tau_fault, mu):
    return np.asarray(tau_fault, dtype=float) - float(mu) * np.asarray(sig_fault, dtype=float)


def analyze_fault(strike: float, dip: float, friction: float,
                  stress_state: StressState, p0: float, dp: float = 0.0) -> dict:
    """Deterministic single-fault analysis. Returns dict of stability metrics."""
    sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(
        strike, dip, stress_state, p0, dp
    )
    slip_p = float(np.maximum(
        calculate_slip_pressure(sig, tau, friction, p0, 1.0, 0.5, dp,
                                s11, s22, s33, s12, n1, n2),
        0.0
    ))
    return {
        "normal_stress": float(sig),
        "shear_stress": float(tau),
        "slip_pressure": slip_p,
        "slip_tendency": round(float(calculate_slip_tendency(sig, tau)), 4),
        "coulomb_failure_function": round(float(calculate_cff(sig, tau, friction))),
        "shear_capacity_utilization": round(float(calculate_scu(sig, tau, friction)), 4),
    }


def analyze_fault_hydro(strike: float, dip: float, friction: float,
                        stress_state: StressState, p0: float, dp: float) -> dict:
    """Fault analysis including hydrology pressure perturbation dp."""
    sig, tau, s11, s22, s33, s12, n1, n2 = calculate_fault_effective_stresses(
        strike, dip, stress_state, p0, dp
    )
    slip_p = float(
        calculate_slip_pressure(sig, tau, friction, p0, 1.0, 0.5, dp,
                                s11, s22, s33, s12, n1, n2)
    )
    return {
        "normal_stress": float(sig),
        "shear_stress": float(tau),
        "slip_pressure": slip_p,
    }
