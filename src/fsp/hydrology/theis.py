"""
Theis equation kernels for radial pressure front calculation.
Port of FSP/core/bill_pfront.jl.

Julia expint(u) = E₁(u) → Python scipy.special.e1(u).
The numba-JIT scalar kernel is used in the Monte Carlo inner loop.
"""
import numpy as np
from scipy.special import exp1 as scipy_e1

try:
    from numba import njit as _njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def _njit(fn):
        return fn


_BBL_PER_DAY_TO_M3_PER_S = 1.84013e-6
_G = 9.81
_PSI_CONVERT = _G / 6894.76   # rho*g/6894.76 becomes rho * this


def pressureScenario_constant_rate(bpds, days, r_meters, STRho):
    """Constant-rate Theis: pressure change (PSI) at each radius in r_meters.

    Port of Julia pressureScenario_constant_rate.
    """
    S, T, rho = STRho
    bpds = np.asarray(bpds, dtype=float)
    days = np.asarray(days, dtype=float)
    r_meters = np.asarray(r_meters, dtype=float)

    if len(bpds) == 0:
        return np.zeros(len(r_meters))

    t_final_sec = float(np.max(days)) * 86400.0
    if t_final_sec <= 0:
        return np.zeros(len(r_meters))

    Q_bpd = float(bpds[-1])
    Q_m3s = Q_bpd * _BBL_PER_DAY_TO_M3_PER_S

    ppp = (r_meters ** 2 * S) / (4.0 * T * t_final_sec)
    head = (Q_m3s / (4.0 * np.pi * T)) * scipy_e1(ppp)

    dp_psi = head * (rho * _G / 6894.76)
    dp_psi = np.where(np.isfinite(dp_psi), dp_psi, 0.0)
    return dp_psi


def pressureScenario_Rall(bpds, days, r_meters, STRho, evaluation_days=None):
    """Variable-rate step-superposition Theis (vectorised over r_meters).

    Port of Julia pressureScenario_Rall.

    Parameters
    ----------
    evaluation_days : float or None
        Days from injection start to evaluate at.  None → use max(days).
    """
    bpds = np.asarray(bpds, dtype=float)
    days = np.asarray(days, dtype=float)
    r_meters = np.asarray(r_meters, dtype=float)

    if len(bpds) == 0 or len(days) == 0:
        return np.zeros(r_meters.shape)

    S, T, rho = STRho

    t_final_sec = (float(np.max(days)) if evaluation_days is None else float(evaluation_days)) * 86400.0

    Q_m3s = bpds * _BBL_PER_DAY_TO_M3_PER_S
    n = len(Q_m3s)

    # Build ΔQ and step times
    dQ = np.empty(n, dtype=float)
    dQ[0] = Q_m3s[0]
    dQ[1:] = Q_m3s[1:] - Q_m3s[:-1]

    r2 = r_meters.ravel() ** 2
    inv_4T = 1.0 / (4.0 * T)

    # Pre-filter to valid steps (non-zero rate change, positive elapsed time)
    dt_all = t_final_sec - days * 86400.0
    valid = (dQ != 0.0) & (dt_all > 0.0)

    if not np.any(valid):
        dp_psi = np.zeros(len(r2), dtype=float)
    else:
        dQ_v = dQ[valid]
        coeffs = S * inv_4T / dt_all[valid]              # (n_valid,)
        u_2d = r2[:, np.newaxis] * coeffs[np.newaxis, :] # (n_radii, n_valid)
        wf_2d = scipy_e1(u_2d)                           # single scipy call
        tstep_sum = wf_2d @ dQ_v                          # matrix-vector product → (n_radii,)

        inv_4piT = 1.0 / (4.0 * np.pi * T)
        dp_psi = tstep_sum * inv_4piT * (rho * _G / 6894.76)

    dp_psi = np.where(np.isfinite(dp_psi) & (dp_psi > 0.0), dp_psi, 0.0)

    return dp_psi.reshape(r_meters.shape)


if _NUMBA_AVAILABLE:
    from numba import njit as _njit_real
    import math as _math

    @_njit_real
    def _exp1_jit(x: float) -> float:
        """E₁(x) using Abramowitz & Stegun approximations.

        For 0 < x ≤ 1: A&S 5.1.53 (polynomial + log)
        For x > 1: A&S 5.1.56 (rational approximation)
        """
        if x <= 0.0:
            return 0.0

        if x <= 1.0:
            # A&S 5.1.53: E1(x) = -ln(x) + a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
            a0 = -0.57721566
            a1 =  0.99999193
            a2 = -0.24991055
            a3 =  0.05519968
            a4 = -0.00976004
            a5 =  0.00107857
            poly = a0 + x * (a1 + x * (a2 + x * (a3 + x * (a4 + x * a5))))
            return -_math.log(x) + poly

        else:
            # A&S 5.1.56: E1(x) = exp(-x)/x * (x^4+a1*x^3+a2*x^2+a3*x+a4) /
            #                                  (x^4+b1*x^3+b2*x^2+b3*x+b4)
            a1 = 8.5733287401
            a2 = 18.0590169700
            a3 = 8.6347608926
            a4 = 0.2677737343
            b1 = 9.5733223454
            b2 = 25.6329560884
            b3 = 21.0996530827
            b4 = 3.9584969228
            x2 = x * x
            x3 = x2 * x
            x4 = x3 * x
            num = x4 + a1 * x3 + a2 * x2 + a3 * x + a4
            den = x4 + b1 * x3 + b2 * x2 + b3 * x + b4
            return _math.exp(-x) * (num / den) / x

    @_njit_real
    def _theis_scalar_jit(bpds, days, r_m, S, T, rho, t_final_sec):
        """Numba JIT inner loop — scalar Theis superposition."""
        BBL = 1.84013e-6
        G = 9.81
        inv_4T = 1.0 / (4.0 * T)
        r2 = r_m * r_m
        total = 0.0
        n = len(bpds)
        prev_Q = 0.0
        for i in range(n):
            cur_Q = bpds[i] * BBL
            dQ = cur_Q - prev_Q
            prev_Q = cur_Q
            if dQ == 0.0:
                continue
            t_i_sec = days[i] * 86400.0
            dt = t_final_sec - t_i_sec
            if dt <= 0.0:
                continue
            u = r2 * S * inv_4T / dt
            if u <= 0.0:
                continue
            wf = _exp1_jit(u)
            total += wf * dQ

        inv_4piT = 1.0 / (4.0 * _math.pi * T)
        dp = total * inv_4piT * (rho * G / 6894.76)
        return dp if (dp > 0.0 and _math.isfinite(dp)) else 0.0

    def pressureScenario_Rall_scalar(bpds, days, r_m, STRho, evaluation_days=None):
        """Scalar Theis using numba JIT kernel (fast for MC loops)."""
        S, T, rho = STRho
        bpds = np.asarray(bpds, dtype=float)
        days = np.asarray(days, dtype=float)
        r_m = float(r_m)
        if len(bpds) == 0 or len(days) == 0:
            return 0.0
        t_final_sec = (float(np.max(days)) if evaluation_days is None else float(evaluation_days)) * 86400.0
        return float(_theis_scalar_jit(bpds, days, r_m, S, T, rho, t_final_sec))

else:
    def pressureScenario_Rall_scalar(bpds, days, r_m, STRho, evaluation_days=None):
        """Scalar Theis (pure Python fallback when numba not available)."""
        result = pressureScenario_Rall(
            bpds, days, np.array([float(r_m)]), STRho, evaluation_days
        )
        return float(result[0])
