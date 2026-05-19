"""
Stress model calculations: gradients, aphi_min, aphi_no_min.
Direct port of FSP/core/geomechanics_model.jl (calculate_absolute_stresses and helpers).
"""
import numpy as np
from ..models.stress import StressState


def calculate_n_phi(aphi: float):
    """Convert A-Phi value (0-3) to (n, phi) pair.

    n: faulting regime index (0=normal, 1=strike-slip, 2=reverse)
    phi: normalised position within regime
    """
    if 0 <= aphi < 1:
        n = 0
    elif 1 <= aphi < 2:
        n = 1
    elif 2 <= aphi <= 3:
        n = 2
    else:
        raise ValueError(f"APhi value must be in range [0,3]. Got: {aphi}")

    phi = (aphi - (n + 0.5)) / ((-1) ** n) + 0.5
    return n, phi


def calculate_modified_aphi_stresses(n: int, phi: float, sv: float, sh: float, p0: float):
    """Modified A-Phi model: derive SHmax given Shmin (sh).

    Matches MATLAB getHorFromAPhi.m.
    Returns (sH, sh).
    """
    sv_eff = sv - p0
    sh_eff = sh - p0

    if n == 0:
        sH = phi * (sv_eff - sh_eff) + sh_eff + p0
    elif n == 1:
        sH = (sv_eff - sh_eff + phi * sh_eff) / phi + p0
    elif n == 2:
        sH = (sh_eff - sv_eff + phi * sv_eff) / phi + p0
    else:
        raise ValueError(f"Invalid n value for Modified A-Phi model: {n}")

    return sH, sh


def calculate_standard_aphi_stresses(n: int, phi: float, sv: float, p0: float, mu: float):
    """Standard A-Phi model with friction: derive both horizontal stresses.

    Matches MATLAB getHorFromAPhi.m.
    Returns (sH, sh).
    """
    if mu <= 0:
        return sv, sv

    k = (mu + np.sqrt(1 + mu ** 2)) ** 2

    if n == 0:
        sh = (sv - p0) / k + p0
        sH = phi * (sv - sh) + sh
    elif n == 1:
        A = np.array([[1.0, -k], [phi, (1 - phi)]])
        b = np.array([p0 - k * p0, sv])
        x = np.linalg.solve(A, b)
        sH, sh = x[0], x[1]
    elif n == 2:
        sH = k * (sv - p0) + p0
        sh = phi * (sH - sv) + sv
    else:
        raise ValueError(f"Invalid n value: {n}")

    return sH, sh


def calculate_absolute_stresses(stress_data: dict, friction_coefficient: float, stress_model_type: str):
    """Calculate absolute principal stresses at reference depth.

    Parameters
    ----------
    stress_data : dict
        Keys: reference_depth, vertical_stress, pore_pressure, max_stress_azimuth,
              min_horizontal_stress (optional), max_horizontal_stress (optional),
              aphi_value (optional)
    friction_coefficient : float
    stress_model_type : str
        One of: 'gradients', 'all_gradients', 'aphi_min', 'aphi_no_min', 'aphi_model'

    Returns
    -------
    (StressState, p0)
    """
    reference_depth = stress_data["reference_depth"]
    vertical_gradient = stress_data["vertical_stress"]
    pore_pressure_gradient = stress_data["pore_pressure"]
    max_stress_azimuth = stress_data["max_stress_azimuth"]

    sh_grad = stress_data.get("min_horizontal_stress")
    sH_grad = stress_data.get("max_horizontal_stress")
    aphi = stress_data.get("aphi_value")

    sV = round(vertical_gradient * reference_depth, 4)
    p0 = round(pore_pressure_gradient * reference_depth, 4)
    mu = friction_coefficient

    if stress_model_type in ("gradients", "all_gradients"):
        sH = round(sH_grad * reference_depth, 2)
        sh = round(sh_grad * reference_depth, 2)

    elif stress_model_type in ("aphi_model", "aphi_no_min", "aphi_min"):
        n, phi = calculate_n_phi(float(aphi))

        if sh_grad is not None:
            sh = sh_grad * reference_depth
            sH, _ = calculate_modified_aphi_stresses(n, phi, sV, sh, p0)
        else:
            sH, sh = calculate_standard_aphi_stresses(n, phi, sV, p0, mu)
    else:
        raise ValueError(f"Invalid stress model type: {stress_model_type}")

    return StressState(np.array([sV, sh, sH], dtype=float), float(max_stress_azimuth)), p0
