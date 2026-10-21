"""
Geomechanics input checks ported from legacy MATLAB FSP.

MATLAB refused these inputs with an error dialog (support code/checkdata.m and
data entry functions/dataentryProbabilistic.m) instead of running on them. The
messages keep MATLAB's wording so users see the same guidance.
"""
import math

GRADIENT_MODES = ("gradients", "all_gradients")

# Uncertainty key -> stress_inputs key for parameters that can never be negative
# in any Monte Carlo draw (MATLAB indices 1, 2, 3, 5 and 13). Friction (index 10)
# is checked separately against the global friction coefficient.
_NON_NEGATIVE_UNCERTAINTIES = {
    "vertical_stress_gradient_uncertainty": "vertical_stress",
    "min_horizontal_stress_uncertainty": "min_horizontal_stress",
    "max_horizontal_stress_uncertainty": "max_horizontal_stress",
    "initial_pore_pressure_gradient_uncertainty": "pore_pressure",
    "aphi_value_uncertainty": "aphi_value",
}


def _number(value):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    return float(value)


def validate_stress_inputs(stress_inputs: dict, friction_coefficient, stress_model_type: str) -> None:
    """Port of MATLAB checkdata.m stress and friction checks, in MATLAB's order."""
    sv = _number(stress_inputs.get("vertical_stress"))
    sh = _number(stress_inputs.get("min_horizontal_stress"))
    sH = _number(stress_inputs.get("max_horizontal_stress"))
    aphi = _number(stress_inputs.get("aphi_value"))
    azimuth = _number(stress_inputs.get("max_stress_azimuth"))
    depth = _number(stress_inputs.get("reference_depth"))
    pp = _number(stress_inputs.get("pore_pressure"))
    mu = _number(friction_coefficient)
    gradients = stress_model_type in GRADIENT_MODES

    if gradients and any(v is None or v <= 0 for v in (sv, sh, sH)):
        raise ValueError("Check stress gradients")
    if gradients and sh > sH:
        raise ValueError("Check horizontal stress gradients")
    if not gradients and (sv is None or sv <= 0):
        raise ValueError("Check vertical stress gradient")
    if not gradients and (aphi is None or aphi < 0 or aphi > 3):
        raise ValueError("Check A-Phi parameter value")
    if stress_model_type == "aphi_min" and (sh is None or sh <= 0):
        raise ValueError("Check minimum horizontal stress gradient")
    if azimuth is None or azimuth < 0 or azimuth >= 360:
        raise ValueError("Check horizontal stress direction in range [0,360)")
    if depth is None or depth <= 0:
        raise ValueError("Check reference depth")
    if pp is None or pp <= 0:
        raise ValueError("Check initial reservoir pressure gradient")
    if sv > 1.5:  # sV_grad > 1.5 psi/ft is not realistic
        raise ValueError("Check vertical stress gradient value")
    if mu is None or mu <= 0:
        raise ValueError("Check friction coefficient for faults <=0")


def validate_shmax_gradient(sH, reference_depth, stress_model_type: str) -> None:
    """MATLAB checkdata.m: SHmax gradient > 8 psi/ft is not realistic.

    In the A-Phi modes SHmax is calculated, so this runs after the stresses are built.
    """
    if float(sH) / float(reference_depth) <= 8:
        return
    if stress_model_type in GRADIENT_MODES:
        raise ValueError("Check horizontal stress gradient value(s)")
    if stress_model_type == "aphi_min":
        raise ValueError("Check minimum horizontal stress gradient value")
    raise ValueError(
        "Calculated horizontal stresses too high; check vertical stress gradient "
        "and friction coefficient values"
    )


def validate_geomechanics_uncertainties(
    stress_inputs: dict,
    friction_coefficient,
    uncertainties: dict,
    labels: dict,
) -> None:
    """Port of MATLAB dataentryProbabilistic.m plus/minus checks.

    Only keys present in ``uncertainties`` are checked, mirroring MATLAB's hidden
    (noshow) fields; ``None`` means the value was not entered. ``labels`` maps an
    uncertainty key to its user-facing name.
    """
    for key, raw in uncertainties.items():
        label = labels.get(key, key)
        value = _number(raw)
        if value is None:
            continue
        if math.isnan(value) or value < 0:
            raise ValueError(
                f"Check {label}: you entered a negative or non-number plus/minus value: {raw}. "
                "FSP adds and subtracts a positive number from the deterministic value "
                "to get the bounds of the uniform distribution."
            )

        if key == "friction_coefficient_uncertainty":
            base = _number(friction_coefficient)
        elif key in _NON_NEGATIVE_UNCERTAINTIES:
            base = _number(stress_inputs.get(_NON_NEGATIVE_UNCERTAINTIES[key]))
        else:
            base = None
        if base is not None and value >= base:
            raise ValueError(
                f"Check {label}: the plus/minus value {value:g} is not smaller than the "
                f"deterministic value {base:g}. This parameter can't be negative in any "
                "Monte Carlo draw."
            )

        if key == "aphi_value_uncertainty" and base is not None and base + value > 3:
            raise ValueError(
                f"Check {label}: the plus/minus value {value:g} added to A-Phi {base:g} "
                "is greater than 3, which is impossible (see Hurd and Zoback 2012)."
            )
