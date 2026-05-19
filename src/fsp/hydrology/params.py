"""
Aquifer parameter calculations: storativity, transmissivity.
Port of FSP/core/hydrology_calculations.jl calcST.
"""

_MD_TO_M2 = 1e-3 * 9.9e-13  # 1 mD → m²
_FT_TO_M = 0.3048


def calcST(h_feet, porosity, kap_md, rho, mu, g, beta, alphav):
    """Calculate storativity S, transmissivity T, and fluid density rho.

    Parameters match Julia calcST exactly.

    Returns
    -------
    (S, T, rho)
    """
    h_m = h_feet * _FT_TO_M
    kap_m2 = kap_md * _MD_TO_M2

    S = rho * g * h_m * (alphav + porosity * beta)
    K = kap_m2 * rho * g / mu
    T = K * h_m

    return S, T, rho
