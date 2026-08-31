"""Legacy-compatible helpers for probabilistic fault slip potential."""

from datetime import date

import numpy as np

# MATLAB UI displayed FSP with two decimal places (%0.2f).
_FSP_DISPLAY_DECIMALS = 2


def normalize_year_of_interest(
    requested_year: int,
    injection_start_date: date,
    injection_end_date: date,
    diffusion_years: int = 3,
) -> tuple[int, int, int]:
    """Clamp a selected year to the injection and diffusion model window."""
    start_year = injection_start_date.year
    end_year = injection_end_date.year + diffusion_years
    effective_year = min(max(int(requested_year), start_year), end_year)
    return effective_year, start_year, end_year


def selected_year_message(requested_year: int, effective_year: int, start_year: int, end_year: int) -> str | None:
    """Describe a selected-year adjustment for the portal, when one occurred."""
    if requested_year == effective_year:
        return None
    return (
        f"Year of interest {requested_year} is outside the supported range "
        f"{start_year}–{end_year}; using {effective_year}."
    )


def _matlab_ecdf_lookup(geomechanics_pressures) -> tuple[np.ndarray, np.ndarray]:
    """Build the CDF points retained by legacy MATLAB's calcFaultFSP."""
    samples = np.asarray(geomechanics_pressures, dtype=float)
    samples = np.sort(samples[np.isfinite(samples)])
    if len(samples) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x, counts = np.unique(samples, return_counts=True)
    cdf = np.cumsum(counts, dtype=float) / float(len(samples))

    # MATLAB ecdf duplicates the minimum point with an initial zero probability.
    x = np.concatenate(([x[0]], x))
    cdf = np.concatenate(([0.0], cdf))
    x = np.maximum(x, 0.0)

    # MATLAB unique keeps the first value after negative pressures are clipped.
    _, first_indices = np.unique(x, return_index=True)
    first_indices.sort()
    return x[first_indices], cdf[first_indices]


def _matlab_hydro_ecdf_queries(hydrology_pressures) -> np.ndarray:
    """Hydro query points used by MATLAB calcFaultFSP.

    MATLAB plots the hydrology exceedance curve with ecdf unique x-values,
    duplicating the minimum (the leading f=0 point). calcFaultFSP then
    averages interp1 lookups at those x-values, not at every MC draw.

    hydro1D replaces Inf/NaN with 0 psi before that CDF is built.
    Negative hydrology pressures are not clipped.
    """
    hydro = np.asarray(hydrology_pressures, dtype=float).reshape(-1)
    if hydro.size == 0:
        return np.array([], dtype=float)

    hydro = np.where(np.isfinite(hydro), hydro, 0.0)
    unique_x = np.unique(hydro)
    return np.concatenate(([unique_x[0]], unique_x))


def _matlab_nearest_with_extrapolation(x, y, query) -> np.ndarray:
    """Evaluate MATLAB interp1(..., 'nearest', 'extrap') for sorted x values.

    A single retained geomechanics CDF point is the deterministic-geo branch
    in calcFaultFSP: FSP is 0 if the query is strictly below that pressure,
    otherwise 1.
    """
    query = np.asarray(query, dtype=float)
    if len(x) == 0:
        return np.zeros(len(query), dtype=float)
    if len(x) == 1:
        return np.where(query < x[0], 0.0, 1.0)

    right = np.searchsorted(x, query, side="left")
    right = np.clip(right, 1, len(x) - 1)
    left = right - 1
    # At an exact midpoint, MATLAB nearest selects the higher x value.
    use_right = (query - x[left]) >= (x[right] - query)
    indices = np.where(use_right, right, left)
    return y[indices]


def legacy_fsp_probabilities(geomechanics_pressures, hydrology_pressures) -> np.ndarray:
    """Evaluate MATLAB calcFaultFSP lookups at hydrology ecdf unique x-values."""
    hydro_queries = _matlab_hydro_ecdf_queries(hydrology_pressures)
    if len(hydro_queries) == 0:
        return np.array([], dtype=float)

    x, cdf = _matlab_ecdf_lookup(geomechanics_pressures)
    return _matlab_nearest_with_extrapolation(x, cdf, hydro_queries)


def legacy_fsp(geomechanics_pressures, hydrology_pressures) -> float:
    """Average the legacy lookup probabilities for one fault."""
    probabilities = legacy_fsp_probabilities(geomechanics_pressures, hydrology_pressures)
    return float(probabilities.mean()) if len(probabilities) else 0.0


def displayed_legacy_fsp(geomechanics_pressures, hydrology_pressures) -> float:
    """MATLAB-style FSP rounded the way the MATLAB UI displayed it."""
    return round(legacy_fsp(geomechanics_pressures, hydrology_pressures), _FSP_DISPLAY_DECIMALS)
