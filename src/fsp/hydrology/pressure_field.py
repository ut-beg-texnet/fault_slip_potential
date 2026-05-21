"""
Grid and fault pressure field calculations.
Port of FSP/core/hydrology_calculations.jl pfieldcalc_all_rates (grid and scalar variants).
"""
import numpy as np
from .theis import pressureScenario_Rall, pressureScenario_Rall_scalar
from ..io.coords import haversine_distance


def pressure_from_distances_m(r_meters, STRho, days, bpds, evaluation_days=None):
    """Pressure for one well at many precomputed radial distances."""
    r_meters = np.asarray(r_meters, dtype=float)
    if r_meters.size == 0:
        return np.zeros(r_meters.shape, dtype=float)
    pfront = pressureScenario_Rall(bpds, days, r_meters, STRho, evaluation_days)
    return np.maximum(pfront, 0.0)


def well_fault_distances_m(well_data_list, fault_lats, fault_lons):
    """Precompute well-to-fault distances in metres.

    Returns a matrix shaped (n_wells, n_faults). Rows correspond to
    well_data_list order, so callers can reuse the matrix across years and
    Monte Carlo samples.
    """
    fault_lats = np.asarray(fault_lats, dtype=float)[np.newaxis, :]   # (1, n_faults)
    fault_lons = np.asarray(fault_lons, dtype=float)[np.newaxis, :]
    well_lats = np.array([wd.latitude for wd in well_data_list])[:, np.newaxis]  # (n_wells, 1)
    well_lons = np.array([wd.longitude for wd in well_data_list])[:, np.newaxis]
    return haversine_distance(fault_lats, fault_lons, well_lats, well_lons) * 1000.0


def pfieldcalc_all_rates(lat_grid, lon_grid, STRho, days, bpds,
                          well_lon, well_lat, evaluation_days=None):
    """Pressure field on a 2-D lat/lon grid for a single well.

    NOTE: matches Julia's 'flipped meshgrid' convention where lat_grid is the
    first argument (xGrid) and lon_grid is the second (yGrid).

    Parameters
    ----------
    lat_grid, lon_grid : 2-D numpy arrays (same shape, from np.meshgrid)
    days, bpds : 1-D arrays — injection time series
    well_lon, well_lat : well coordinates
    evaluation_days : float or None

    Returns
    -------
    2-D numpy array of pressure change in PSI (same shape as lat_grid)
    """
    # R_km[i,j] = haversine(lat_grid[i,j], lon_grid[i,j], well_lat, well_lon)
    # Vectorised haversine over the grid
    R_km = haversine_distance(lat_grid, lon_grid, well_lat, well_lon)
    R_meters = R_km * 1000.0

    R_flat = R_meters.ravel()
    pfront_flat = pressure_from_distances_m(
        R_flat, STRho, days, bpds, evaluation_days
    )
    return pfront_flat.reshape(lat_grid.shape)


def pfieldcalc_all_rates_for_distances(distance_meters, STRho, days, bpds,
                                       evaluation_days=None):
    """Pressure for a single well using precomputed distances."""
    return pressure_from_distances_m(
        distance_meters, STRho, days, bpds, evaluation_days
    )


def pfieldcalc_all_rates_scalar(fault_lon, fault_lat, STRho, days, bpds,
                                  well_lon, well_lat, evaluation_days=None):
    """Pressure at a single fault centroid from one well.

    Port of Julia pfieldcalc_all_rates scalar dispatch.
    """
    R_km = haversine_distance(float(fault_lat), float(fault_lon),
                               float(well_lat), float(well_lon))
    R_m = float(R_km) * 1000.0
    return pressureScenario_Rall_scalar(bpds, days, R_m, STRho, evaluation_days)


def compute_fault_pressures(well_data_list, STRho, fault_lats, fault_lons,
                             evaluation_days, distance_matrix_m=None):
    """Sum pressure contributions from all wells at each fault centroid.

    Parameters
    ----------
    well_data_list : list of WellData
    STRho : (S, T, rho)
    fault_lats, fault_lons : arrays of fault centroid coordinates
    evaluation_days : float — days from first injection to evaluate at

    Returns
    -------
    numpy array of total pressure (PSI) per fault
    """
    fault_lats = np.asarray(fault_lats, dtype=float)
    fault_lons = np.asarray(fault_lons, dtype=float)
    total = np.zeros(len(fault_lats), dtype=float)
    if distance_matrix_m is None:
        distance_matrix_m = well_fault_distances_m(
            well_data_list, fault_lats, fault_lons
        )

    for wi, wd in enumerate(well_data_list):
        if len(wd.days) == 0:
            continue
        total += pfieldcalc_all_rates_for_distances(
            distance_matrix_m[wi], STRho, wd.days, wd.rates, evaluation_days
        )

    return np.maximum(total, 0.0)
