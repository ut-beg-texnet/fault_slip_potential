"""
Mohr diagram D3-compatible data formatters.
Port of FSP/graphs/julia_fsp_graphs.jl mohr_diagram_data_to_d3_portal and
mohr_diagram_hydro_data_to_d3_portal.
"""
import numpy as np
import pandas as pd
from ..models.stress import StressState


def _mohr_arcs(sh, sH, sV, p0, dp=0.0, biot=1.0, nu=0.5):
    """Generate Mohr circle arc data (three semicircles) as a DataFrame.

    Returns arcs_df with columns: id, x, y
    """
    Sig0sorted = np.sort([sh, sH, sV])[::-1]   # descending

    # index of vertical stress
    ixSv = np.argmin(np.abs(Sig0sorted - sV))

    Ds = np.full(3, biot * (1.0 - 2.0 * nu) / (1.0 - nu) * dp)
    Ds[ixSv] = 0.0
    Sig = Sig0sorted + Ds

    a = np.linspace(0, np.pi, 100)
    c = np.exp(1j * a)

    R1 = 0.5 * (Sig[0] - Sig[2])
    R2 = 0.5 * (Sig[1] - Sig[2])
    R3 = 0.5 * (Sig[0] - Sig[1])

    centre_pp = p0 + dp

    C1 = R1 * c + (Sig[0] + Sig[2]) / 2.0 - centre_pp
    C2 = R2 * c + (Sig[1] + Sig[2]) / 2.0 - centre_pp
    C3 = R3 * c + (Sig[0] + Sig[1]) / 2.0 - centre_pp

    rows = []
    for cid, circle in [("circle1", C1), ("circle2", C2), ("circle3", C3)]:
        for x, y in zip(circle.real, circle.imag):
            rows.append({"id": cid, "x": float(x), "y": float(y)})

    return pd.DataFrame(rows)


def _friction_line(arcs_df, mu):
    """Generate the Mohr-Coulomb failure envelope (frictional slip line)."""
    x_max = arcs_df["x"].max() if not arcs_df.empty else 5000.0
    x_range = np.linspace(0, x_max * 1.05, 100)
    rows = [{"id": "friction_line", "x": float(x), "y": float(mu * x)} for x in x_range]
    return pd.DataFrame(rows)


def mohr_diagram_data_to_d3(sh, sH, sV, tau_faults, sigma_faults,
                              p0, biot, nu, dp, strikes, mu,
                              stress_regime, slip_pressures, fault_ids):
    """Build D3 data for Mohr diagram (geomechanics step, scalar dp).

    Port of julia_fsp_graphs.jl mohr_diagram_data_to_d3_portal.

    Returns (arcs_df, slip_df, fault_df)
    """
    arcs_df = _mohr_arcs(sh, sH, sV, p0, dp, biot, nu)
    arcs_df = pd.concat([arcs_df, _friction_line(arcs_df, mu)], ignore_index=True)

    # Slip line: fault points on Mohr circle
    slip_rows = []
    for fid, sp in zip(fault_ids, slip_pressures):
        slip_rows.append({"id": str(fid), "slip_pressure": float(sp)})
    slip_df = pd.DataFrame(slip_rows)

    # Fault points: sigma_effective, tau_effective
    fault_rows = []
    for fid, sigma, tau, sp in zip(fault_ids, sigma_faults, tau_faults, slip_pressures):
        fault_rows.append({
            "id": str(fid),
            "x": float(sigma),
            "y": float(tau),
            "slip_pressure": float(sp),
        })
    fault_df = pd.DataFrame(fault_rows)

    return arcs_df, slip_df, fault_df


def mohr_diagram_data_to_d3_portal(sh, sH, sV, tau_faults, sigma_faults,
                                    p0, biot, nu, dp, strikes, mu,
                                    stress_regime, slip_pressures, fault_ids):
    """Alias used by step 2 — same as mohr_diagram_data_to_d3."""
    return mohr_diagram_data_to_d3(sh, sH, sV, tau_faults, sigma_faults,
                                    p0, biot, nu, dp, strikes, mu,
                                    stress_regime, slip_pressures, fault_ids)


def mohr_diagram_hydro_data_to_d3(sh, sH, sV, tau_faults, sigma_faults,
                                   p0, dp_array, strikes, mu, fault_ids,
                                   slip_pressures=None):
    """Build D3 Mohr data for hydrology step using per-fault pressure shifts."""
    if slip_pressures is None:
        slip_pressures = dp_array

    arc_frames = []
    for fid, dp_val in zip(fault_ids, dp_array):
        # Each fault has its own pressure-shifted Mohr circles.
        fault_arcs = _mohr_arcs(sh, sH, sV, p0, float(dp_val))
        fault_arcs["fault_id"] = str(fid)
        arc_frames.append(fault_arcs)
    arcs_df = pd.concat(arc_frames, ignore_index=True) if arc_frames else pd.DataFrame(columns=["id", "x", "y", "fault_id"])
    arcs_df = pd.concat([arcs_df, _friction_line(arcs_df, mu)], ignore_index=True)

    slip_rows = []
    for fid, dp_val, sp in zip(fault_ids, dp_array, slip_pressures):
        slip_rows.append({"id": str(fid), "dp": float(dp_val), "slip_pressure": float(sp)})
    slip_df = pd.DataFrame(slip_rows)

    fault_rows = []
    for fid, sigma, tau, dp_val, sp in zip(fault_ids, sigma_faults, tau_faults, dp_array, slip_pressures):
        fault_rows.append({
            "id": str(fid),
            "x": float(sigma),
            "y": float(tau),
            "dp": float(dp_val),
            "slip_pressure": float(sp),
        })
    fault_df = pd.DataFrame(fault_rows)

    return arcs_df, slip_df, fault_df


def mohr_diagram_hydro_data_to_d3_portal(sh, sH, sV, tau_faults, sigma_faults,
                                          p0, dp_array, strikes, mu, fault_ids,
                                          slip_pressures=None):
    return mohr_diagram_hydro_data_to_d3(sh, sH, sV, tau_faults, sigma_faults,
                                          p0, dp_array, strikes, mu, fault_ids,
                                          slip_pressures)
