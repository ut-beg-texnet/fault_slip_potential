"""
Unit tests for hydrology calculations.
"""
import sys
import os
from datetime import date
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from fsp.hydrology.params import calcST
from fsp.hydrology.theis import pressureScenario_Rall, pressureScenario_Rall_scalar
from fsp.hydrology.pressure_field import (
    pfieldcalc_all_rates_for_distances,
    pfieldcalc_all_rates_scalar,
    well_fault_distances_m,
)
from fsp.io.coords import create_projected_spatial_grid
from fsp.models.hydrology import HydrologyParams
from fsp.monte_carlo.hydrology_mc import run_hydrology_mc_time_series


class TestCalcST:
    def test_positive_outputs(self):
        S, T, rho = calcST(
            h_feet=300.0,
            porosity=0.1,
            kap_md=50.0,
            rho=1000.0,
            mu=1e-3,
            g=9.81,
            beta=4.4e-10,
            alphav=1e-10,
        )
        assert S > 0
        assert T > 0
        assert rho == 1000.0

    def test_unit_conversion(self):
        # h_feet=1 → h_m=0.3048
        S, T, _ = calcST(1.0, 0.1, 1.0, 1000.0, 1e-3, 9.81, 4.4e-10, 1e-10)
        # Verify S depends on h in metres
        S2, T2, _ = calcST(2.0, 0.1, 1.0, 1000.0, 1e-3, 9.81, 4.4e-10, 1e-10)
        assert abs(S2 / S - 2.0) < 1e-6
        assert abs(T2 / T - 2.0) < 1e-6


class TestPressureScenarioRall:
    def setup_method(self):
        self.STRho = (5e-5, 4.5e-6, 1000.0)
        self.bpds = np.array([10000.0, 10000.0])
        self.days = np.array([1.0, 3650.0])

    def test_pressure_positive(self):
        r = np.array([100.0, 500.0, 1000.0, 5000.0])
        dp = pressureScenario_Rall(self.bpds, self.days, r, self.STRho)
        assert np.all(dp >= 0.0)

    def test_pressure_decreases_with_distance(self):
        r = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
        dp = pressureScenario_Rall(self.bpds, self.days, r, self.STRho)
        assert np.all(np.diff(dp) <= 0.0)

    def test_scalar_consistent_with_vector(self):
        r_scalar = 500.0
        dp_vec = pressureScenario_Rall(self.bpds, self.days,
                                        np.array([r_scalar]), self.STRho)
        dp_sc = pressureScenario_Rall_scalar(self.bpds, self.days, r_scalar, self.STRho)
        # scalar JIT kernel uses different E1 approx — allow 5% tolerance
        assert abs(float(dp_vec[0]) - dp_sc) / max(float(dp_vec[0]), 1e-10) < 0.05

    def test_empty_input_returns_zeros(self):
        dp = pressureScenario_Rall(np.array([]), np.array([]),
                                    np.array([100.0, 500.0]), self.STRho)
        assert np.all(dp == 0.0)

    def test_evaluation_days_parameter(self):
        r = np.array([1000.0])
        dp_full = pressureScenario_Rall(self.bpds, self.days, r, self.STRho)
        dp_half = pressureScenario_Rall(self.bpds, self.days, r, self.STRho,
                                         evaluation_days=1000.0)
        # Evaluating at earlier date should give less or equal pressure
        assert float(dp_half[0]) <= float(dp_full[0]) + 1e-6


class TestProjectedSpatialGrid:
    def test_grid_bounds_include_faults_and_wells(self):
        lats = np.array([30.0, 30.2, 30.1])
        lons = np.array([-101.0, -100.7, -100.9])

        lat_grid, lon_grid, bounds = create_projected_spatial_grid(lats, lons, n=150)

        assert lat_grid.shape == (150, 150)
        assert lon_grid.shape == (150, 150)
        assert bounds[0][0] < float(lats.min())
        assert bounds[1][0] > float(lats.max())
        assert bounds[0][1] < float(lons.min())
        assert bounds[1][1] > float(lons.max())


class TestHydrologyMonteCarloSamples:
    def test_returns_sampled_hydrology_inputs(self):
        params = HydrologyParams(
            aquifer_thickness=100.0,
            porosity=0.1,
            permeability=200.0,
            fluid_density=1000.0,
            dynamic_viscosity=8e-4,
            fluid_compressibility=3.6e-10,
            rock_compressibility=1.08e-9,
            plus_minus={
                "aquifer_thickness": 5.0,
                "porosity": 0.01,
                "permeability": 10.0,
                "fluid_density": 1.0,
                "dynamic_viscosity": 1e-5,
                "fluid_compressibility": 1e-12,
                "rock_compressibility": 1e-11,
            },
            n_iterations=3,
        )
        well = SimpleNamespace(
            well_id="well-1",
            latitude=30.0,
            longitude=-97.0,
            start_date=date(2020, 1, 1),
            start_year=2020,
            days=np.array([1.0, 365.0]),
            rates=np.array([1000.0, 1200.0]),
        )
        faults = pd.DataFrame({
            "FaultID": ["fault-1"],
            "Latitude(WGS84)": [30.05],
            "Longitude(WGS84)": [-97.05],
        })

        results_df, sample_inputs_df = run_hydrology_mc_time_series(
            params,
            [well],
            faults,
            [2021],
            n_jobs=1,
            return_sample_inputs=True,
        )

        assert len(sample_inputs_df) == 3
        assert set([
            "SimulationID",
            "aquifer_thickness",
            "porosity",
            "permeability",
            "fluid_density",
            "dynamic_viscosity",
            "fluid_compressibility",
            "rock_compressibility",
        ]).issubset(sample_inputs_df.columns)
        assert "SimulationID" in results_df.columns
        assert results_df["Pressure"].ge(0.0).all()

    def test_mean_result_mode_matches_grouped_raw_results(self):
        params = HydrologyParams(
            aquifer_thickness=100.0,
            porosity=0.1,
            permeability=200.0,
            fluid_density=1000.0,
            dynamic_viscosity=8e-4,
            fluid_compressibility=3.6e-10,
            rock_compressibility=1.08e-9,
            plus_minus={},
            n_iterations=3,
        )
        well = SimpleNamespace(
            well_id="well-1",
            latitude=30.0,
            longitude=-97.0,
            start_date=date(2020, 1, 1),
            start_year=2020,
            days=np.array([1.0, 365.0]),
            rates=np.array([1000.0, 1200.0]),
        )
        faults = pd.DataFrame({
            "FaultID": ["fault-1", "fault-2"],
            "Latitude(WGS84)": [30.05, 30.1],
            "Longitude(WGS84)": [-97.05, -97.1],
        })

        raw_df = run_hydrology_mc_time_series(
            params, [well], faults, [2021, 2022], n_jobs=1
        )
        mean_df = run_hydrology_mc_time_series(
            params, [well], faults, [2021, 2022], n_jobs=1, result_mode="mean"
        )

        expected = (
            raw_df.groupby(["ID", "Year"])["Pressure"]
            .mean()
            .reset_index()
            .sort_values(["ID", "Year"])
            .reset_index(drop=True)
        )
        actual = (
            mean_df[expected.columns]
            .sort_values(["ID", "Year"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_year_samples_result_mode_limits_raw_rows(self):
        params = HydrologyParams(
            aquifer_thickness=100.0,
            porosity=0.1,
            permeability=200.0,
            fluid_density=1000.0,
            dynamic_viscosity=8e-4,
            fluid_compressibility=3.6e-10,
            rock_compressibility=1.08e-9,
            plus_minus={},
            n_iterations=3,
        )
        well = SimpleNamespace(
            well_id="well-1",
            latitude=30.0,
            longitude=-97.0,
            start_date=date(2020, 1, 1),
            start_year=2020,
            days=np.array([1.0, 365.0]),
            rates=np.array([1000.0, 1200.0]),
        )
        faults = pd.DataFrame({
            "FaultID": ["fault-1", "fault-2"],
            "Latitude(WGS84)": [30.05, 30.1],
            "Longitude(WGS84)": [-97.05, -97.1],
        })

        samples_df = run_hydrology_mc_time_series(
            params,
            [well],
            faults,
            [2021, 2022],
            n_jobs=1,
            result_mode="year_samples",
            sample_year=2022,
        )

        assert set(samples_df["Year"]) == {2022}
        assert len(samples_df) == params.n_iterations * len(faults)


class TestBatchedFaultPressure:
    def test_batched_pressure_matches_scalar_pressure_for_multiple_faults(self):
        STRho = (5e-5, 4.5e-6, 1000.0)
        well = SimpleNamespace(
            well_id="well-1",
            latitude=30.0,
            longitude=-97.0,
            days=np.array([1.0, 365.0, 730.0]),
            rates=np.array([1000.0, 1200.0, 800.0]),
        )
        fault_lats = np.array([30.05, 30.1, 30.2])
        fault_lons = np.array([-97.05, -97.1, -97.2])

        distances = well_fault_distances_m([well], fault_lats, fault_lons)[0]
        batched = pfieldcalc_all_rates_for_distances(
            distances, STRho, well.days, well.rates, evaluation_days=900.0
        )
        scalar = np.array([
            pfieldcalc_all_rates_scalar(
                lon, lat, STRho, well.days, well.rates,
                well.longitude, well.latitude, evaluation_days=900.0
            )
            for lat, lon in zip(fault_lats, fault_lons)
        ])

        assert np.allclose(batched, scalar, rtol=0.05, atol=1e-9)


def test_probabilistic_hydrology_step_does_not_emit_slip_potential_graph():
    step5_path = os.path.join(os.path.dirname(__file__), "..", "src", "fsp_step5.py")

    with open(step5_path, encoding="utf-8") as step5_file:
        step5_source = step5_file.read()

    assert "save_slip_potential_artifact" not in step5_source
    assert "fsp-probabilistic-hydrology-slip-potential" not in step5_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
