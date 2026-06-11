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
from fsp.geomechanics.stress import calculate_absolute_stresses
from fsp.geomechanics.slip import (
    calculate_fault_effective_stresses,
    ComputeCriticalPorePressureForFailure,
)
from fsp.io.wells import preprocess_well_data, normalize_wells_to_well_data
from fsp.io.coords import create_projected_spatial_grid
from fsp.models.hydrology import HydrologyParams
from fsp.monte_carlo.hydrology_mc import run_hydrology_mc_time_series
from fsp_step5 import (
    _combined_slip_potential_rows,
    _empirical_geomechanics_probabilities,
    _has_geomechanics_cdf as _step5_has_geomechanics_cdf,
)
from fsp_step6 import (
    _calculate_fsp as _calculate_summary_fsp,
    _fault_summary_for_year,
    _has_geomechanics_cdf as _step6_has_geomechanics_cdf,
    _summary_map_configuration,
)
from graphs.artifacts import FSP_COLOR_SCALE, SLIP_PRESSURE_COLOR_SCALE


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

    def test_empty_faults_return_no_pressure_rows_but_keep_sample_inputs(self):
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
        faults = pd.DataFrame(columns=["FaultID", "Latitude(WGS84)", "Longitude(WGS84)"])

        results_df, sample_inputs_df = run_hydrology_mc_time_series(
            params,
            [well],
            faults,
            [2021],
            n_jobs=1,
            return_sample_inputs=True,
            result_mode="year_samples",
            sample_year=2021,
        )

        assert list(results_df.columns) == ["SimulationID", "ID", "Pressure", "Year"]
        assert results_df.empty
        assert len(sample_inputs_df) == params.n_iterations


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


def test_empirical_geomechanics_lookup_counts_slip_pressures_at_or_below_hydro_pressure():
    geo_pressures = np.array([10.0, 20.0, 30.0, 40.0])
    hydro_pressures = np.array([5.0, 10.0, 25.0, 40.0, 50.0])

    probabilities = _empirical_geomechanics_probabilities(geo_pressures, hydro_pressures)

    assert np.allclose(probabilities, [0.0, 0.25, 0.5, 1.0, 1.0])


def test_combined_slip_potential_averages_empirical_geomechanics_probability():
    pressure_groups = {
        "A": pd.Series([5.0, 15.0, 25.0, 45.0]),
    }
    geo_groups = {
        "A": np.array([10.0, 20.0, 30.0, 40.0]),
    }

    rows, probabilities = _combined_slip_potential_rows(["A"], pressure_groups, geo_groups, 2025)

    assert probabilities["A"] == pytest.approx(0.4375)
    assert rows[0]["ID"] == "A"
    assert rows[0]["slip_pressure"] == pytest.approx(25.0)
    assert rows[0]["probability"] == pytest.approx(0.4375)
    assert rows[0]["Pressure"] == pytest.approx(22.5)
    assert rows[0]["Year"] == 2025


def test_probabilistic_hydrology_recognizes_missing_geomechanics_cdf_as_pressure_only():
    assert not _step5_has_geomechanics_cdf(pd.DataFrame())
    assert not _step5_has_geomechanics_cdf(pd.DataFrame({"ID": ["A"], "Pressure": [12.0]}))
    assert _step5_has_geomechanics_cdf(pd.DataFrame({
        "ID": ["A"],
        "slip_pressure": [10.0],
    }))


def test_summary_fsp_returns_empty_frame_without_geomechanics_cdf():
    hydro_df = pd.DataFrame({
        "ID": ["A"],
        "Pressure": [25.0],
        "Year": [2032],
    })

    fsp_df = _calculate_summary_fsp(pd.DataFrame(), hydro_df)

    assert list(fsp_df.columns) == ["ID", "Year", "FSP", "epoch_time"]
    assert fsp_df.empty
    assert not _step6_has_geomechanics_cdf(pd.DataFrame())


def test_summary_fsp_uses_empirical_samples_for_probabilistic_hydrology():
    geo_cdf_df = pd.DataFrame({
        "ID": ["A", "A", "A", "A"],
        "slip_pressure": [10.0, 20.0, 30.0, 40.0],
        "probability": [0.25, 0.50, 0.75, 1.0],
    })
    hydro_df = pd.DataFrame({
        "SimulationID": [1, 2, 3],
        "ID": ["A", "A", "A"],
        "Pressure": [5.0, 15.0, 45.0],
        "Year": [2032, 2032, 2032],
    })

    fsp_df = _calculate_summary_fsp(geo_cdf_df, hydro_df)

    assert fsp_df.loc[0, "FSP"] == pytest.approx(0.42)


def test_summary_fault_values_use_only_the_selected_year():
    faults = pd.DataFrame({"FaultID": ["A", "B", "C"]})
    fsp_df = pd.DataFrame({
        "ID": ["A", "A", "B"],
        "Year": [2031, 2032, 2032],
        "FSP": [0.1, 0.8, 0.4],
    })
    pressure_df = pd.DataFrame({
        "ID": ["A", "A", "B"],
        "Year": [2031, 2032, 2032],
        "Pressure": [10.0, 80.0, 40.0],
    })

    summary = _fault_summary_for_year(
        faults,
        fsp_df,
        pressure_df,
        2032,
        include_fsp=True,
    ).set_index("FaultID")

    assert summary.loc["A", "summary_fsp"] == pytest.approx(0.8)
    assert summary.loc["A", "summary_pressure"] == pytest.approx(80.0)
    assert summary.loc["B", "summary_fsp"] == pytest.approx(0.4)
    assert summary.loc["C", "summary_fsp"] == pytest.approx(0.0)
    assert summary.loc["C", "summary_pressure"] == pytest.approx(0.0)


def test_summary_map_configuration_colors_fsp_on_fixed_probability_range():
    config = _summary_map_configuration(True)

    assert config["result_fields"] == ["summary_fsp", "summary_pressure"]
    assert config["value_column"] == "summary_fsp"
    assert config["legend_title"] == "Summary FSP"
    assert config["color_scale"] == FSP_COLOR_SCALE
    assert config["value_min_default"] == 0.0
    assert config["value_max_default"] == 1.0


def test_summary_map_configuration_falls_back_to_pressure_without_geomechanics():
    config = _summary_map_configuration(False)

    assert config["result_fields"] == ["summary_pressure"]
    assert config["value_column"] == "summary_pressure"
    assert config["legend_title"] == "Summary Pressure"
    assert config["color_scale"] == SLIP_PRESSURE_COLOR_SCALE
    assert config["value_min_default"] is None
    assert config["value_max_default"] is None


def test_variable_fsp_demo_csvs_have_expected_schema_and_calibration():
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    faults_path = os.path.join(examples_dir, "demo_texas_faults_fsp_100_variable_fsp.csv")
    injection_path = os.path.join(examples_dir, "demo_texas_injection_wells_monthly_fsp_20wells_variable_fsp.csv")
    faults = pd.read_csv(faults_path, dtype={"FaultID": str})
    injection = pd.read_csv(injection_path, dtype={"WellID": str})

    assert list(faults.columns) == [
        "FaultID",
        "Latitude(WGS84)",
        "Longitude(WGS84)",
        "Strike",
        "Dip",
        "LengthKm",
        "FrictionCoefficient",
    ]
    assert list(injection.columns) == [
        "WellID",
        "Latitude(WGS84)",
        "Longitude(WGS84)",
        "Year",
        "Month",
        "InjectionRate(bbl/month)",
    ]
    assert len(faults) == 100
    assert len(injection) == 20 * 15 * 12
    assert injection["WellID"].nunique() == 20
    assert injection["Year"].min() == 2018
    assert injection["Year"].max() == 2032
    assert injection["InjectionRate(bbl/month)"].max() <= 650000
    assert injection["InjectionRate(bbl/month)"].min() >= 15000

    stress_inputs = {
        "reference_depth": 11000.0,
        "vertical_stress": 1.1,
        "min_horizontal_stress": 0.693,
        "max_horizontal_stress": 1.22,
        "pore_pressure": 0.43,
        "max_stress_azimuth": 70.0,
    }
    friction = 0.58
    stress_state, p0 = calculate_absolute_stresses(stress_inputs, friction, "gradients")
    slip_pressures = []
    for _, fault in faults.iterrows():
        sig_normal, tau_normal, *_ = calculate_fault_effective_stresses(
            float(fault["Strike"]),
            float(fault["Dip"]),
            stress_state,
            p0,
            0.0,
        )
        slip_pressures.append(float(np.asarray(
            ComputeCriticalPorePressureForFailure(sig_normal, tau_normal, friction, p0)
        ).item()))
    slip_pressures = np.asarray(slip_pressures)
    assert np.sum(slip_pressures < 100.0) >= 10
    assert np.sum((slip_pressures >= 100.0) & (slip_pressures < 500.0)) >= 10
    assert np.sum(slip_pressures > 1000.0) >= 20

    well_info = preprocess_well_data(injection, "monthly_fsp")
    well_data_list = normalize_wells_to_well_data(well_info, "monthly_fsp", date(2031, 12, 31))
    STRho = calcST(100.0, 0.1, 200.0, 1000.0, 0.0008, 9.81, 3.6e-10, 1.08e-9)
    distances = well_fault_distances_m(
        well_data_list,
        faults["Latitude(WGS84)"].values.astype(float),
        faults["Longitude(WGS84)"].values.astype(float),
    )
    pressure = np.zeros(len(faults), dtype=float)
    for well_index, well_data in enumerate(well_data_list):
        evaluation_days = float((date(2031, 12, 31) - well_data.start_date).days + 1)
        pressure += pfieldcalc_all_rates_for_distances(
            distances[well_index],
            STRho,
            well_data.days,
            well_data.rates,
            evaluation_days,
        )

    overlap_count = int(np.sum(pressure >= slip_pressures))
    assert pressure.max() > 300.0
    assert np.quantile(pressure, 0.75) > 150.0
    assert 10 <= overlap_count <= 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
