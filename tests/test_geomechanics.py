"""
Unit tests for geomechanics calculations.
Reference values derived from FSP Julia tests/computeStressTensorTests.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from fsp.models.stress import StressState
from fsp.geomechanics.stress import (
    calculate_n_phi,
    calculate_absolute_stresses,
)
from fsp.geomechanics.slip import (
    calculate_fault_effective_stresses,
    ComputeCriticalPorePressureForFailure,
    analyze_fault,
)
from fsp.monte_carlo.geomechanics_mc import run_geomechanics_mc
from fsp_step3 import (
    PROBABILISTIC_GEOMECHANICS_FIELD_LABELS,
    _mc_uncertainty_sensitivity_data,
    _prob_geomechanics_cdf,
    _probabilistic_geomechanics_fault_map_data,
)
from graphs.scientific import (
    save_cdf_artifact,
    save_input_distribution_histograms_artifact,
    save_radial_curves_artifact,
    save_uncertainty_tornado_artifact,
)


class TestCalculateNPhi:
    def test_normal_faulting(self):
        n, phi = calculate_n_phi(0.5)
        assert n == 0
        assert abs(phi - 0.5) < 1e-10

    def test_strike_slip(self):
        n, phi = calculate_n_phi(1.5)
        assert n == 1
        assert abs(phi - 0.5) < 1e-10

    def test_reverse(self):
        n, phi = calculate_n_phi(2.5)
        assert n == 2
        assert abs(phi - 0.5) < 1e-10

    def test_boundaries(self):
        n0, _ = calculate_n_phi(0.0)
        assert n0 == 0
        n2, _ = calculate_n_phi(2.0)
        assert n2 == 2
        n3, _ = calculate_n_phi(3.0)
        assert n3 == 2

    def test_invalid(self):
        with pytest.raises(ValueError):
            calculate_n_phi(-0.1)
        with pytest.raises(ValueError):
            calculate_n_phi(3.1)


class TestCalculateAbsoluteStresses:
    def test_gradients_model(self):
        stress_data = {
            "reference_depth": 5000.0,
            "vertical_stress": 1.0,
            "pore_pressure": 0.45,
            "max_stress_azimuth": 60.0,
            "min_horizontal_stress": 0.7,
            "max_horizontal_stress": 0.9,
        }
        state, p0 = calculate_absolute_stresses(stress_data, 0.6, "gradients")
        assert abs(state.principal_stresses[0] - 5000.0) < 0.1   # Svert
        assert abs(state.principal_stresses[1] - 3500.0) < 0.1   # Shmin
        assert abs(state.principal_stresses[2] - 4500.0) < 0.1   # SHmax
        assert abs(p0 - 2250.0) < 0.1


class TestFaultEffectiveStresses:
    def setup_method(self):
        # Simple normal faulting stress state
        self.state = StressState(np.array([5000.0, 3500.0, 4500.0]), 60.0)
        self.p0 = 2250.0

    def test_scalar_inputs(self):
        sig, tau, *_ = calculate_fault_effective_stresses(30.0, 75.0, self.state, self.p0, 0.0)
        assert float(sig) >= 0.0
        assert float(tau) >= 0.0

    def test_vectorised_over_faults(self):
        strikes = np.array([30.0, 60.0, 120.0])
        dips = np.array([75.0, 60.0, 45.0])
        sig, tau, *_ = calculate_fault_effective_stresses(strikes, dips, self.state, self.p0, 0.0)
        assert sig.shape == (3,)
        assert tau.shape == (3,)
        assert np.all(sig >= 0.0)
        assert np.all(tau >= 0.0)

    def test_hydro_dp_increases_pressure(self):
        sig0, tau0, *_ = calculate_fault_effective_stresses(30.0, 75.0, self.state, self.p0, 0.0)
        sig1, tau1, *_ = calculate_fault_effective_stresses(30.0, 75.0, self.state, self.p0, 500.0)
        # Higher pore pressure reduces effective normal stress
        assert float(sig1) <= float(sig0)


class TestComputeCriticalPorePressure:
    def test_positive_pressure_needed(self):
        pp = ComputeCriticalPorePressureForFailure(1000.0, 300.0, 0.6)
        assert float(pp) > 0.0

    def test_already_slipping(self):
        # tau/sig > mu → already above failure → Pcritical = 0
        pp = ComputeCriticalPorePressureForFailure(100.0, 100.0, 0.6)
        assert float(pp) == 0.0

    def test_vectorised(self):
        sig = np.array([1000.0, 2000.0, 500.0])
        tau = np.array([300.0, 800.0, 100.0])
        pp = ComputeCriticalPorePressureForFailure(sig, tau, 0.6)
        assert pp.shape == (3,)
        assert np.all(pp >= 0.0)


class TestAnalyzeFault:
    def test_returns_expected_keys(self):
        state = StressState(np.array([5000.0, 3500.0, 4500.0]), 60.0)
        result = analyze_fault(30.0, 75.0, 0.6, state, 2250.0, 0.0)
        for key in ["normal_stress", "shear_stress", "slip_pressure",
                     "slip_tendency", "coulomb_failure_function", "shear_capacity_utilization"]:
            assert key in result
        assert result["slip_pressure"] >= 0.0
        assert 0.0 <= result["shear_capacity_utilization"] <= 1.0


def _mc_inputs():
    stress_data = {
        "reference_depth": 5000.0,
        "vertical_stress": 1.0,
        "pore_pressure": 0.45,
        "max_stress_azimuth": 60.0,
        "min_horizontal_stress": 0.7,
        "max_horizontal_stress": 0.9,
        "aphi_value": None,
    }
    faults = pd.DataFrame({
        "FaultID": ["A", "B"],
        "Strike": [30.0, 75.0],
        "Dip": [70.0, 55.0],
    })
    uncertainties = {
        "vertical_stress_gradient_uncertainty": 0.05,
        "initial_pore_pressure_gradient_uncertainty": 0.02,
        "max_stress_azimuth_uncertainty": 5.0,
        "max_horizontal_stress_uncertainty": 0.04,
        "min_horizontal_stress_uncertainty": 0.03,
        "strike_angles_uncertainty": 4.0,
        "dip_angles_uncertainty": 3.0,
        "friction_coefficient_uncertainty": 0.02,
    }
    return stress_data, faults, uncertainties


class TestGeomechanicsMonteCarloMetadata:
    def test_default_return_shape_is_preserved(self):
        stress_data, faults, uncertainties = _mc_inputs()
        results = run_geomechanics_mc(
            stress_data, faults, 5, uncertainties, "gradients", 0.6, random_seed=42
        )
        assert list(results.columns) == ["SimulationID", "FaultID", "SlipPressure"]
        assert len(results) == 10

    def test_can_return_sample_inputs(self):
        stress_data, faults, uncertainties = _mc_inputs()
        results, sample_inputs = run_geomechanics_mc(
            stress_data, faults, 5, uncertainties, "gradients", 0.6,
            random_seed=42, return_sample_inputs=True
        )
        assert len(results) == 10
        assert len(sample_inputs) == 10
        for column in [
            "SimulationID",
            "FaultID",
            "vertical_stress_gradient",
            "initial_pore_pressure_gradient",
            "max_stress_azimuth",
            "max_horizontal_stress_gradient",
            "min_horizontal_stress_gradient",
            "friction_coefficient",
            "strike_angle",
            "dip_angle",
        ]:
            assert column in sample_inputs.columns


class TestMCUncertaintySensitivity:
    def test_builds_per_fault_parameter_rows_sorted_by_impact(self):
        mc_df = pd.DataFrame({
            "SimulationID": list(range(1, 11)) * 2,
            "FaultID": ["A"] * 10 + ["B"] * 10,
            "SlipPressure": [1, 1, 2, 2, 3, 10, 11, 12, 12, 13] + [5] * 10,
        })
        sample_inputs = pd.DataFrame({
            "SimulationID": list(range(1, 11)) * 2,
            "FaultID": ["A"] * 10 + ["B"] * 10,
            "vertical_stress_gradient": list(range(10)) + list(range(10)),
            "max_stress_azimuth": [60.0] * 20,
        })
        uncertainties = {
            "vertical_stress_gradient_uncertainty": 0.1,
            "max_stress_azimuth_uncertainty": 5.0,
        }

        sensitivity = _mc_uncertainty_sensitivity_data(mc_df, sample_inputs, uncertainties, quantile=0.2)

        assert list(sensitivity["FaultID"].unique()) == ["A", "B"]
        assert set(sensitivity["parameter"]) == {"vertical_stress_gradient_uncertainty"}
        fault_a = sensitivity[sensitivity["FaultID"] == "A"].iloc[0]
        fault_b = sensitivity[sensitivity["FaultID"] == "B"].iloc[0]
        assert fault_a["low_slip_pressure"] == 1.0
        assert fault_a["high_slip_pressure"] == 12.5
        assert fault_a["impact"] == 11.5
        assert fault_b["impact"] == 0.0


class TestProbabilisticGeomechanicsMapData:
    def test_includes_prior_deterministic_answers_with_explicit_labels(self):
        faults = pd.DataFrame({
            "FaultID": ["A", "B"],
            "Latitude(WGS84)": [31.0, 31.1],
            "Longitude(WGS84)": [-100.0, -100.1],
        })
        stats = pd.DataFrame({
            "FaultID": ["A", "B"],
            "Mean": [125.0, 225.0],
            "StdDev": [5.0, 7.0],
            "Median": [120.0, 220.0],
            "Min": [100.0, 200.0],
            "Max": [150.0, 250.0],
        })
        deterministic = pd.DataFrame({
            "FaultID": ["A", "B"],
            "slip_pressure": [111.0, 222.0],
            "coulomb_failure_function": [1.1, 2.2],
            "shear_capacity_utilization": [0.11, 0.22],
            "normal_stress": [3000.0, 4000.0],
        })

        map_df = _probabilistic_geomechanics_fault_map_data(faults, stats, deterministic)

        fault_a = map_df[map_df["FaultID"] == "A"].iloc[0]
        assert fault_a["Mean"] == 125.0
        assert fault_a["det_slip_pressure"] == 111.0
        assert fault_a["det_coulomb_failure_function"] == 1.1
        assert fault_a["det_shear_capacity_utilization"] == 0.11
        assert "normal_stress" not in map_df.columns
        assert PROBABILISTIC_GEOMECHANICS_FIELD_LABELS["det_slip_pressure"] == "Deterministic Pore Pressure to Slip"
        assert PROBABILISTIC_GEOMECHANICS_FIELD_LABELS["det_coulomb_failure_function"] == "Deterministic Coulomb Failure Function"
        assert PROBABILISTIC_GEOMECHANICS_FIELD_LABELS["det_shear_capacity_utilization"] == "Deterministic Shear Capacity Utilization"


class TestProbabilisticGeomechanicsCDFData:
    def test_cdf_carries_deterministic_slip_pressure_per_fault(self):
        mc_df = pd.DataFrame({
            "FaultID": ["A", "A", "B", "B"],
            "SlipPressure": [12.0, 10.0, 35.0, 30.0],
        })
        det_df = pd.DataFrame({
            "FaultID": ["A", "B"],
            "slip_pressure": [111.0, 222.0],
        })

        cdf_df = _prob_geomechanics_cdf(mc_df, det_df)

        fault_a = cdf_df[cdf_df["ID"] == "A"]
        fault_b = cdf_df[cdf_df["ID"] == "B"]
        assert list(fault_a["slip_pressure"]) == [10.0, 12.0]
        assert list(fault_b["slip_pressure"]) == [30.0, 35.0]
        assert set(fault_a["det_slip_pressure"]) == {111.0}
        assert set(fault_b["det_slip_pressure"]) == {222.0}


class _FakeHelper:
    def __init__(self, scratch_path):
        self.scratchPath = str(scratch_path)
        self.origArgsData = {"SessionState": {"StepState": [{"Messages": []}]}, "GraphArtifacts": []}
        self.artifacts = []

    def addMessageWithStepIndex(self, step_index, message, severity):
        self.origArgsData["SessionState"]["StepState"][step_index]["Messages"].append({
            "MessageContent": message,
            "Severity": severity,
        })

    def saveGraphArtifact(self, **kwargs):
        self.artifacts.append(kwargs)


class TestScientificGraphArtifacts:
    def test_radial_pressure_artifact_contains_well_selector_controls(self, tmp_path):
        helper = _FakeHelper(tmp_path)
        radial_df = pd.DataFrame({
            "ID": ["WELL_A", "WELL_A", "WELL_B", "WELL_B"],
            "Distance_km": [0.0, 10.0, 0.0, 10.0],
            "Pressure_psi": [50.0, 5.0, 35.0, 3.0],
        })

        path = save_radial_curves_artifact(helper, 0, radial_df)

        html = open(path, encoding="utf-8").read()
        assert helper.artifacts[0]["renderer"] == "html"
        assert "Injection Wells" in html
        assert "series-legend" in html
        assert "selector-body" in html
        assert "selector-master" in html
        assert "Show All" in html
        assert "Hide All" in html
        assert "All wells" in html
        assert "Select one or more injection wells from the legend" in html
        assert "toggleSeries" in html
        assert "toggleAllFromMaster" in html
        assert "setAllSelection(false)" in html
        assert "selectedSeriesIds = new Set(seriesIds)" in html
        assert '"WELL_A"' in html
        assert '"WELL_B"' in html
        assert '"label":"Well WELL_A"' in html
        assert '"label":"Well WELL_B"' in html
        assert '"color":"#2563eb"' in html
        assert '"color":"#e11d48"' in html
        assert "flex-direction: row;" in html
        assert "flex: 0 0 clamp(300px, 28vw, 360px);" in html
        assert "overflow-y: auto;" in html
        assert "@media (max-width: 780px)" in html
        assert "flex: 1 1 55%;" in html
        assert "flex: 1 1 45%;" in html
        assert "min-height: 140px;" in html
        assert "flex: 1 1 auto;" in html
        assert "flex: 0 0 260px;" not in html
        assert "showlegend: false" in html
        assert "Well: Well WELL_A" in html
        assert "Plotly.react" in html
        assert "All wells" in html
        assert "No wells" not in html
        assert helper.artifacts[0]["preferredHeight"] == 800

    def test_cdf_artifact_contains_fault_filter_controls(self, tmp_path):
        helper = _FakeHelper(tmp_path)
        cdf_df = pd.DataFrame({
            "ID": ["A", "A", "B", "B"],
            "slip_pressure": [1.0, 2.0, 1.5, 2.5],
            "probability": [0.5, 1.0, 0.5, 1.0],
            "det_slip_pressure": [100.0, 100.0, 300.0, 300.0],
        })

        path = save_cdf_artifact(
            helper,
            0,
            cdf_df,
            artifact_key="cdf-test",
            title="Probabilistic Geomechanics CDF",
            pressure_label="Pressure",
            probability_label="Probability",
            display_order=1,
        )

        html = open(path, encoding="utf-8").read()
        assert helper.artifacts[0]["renderer"] == "html"
        assert "series-legend" in html
        assert "Show All" in html
        assert "All faults" in html
        assert "Fault Curves" in html
        assert "Color Range" in html
        assert "faults-tab-button" in html
        assert "colors-tab-button" in html
        assert "switchControlTab" in html
        assert "faults-tab" in html
        assert "colors-tab" in html
        assert "tab-panel is-active" in html
        assert "Pore Pressure to Slip" in html
        assert "pressure-min" in html
        assert "pressure-max" in html
        assert "Reset Range" in html
        assert "colorbar-min" in html
        assert "colorbar-max" in html
        assert "handleColorRangeInput" in html
        assert "resetColorRange" in html
        assert "selectedColorRange" in html
        assert "scaledColor" in html
        assert "currentSeriesColor" in html
        assert "legend-panel" in html
        assert "plot-stage" in html
        assert "flex-direction: column;" in html
        assert "flex: 0 0 clamp(430px, 56vh, 560px);" in html
        assert "height: 8px;" in html
        assert "overflow-y: auto;" in html
        assert "toggleSeries" in html
        assert '"A":' in html
        assert '"B":' in html
        assert '"color":"#800000"' in html
        assert '"color":"#007f00"' in html
        assert "detSlipPressure" in html
        assert "const autoColorMin = 100.0" in html
        assert "const autoColorMax = 300.0" in html
        assert "@media (max-width: 780px)" in html
        assert "Inter, Segoe UI" in html
        assert "Plotly.react" in html
        assert helper.artifacts[0]["preferredHeight"] == 700

    def test_uncertainty_artifact_contains_fault_dropdown(self, tmp_path):
        helper = _FakeHelper(tmp_path)
        sensitivity_df = pd.DataFrame({
            "FaultID": ["A", "A", "B"],
            "label": ["Pore Press Grad", "SHmax Azimuth", "Pore Press Grad"],
            "low_slip_pressure": [10.0, 11.0, 7.0],
            "high_slip_pressure": [15.0, 9.0, 8.0],
            "impact": [5.0, -2.0, 1.0],
            "low_delta": [0.0, -2.0, 0.0],
            "high_delta": [5.0, 0.0, 1.0],
            "method": ["P10 vs P90 sample-band median"] * 3,
        })

        path = save_uncertainty_tornado_artifact(
            helper,
            0,
            sensitivity_df,
            artifact_key="tornado-test",
            title="Geomechanics Uncertainty Variability",
            x_label="Impact",
            display_order=1,
        )

        html = open(path, encoding="utf-8").read()
        assert "Fault A" in html
        assert "Fault B" in html
        assert "P10 vs P90 sample-band median" in html
        assert "rgba(255, 255, 255, 0.94)" in html

    def test_input_distribution_histograms_use_per_fault_window(self, tmp_path):
        helper = _FakeHelper(tmp_path)
        sample_inputs = pd.DataFrame({
            "SimulationID": [1, 2, 1, 2],
            "FaultID": ["A", "A", "B", "B"],
            "vertical_stress_gradient": [1.09, 1.11, 1.08, 1.10],
            "min_horizontal_stress_gradient": [0.68, 0.70, 0.67, 0.69],
            "max_horizontal_stress_gradient": [1.21, 1.23, 1.20, 1.22],
            "initial_pore_pressure_gradient": [0.42, 0.44, 0.41, 0.43],
            "strike_angle": [33.0, 43.0, 50.0, 55.0],
            "dip_angle": [56.0, 66.0, 60.0, 65.0],
            "max_stress_azimuth": [65.0, 75.0, 62.0, 72.0],
            "friction_coefficient": [0.57, 0.59, 0.56, 0.58],
        })
        mc_results = pd.DataFrame({
            "SimulationID": [1, 2, 1, 2],
            "FaultID": ["A", "A", "B", "B"],
            "SlipPressure": [200.0, 800.0, 300.0, 900.0],
        })

        path = save_input_distribution_histograms_artifact(
            helper,
            0,
            sample_inputs,
            mc_results,
            artifact_key="histogram-test",
            title="Probabilistic Geomechanics Histogram",
            display_order=31,
        )

        html = open(path, encoding="utf-8").read()
        assert "Use the graph fullscreen button" in html
        assert "function syncViewMode()" in html
        assert "isLargeEnoughForFullHistogram" in html
        assert "window.open" not in html
        assert "previousSlide" not in html
        assert "nextSlide" not in html
        assert "Fault A" in html
        assert "Fault B" in html
        assert "All faults" not in html
        assert "S&lt;sub&gt;vertical&lt;/sub&gt;" in html
        assert "Natural Pore Pressure" in html
        assert "result: pore pressure to slip for fault A" in html
        assert "Number of Realizations" in html
        assert "Inter, Segoe UI" in html
        assert "border-radius: 8px" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
