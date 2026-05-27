import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fsp.models.stress import StressState
from graphs.artifacts import SLIP_PRESSURE_COLOR_SCALE
from graphs.stereonet import (
    fault_normal_projection,
    normal_composite_grid,
    projected_curve,
    save_stereonet_graph_artifact,
)


class DummyHelper:
    def __init__(self, scratch_path):
        self.scratchPath = str(scratch_path)
        self.origArgsData = {
            "GraphArtifacts": [],
            "SessionState": {
                "StepState": [
                    {"Messages": []},
                    {"Messages": []},
                ],
            },
        }

    def saveGraphArtifact(self, **artifact):
        self.origArgsData["GraphArtifacts"].append(artifact)

    def addMessageWithStepIndex(self, step_index, message_content, message_level):
        self.origArgsData["SessionState"]["StepState"][step_index]["Messages"].append({
            "MessageContent": message_content,
            "MessageLevel": message_level,
        })


def test_fault_normal_projection_uses_matlab_lower_hemisphere_pole_formula():
    x, y = fault_normal_projection([90.0], [45.0])

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert math.isclose(float(x[0]), 0.0, abs_tol=1e-12)
    assert math.isclose(float(y[0]), 0.5, abs_tol=1e-12)


def test_projected_curve_coordinates_are_finite_inside_stereonet():
    x, y = projected_curve(45.0, 60.0)
    radius = np.sqrt(x ** 2 + y ** 2)

    assert len(x) == 181
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert float(radius.max()) <= 1.0000001


def test_normal_composite_grid_returns_non_negative_slip_pressure_values():
    stress_state = StressState(np.array([7000.0, 4300.0, 6000.0]), 60.0)
    composite = normal_composite_grid(stress_state, 2500.0, 0.6, grid_size=8)

    assert len(composite) == 64
    assert {"strike", "dip", "x", "y", "slip_pressure"}.issubset(composite.columns)
    assert composite["slip_pressure"].ge(0.0).all()
    assert np.isfinite(composite[["x", "y", "slip_pressure"]].to_numpy()).all()


def test_stereonet_artifact_registers_html_and_dropdown_modes(tmp_path):
    helper = DummyHelper(tmp_path)
    faults_df = pd.DataFrame({
        "FaultID": ["A", "B"],
        "Strike": [45.0, 210.0],
        "Dip": [60.0, 35.0],
        "slip_pressure": [100.0, 500.0],
    })
    stress_state = StressState(np.array([7000.0, 4300.0, 6000.0]), 60.0)

    output_path = save_stereonet_graph_artifact(
        helper,
        faults_df,
        stress_state,
        2500.0,
        0.6,
        60.0,
    )

    assert output_path is not None
    artifact = helper.origArgsData["GraphArtifacts"][0]
    assert artifact["key"] == "fsp-deterministic-geomechanics-stereonet"
    assert artifact["renderer"] == "html"
    assert artifact["contentType"] == "text/html"
    assert artifact["displayOrder"] == 22

    html_text = open(output_path, encoding="utf-8").read()
    assert "Fault Normals" in html_text
    assert "Projected Curves" in html_text
    assert "Normal Composite" in html_text
    assert "Delta PP to slip (PSI)" in html_text
    assert "Min PSI" in html_text
    assert 'id="stereonet-min-psi" type="number" step="any" value="0"' in html_text
    assert "Max PSI" in html_text
    assert "stereonet-controls" in html_text
    assert "Inter, Segoe UI" in html_text
    assert "scattergl" in html_text
    assert SLIP_PRESSURE_COLOR_SCALE[0][1] in html_text
