import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from graphs.leaflet_map import (
    DETERMINISTIC_GEOMECHANICS_FIELD_LABELS,
    _pressure_grid_to_rgba,
    save_fault_results_map_artifact,
    save_hydrology_heatmap_fault_map_artifact,
    save_model_inputs_map_artifact,
)
from graphs.hydrology_map import (
    _grid_payload,
    save_direct_hydrology_pressure_map_artifact,
)
from graphs.injection_rate import save_injection_rate_graph_artifact
from graphs.mohr_diagram import save_mohr_diagram_graph_artifact


class DummyHelper:
    def __init__(self, scratch_path):
        self.scratchPath = str(scratch_path)
        self.origArgsData = {
            "GraphArtifacts": [],
            "SessionState": {
                "StepState": [
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


def test_model_inputs_map_uses_single_html_artifact_with_embedded_payload(tmp_path):
    helper = DummyHelper(tmp_path)
    faults_df = pd.DataFrame({
        "FaultID": ["fault-row-value"],
        "Latitude(WGS84)": [31.2],
        "Longitude(WGS84)": [-100.4],
        "Strike": [45],
        "Dip": [60],
    })
    injection_df = pd.DataFrame({
        "WellID": ["well-row-value"],
        "Latitude(WGS84)": [31.1],
        "Longitude(WGS84)": [-100.2],
    })

    output_path = save_model_inputs_map_artifact(helper, 0, faults_df, injection_df)

    assert output_path is not None
    assert helper.origArgsData["GraphArtifacts"][0]["renderer"] == "html"
    assert helper.origArgsData["GraphArtifacts"][0]["contentType"] == "text/html"
    html_text = open(output_path, encoding="utf-8").read()
    assert "fault-row-value" in html_text
    assert "well-row-value" in html_text
    assert "setAllFaults(true)" in html_text
    assert "setAllWells(true)" in html_text
    assert "fault-search" in html_text
    assert "well-search" in html_text
    assert "preferCanvas: true" in html_text
    assert not (tmp_path / "graphs" / "map_data").exists()


def test_leaflet_missing_geometry_columns_adds_warning_and_keeps_valid_layers(tmp_path):
    helper = DummyHelper(tmp_path)
    faults_df = pd.DataFrame({
        "FaultID": ["fault-1"],
        "Latitude(WGS84)": [31.2],
        "Longitude(WGS84)": [-100.4],
    })
    injection_df = pd.DataFrame({
        "WellID": ["well-without-location"],
    })

    output_path = save_model_inputs_map_artifact(helper, 0, faults_df, injection_df)

    assert output_path is not None
    html_text = open(output_path, encoding="utf-8").read()
    assert "fault-1" in html_text
    assert "well-without-location" not in html_text
    messages = helper.origArgsData["SessionState"]["StepState"][0]["Messages"]
    assert len(messages) == 1
    assert "injection wells layer skipped" in messages[0]["MessageContent"]
    assert messages[0]["MessageLevel"] == 1


def test_geomechanics_map_manifest_includes_faults_wells_and_dynamic_range_controls(tmp_path):
    helper = DummyHelper(tmp_path)
    faults_df = pd.DataFrame({
        "FaultID": ["fault-row-value-1", "fault-row-value-2"],
        "wkt": ["LINESTRING (-100 31,-99.9 31.1)", "LINESTRING (-100.2 31.2,-100.1 31.3)"],
        "Latitude(WGS84)": [31.0, 31.2],
        "Longitude(WGS84)": [-100.0, -100.2],
        "Strike": [45.0, 50.0],
        "Dip": [60.0, 65.0],
        "slip_pressure": [100.0, 500.0],
        "coulomb_failure_function": [1.0, 2.0],
        "shear_capacity_utilization": [0.2, 0.4],
    })
    wells_df = pd.DataFrame({
        "WellID": ["well-row-value-1", "well-row-value-2"],
        "Latitude(WGS84)": [31.05, 31.25],
        "Longitude(WGS84)": [-100.05, -100.25],
    })

    output_path = save_fault_results_map_artifact(
        helper,
        0,
        faults_df,
        artifact_key="fsp-deterministic-geomechanics-map",
        title="Deterministic Geomechanics Map",
        caption="Leaflet map of deterministic geomechanics fault results.",
        display_order=21,
        result_fields=["slip_pressure", "coulomb_failure_function", "shear_capacity_utilization"],
        color="#7c3aed",
        value_column="slip_pressure",
        legend_title="Deterministic Pore Pressure to Slip",
        value_min_default=0.0,
        well_df=wells_df,
        field_labels=DETERMINISTIC_GEOMECHANICS_FIELD_LABELS,
    )

    assert output_path is not None
    assert helper.origArgsData["GraphArtifacts"][0]["renderer"] == "leaflet-map"
    assert helper.origArgsData["GraphArtifacts"][0]["contentType"] == "application/json"
    import json
    manifest = json.load(open(output_path, encoding="utf-8"))
    layer_keys = [layer["key"] for layer in manifest["layers"]]
    assert "fault-results" in layer_keys
    assert "fault-midpoints" in layer_keys
    assert "fault-result-1" in layer_keys
    assert "fault-midpoint-1" in layer_keys
    assert "fault-result-2" in layer_keys
    assert "fault-midpoint-2" in layer_keys
    assert "injection-wells" in layer_keys
    assert "injection-well-1" in layer_keys
    assert "injection-well-2" in layer_keys
    assert {group["key"] for group in manifest["filterControls"]["groups"]} == {"fault-results", "injection-wells"}
    fault_layer = next(layer for layer in manifest["layers"] if layer["key"] == "fault-results")
    midpoint_layer = next(layer for layer in manifest["layers"] if layer["key"] == "fault-midpoints")
    wells_layer = next(layer for layer in manifest["layers"] if layer["key"] == "injection-wells")
    fault_item_layer = next(layer for layer in manifest["layers"] if layer["key"] == "fault-result-1")
    midpoint_item_layer = next(layer for layer in manifest["layers"] if layer["key"] == "fault-midpoint-1")
    assert fault_layer["type"] == "line"
    assert fault_layer["style"]["valueColumn"] == "slip_pressure"
    assert fault_layer["style"]["legendTitle"] == "Deterministic Pore Pressure to Slip"
    assert fault_layer["style"]["minValue"] == 0.0
    assert fault_layer["style"]["maxValue"] == 500.0
    assert fault_layer["style"]["allowUserRange"] is True
    assert fault_layer["fieldLabels"]["slip_pressure"] == "Deterministic Pore Pressure to Slip"
    assert fault_layer["fieldLabels"]["coulomb_failure_function"] == "Coulomb Failure Function"
    assert "legendTitle" not in fault_item_layer["style"]
    assert "allowUserRange" not in fault_item_layer["style"]
    assert fault_item_layer["style"]["valueColumn"] == "slip_pressure"
    assert midpoint_layer["type"] == "point"
    assert midpoint_layer["filter"]["groupKey"] == "fault-results"
    assert midpoint_layer["filter"]["isAll"] is True
    assert midpoint_layer["style"]["valueColumn"] == "slip_pressure"
    assert midpoint_layer["style"]["colorScale"] == fault_layer["style"]["colorScale"]
    assert "legendTitle" not in midpoint_layer["style"]
    assert "allowUserRange" not in midpoint_layer["style"]
    assert midpoint_item_layer["filter"]["itemValue"] == fault_item_layer["filter"]["itemValue"]
    assert midpoint_item_layer["style"]["valueColumn"] == "slip_pressure"
    assert wells_layer["key"] == "injection-wells"
    assert wells_layer["type"] == "point"
    fault_layer_df = pd.read_csv(tmp_path / fault_layer["source"]["path"])
    wells_layer_df = pd.read_csv(tmp_path / wells_layer["source"]["path"])
    assert "fault-row-value-1" in set(fault_layer_df["FaultID"])
    assert "fault-row-value-2" in set(fault_layer_df["FaultID"])
    assert "well-row-value-1" in set(wells_layer_df["WellID"])
    assert "well-row-value-2" in set(wells_layer_df["WellID"])


def test_large_fault_segment_geomechanics_map_stays_single_leaflet_artifact(tmp_path):
    helper = DummyHelper(tmp_path)
    faults_df = pd.DataFrame({
        "FaultID": [f"DEMO_TX_TRACE_SEG_{i:03d}" for i in range(500)],
        "Latitude(WGS84)": [31.0 + (i // 25) * 0.01 for i in range(500)],
        "Longitude(WGS84)": [-104.0 + (i % 25) * 0.01 for i in range(500)],
        "Strike": [65.0 for _ in range(500)],
        "Dip": [60.0 for _ in range(500)],
        "LengthKm": [1.2 for _ in range(500)],
        "slip_pressure": [float(i) for i in range(500)],
    })

    output_path = save_fault_results_map_artifact(
        helper,
        0,
        faults_df,
        artifact_key="fsp-deterministic-geomechanics-map",
        title="Deterministic Geomechanics Map",
        caption="Leaflet map of deterministic geomechanics fault results.",
        display_order=21,
        result_fields=["slip_pressure"],
        color="#7c3aed",
        value_column="slip_pressure",
        legend_title="Deterministic Pore Pressure to Slip",
    )

    assert output_path is not None
    assert len(helper.origArgsData["GraphArtifacts"]) == 1
    assert helper.origArgsData["GraphArtifacts"][0]["renderer"] == "leaflet-map"
    assert helper.origArgsData["GraphArtifacts"][0]["key"] == "fsp-deterministic-geomechanics-map"
    import json
    manifest = json.load(open(output_path, encoding="utf-8"))
    assert "filterControls" in manifest
    fault_layer = next(layer for layer in manifest["layers"] if layer["key"] == "fault-results")
    layer_df = pd.read_csv(tmp_path / fault_layer["source"]["path"])
    assert "DEMO_TX_TRACE_SEG_000" in set(layer_df["FaultID"])
    assert "DEMO_TX_TRACE_SEG_499" in set(layer_df["FaultID"])
    assert len(layer_df) == 500


def test_geomechanics_map_warns_for_missing_well_geometry_but_keeps_faults(tmp_path):
    helper = DummyHelper(tmp_path)
    faults_df = pd.DataFrame({
        "FaultID": ["fault-1"],
        "Latitude(WGS84)": [31.2],
        "Longitude(WGS84)": [-100.4],
        "Strike": [45.0],
        "Dip": [60.0],
        "LengthKm": [1.5],
        "slip_pressure": [250.0],
    })
    wells_df = pd.DataFrame({
        "WellID": ["well-without-location"],
    })

    output_path = save_fault_results_map_artifact(
        helper,
        0,
        faults_df,
        artifact_key="fsp-deterministic-geomechanics-map",
        title="Deterministic Geomechanics Map",
        caption="Leaflet map of deterministic geomechanics fault results.",
        display_order=21,
        result_fields=["slip_pressure"],
        color="#7c3aed",
        value_column="slip_pressure",
        legend_title="Deterministic Pore Pressure to Slip",
        well_df=wells_df,
    )

    assert output_path is not None
    import json
    manifest = json.load(open(output_path, encoding="utf-8"))
    assert any(layer["key"] == "fault-results" for layer in manifest["layers"])
    assert not any(layer["key"] == "injection-wells" for layer in manifest["layers"])
    messages = helper.origArgsData["SessionState"]["StepState"][0]["Messages"]
    assert len(messages) == 1
    assert "injection wells layer skipped" in messages[0]["MessageContent"]


def test_hydrology_leaflet_manifest_uses_external_heatmap_csv(tmp_path):
    helper = DummyHelper(tmp_path)
    heatmap_df = pd.DataFrame({
        "Latitude": [31.0, 31.0, 31.1, 31.1],
        "Longitude": [-100.1, -100.0, -100.1, -100.0],
        "Pressure_psi": [0.1, 25.0, 75.0, 100.0],
    })
    faults_df = pd.DataFrame({
        "FaultID": ["fault-row-value"],
        "Latitude(WGS84)": [31.0],
        "Longitude(WGS84)": [-100.0],
        "pressure_psi": [42.0],
        "year": [2026],
    })
    wells_df = pd.DataFrame({
        "WellID": ["well-1", "well-2"],
        "Latitude": [31.01, 31.02],
        "Longitude": [-100.01, -100.02],
        "StartDate": ["2020-01-01", "2021-01-01"],
        "EndDate": ["2025-12-31", "2025-12-31"],
        "MaxRate_bbl_day": [1000.0, 2000.0],
        "MeanRate_bbl_day": [800.0, 1500.0],
    })

    manifest_path = save_hydrology_heatmap_fault_map_artifact(
        helper,
        0,
        heatmap_df,
        faults_df,
        wells_df,
        artifact_key="fsp-deterministic-hydrology-map",
        title="Hydrology Pressure Map",
        caption="Leaflet heatmap of deterministic hydrology pressure with fault pressure results.",
        display_order=41,
    )

    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    heat_layer = manifest["layers"][0]
    assert heat_layer["key"] == "pressure-grid"
    assert heat_layer["type"] == "imageOverlay"
    assert heat_layer["source"]["path"].endswith("pressure-grid.png")
    assert heat_layer["source"]["contentType"] == "image/png"
    assert heat_layer["bounds"] == [[31.0, -100.1], [31.1, -100.0]]
    assert heat_layer["style"]["maxValue"] == 100.0
    assert "Leaflet.heat" not in json.dumps(manifest)
    assert "fault-row-value" not in json.dumps(manifest)
    image_path = tmp_path / heat_layer["source"]["path"]
    assert image_path.exists()
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    layer_keys = [layer["key"] for layer in manifest["layers"]]
    assert "injection-wells" in layer_keys
    assert "injection-well-1" in layer_keys
    assert "injection-well-2" in layer_keys
    assert "fault-result-1" in layer_keys
    assert {group["key"] for group in manifest["filterControls"]["groups"]} == {"fault-results", "injection-wells"}
    well_layers = {layer["key"]: layer for layer in manifest["layers"] if layer["key"].startswith("injection-well")}
    assert well_layers["injection-wells"]["visible"] is True
    assert well_layers["injection-well-1"]["visible"] is False
    assert well_layers["injection-well-2"]["visible"] is False


def test_hydrology_leaflet_manifest_converts_fault_csv_rows_to_line_wkt(tmp_path):
    helper = DummyHelper(tmp_path)
    heatmap_df = pd.DataFrame({
        "Latitude": [30.0, 30.0, 30.1, 30.1],
        "Longitude": [-97.1, -97.0, -97.1, -97.0],
        "Pressure_psi": [1.0, 2.0, 3.0, 4.0],
    })
    faults_df = pd.DataFrame({
        "FaultID": ["AUS-F01"],
        "Latitude(WGS84)": [30.2508],
        "Longitude(WGS84)": [-97.7652],
        "Strike": [32.0],
        "Dip": [58.0],
        "LengthKm": [3.2],
        "pressure_psi": [42.0],
        "year": [2025],
    })

    manifest_path = save_hydrology_heatmap_fault_map_artifact(
        helper,
        0,
        heatmap_df,
        faults_df,
        artifact_key="fsp-deterministic-hydrology-map",
        title="Hydrology Pressure Map",
        caption="Leaflet heatmap of deterministic hydrology pressure with fault pressure results.",
        display_order=41,
    )

    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    fault_layer = next(layer for layer in manifest["layers"] if layer["key"] == "fault-results")
    assert fault_layer["type"] == "line"
    assert fault_layer["geometry"]["wktColumn"] == "wkt"

    fault_layer_path = tmp_path / fault_layer["source"]["path"]
    layer_df = pd.read_csv(fault_layer_path)
    assert layer_df.loc[0, "wkt"].startswith("LINESTRING")


def test_direct_hydrology_map_html_contains_well_filters_and_fault_lines(tmp_path):
    helper = DummyHelper(tmp_path)
    per_well_grid_df = pd.DataFrame({
        "WellID": ["well-1", "well-1", "well-2", "well-2"],
        "Latitude": [30.0, 30.1, 30.0, 30.1],
        "Longitude": [-97.0, -97.0, -97.0, -97.0],
        "Pressure_psi": [1.0, 3.0, 2.0, 4.0],
    })
    faults_df = pd.DataFrame({
        "FaultID": ["AUS-F01"],
        "Latitude(WGS84)": [30.05],
        "Longitude(WGS84)": [-97.02],
        "Strike": [35.0],
        "Dip": [60.0],
        "LengthKm": [2.0],
        "pressure_psi": [5.0],
        "year": [2025],
    })
    wells_df = pd.DataFrame({
        "WellID": ["well-1", "well-2"],
        "Latitude": [30.0, 30.1],
        "Longitude": [-97.0, -97.0],
    })

    output_path = save_direct_hydrology_pressure_map_artifact(
        helper,
        0,
        per_well_grid_df,
        faults_df,
        wells_df,
        artifact_key="fsp-deterministic-hydrology-map",
        title="Hydrology Pressure Map",
        caption="Interactive hydrology pressure map with selected-well pressure grid summation.",
        display_order=41,
    )

    assert output_path is not None
    assert helper.origArgsData["GraphArtifacts"][0]["renderer"] == "html"
    html_text = open(output_path, encoding="utf-8").read()
    assert "well-1" in html_text
    assert "well-2" in html_text
    assert "setAllWells(true)" in html_text
    assert "setAllFaults(true)" in html_text
    assert "setAllWellMarkers(true)" in html_text
    assert "fault-controls" in html_text
    assert "well-marker-controls" in html_text
    assert "map.attributionControl.setPrefix(false)" in html_text
    assert 'class="section-toggle" aria-expanded="false" aria-controls="fault-section-content"' in html_text
    assert 'class="section-toggle" aria-expanded="false" aria-controls="well-marker-section-content"' in html_text
    assert 'id="fault-section-content" class="section-content" hidden' in html_text
    assert 'id="well-marker-section-content" class="section-content" hidden' in html_text
    assert "sumSelectedGrid" in html_text
    assert "Pressure Front (PSI)" in html_text
    assert "pressure-min" in html_text
    assert "pressure-max" in html_text
    legend_start = html_text.index('<div class="legend">')
    toolbar_start = html_text.index('<div class="toolbar">')
    assert html_text.index('id="pressure-min"', legend_start) > toolbar_start
    assert html_text.index('id="pressure-max"', legend_start) > toolbar_start
    assert "backdrop-filter: blur(10px)" in html_text
    assert "Inter, Segoe UI" in html_text
    assert "selectedColorRange" in html_text
    assert "defaultMaxPressure" in html_text
    assert "autoMaxValue > 1000 ? autoMaxValue : 1000" in html_text
    assert "LINESTRING" in html_text


def test_direct_hydrology_grid_payload_sums_per_well_pressure():
    payload = _grid_payload(pd.DataFrame({
        "WellID": ["well-1", "well-1", "well-2", "well-2"],
        "Latitude": [30.0, 30.1, 30.0, 30.1],
        "Longitude": [-97.0, -97.0, -97.0, -97.0],
        "Pressure_psi": [1.0, 3.0, 2.0, 4.0],
    }))

    well_1 = payload["wellGrids"]["well-1"]
    well_2 = payload["wellGrids"]["well-2"]
    summed = [
        [well_1[row][col] + well_2[row][col] for col in range(len(well_1[row]))]
        for row in range(len(well_1))
    ]
    assert summed == [[3.0], [7.0]]
    assert payload["maxPressure"] == 7.0


def test_injection_rate_graph_contains_well_filter_dropdown(tmp_path):
    helper = DummyHelper(tmp_path)
    rate_df = pd.DataFrame({
        "WellID": ["well-1", "well-1", "well-2", "well-2"],
        "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "rate_bbl_day": [100.0, 150.0, 200.0, 250.0],
    })

    output_path = save_injection_rate_graph_artifact(helper, rate_df)

    assert output_path is not None
    html_text = open(output_path, encoding="utf-8").read()
    assert "All wells" in html_text
    assert "No wells" in html_text
    assert "Well well-1" in html_text
    assert "Well well-2" in html_text
    assert "updatemenus" in html_text
    assert "Inter, Segoe UI" in html_text
    assert "hoverlabel" in html_text


def test_mohr_diagram_labels_circles_by_stress_regime(tmp_path):
    helper = DummyHelper(tmp_path)
    arcs_df = pd.DataFrame({
        "id": ["circle1", "circle1", "circle2", "circle2", "circle3", "circle3", "friction_line", "friction_line"],
        "x": [0.0, 10.0, 2.0, 8.0, 4.0, 6.0, 0.0, 10.0],
        "y": [0.0, 5.0, 0.0, 3.0, 0.0, 1.0, 0.0, 6.0],
    })
    fault_df = pd.DataFrame({
        "id": ["fault-1"],
        "x": [5.0],
        "y": [2.5],
        "slip_pressure": [100.0],
    })
    slip_df = pd.DataFrame({"id": ["fault-1"], "slip_pressure": [100.0]})

    output_path = save_mohr_diagram_graph_artifact(
        helper,
        arcs_df,
        slip_df,
        fault_df,
        step_index=0,
        stress_regime="normal",
    )

    assert output_path is not None
    html_text = open(output_path, encoding="utf-8").read()
    assert "\\u03c3V - \\u03c3h" in html_text
    assert "\\u03c3H - \\u03c3h" in html_text
    assert "\\u03c3V - \\u03c3H" in html_text
    assert "Circle 1" not in html_text
    assert "mohr-controls" in html_text
    assert 'id="mohr-min-psi" type="number" step="any" value="0.0"' in html_text
    assert "border-radius: 6px" in html_text


def test_pressure_raster_masks_low_pressure_values():
    rgba, min_value, max_value = _pressure_grid_to_rgba(
        [[0.0, 0.5], [2.0, 100.0]],
        transparent_fraction=0.01,
    )

    assert min_value == 0.0
    assert max_value == 100.0
    assert rgba is not None
    # Output rows are flipped for Leaflet imageOverlay orientation.
    assert rgba[1, 0, 3] == 0
    assert rgba[1, 1, 3] == 0
    assert rgba[0, 0, 3] > 0
    assert rgba[0, 1, 3] > rgba[0, 0, 3]


def test_hydrology_step_does_not_emit_legacy_arcgis_heatmap_parameter():
    step4_path = os.path.join(os.path.dirname(__file__), "..", "src", "fsp_step4.py")

    with open(step4_path, encoding="utf-8") as step4_file:
        step4_source = step4_file.read()

    assert "hydrology_heatmap_data_arcgis" not in step4_source
