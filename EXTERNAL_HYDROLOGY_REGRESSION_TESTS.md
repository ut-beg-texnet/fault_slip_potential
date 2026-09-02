# External hydrology regression tests (MATLAB vs Python)

This is a **parity check** for FSP *imported* hydrology: an uploaded pressure-snapshot CSV interpolated onto faults.

It does **not** run Theis, injection wells, or Monte Carlo. For that path see [`HYDROLOGY_REGRESSION_TESTS.md`](HYDROLOGY_REGRESSION_TESTS.md).

Each side independently interpolates the same snapshots:

1. **Python** — production `load_external_hydrology` + `interpolate_fault_pressures` (same as portal Steps 4 and 6; unrounded)
2. **MATLAB R2012b** — production CSV load, then `griddata`
   - load: `textscan` + `SpreadsheetStrings2HydrologyData` (from `support code/`, as in `dataentryHydrology.m`)
   - interpolate: `griddata(thisYearX, thisYearY, thisYearPSI, fault_x, fault_y)` (as in `ShowAllIntegratedCurves.m` / `calcengine.m`)

The portal CSV is WGS84; MATLAB’s imported-model CSV is East/North km. The harness projects each snapshot year with production `_project_points` / `_project_targets` (mean lat/lon origin, `111.0 * cos(lat0)` km/deg lon, `111.0` km/deg lat) so both interpolators see the same Cartesian frame. Python I/O stays WGS84.

On a **regular rectangular lattice**, Delaunay is not unique. MATLAB R2012b `griddata` (`TriScatteredInterp` / CGAL) splits each cell **NW–SE**; SciPy `LinearNDInterpolator` (Qhull) splits **SW–NE**. Production `interpolate_fault_pressures` uses the MATLAB NW–SE split on complete rectilinear km grids and falls back to `LinearNDInterpolator` for scattered snapshots.

The entry point is [`run_matlab_python_external_hydrology_regression.py`](run_matlab_python_external_hydrology_regression.py).

## Prerequisites

- **Python 3.12.x** with this repo’s venv and `requirements.txt`
- **MATLAB R2012b** (the comparison driver is written for that release and legacy code requires this specific version)
- A copy of the **legacy MATLAB FSP source** (this is not provided in this repository)

### Python environment

```powershell
cd C:\path\to\fsp_python
python -m venv fsp_python_venv
.\fsp_python_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### MATLAB FSP layout

`--matlab-code` (also `--matlab_code`) points at the **root** of the MATLAB tree. The folder can have any name. It must contain:

```text
<matlab-root>/
  support code/
    SpreadsheetStrings2HydrologyData.m
```

This harness does **not** need `technical code/` (`pfront.m` / `calcST.m`). If MATLAB is not at the default Windows path (`C:\Program Files\MATLAB\R2012b\bin\matlab.exe`), pass `--matlab-executable`.

## Input CSVs

### Faults

Required columns: `FaultID`, `Latitude(WGS84)`, `Longitude(WGS84)`.

Every fault must lie inside each snapshot’s convex hull. Python raises if a fault is uncovered; MATLAB `griddata` returns NaN, which fails the comparison.

### External hydrologic model

Portal format (`--model`). Required columns:

- `Latitude (WGS84)`
- `Longitude (WGS84)`
- `Change in PSI`
- `Year` (whole calendar years)

Step 1’s normalized names (`Latitude(WGS84)`, `Longitude(WGS84)`, `Pressure_psi`, `Year`) are also accepted. At least three non-collinear points are required per year. Duplicate latitude/longitude/year rows are rejected.

The comparison interpolates **every snapshot year** that appears in the CSV (MATLAB summary / Python Step 6). It does not exercise the year-slider gap-snapping path.

## How to run

From the repo root, with the venv activated (or using the venv’s `python.exe`):

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_external_hydrology_regression.py
```

That uses the default MATLAB code (`reference_old_code`), default MATLAB executable, and default example CSVs that we use during testing. Those will be different on your machine and you need to provide the corresponding flags to overwrite them with your files/setup.

### Point at MATLAB source on another machine

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_external_hydrology_regression.py `
  --matlab-code C:\path\to\matlab_fsp_root
```

`--matlab_code` is the same flag.

### Other useful flags

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_external_hydrology_regression.py `
  --matlab-executable "C:\Program Files\MATLAB\R2012b\bin\matlab.exe" `
  --matlab-code C:\path\to\matlab_fsp_root `
  --faults C:\path\to\faults.csv `
  --model C:\path\to\external_hydrology_model.csv `
  --output-dir C:\path\to\external_hydrology_regression_output
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--matlab-code` / `--matlab_code` | MATLAB FSP **root** (must contain `support code`) | `<repo>/reference_old_code` |
| `--matlab-executable` | `matlab.exe` | `C:\Program Files\MATLAB\R2012b\bin\matlab.exe` |
| `--faults` | Fault CSV | `examples/demo_texas_faults_fsp_100_variable_fsp.csv` |
| `--model` | Portal-format external hydrology CSV | `examples/demo_external_hydrology_model.csv` |
| `--output-dir` | Where CSVs, HTML, and `summary.json` are written | `<repo>/external_hydrology_regression_output` |

## What it compares

- **Fault pressure at every snapshot year** — linear interpolation of that year’s points onto each fault

Python and MATLAB share the same per-year map origin (mean lat/lon of that snapshot). They do **not** share a pre-interpolated pressure field.

On a complete rectilinear snapshot, both sides use the same NW–SE triangle split that MATLAB `griddata` chose on the comparison 3×3 lattice. Scattered (non-grid) snapshots still use SciPy linear interpolation.

## Outputs

Written under `--output-dir`:

| File | Contents |
|------|----------|
| `fault_pressure_comparison.csv` | Fault ID, year, Python/MATLAB pressure, errors |
| `python_fault_pressure.csv` | Raw Python interpolator output |
| `matlab_fault_pressure.csv` | Raw MATLAB `griddata` output |
| `matlab_model_input.csv` | MATLAB-layout CSV (East km, North km, PSI, Year) |
| `matlab_faults_xy.csv` | Per-year projected fault coordinates (km) |
| `fault_pressure_parity.html` | Plotly Python vs MATLAB scatter (1:1 line, colored by year) |
| `summary.json` | Pass/fail, max error, inputs (including `matlab_code` and years), output paths |
| `matlab_external_hydrology_regression_driver.m` | Generated MATLAB driver (for debugging) |

## Pass / fail

**Pressure** (every fault × snapshot year)

- If MATLAB pressure is **≥ 1 psi**: relative error must be **≤ 1%**
- If MATLAB pressure is **< 1 psi**: absolute error must be **≤ 0.01 psi**
- Any non-finite pressure (NaN outside the convex hull) fails

The process exits **0** on pass and **1** on fail. `summary.json` has `"passed"`, `"max_relative_error"`, and `"nan_count"`.

## Common failures

| Symptom | Likely cause |
|---------|----------------|
| `SpreadsheetStrings2HydrologyData.m` missing | `--matlab-code` is not the MATLAB FSP **root**, or `support code/` is missing |
| `MATLAB R2012b was not found` | Wrong `--matlab-executable`, or R2012b is not installed |
| MATLAB did not produce all outputs | MATLAB failed to run the generated `.m` driver; inspect `matlab_run.log`, `matlab_error.txt`, and `matlab_external_hydrology_regression_driver.m` |
| External hydrologic model CSV is missing required columns | File is not portal format; need `Latitude (WGS84)`, `Longitude (WGS84)`, `Change in PSI`, `Year` |
| `does not cover fault row(s)` | A fault lies outside that year’s convex hull; move faults inside the snapshot or expand the model |
| Python/MATLAB rows did not line up | Stale MATLAB CSVs from a previous run in `--output-dir`, or a year failed to parse |
| Default faults/model file not found | `examples/` is not in git; pass `--faults` and `--model` explicitly |
| Pressure fail with NaNs | MATLAB `griddata` returned NaN for a fault outside the hull |
