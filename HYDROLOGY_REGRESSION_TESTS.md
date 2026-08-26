# Hydrology regression tests (MATLAB vs Python)

This is a **parity check** for FSP hydrology. It runs the same injection scenario through:

1. **Python** — `calcST` + `pressureScenario_Rall` in this repo
2. **MATLAB R2012b** — `calcST` + `pfront` from the legacy FSP MATLAB tree

Then it compares pore pressure and reports pass/fail.

It tests the core Theis hydrology kernel.

The entry point is [`run_matlab_python_hydrology_regression.py`](run_matlab_python_hydrology_regression.py).

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
  technical code/
    pfront.m
    calcST.m
```

If MATLAB is not at the default Windows path (C:\Program Files\MATLAB\R2012b\bin\matlab.exe) , pass `--matlab-executable`.

## Input CSVs

### Faults

Required columns: `FaultID`, `Latitude(WGS84)`, `Longitude(WGS84)`.

### Wells (monthly injection rates)

Required columns:

- `WellID`
- `Latitude(WGS84)`
- `Longitude(WGS84)`
- `Year`
- `Month`
- `InjectionRate(bbl/month)`

## How to run

From the repo root, with the venv activated (or using the venv’s `python.exe`):

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_hydrology_regression.py
```

That uses the default MATLAB code (`reference_old_code`), default MATLAB executable, and default example CSVs that we use during testing. Those will be different in your machine and you need to provide the corresponding flags to overwrite them with your files/setup.

### Point at MATLAB source on another machine

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_hydrology_regression.py `
  --matlab-code C:\path\to\matlab_fsp_root
```

`--matlab_code` is the same flag.

### Other useful flags

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_hydrology_regression.py `
  --matlab-executable "C:\Program Files\MATLAB\R2012b\bin\matlab.exe" `
  --matlab-code C:\path\to\matlab_fsp_root `
  --faults C:\path\to\faults.csv `
  --wells C:\path\to\monthly_wells.csv `
  --year 2031 `
  --output-dir C:\path\to\hydrology_regression_output
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--matlab-code` / `--matlab_code` | MATLAB FSP **root** (must contain `technical code`) | `<repo>/reference_old_code` |
| `--matlab-executable` | `matlab.exe` | `C:\Program Files\MATLAB\R2012b\bin\matlab.exe` |
| `--faults` | Fault CSV | `examples/demo_texas_faults_fsp_100_variable_fsp.csv` |
| `--wells` | Monthly wells CSV | `examples/demo_texas_injection_wells_monthly_fsp_20wells_variable_fsp.csv` |
| `--year` | Evaluation year (pressure at end of previous calendar year) | `2031` |
| `--well-id` | Well used for the radial (distance) plot | First active well |
| `--output-dir` | Where CSVs, HTML, and `summary.json` are written | `<repo>/hydrology_regression_output` |

Hydrology parameters (`--porosity-fraction`, `--aquifer-thickness-ft`, `--permeability-md`, fluid/rock properties) have the same defaults as a typical demo run. Porosity is a **fraction** in Python (for example `0.10`); the script converts it to percent for MATLAB.

Optional Monte Carlo: `--mc-iterations` (default `0`) plus `--mc-uncertainty-json` and `--random-seed`.

## What it compares

- **Radial pressure** — pressure vs distance (0.1–20 km) from one well
- **Fault pressure** — summed pressure at each fault from all wells
- **Optional MC** — per-sample fault pressures and a CDF comparison, if `--mc-iterations > 0`

Python and MATLAB share the same map origin (mean lat/lon), the same monthly-to-step-rate series, and the same evaluation time.

## Outputs

Written under `--output-dir`:

| File | Contents |
|------|----------|
| `radial_pressure_comparison.csv` | Distance, Python/MATLAB pressure, errors |
| `fault_pressure_comparison.csv` | Fault ID, Python/MATLAB pressure, errors |
| `pressure_distance_side_by_side.html` | Plotly radial curves |
| `summary.json` | Pass/fail, max relative error, inputs (including `matlab_code`), output paths |
| `matlab_hydrology_regression_driver.m` | Generated MATLAB driver (for debugging) |

If Monte Carlo ran: `mc_fault_pressure_comparison.csv`, `mc_cdf_comparison.csv`.

## Pass / fail

- If MATLAB pressure is **≥ 1 psi**: relative error must be **≤ 1%**
- If MATLAB pressure is **< 1 psi**: absolute error must be **≤ 0.01 psi**

Every comparison table must pass. The process exits **0** on pass and **1** on fail. `summary.json` has `"passed": true/false` and `"max_relative_error"`.

## Common failures

| Symptom | Likely cause |
|---------|----------------|
| `MATLAB technical code was not found` | `--matlab-code` is not the MATLAB FSP **root**, or `technical code/` is missing |
| `pfront.m` / `calcST.m` missing | Incomplete MATLAB tree |
| `MATLAB R2012b was not found` | Wrong `--matlab-executable`, or R2012b is not installed |
| MATLAB did not produce all outputs | MATLAB failed to run the generated `.m` driver; inspect stdout/stderr and `matlab_hydrology_regression_driver.m` |
| Monthly wells CSV is missing columns | File is not monthly FSP format |
| Length mismatch on comparison | Stale MATLAB CSVs from a previous run in `--output-dir` (the script deletes them at the start of a run; use a fresh `--output-dir` if this persists) |
| Default faults/wells file not found | `examples/` is not in git; pass `--faults` and `--wells` explicitly |
