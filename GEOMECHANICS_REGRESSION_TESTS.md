# Geomechanics regression tests (MATLAB vs Python)

This is a **parity check** for FSP geomechanics. It runs the same stress field and faults through:

1. **Python production functions** — the same calls as the portal steps
   - Deterministic (Step 2): `calculate_absolute_stresses`, `calculate_slip_pressure`, `calculate_cff`, `calculate_scu`, `mohr_diagram_data_to_d3_portal`
   - Probabilistic (Step 3, optional): `run_geomechanics_mc` (`ComputeCriticalPorePressureForFailure`)
2. **MATLAB R2012b production kernel** — `mohrs_3D` (deterministic and MC)

Then it compares outputs and reports pass/fail.

Optional MC uses **shared samples**: Python `run_geomechanics_mc` draws them (production sampler, including its clamps), and the same rows are sent to MATLAB `mohrs_3D`.

The entry point is [`run_matlab_python_geomechanics_regression.py`](run_matlab_python_geomechanics_regression.py).

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
    mohrs_3D.m
  data entry functions/          (required only for A-Phi modes)
    getHorFromAPhi.m
```

If MATLAB is not at the default Windows path (`C:\Program Files\MATLAB\R2012b\bin\matlab.exe`), pass `--matlab-executable`.

## Input CSV

### Faults

Required columns: `FaultID`, `Strike`, `Dip`.

All faults use the same friction coefficient from `--friction-coefficient` (FSP 2.0).

## How to run

From the repo root, with the venv activated (or using the venv’s `python.exe`):

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_geomechanics_regression.py
```

That uses the default MATLAB code (in our testing environment it's in a `reference_old_code` directory), default MATLAB executable, default example faults, and the default stress field (gradients). Those will be different on your machine and you need to provide the corresponding flags to overwrite them with your files/setup.

### Point at MATLAB source on another machine

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_geomechanics_regression.py `
  --matlab-code C:\path\to\matlab_fsp_root
```

`--matlab_code` is the same flag.

### Other useful flags

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_geomechanics_regression.py `
  --matlab-executable "C:\Program Files\MATLAB\R2012b\bin\matlab.exe" `
  --matlab-code C:\path\to\matlab_fsp_root `
  --faults C:\path\to\faults.csv `
  --stress-mode gradients `
  --reference-depth 5000 `
  --vertical-stress 1.0 `
  --min-horizontal-stress 0.7 `
  --max-horizontal-stress 0.9 `
  --pore-pressure 0.45 `
  --max-stress-azimuth 60 `
  --friction-coefficient 0.6 `
  --output-dir C:\path\to\geomechanics_regression_output
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--matlab-code` / `--matlab_code` | MATLAB FSP **root** (must contain `technical code`) | `<repo>/reference_old_code` |
| `--matlab-executable` | `matlab.exe` | `C:\Program Files\MATLAB\R2012b\bin\matlab.exe` |
| `--faults` | Fault CSV | `examples/demo_texas_faults_fsp_100_variable_fsp.csv` |
| `--output-dir` | Where CSVs, HTML, and `summary.json` are written | `<repo>/geomechanics_regression_output` |
| `--stress-mode` | `gradients`, `aphi_min`, or `aphi_no_min` | `gradients` |
| `--reference-depth` | Depth (ft) | `11000` |
| `--vertical-stress` | Sv gradient (psi/ft) | `1.1` |
| `--min-horizontal-stress` | Shmin gradient (psi/ft) | `0.693` |
| `--max-horizontal-stress` | SHmax gradient (psi/ft) | `1.22` |
| `--pore-pressure` | Pp gradient (psi/ft) | `0.43` |
| `--max-stress-azimuth` | SHmax azimuth (deg CW from N) | `70` |
| `--aphi-value` | A-Phi in [0, 3]; used only for A-Phi modes | `1.22` |
| `--friction-coefficient` | Fault μ | `0.58` |
| `--poissons-ratio` | ν (both codes use 0.5 on the geo tab) | `0.5` |

Optional Monte Carlo: `--mc-iterations` (default `0`) plus `--mc-uncertainty-json` and `--random-seed`.

`--mc-uncertainty-json` is a JSON object of **plus/minus half-widths** in Python units. Keys:

- `vertical_stress_gradient_uncertainty`
- `min_horizontal_stress_uncertainty`
- `max_horizontal_stress_uncertainty`
- `initial_pore_pressure_gradient_uncertainty`
- `max_stress_azimuth_uncertainty`
- `strike_angles_uncertainty`
- `dip_angles_uncertainty`
- `friction_coefficient_uncertainty`
- `aphi_value_uncertainty`

Example:

```powershell
.\fsp_python_venv\Scripts\python.exe run_matlab_python_geomechanics_regression.py `
  --mc-iterations 100 `
  --mc-uncertainty-json '{"strike_angles_uncertainty": 10, "friction_coefficient_uncertainty": 0.05}'
```

Samples come from production `run_geomechanics_mc` (uniform around the nominal value, with that function’s clamps). Gradients are converted to absolute psi for MATLAB (`gradient × depth`). There is one sample row per (fault, simulation), in the same order Python returns.

## What it compares

- **PPF** — Step 2: delta pore pressure to slip (psi), clamped at 0 like the portal
- **CFF** — Step 2: Coulomb failure function, integer-rounded like the portal
- **SCU** — Step 2: `calculate_scu` (clamped at 1 like the portal)
- **Mohr arcs** — 100-point semicircles C1/C2/C3 (effective σ, τ)
- **Mohr fault points** — resolved `(σ, τ)` on each fault
- **Optional MC** — Step 3 `run_geomechanics_mc` slip pressure vs MATLAB `mohrs_3D` on the same samples, plus a per-fault CDF, if `--mc-iterations > 0`

Python and MATLAB share the same faults, the same nominal stress field, and (for MC) the same sample matrix.

`--stress-mode` maps to MATLAB `aphi.use`: `gradients` → 0, `aphi_no_min` → 11, `aphi_min` → 12.

## Outputs

Written under `--output-dir`:

| File | Contents |
|------|----------|
| `fault_metrics_comparison.csv` | Fault ID, Python/MATLAB PPF, CFF, SCU, σ, τ, errors |
| `mohr_arcs_comparison.csv` | Circle id, point index, Python/MATLAB x/y, errors |
| `mohr_fault_points_comparison.csv` | Fault ID, Python/MATLAB σ/τ, errors |
| `mohr_side_by_side.html` | Plotly Mohr diagrams (circles + fault points + μ-line) |
| `summary.json` | Pass/fail, max relative error, inputs (including `matlab_code`), output paths |
| `matlab_geomechanics_regression_driver.m` | Generated MATLAB driver (for debugging) |

If Monte Carlo ran: `mc_slip_pressure_comparison.csv`, `mc_cdf_comparison.csv`.

## Pass / fail

**PPF, CFF, Mohr x/y, σ, τ (psi)**

- If `|MATLAB| ≥ 1 psi`: relative error must be **≤ 1%**
- If `|MATLAB| < 1 psi`: absolute error must be **≤ 0.01 psi**

**SCU (dimensionless)**

- If `|MATLAB| ≥ 0.01`: relative error must be **≤ 1%**
- If `|MATLAB| < 0.01`: absolute error must be **≤ 1e-4**

Every comparison table must pass. The process exits **0** on pass and **1** on fail. `summary.json` has `"passed": true/false` and `"max_relative_error"`.

## Common failures

| Symptom | Likely cause |
|---------|----------------|
| `MATLAB technical code was not found` | `--matlab-code` is not the MATLAB FSP **root**, or `technical code/` is missing |
| `mohrs_3D.m` missing | Incomplete MATLAB tree |
| `getHorFromAPhi.m` missing | A-Phi mode needs `data entry functions/` |
| `MATLAB R2012b was not found` | Wrong `--matlab-executable`, or R2012b is not installed |
| MATLAB did not produce all outputs | MATLAB failed to run the generated `.m` driver; inspect `matlab_run.log`, `matlab_error.txt`, and `matlab_geomechanics_regression_driver.m` |
| Fault CSV is missing columns | File needs `FaultID`, `Strike`, `Dip` |
| Length mismatch on comparison | Stale MATLAB CSVs from a previous run in `--output-dir` (the script deletes them at the start of a run; use a fresh `--output-dir` if this persists) |
| Default faults file not found | `examples/` is not in git; pass `--faults` explicitly |
| Large MC error | Python Step 3 uses Josimar (`ComputeCriticalPorePressureForFailure`); MATLAB MC uses quadratic `mohrs_3D` |
