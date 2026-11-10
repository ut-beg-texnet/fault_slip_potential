# FSP regression: Python vs legacy MATLAB

`run_fsp_matlab_regression.py` runs the current Python FSP code (`src/fsp`) on the webtool (WEB) input files and compares the results with saved legacy MATLAB FSP results for the same inputs. It needs no MATLAB. The test design is in `FSP Documentation Regression Test.pptx`.

## Folder layout

```
regression_testing/
  run_fsp_matlab_regression.py
  requirements-regression.txt
  FSP Documentation Regression Test.pptx
  Input Data/
    Fault Data/
      WEB/      FSP_testing_faults_data_WEB_{NORMAL,REVERSE,STRIKESLIP}.csv      lat/lon, what Python runs on
      MATLAB/   FSP_testing_faults_data_MATLAB_{NORMAL,REVERSE,STRIKESLIP}.csv   km (UTM 13N), what MATLAB ran on
    Injection Data/
      WEB/      Annual Injection on 4 Wells/, Monthly Injection on 1 Well/
      MATLAB/   Annual Injection on 4 Wells/, Monthly Injection on 1 Well/
  MATLAB Results/
    Annual Injection on 4 Wells/{Normal,Reverse,StrikeSlip}/
    Monthly Injection on 1 Well/{Normal,Reverse,StrikeSlip}/
  output/                                                                     written by the script
```

The WEB and MATLAB input files describe the same faults and wells. The MATLAB km coordinates are the WEB lat/lons projected to UTM zone 13N (EPSG:32613).

Each case folder under `MATLAB Results/` holds one saved MATLAB session and, optionally, MATLAB exports:

| File | What it is | Used by |
|---|---|---|
| `FSP_<Regime>_<Annual\|Monthly>_MATLAB` (any `FSP_*_MATLAB` name that contains the regime and Annual/Monthly) | The saved MATLAB session (MATLAB v5 `.mat`, no extension) | all checks |
| `FSPvalues_*.csv` | Displayed FSP per fault (`Fault #1, 0.36 FSP`) | G, H |
| `CDF_*.csv` | Probabilistic geomechanics CDF export (pore pressure to slip at probabilities 0, 0.001, …, 1) | F |
| `PP2SLIP_*.jpg` | Screenshot of the deterministic pore pressure to slip map | not used by any check |

Export them from the same MATLAB run as the saved session: re-running an analysis before exporting changes the Monte Carlo draws, and then the exports no longer match the session.

To replace a case, swap the files in its folder. Each case must have exactly one session; the script stops if it finds two.

## Run

From the repository root:

```powershell
.\fsp_python_14_venv\Scripts\python.exe -m pip install -r regression_testing\requirements-regression.txt   # once
.\fsp_python_14_venv\Scripts\python.exe regression_testing\run_fsp_matlab_regression.py
```

It takes about a minute for all six cases. The exit code is 0 only when checks A, B1, B2 and G (where available) pass for every case.

| Flag | Default | Meaning |
|---|---|---|
| `--cases` | `all` | Substrings of case or session file names to run, e.g. `--cases Reverse Monthly`. The run's summary and graphs then cover only those cases. |
| `--reference-dir` | `MATLAB Results` | Searched recursively for `FSP_*_MATLAB` sessions |
| `--inputs-dir` | `Input Data` | The WEB and MATLAB input CSVs |
| `--out` | `output` | Base folder; each run writes into a new `regression_results_DDMMYYYY_HHMMSS` folder inside it |
| `--epsg` | `32613` | Projected CRS of MATLAB's km coordinates (UTM 13N), used to compare positions in check I |
| `--seed` | `200` | Seed for Python's own Monte Carlo runs (checks D, E; H starts here) |
| `--fsp-seeds` | `30` | Number of Python Monte Carlo repetitions in check H (0 skips it) |
| `--geo-iterations` | `1000` | Python geomechanics MC draws |
| `--hydro-iterations` | MATLAB's count | Python hydrology MC draws |
| `--schedule-days` | `1` | A passes when no well's daily rate differs on more than this many days |
| `--kernel-rtol` / `--kernel-atol` | `0.002` / `0.01` psi | B1/B2 pass when \|Δ\| ≤ 0.01 psi + 0.2% of MATLAB's pressure, at every sim and fault |
| `--geometry-rtol` | `0.05` | C warns above a 5% pressure difference |
| `--fsp-tol` | `0.01` | G passes when every displayed FSP is within 0.01 of MATLAB's |
| `--no-plots` | off | Skip PNGs |

## What a MATLAB session contains

- **Inputs:** stress (A-Phi with Shmin, `aphi.use = 12`), 50 faults in km, wells, hydrology, and all uncertainties.
- **Geomechanics Monte Carlo input samples:** 1000 per fault. MATLAB saves no slip pressures.
- **Hydrology Monte Carlo input samples:** 200 sims. Thickness is stored in **metres**.
- **One MATLAB output:** `probHydrology.outsAllFaults`, the pressure in psi for each sim and fault at the hydrology year.

Deterministic geomechanics, the geomechanics CDF and the final FSP are not in the session; they come from the exports above. Checks F, G and H show `n/a` for a case without them.

## Checks

| | Gates exit | What it compares |
|---|---|---|
| **I** | no (WARN) | The WEB input files vs what the MATLAB session ran with: which MATLAB fault file the session matches, strike/dip/length, fault and well positions (WEB lat/lon projected to `--epsg`), and the injection columns. |
| **A** | yes | Python well loader (`normalize_wells_to_well_data`) vs MATLAB's saved rate schedule, day by day |
| **B1** | yes | Python Theis kernel with MATLAB's schedule, distances and samples, vs `outsAllFaults`. It also identifies MATLAB's evaluation instant (Dec 31 of Y−1 or Jan 1 of Y). |
| **B2** | yes | Python hydrology pipeline (its own well schedule and year convention) with MATLAB's distances and samples. Runs both the scipy and the numba Theis paths. |
| **C** | no (WARN) | Same as B2, but with production geometry: WEB lat/lon → haversine. Isolates the effect of the distance calculation. |
| **D** | no | Python's own hydrology Monte Carlo vs MATLAB's pressure distribution per fault (KS test, medians, P10/P90) |
| **E** | no | Python's geomechanics Monte Carlo input sampling vs MATLAB's saved samples (8 parameters, KS test). Dip is compared after Python's folding into [0, 90]. |
| **F** | no | Python's slip pressure for each of MATLAB's saved geomechanics draws vs the CDF export. Exact if the export came from the same Monte Carlo run as the session; otherwise a KS test per fault. |
| **G** | yes | Python's FSP code (`probabilistic_fsp.py`) fed MATLAB's geomechanics draws and MATLAB's hydrology pressures, vs the displayed FSP export (2 decimals). Also reported with Python's hydrology from B2 and C. |
| **H** | no | Python's own Monte Carlo, end to end, repeated over `--fsp-seeds` seeds. A fault is flagged when MATLAB's FSP is outside Python's mean ± 3σ (+0.005 display rounding). Shows whether a webtool-vs-MATLAB FSP difference is Monte Carlo noise. |

## Tolerances

The script is meant to be re-run regularly, so the tolerances allow for tiny differences between the two languages and catch real changes:

| Check | Passes when |
|---|---|
| A injection schedule | No well's daily rate differs on more than 1 day (MATLAB and Python put an annual well's last day 1 day apart) |
| B1/B2 pressure | Every pressure is within 0.01 psi + 0.2% of MATLAB's (the known unit and date differences are about 0.03%) |
| G fault slip potential | Every displayed FSP is within 0.01 of MATLAB's (one display step, so a value on a rounding boundary can't fail) |
| C pressure with WEB lat/lon | Warns above 5% (distance-method difference is up to about 3%) |
| I inputs | Warns when a position is more than 1 m off, or strike/dip/length/rates differ |
| H fault slip potential, independent runs | Flags a fault outside Python's mean ± 3 standard deviations over 30 runs + 0.005 |

D, E and F report KS-test counts (p < 0.01) only.

## Output

Each run writes into its own folder, `output/regression_results_<DDMMYYYY_HHMMSS>/` (date and time the run started), so earlier results are never overwritten:

- `presentation/`: six plain-language comparison graphs of MATLAB vs Python (web tool). Each combined graph puts the three fault types (normal, reverse, strike-slip) on one set of axes:
  - `00_overview.png`: one slide with every case, in four columns (injection schedule, pressure, fault slip potential with shared samples, with independent runs)
  - `01_fsp_annual.png` and `02_fsp_monthly.png`: FSP per fault, MATLAB vs Python with MATLAB's samples (check G). Vertical bars show the range of Python's own Monte Carlo runs (check H). A fault type without an `FSPvalues` export is left out and named in the subtitle. The graph is skipped when no case in the group has that export.
  - `03_pressure_annual.png` and `04_pressure_monthly.png`: pressure at every fault for every MATLAB hydrology sample, MATLAB vs Python's pipeline (check B2)
  - `05_fsp_map.png`: fault maps colored by FSP (web tool color scale), MATLAB on the top row and Python on the bottom row, one column per case with an `FSPvalues` export
- `summary/summary.md` and `summary.json`: the status table, session files used, the kernel scale factor, the evaluation-instant details and per-case notes.
- `<case>/` for each case (technical detail):
  - `inputs/`: the fault and well CSVs Python was run on (copies of the WEB files, ready to upload to the webtool)
  - CSVs with every compared number

## Known MATLAB vs Python differences

- **Pa → psi:** MATLAB converts with ×1e-5 × 14.5 = ×1.45e-4 (`pfront.m`, `hydro1D.m`); Python uses 1/6894.76. Python pressures are 1.00026× MATLAB's.
- **Evaluation instant:** MATLAB evaluates hydrology at Jan 1 of the year of interest; Python at Dec 31 of the year before. With the factor above, this explains B1/B2.
- **Annual shut-in:** MATLAB shuts an annual well in on Jan 1 of `EndYear`; Python on Dec 31 of `EndYear − 1` (check A, one day per well).
- **Year cap:** Python caps the year of interest at the last injection data year + 3. With "Extrapolate Injection?" ticked, MATLAB extends the last monthly rate (to 2045 here) and accepts later years. When MATLAB's year is past Python's cap, B2, C, D and H compare different years (the summary notes when this happens).
- **Distances:** MATLAB uses straight-line distance on the km grid, Python haversine on a sphere (R = 6371 km); they differ by up to 0.28% here (check C).
- **Stress sampling:** MATLAB draws stress parameters independently per fault; Python shares one draw per simulation across faults. Per-fault distributions are the same.
- "Extrapolate Injection?" only affects monthly data. MATLAB's `pfront.m` always shuts a well in after its last saved schedule row; any extrapolation is already in the saved rows.

## Notes on the current MATLAB results

- Every case evaluates 2052 (annual) or 2032 (monthly). The monthly sessions were run with "Extrapolate Injection?" off.
- The Normal monthly case (session, CDF and FSP exports) was replaced with a full MATLAB re-run on 10 Nov 2026.
- The Reverse monthly CDF export matches its session exactly. The other CDF exports come from a different MATLAB Monte Carlo run than their sessions, so F compares them statistically.
- All six cases have an FSP export, and each one matches its session (G passes).
