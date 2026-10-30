# Fault Slip Potential (FSP) 4.0

### A Program for Probabilistic Estimation of Fault Slip Potential Resulting from Fluid Injection

---

### Overview

FSP is a tool developed to screen faults near injection wells and estimate the potential for fault slip induced by fluid injection. It uses deterministic and probabilistic methods to estimate the cumulative probability of a fault slipping due to increased pore pressure.

### Features

- **Probabilistic and Deterministic Models**: FSP provides both deterministic and Monte Carlo probabilistic approaches for estimating fault slip.
- **Input Parameters**: Supports input of fault strike, dip, well locations, injection rates, hydrologic parameters, and mechanical stress state parameters.
- **Hydrology Model**: Includes a simplified radial flow hydrology model, allowing users to relate injection to pore pressure changes.
- **Monte Carlo Simulations**: Uses Monte Carlo analysis to calculate probabilities of fault slip as a function of pore pressure increase.

### How It Works

1. **Mohr-Coulomb Slip Criteria**: FSP calculates pore pressure to slip on each fault using deterministic geomechanical modeling.
2. **Monte Carlo Analysis**: Probabilistic simulations are run to yield the probability of each fault slipping, considering uncertainties in input parameters.
3. **Hydrology Model**: The hydrology model assesses specific injection scenarios, providing pore pressure changes to be used in the fault slip analysis.
4. **Results**: FSP produces visual outputs including Mohr diagrams, fault maps, and cumulative distribution functions (CDF) of fault slip probability.

### Requirements

- **Python:** 3.14.8 (Windows x64)
- **Packages:** `requirements.txt` pins the direct dependencies; `requirements-lock.txt` is the full `pip freeze` of the venv.

The venv is named `fsp_python_14_venv` and lives at the repo root (it is git-ignored).

```powershell
py -3.14 -m venv fsp_python_14_venv
.\fsp_python_14_venv\Scripts\python.exe -m pip install --upgrade pip
.\fsp_python_14_venv\Scripts\python.exe -m pip install -r requirements-lock.txt

# Run the tests from the repo root
.\fsp_python_14_venv\Scripts\python.exe -m pytest
```

After upgrading any package, regenerate the lock with
`.\fsp_python_14_venv\Scripts\python.exe -m pip freeze > requirements-lock.txt`.


### Licensing

FSP 3 is licensed under the **BSD-3-Clause**. Please refer to the `LICENSE` file for more details.


---


