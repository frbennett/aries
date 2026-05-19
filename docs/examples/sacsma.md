# Example: SAC-SMA calibration

Full example calibrating the 14-parameter Sacramento Soil Moisture
Accounting model to the St.~Helens catchment (Queensland, Australia).

```python
import numpy as np
import pandas as pd
from aries import esmda
from joblib import Parallel, delayed

# ── Model runner ──
# (assumes sacsma is available — Numba JIT-compiled)
from sacramento import sacsma

def run_model(parameter_set):
    """Run SAC-SMA and return discharge in transformed space."""
    flow_mm = sacsma(precip, pet, parameter_set)
    flow_cumecs = (flow_mm * area / 1000) / (3600 * 24)
    return flow_cumecs ** 0.5   # power transform λ = 0.5

# ── Ensemble filler ──
def fill_ensemble(m, nEnsemble, mLength, dLength):
    y = np.zeros([dLength, nEnsemble])
    jobs = [delayed(run_model)(m[:, j]) for j in range(nEnsemble)]
    results = Parallel(n_jobs=8, backend="threading")(jobs)
    for j in range(nEnsemble):
        y[:, j] = results[j]
    return y

# ── Run ──
solver = esmda(
    parameter_file_name="es_parameters.csv",    # 14 SAC-SMA params
    observation_file_name="es_data.csv",        # observed discharge
    nEnsemble=1000,
    maxIter=12,
    error={"FLOW": 10.0},
    job_name="sacsma_calibration",
    inversion_type="efast_subspace",
    calculation_type="ikea",                    # adaptive noise
    phi_update="laplace",                       # Newton-Raphson σ estimate
    inflation_schedule="fixed",                 # standard ES-MDA
)
solver.run_esmda(fill_ensemble)
```

## Output

After completion, the job directory contains:

| File | Contents |
|---|---|
| `*_11_parameters.csv` | Final parameter ensemble (14 params × 1000 members) |
| `*_11_data.csv` | Ensemble predictions (14,610 timesteps × 1000 members) |
| `*_final_phi.csv` | Per-observation noise standard deviations |

## Diagnostics

With `phi_update="mcmc"`:

```
═══ Iteration 3/12 ═══
  Params: 14.10  47.21  … (14 params)
  Inversion: efast_subspace  |  Time: 10.14s
  φ = 0.5791
  MCMC:
    FLOW: Chain ESS 37 (min 15), φ = 0.5791 ± 0.0280
```

With `inflation_schedule="ess"`:

```
═══ Iteration 3 ═══
  Params: 14.10  47.21  … (14 params)
  Inversion: efast_subspace  |  Time: 10.14s
  φ = 0.5791
  α = 0.7427  Δα = 0.3713  Ensemble ESS(tgt) = 500
```
