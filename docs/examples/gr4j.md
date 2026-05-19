# Example: GR4J comparison

The 4-parameter GR4J model (Perrin et al., 2003) provides a simpler
test case with substantially less equifinality than SAC-SMA.

```python
import numpy as np
import pandas as pd
from aries import esmda
from joblib import Parallel, delayed
from gr4j import gr4j   # Numba JIT-compiled

def run_gr4j(parameter_set):
    """Run GR4J and return discharge in transformed space."""
    flow_mm = gr4j(precip, pet, parameter_set, warmup_days=365)
    flow_cumecs = (flow_mm * area / 1000) / (3600 * 24)
    return flow_cumecs ** 0.5   # λ = 0.5

def fill_ensemble(m, nEnsemble, mLength, dLength):
    y = np.zeros([dLength, nEnsemble])
    jobs = [delayed(run_gr4j)(m[:, j]) for j in range(nEnsemble)]
    results = Parallel(n_jobs=8, backend="threading")(jobs)
    for j in range(nEnsemble):
        y[:, j] = results[j]
    return y

solver = esmda(
    parameter_file_name="es_parameters_gr4j.csv",  # 4 params
    observation_file_name="es_data.csv",
    nEnsemble=1000,
    maxIter=12,
    error={"FLOW": 10.0},
    job_name="gr4j_calibration",
    inversion_type="efast_subspace",
    calculation_type="ikea",
    phi_update="laplace",
)
solver.run_esmda(fill_ensemble)
```

## Parameter file for GR4J

**`es_parameters_gr4j.csv`**:

| parameter | lower | upper | mean | width |
|---|---|---|---|---|
| x1 | 100.0 | 1200.0 | 350.0 | 4 |
| x2 | −5.0 | 3.0 | 0.0 | 4 |
| x3 | 20.0 | 300.0 | 90.0 | 4 |
| x4 | 1.1 | 3.9 | 2.0 | 4 |

x1–x4 correspond to the production store capacity, groundwater
exchange coefficient, routing store capacity, and unit hydrograph
time base, respectively.

## Comparison with SAC-SMA

| | SAC-SMA | GR4J |
|---|---|---|
| Parameters | 14 | 4 |
| Equifinality | Severe (mean |z| = 1.29) | Mild |
| Posterior width (CV) | 3–40% | 2–5% |
| Predictive skill (NSE) | 0.868 | 0.868 |
