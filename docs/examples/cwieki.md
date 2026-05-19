# Example: CWIEKI adaptive tempering

Full CWIEKI calibration — MCMC noise estimation (slice sampler) with
ESS-adaptive likelihood tempering, as described in Botha et al. (2023).

```python
import numpy as np
import pandas as pd
from aries import esmda
from joblib import Parallel, delayed

# ── Model runner (same as SAC-SMA example) ──
...

# ── Run CWIEKI ──
solver = esmda(
    parameter_file_name="es_parameters.csv",
    observation_file_name="es_data.csv",
    nEnsemble=1000,
    maxIter=30,                    # safety cap — actual steps determined by ESS
    error={"FLOW": 10.0},
    job_name="cwieki_calibration",
    inversion_type="efast_subspace",
    calculation_type="ikea",       # adaptive noise
    phi_update="mcmc",             # slice-sampling MCMC for σ
    inflation_schedule="ess",      # adaptive tempering via ensemble ESS
    target_ess=0.5,                # maintain ≥ 50% effective particles
)
solver.run_esmda(fill_ensemble)
```

## How it works

1. **θ update**: IEKI Kalman step with inflation factor 1/Δα, where Δα is
   the tempering step size.
2. **φ update**: Per-ensemble-member slice-sampling MCMC for σ (replaces
   PyMC/NUTS — zero dependencies).
3. **α adaptation**: At each iteration, binary search finds the largest
   Δα such that the ensemble ESS stays above `target_ess × nEnsemble`.
4. **Stopping**: The algorithm stops when α = 1 (full posterior reached),
   or when `maxIter` safety cap is hit.

## Computational cost

CWIEKI requires more iterations than standard ARIES (25 vs 12 for
SAC-SMA real data) because the ESS target constrains the tempering step
sizes to be smaller. Each iteration also runs 50+50 MCMC draws per
ensemble member for σ. The total cost is dominated by model evaluations
(each iteration evaluates the full 1000-member ensemble), not by the
MCMC overhead.

## Comparison with ARIES

| | ARIES | CWIEKI |
|---|---|---|
| Noise estimation | Laplace approximation | Slice-sampling MCMC |
| Tempering | Fixed schedule (12 steps) | ESS-adaptive (25 steps) |
| Parameter accuracy (mean |z| vs SMC) | 1.29 | 0.84 |
| Predictive performance | Identical | Identical |
| Runtime | 1× (baseline) | ~12× (dominated by extra model evals) |
