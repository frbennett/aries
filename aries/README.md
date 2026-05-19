# ARIES

**A**daptive **R**esidual-based **I**terative **E**nsemble **S**moother

Ensemble-based Bayesian inference for hydrological model calibration. ARIES
descends from CWIEKI (Botha et al., 2023, *Inverse Problems* 39 125014),
replacing the component-wise MCMC noise update with a Laplace approximation
— gaining orders of magnitude in speed while maintaining predictive accuracy.

## Quick start

```python
from aries import esmda

# Build your ensemble runner
def fill_ensemble(parameter_set, n_ensemble, n_params, n_obs):
    """Run your model for each ensemble member, return (n_obs, n_ensemble) array."""
    ...

solver = esmda(
    parameter_file_name="es_parameters.csv",
    observation_file_name="es_data.csv",
    nEnsemble=1000,
    maxIter=12,
    inversion_type="efast_subspace",
    calculation_type="ikea",
)
solver.run_esmda(fill_ensemble)
```

## Installation

```bash
pip install git+https://github.com/frbennett/aries.git
```

Or from a local checkout:

```bash
git clone https://github.com/frbennett/aries.git
cd aries
pip install -e .
```

## Parameter files

**`es_parameters.csv`** — prior definitions for each model parameter:

| parameter | lower | upper | mean | width |
|-----------|-------|-------|------|-------|
| Uztwm     | 1.0   | 150.0 | 75.0 | 6     |
| Uzfwm     | 1.0   | 150.0 | 75.0 | 6     |
| ...       | ...   | ...   | ...  | ...   |

**`es_data.csv`** — observation data:

| label | Y | noise | group |
|-------|---|-------|-------|
| FLOW_1 | 12.3 | 0.54 | FLOW |
| FLOW_2 | 11.8 | 0.54 | FLOW |
| ... | ... | ... | ... |

## Phi update methods

| `phi_update` | Description |
|---|---|
| `"laplace"` | **(default)** Laplace approximation in log(σ) space — Newton–Raphson mode-finding. Fast, includes prior regularisation. |
| `"residual"` | Residual std: σ = std(Y − G). Fastest, no prior regularisation. |
| `"mcmc"` | Slice-sampling MCMC for σ per group (replaces PyMC/NUTS). |
| `"grid"` | Exact posterior on a fine log-space grid. |

## Inflation schedules

| `inflation_schedule` | Description |
|---|---|
| `"fixed"` | **(default)** Standard ES-MDA schedule. `maxIter` pre-determined steps. |
| `"ess"` | CWIEKI-style adaptive likelihood tempering. Step sizes chosen to maintain a target ensemble ESS. Number of iterations determined by the data. |

With `inflation_schedule="ess"`:

```python
solver = esmda(
    ...,
    inflation_schedule="ess",
    target_ess=0.5,    # keep ≥ 50% of ensemble effective per step
)
```

## Inversion types

| `inversion_type` | Description |
|---|---|
| `"efast_subspace"` | **(default)** Efficient subspace pseudo-inverse with randomised ϕ. |
| `"svd"` | Full SVD per ensemble member (robust, slower). |
| `"fast_subspace"` | Subspace method with per-member noise. |
| `"subspace"` | Original subspace method. |
| `"esmda"` | Standard ES-MDA Kalman update (vectorised). |
| `"dask"` / `"esmda_dask"` | Dask-based for out-of-core ensembles. |

## Diagnostics output

```
═══ Iteration 3 ═══
  Params: 30.53  46.58  … (14 params)
  Inversion: efast_subspace  |  Time: 10.14s
  φ = 0.5791
  α = 0.7427  Δα = 0.3713  Ensemble ESS(tgt) = 500
```

With `phi_update="mcmc"`:

```
  MCMC:
    FLOW: Chain ESS 37 (min 15), φ = 0.5791 ± 0.0280
```

## Submodules

```python
from aries import noise, linalg, metrics

# Noise estimation functions
noise.covariance_matrix(D, data, error, ...)
noise.ess(samples)           # MCMC effective sample size
noise.build_prior(params, n_ensemble)
noise.inv_trans(params, M)   # inverse logit transform

# Linear algebra
linalg.efast_inverse(M, Cmd, Duc, D, del_D, phi, alpha, Ne)

# Metrics
metrics.PICP(posterior_path, data_path, CI)   # coverage probability
```

## Citation

If you use ARIES in published work, please cite the paper:

> Bennett, F.R. (2025) ARIES vs SMC for hydrological model calibration.
> *Modeling Earth Systems and Environment*. (submitted)

And the underlying method:

> Botha, I., Adams, M.P., Frazier, D., Tran, D.K., Bennett, F.R., &
> Drovandi, C. (2023) Component-wise iterative ensemble Kalman inversion
> for static Bayesian models with unknown measurement error covariance.
> *Inverse Problems* 39, 125014.

## License

MIT
