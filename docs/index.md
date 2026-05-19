# ARIES

**A**daptive **R**esidual-based **I**terative **E**nsemble **S**moother

Ensemble-based Bayesian inference for hydrological model calibration.
ARIES descends from
[CWIEKI](https://doi.org/10.1088/1361-6420/ad05df)
(Botha et al., 2023, *Inverse Problems* 39 125014),
replacing the component-wise MCMC noise update with a Laplace
approximation — gaining orders of magnitude in speed while
maintaining predictive accuracy.

## Quick start

```python
from aries import esmda

def fill_ensemble(parameter_set, n_ensemble, n_params, n_obs):
    """Run your model for each ensemble member; return (n_obs, n_ensemble) array."""
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

## Key features

| Feature | Description |
|---|---|
| **Laplace noise estimation** | Newton–Raphson mode-finding on the exact posterior of σ, with HalfNormal prior regularisation |
| **Slice-sampling MCMC** | Optional full MCMC for σ per ensemble member — zero PyMC dependency |
| **ESS-adaptive tempering** | CWIEKI-style likelihood tempering with adaptive step size from ensemble ESS |
| **Six inversion types** | SVD, subspace, fast-subspace, efast-subspace, Dask, and standard ES-MDA |
| **Parallel model execution** | joblib-based parallelisation — embarrassingly parallel iterations |

## Three noise estimation methods

```python
# Default: Laplace approximation (fast, includes prior regularisation)
solver = esmda(phi_update="laplace")

# Full MCMC: slice-sampling for σ per ensemble member
solver = esmda(phi_update="mcmc")

# Grid: exact posterior mean via fine-grid integration
solver = esmda(phi_update="grid")

# Residual: std of Y - G (fastest, no prior)
solver = esmda(phi_update="residual")
```

## Two tempering schedules

```python
# Fixed: standard ES-MDA schedule (12 equal steps)
solver = esmda(inflation_schedule="fixed", maxIter=12)

# ESS: CWIEKI-style adaptive (steps chosen to maintain ensemble ESS target)
solver = esmda(inflation_schedule="ess", target_ess=0.5, maxIter=30)
```

## Citation

If you use ARIES in published work, please cite:

> Bennett, F.R. (2025) ARIES vs SMC for hydrological model calibration.
> *Modeling Earth Systems and Environment*. (submitted)

And the underlying method:

> Botha, I., Adams, M.P., Frazier, D., Tran, D.K., Bennett, F.R.,
> & Drovandi, C. (2023) Component-wise iterative ensemble Kalman
> inversion for static Bayesian models with unknown measurement
> error covariance. *Inverse Problems* **39**, 125014.
