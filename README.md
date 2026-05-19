# ARIES

**A**daptive **R**esidual-based **I**terative **E**nsemble **Smoother**

Ensemble-based Bayesian inference for hydrological model calibration,
descended from
[CWIEKI](https://doi.org/10.1088/1361-6420/ad05df)
(Botha et al., 2023, *Inverse Problems* 39 125014).

## Install

```bash
pip install git+https://github.com/frbennett/aries.git
```

Or locally:

```bash
git clone https://github.com/frbennett/aries.git
cd aries
pip install -e .
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas. No MCMC framework needed.

## Quick start

```python
from aries import esmda

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

## Documentation

Full docs at **[frbennett.github.io/aries](https://frbennett.github.io/aries)**

- [API reference](https://frbennett.github.io/aries/api/core/) — esmda class parameters
- [Noise estimation](https://frbennett.github.io/aries/api/noise/) — laplace, mcmc, grid, residual
- [Examples](https://frbennett.github.io/aries/examples/sacsma/) — SAC-SMA, GR4J, CWIEKI
- [Theory](https://frbennett.github.io/aries/reference/theory/) — ES-MDA, tempering, bias sources

## Citation

> Bennett, F.R. (2025) ARIES vs SMC for hydrological model calibration.
> *Modeling Earth Systems and Environment*. (submitted)

> Botha, I., Adams, M.P., Frazier, D., Tran, D.K., Bennett, F.R.,
> & Drovandi, C. (2023) Component-wise iterative ensemble Kalman
> inversion for static Bayesian models with unknown measurement
> error covariance. *Inverse Problems* **39**, 125014.
