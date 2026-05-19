# Installation

```bash
pip install git+https://github.com/frbennett/aries.git
```

Or from a local checkout:

```bash
git clone https://github.com/frbennett/aries.git
cd aries
pip install -e .
```

## Dependencies

ARIES requires Python ≥ 3.10 and:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Array operations |
| `scipy` | ≥ 1.7 | Linear algebra, SVD, statistics |
| `pandas` | ≥ 1.3 | Data I/O for parameter and observation files |

No MCMC framework (PyMC, Stan, etc.) is required. The optional `mcmc` phi-update
uses a bespoke slice sampler written in ~50 lines of NumPy.

## Optional dependencies

| Package | Purpose |
|---|---|
| `joblib` | Parallel model execution across CPU cores |
| `matplotlib` | Visualisation of diagnostics |
| `dask` | Out-of-core ensemble inversion for very large ensembles |
| `numba` | JIT-compiled model functions (e.g., SAC-SMA, GR4J) |

## Verify installation

```python
>>> from aries import esmda
>>> print(esmda.__name__)
esmda
```
