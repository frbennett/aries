# Linear algebra (`aries.linalg`)

Matrix inversion utilities for the Kalman gain computation.

## Inversion types

| Type | Description |
|---|---|
| `"svd"` | Full SVD per ensemble member — most robust, slower |
| `"subspace"` | Subspace method with per-member noise |
| `"fast_subspace"` | Efficient subspace pseudo-inverse (recommended) |
| `"efast_subspace"` | Fast subspace with randomised φ draw — fastest with ensemble noise |
| `"esmda"` | Standard ES-MDA vectorised Kalman update (single φ for all members) |
| `"dask"` / `"esmda_dask"` | Dask-based for out-of-core ensembles (large ensemble, limited RAM) |

## Recommendation

```python
solver = esmda(inversion_type="efast_subspace")
```

`"efast_subspace"` is the fastest method for typical use cases
(100–2000 ensemble members, 14–100 parameters, 10k+ observations).
It uses a randomised per-member φ draw truncated to ±1σ for
numerical stability, then applies the Woodbury matrix identity for
efficient subspace inversion.

Use `"svd"` when debugging or when numerical stability is critical.
Use `"dask"` when the ensemble is too large to fit in memory.

## Functions

::: aries._linalg
    options:
      members: [efast_inverse, dask_inverse, pseudo_inverse]
