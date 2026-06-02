# ARIES with Adaptive Student-t Likelihood — Implementation Plan

## Overview

Add a Student-t observation model to ARIES via the scale-mixture-of-normals
representation. Each observation gets a latent weight λ_i that down-weights
outliers. The degrees of freedom ν is estimated from ensemble residuals at
each iteration and can adapt as the ensemble converges.

## Theory

Student-t(y_i | μ_i, σ_i, ν) ≡ ∫ Normal(y_i | μ_i, σ_i/√λ_i) × InvGamma(λ_i | ν/2, ν/2) dλ_i

Conditional on λ_i, the likelihood is Gaussian → the Kalman update is exact.

## Changes to `aries/_core.py`

### 1. New `__init__` parameters

```python
self.likelihood = kwargs.get("likelihood", "gaussian")  # "gaussian" or "student_t"
self.nu_init = kwargs.get("nu_init", 8.0)       # starting ν
self.nu_min  = kwargs.get("nu_min", 3.0)        # lower bound for ν
self.nu_adapt = kwargs.get("nu_adapt", True)     # adapt ν each iteration?
self.nu_smooth = kwargs.get("nu_smooth", 0.8)    # EMA smoothing factor
```

### 2. ν estimation (after fill_ensemble, before perturbation)

```python
def _estimate_nu(self, residuals, phi_mean):
    """
    Estimate ν from standardised residuals via kurtosis method-of-moments.
    
    For t_ν: excess kurtosis = 6/(ν-4) for ν > 4
    → ν ≈ 4 + 6 / (sample_kurtosis - 3)
    
    Clamp ν ∈ [nu_min, 100]. Values > 100 are effectively Gaussian.
    """
    r_std = residuals / np.maximum(phi_mean, 1e-8)  # standardised residuals
    # Remove extreme outliers from estimation (they bias kurtosis)
    r_clean = np.clip(r_std, -10, 10)
    k = np.mean(r_clean ** 4) / (np.mean(r_clean ** 2) ** 2)  # sample kurtosis
    
    if k > 3.1:  # heavier tails than Gaussian (k=3)
        nu_est = 4.0 + 6.0 / max(k - 3.0, 0.01)
    else:
        nu_est = 100.0  # effectively Gaussian
    
    return np.clip(nu_est, self.nu_min, 100.0)
```

### 3. Latent weight draws (once per iteration)

```python
def _draw_lambda(self, residuals, phi_mean, nu):
    """Draw λ_i ~ InvGamma(ν/2, ν/2) with observation-specific priors."""
    r_std = residuals / np.maximum(phi_mean, 1e-8)
    # Shape parameters per observation — down-weight large residuals
    alpha = nu / 2.0 + 0.5 * r_std ** 2     # posterior shape
    beta  = nu / 2.0 + 0.0                    # posterior rate = prior rate
    # Mode of InvGamma(alpha, beta) = beta/(alpha+1)
    # Mean = beta/(alpha-1) for alpha > 1
    lam = np.random.gamma(shape=alpha, scale=1.0/beta, size=len(residuals))
    lam = 1.0 / lam  # InvGamma
    return np.clip(lam, 1e-4, 1e4)
```

Actually — simpler: just draw from the prior InvGamma(ν/2, ν/2). The
posterior update requires marginalising over the model predictions which
adds complexity. The prior draw is correct for the perturbation step
because the λ_i are drawn independently of the current ensemble.

```python
def _draw_lambda_prior(self, Nd, nu):
    """Draw λ_i ~ InvGamma(ν/2, ν/2) — prior draws, correct for perturbation."""
    lam = np.random.gamma(shape=nu/2.0, scale=2.0/nu, size=Nd)
    return 1.0 / lam  # InvGamma via Gamma reciprocal
```

Wait — that's wrong. Gamma(shape=ν/2, scale=2/ν) has mean 1, and 1/Gamma has
mean ν/(ν-2) for ν>2. Let me use the correct parameterisation:

InvGamma(α, β) where α=ν/2, β=ν/2 has mean β/(α-1) = 1 for ν>2.
So λ_i ~ InvGamma(ν/2, ν/2) has mean 1 (typical observation gets weight ~1).

### 4. Perturbation step (lines 244-257)

Replace:
```python
Duc[:, i] = (
    np.sqrt(step_inflation) * phi[:, i]
    * np.random.normal(0, 1, Nd)
    + d_obs
)
```

With:
```python
if self.likelihood == "student_t":
    Duc[:, i] = (
        np.sqrt(step_inflation) * phi[:, i] / np.sqrt(self._lambda)
        * np.random.normal(0, 1, Nd)
        + d_obs
    )
else:
    Duc[:, i] = (
        np.sqrt(step_inflation) * phi[:, i]
        * np.random.normal(0, 1, Nd)
        + d_obs
    )
```

The same change applies to the `else` branch (non-IKEA mode) and all
inversion types (subspace, fast_subspace, efast_subspace).

### 5. Per-iteration reporting

Add ν to the iteration header:
```python
if self.likelihood == "student_t":
    print(f"  ν = {self._nu:.1f}")
```

## Testing

### Sanity check: Gaussian limit

Set `nu_init=100`. Results should match the existing Gaussian ARIES within
Monte Carlo noise (±1% on NSE, parameter means).

### Real-data test: St Helens

Run with `nu_init=8, nu_adapt=True`. Check:
- ν converges to a stable value (should settle by iteration 5-8)
- φ is NOT inflated relative to Gaussian ARIES (λ handles outliers, not φ)
- Parameter posteriors are stable (no blow-up)
- P-factor > 90% on training and validation

### Failure modes to watch

| Symptom | Cause | Fix |
|---------|-------|-----|
| ν → 3 (lower bound) | Residuals extremely heavy-tailed; kurtosis estimate dominated by a few points | Increase nu_min or switch to median-based estimator |
| ν oscillates iteration-to-iteration | Ensemble still converging, residuals unstable | Increase nu_smooth (EMA) to 0.9 |
| φ inflates despite Student-t | IKEA noise model compensates for λ weights | Remove λ from phi update (use Gaussian residuals for φ) |
| Ensemble scatters (parameter variance grows) | Kalman step assumes Gaussian; t-weights break covariance estimation | Fall back to Gaussian for this iteration or reduce step_inflation |

## File changes

Only `aries/_core.py` — approximately 30 lines added, 4 lines modified.

No changes to `_noise.py` (IKEA Laplace approximation stays on Gaussian model),
`_linalg.py`, `_metrics.py`, or `__init__.py`.
