# Noise estimation (`aries.noise`)

Four methods for estimating the observation noise standard deviation σ
at each ES-MDA iteration:

## `laplace` (default)

Laplace approximation in log(σ) space. Newton–Raphson optimisation
finds the exact posterior mode:

$$ \sigma^* = \arg\max_\sigma \, p(\sigma \mid \mathbf{d}_\text{obs}, \bar{\mathbf{d}}_k) $$

The HalfNormal(τ) prior provides mild regularisation, preventing
unrealistically narrow noise estimates in early iterations.

**Strengths**: fast, includes prior regularisation, handles both variance and
systematic bias (uses full sum-of-squares, not centred residuals).

**Method**: `phi_update="laplace"`

## `mcmc`

Slice-sampling MCMC for the exact posterior of σ given each ensemble
member's predictions. A univariate slice sampler on γ = log(σ) with
stepping-out and shrinkage. Zero PyMC dependency — implemented in
~50 lines of NumPy.

### Slice sampler parameters

| Parameter | Default | Description |
|---|---|---|
| `mcmc_draws` | `50` | Posterior draws after warmup |
| `mcmc_tune` | `50` | Warmup (burn-in) draws |

### Diagnostics

When `return_stats=True`, per-group statistics include:

| Field | Description |
|---|---|
| `ess` | Effective sample size (Geyer's initial positive sequence) |
| `mean` | Posterior mean of σ |
| `std` | Posterior standard deviation |

```python
from aries.noise import ess

# Compute ESS for a chain of MCMC samples
n_eff = ess(samples)
```

## `grid`

Exact posterior mean via fine-grid integration in log(σ) space.
Evaluates the unnormalised log-posterior on 200+ grid points spanning
[log(1e−6), log(5τ)], normalises via softmax, and returns the
posterior mean. No approximation, no sampling.

**Grid parameters**: `grid_points=200`

## `residual`

Simple plug-in estimator: σₖ = std(Y − Gₖ(θ)) per group. Fastest
method, but includes no prior regularisation.

## Per-group noise

Group labels in the observation file (column `group`) control
per-group noise estimation. For example, a discharge-only model
with `group=FLOW` for all rows has a single global σ estimate.
Multi-variable calibration (e.g., discharge + stage) would have
separate σ estimates per variable.

## Prior scales

The `error` dictionary passed to `esmda()` sets HalfNormal prior
scales per group:

```python
solver = esmda(
    ...,
    error={"FLOW": 10.0, "STAGE": 5.0},
)
```

These act as regularisation: when data are sparse, the prior
prevents σ from collapsing to zero.
