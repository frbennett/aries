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

## Likelihood

ARIES supports both Gaussian and Student-t observation models. The
Student-t provides robustness to outliers (flood peaks) via latent
per-observation weights that down-weight extreme residuals.

| `likelihood` | Description |
|---|---|
| `"gaussian"` | **(default)** Standard ES-MDA with Gaussian observation noise. |
| `"student_t"` | Adaptive Student-t via scale-mixture-of-normals. Latent weights λᵢ ~ InvGamma(ν/2, ν/2) inflate observation variance for outliers. ν is estimated from ensemble residuals each iteration via kurtosis matching. |

**Student-t parameters:**

| Parameter | Default | Description |
|---|---|---|
| `nu_init` | 8.0 | Initial degrees of freedom |
| `nu_adapt` | `True` | Adapt ν from residuals each iteration |
| `nu_smooth` | 0.7 | EMA smoothing factor (0 = no memory, 1 = frozen) |

Usage:

```python
solver = esmda(
    ...,
    likelihood="student_t",
    nu_init=8.0,
    nu_adapt=True,
)
```

The iteration diagnostics will show `ν = 4.3` (or whatever it converges to)
alongside φ:

## Inversion types

All inversion methods compute the Kalman update

```
Mₚ₊₁  =  Mₚ  +  Cₘᵈ · K⁻¹ · (Dᵤ꜀ − D)
```

where `Cₘᵈ` is the cross-covariance between parameters and predictions,
`Dᵤ꜀` are perturbed observations, `D` is the ensemble, and

```
K  =  Cᴰᴰ  +  α · Cᴰ          (Cᴰᴰ = ensemble covariance, Cᴰ = observation noise)
```

The methods differ in **how they invert `K`** and **how they handle the
per-ensemble-member noise vector `φ`**.

| `inversion_type` | Noise shape | Strategy | When to use |
|---|---|---|---|
| `"efast_subspace"` | **(default)** | `rand_φ` (condensed) | Matrix-free Woodbury via small (r×r) solve. Never forms Nd×Nd. | General use — fast, memory-safe, numerically stable. |
| `"fast_subspace"` | `φ[:, i]` (per-member) | Woodbury identity, Nd×Nd diagonal formed implicitly via column broadcasting. | When per-member noise heterogeneity matters and Nd is moderate (<5000). |
| `"subspace"` | `φ[:, i]` (per-member) | Woodbury identity, explicit Nd×Nd diagonal matrix `A⁻¹`. | Legacy — kept for reproducibility. For new work use `fast_subspace`. |
| `"svd"` (slow) | `φ[:, i]` (per-member) | Forms `K = Cᴰᴰ + α·Cᴰ` explicitly and inverts via `scipy.linalg.pinvh`. | Diagnostic use only — O(Nd³) per member is prohibitive for large Nd. |
| `"esmda"` | 1D `φ` vector only | Same algebra as `fast_subspace` but vectorised over Ne (no loop). | **Only with `calculation_type="standard"`** where `φ` is 1D. Fails with IKEA-mode 2D `φ`. |
| `"esmda_dask"` | 1D `φ` vector only | Dask-based vectorised esmda. | Out-of-core / large-ensemble standard mode. |
| `"dask"` | `rand_φ` (condensed) | Dask-based efast_subspace. | Out-of-core / large-ensemble IKEA mode. |

---

### `efast_subspace` — recommended default

```
for each observation i:
    rand_φ[i] ~ TruncNormal(mean=φ̅ᵢ, sd=σ(φᵢ), bounds=[-1, 1]σ)
```

The Nd×Ne `φ` matrix is condensed to a single Nd-vector `rand_φ` by
sampling from the ensemble distribution of each observation's noise
(truncated to ±1σ for robustness). The update is then computed by
[`efast_inverse`](aries/_linalg.py), which avoids forming the Nd×Nd
inverse altogether:

```
v = (Dᵤ꜀ − D) / aCᴅ                      # (Nd × Ne)
temp = solve(bracket, Uᵀ · v)             # small (r × r) system
action = (Ne−1) · (v − A⁻¹U · temp)       # K⁻¹ · (Dᵤ꜀ − D) as (Nd × Ne)
Mₚ₊₁ = M + Cₘᵈ · action
```

**Peak memory:** O(Nd · Ne) — no Nd×Nd or Nd×r matrices formed.

**Numerical safety:** Singular values of `del_D` below a machine-epsilon
tolerance are set to `∞`, automatically zeroing any collapsed modes.

| Strength | Weakness |
|---|---|
| Fastest option — no per-member loop, no Nd×Nd matrices | Loses per-ensemble-member noise heterogeneity (uses condensed `rand_φ`) |
| Memory-efficient — O(Nd·Ne) | Condensation is stochastic (sampled), not deterministic |
| No Nd×Nd or Nd×r allocations | |
| SVD regularisation protects against ensemble collapse | |

---

### `fast_subspace` — per-member noise, same algebra

Loops over Ne ensemble members, using its own `φ[:, i]` noise vector for
each.  Mathematically identical to `subspace` but avoids forming the full
Nd×Nd diagonal matrix by using numpy column broadcasting:

```
# subspace:
A⁻¹ = diag(1/aCᴅ)              # Nd × Nd  — large!
K⁻¹ = (Ne−1)(A⁻¹ − A⁻¹U · B⁻¹ · UᵀA⁻¹)

# fast_subspace (same result, no large diagonal):
A⁻¹U = U / aCᴅ                  # (Nd × r) via broadcasting
K⁻¹ = (Ne−1)(diag(1/aCᴅ) − A⁻¹U · B⁻¹ · (A⁻¹U)ᵀ)
```

Note that `diag(1/aCᴅ)` is still formed here — but only as an intermediate
in the final assembly, not as the start of a chain of Nd×Nd multiplications.

| Strength | Weakness |
|---|---|
| Preserves per-member noise heterogeneity (±σ² per ensemble member) | ~Ne× slower than `efast_subspace` (loops over all members) |
| Same Woodbury stability as subspace | Lacks explicit SVD regularisation — relies on `inv(∞)` behaviour |
| Memory O(Nd²) only during final `K⁻¹` assembly | For Nd > 5000, the Nd×Nd `diag(1/aCᴅ)` may be large |

---

### `subspace` — full Woodbury with explicit diagonal

Original implementation.  Forms the Nd×Nd diagonal matrix `A⁻¹`
explicitly, then computes the full Kalman inverse via:

```
K⁻¹ = (Ne−1) · (A⁻¹  −  A⁻¹ · U · (diag(W⁻²) + Uᵀ · A⁻¹ · U)⁻¹ · Uᵀ · A⁻¹)
       \______/   \__________________________________________________/
       scaling    Woodbury correction in the (r × r) subspace
```

The (r × r) interior solve avoids inverting a full Nd×Nd matrix, but
the `A⁻¹ · U` multiplications still produce (Nd × r) intermediates.

| Strength | Weakness |
|---|---|
| Pedagogically clear — direct Woodbury formula | Forms full Nd×Nd `A⁻¹` every iteration — memory O(Nd²) |
| Mathematically identical to `fast_subspace` | Loses to `fast_subspace` on memory and speed |
| Preserves per-member noise | |

---

### `svd` — brute-force pinvh (diagnostic only)

For each ensemble member, explicitly constructs the full Kalman matrix:

```
K = Cᴰᴰ + α · diag(φᵢ²)              # Nd × Nd
K⁻¹ = scipy.linalg.pinvh(K)          # explicit inverse
```

| Strength | Weakness |
|---|---|
| Gold standard for correctness | O(Nd³) per member — prohibitive for Nd > 500 |
| Uses numerically robust `pinvh` | ~1000× slower than `fast_subspace` for Nd=5000 |
| Best for debugging / cross-checking other methods | |

---

### `esmda` — vectorised (1D φ only, standard mode)

Same Woodbury algebra as `fast_subspace` but vectorised over Ne — no
per-member loop.  Instead of indexing `φ[:, i]`, it operates on the full
array at once:

```python
aCd = (Ne - 1) * α * φ ** 2     # φ is 1D (Nd,), aCd is (Nd,)
# ... all operations are matrix-level
M_update = M + Cmd @ Kinv @ (Duc - D)
```

**Requires `calculation_type="standard"`** — `φ` must be a 1D vector
(from the CSV's `noise` column).  Fails with IKEA-mode 2D `φ` because the
matrix operations don't broadcast across the ensemble dimension.

| Strength | Weakness |
|---|---|
| Fast — fully vectorised, no per-member loop | Only works in standard mode (1D φ) |
| Same Woodbury stability | Cannot be combined with noise adaptation |

---

### `esmda_dask` / `dask` — out-of-core

`esmda_dask` wraps the esmda computation in Dask arrays for very large
ensembles.  `dask` does the same for the efast_subspace approach.
Use when the full `del_D` or `Kinv` matrices exceed available RAM.

---

### Quick reference: which to choose

| Situation | Recommended |
|---|---|
| General use, Nd < 5000, standard or IKEA mode | `"efast_subspace"` |
| Per-member noise heterogeneity critical | `"fast_subspace"` |
| Standard mode (no noise adaptation) | `"esmda"` (fastest for this case) |
| Nd > 10 000 or Ne > 10 000 | `"dask"` or `"esmda_dask"` |
| Debugging / cross-checking | `"svd"` (verify against fast methods) |
| Legacy reproducibility | `"subspace"` |

## Diagnostics output

```
═══ Iteration 3 ═══
  Params: 30.53  46.58  … (14 params)
  Inversion: efast_subspace  |  Time: 10.14s
  φ = 0.5791
  ν = 4.3                # only with likelihood="student_t"
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
