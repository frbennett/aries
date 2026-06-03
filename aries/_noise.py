"""
Parameter transforms, prior generation, and covariance estimation for ES-MDA.

Phi update methods (controlled by esmda(phi_update=...)):
    residual : σ = std(Y - G) per group  [IKEA default, fastest]
    mcmc     : Adaptive Metropolis-Hastings on log(σ)  [CWIEKI / Botha]
    laplace  : Laplace approx. in log(σ) space  [analytic posterior, no MCMC]
    grid     : Fine-grid posterior approx.  [full Bayesian, no MCMC]
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import halfnorm, truncnorm


# ===================================================================
# Parameter transforms
# ===================================================================

def logit(p):
    """Logit (log-odds) transform."""
    return np.log(p) - np.log(1 - p)


def inv_logit(p):
    """Inverse logit (logistic) transform."""
    return np.exp(p) / (1 + np.exp(p))


def inverse_scale_param(P, a, b):
    """Scale a [0, 1] value back to interval [a, b]."""
    return P * (b - a) + a


def scale_param(P, a, b):
    """Scale a value from interval [a, b] to [0, 1]."""
    return (P - a) / (b - a)


# ===================================================================
# Prior ensemble generation
# ===================================================================

def build_prior(es_parameters, nEnsemble):
    """
    Generate an ensemble of logit-transformed parameters.

    Parameters are sampled from truncated normal distributions, then
    logit-transformed to (-inf, inf) for the ensemble update.
    """
    mLength = len(es_parameters)
    mPrior = np.zeros([mLength, nEnsemble])

    es_parameters["std"] = (
        (es_parameters["upper"] - es_parameters["lower"])
        / es_parameters["width"]
    )
    es_parameters["a"] = (
        (es_parameters["lower"] - es_parameters["mean"])
        / es_parameters["std"]
    )
    es_parameters["b"] = (
        (es_parameters["upper"] - es_parameters["mean"])
        / es_parameters["std"]
    )
    print(es_parameters)

    stdevM = es_parameters["std"].values
    param_mean = es_parameters["mean"].values
    a = es_parameters["a"].values
    b = es_parameters["b"].values

    # Sample prior in physical (untransformed) parameter space
    mPrior_untransformed = np.zeros([mLength, nEnsemble])
    for i in range(mLength):
        mPrior_untransformed[i, :] = truncnorm.rvs(
            a[i], b[i],
            loc=param_mean[i], scale=stdevM[i],
            size=nEnsemble,
            random_state=None,
        )

    # Transform to logit space
    for i in range(nEnsemble):
        scaled = scale_param(
            mPrior_untransformed[:, i],
            es_parameters["lower"].values,
            es_parameters["upper"].values,
        )
        mPrior[:, i] = logit(scaled)

    return mPrior


def inv_trans(es_parameters, mCurrent):
    """Inverse logit transform: back from logit space to physical space."""
    M = np.zeros_like(mCurrent)
    nEnsemble = np.shape(M)[1]
    for i in range(nEnsemble):
        scaled = inv_logit(mCurrent[:, i])
        M[:, i] = inverse_scale_param(
            scaled,
            es_parameters["lower"].values,
            es_parameters["upper"].values,
        )
    return M


# ===================================================================
# Covariance / noise estimation — phi update dispatch
# ===================================================================

def covariance_matrix(D, data, error, alpha_j, phi_std, phi_val,
                      method="residual", return_stats=False,
                      **method_kwargs):
    """
    Build covariance matrix for each ensemble member using the
    selected phi update method.

    Parameters
    ----------
    D : ndarray (Nd, Ne)
        Ensemble model predictions (transformed space).
    data : DataFrame
        Observation data with columns: Y, group, ...
    error : dict
        HalfNormal prior scales per group, e.g. {"FLOW": 10.0}.
    alpha_j : float
        Cumulative inflation factor (1/Na per iteration).
    phi_std, phi_val : dict
        Per-group phi statistics from previous iteration (unused by
        residual, accepted for compatibility).
    method : str
        One of 'residual', 'mcmc', 'laplace', 'grid'.
    return_stats : bool
        If True and method='mcmc', also return aggregated MCMC
        diagnostics (ESS, posterior mean, etc.) per group across
        all ensemble members.
    **method_kwargs
        Passed through to the specific update function.

    Returns
    -------
    phi : ndarray (Nd, Ne)
        Per-observation, per-ensemble-member error standard deviations.
    aggregated : dict, optional
        Only if return_stats=True and method='mcmc'.
        {group: {ess_mean, ess_min, phi_mean, n_members}}.
    """
    Nd, Ne = np.shape(D)

    # Pre-compute per-group prior scales
    group_idx = {}
    for group in data.group.unique():
        group_idx[group] = data.loc[data["group"] == group].index.values

    result = np.zeros_like(D)

    if method == "residual":
        update_fn = _residual_update
    elif method == "mcmc":
        update_fn = _mcmc_update
    elif method == "laplace":
        update_fn = _laplace_update
    elif method == "grid":
        update_fn = _grid_update
    else:
        raise ValueError(f"Unknown phi_update method: {method}. "
                         "Choose from: residual, mcmc, laplace, grid.")

    # Stats collection across ensemble members
    if return_stats and method == "mcmc":
        all_stats = []  # list of per-ensemble-member stats dicts
        for i in range(Ne):
            col, col_stats = update_fn(
                data, D[:, i], error, group_idx,
                return_stats=True, **method_kwargs
            )
            result[:, i] = col
            all_stats.append(col_stats)

        # Aggregate across ensemble members
        groups = list(all_stats[0].keys())
        aggregated = {}
        for grp in groups:
            ess_vals = [s[grp]['ess'] for s in all_stats]
            mean_vals = [s[grp]['mean'] for s in all_stats]
            aggregated[grp] = {
                'ess_mean': float(np.mean(ess_vals)),
                'ess_min': float(np.min(ess_vals)),
                'phi_mean': float(np.mean(mean_vals)),
                'phi_std': float(np.std(mean_vals)),
                'n_members': Ne,
            }
        return result, aggregated

    # Standard path (no stats)
    for i in range(Ne):
        result[:, i] = update_fn(
            data, D[:, i], error, group_idx, **method_kwargs
        )

    return result


# -------------------------------------------------------------------
# Method 1: Residual-based (IKEA default)
# -------------------------------------------------------------------

def _residual_update(data, G, error, group_idx, **kwargs):
    """
    Residual-based phi estimate: σ = std(Y - G) per group.

    This is the IKEA plug-in estimator — replaces the MCMC step from
    CWIEKI with a simple standard deviation of residuals.  Fast and
    empirically unbiased for large N (see paper: error differs from
    SMC by ~0.0004).
    """
    result = np.empty(len(data))
    for group, idx in group_idx.items():
        y = data["Y"].values[idx]
        g = G[idx]
        result[idx] = np.std(y - g)
    return result


# -------------------------------------------------------------------
# Method 2: Full MCMC (CWIEKI / Botha style)
# -------------------------------------------------------------------

def _mcmc_update(data, G, error, group_idx,
                 mcmc_draws=50, mcmc_tune=50, n_jobs=1,
                 prior_scale=None, return_chain=False, return_stats=False,
                 chain_seed=42, **kwargs):
    """
    MCMC for σ per group using univariate slice sampling on log(σ).

    Replaces the original PyMC/NUTS with a bespoke slice sampler on
    γ = log(σ).  For this 1D log-concave posterior, slice sampling
    gives ~90% sampling efficiency with zero tuning — vastly better
    than the ~30% of random-walk Metropolis-Hastings and orders of
    magnitude faster than the full NUTS machinery.

    Parameters (via **method_kwargs)
    --------------------------------
    mcmc_draws   : int  (default 50)   Posterior draws after tuning.
    mcmc_tune    : int  (default 50)   Warmup (burn-in) draws.
    n_jobs       : int  (default 1)    Unused — kept for API compat.
    prior_scale  : dict or float       HalfNormal scale(s).  If None,
                                       uses the ``error`` dict from
                                       the esmda constructor.
    return_chain : bool                If True, return full chain per
                                       group alongside the means.
    return_stats : bool                If True, return per-group ESS
                                       and posterior stats.
    chain_seed   : int                 RNG seed (default 42).

    Returns
    -------
    result   : ndarray (Nd,)
        Posterior mean of σ per observation.
    chains   : dict, optional
        Only if return_chain=True.  {group: ndarray} of posterior draws.
    stats    : dict, optional
        Only if return_stats=True.  {group: {ess, mean, std}}.
    """
    scales = _resolve_prior_scales(error, prior_scale)
    rng = np.random.default_rng(chain_seed)

    result = np.empty(len(data))
    chains = {} if return_chain else None
    stats = {} if return_stats else None

    for group, idx in group_idx.items():
        y = data["Y"].values[idx]
        g = G[idx]
        tau = scales.get(group, 10.0)

        n_obs = len(y)
        SS = np.sum((y - g) ** 2)

        # Log-posterior of γ = log(σ)  (HalfNormal(τ) prior + Normal likelihood)
        def log_target(gamma):
            s2 = np.exp(2 * gamma)
            return -(n_obs - 1) * gamma - 0.5 * SS / s2 - 0.5 * s2 / tau ** 2

        # Posterior width estimate from curvature at starting point
        gamma0 = np.log(max(np.std(y - g), 1e-12))
        e2g = np.exp(2 * gamma0)
        curvature = 2 * SS / e2g + 2 * e2g / tau ** 2
        posterior_sd = 1.0 / max(np.sqrt(curvature), 1e-10)

        # --- Slice sampling ---
        gamma = gamma0
        n_total = mcmc_tune + mcmc_draws
        samples = np.empty(n_total)
        # Interval width: tight enough for efficiency, generous for robustness
        w = 4.0 * posterior_sd
        max_expand = 20

        for t in range(n_total):
            # 1. Sample vertical level
            log_u = log_target(gamma) + np.log(rng.uniform())

            # 2. Stepping-out: bracket the level set
            L = gamma - w * rng.uniform()
            R = L + w
            for _ in range(max_expand):
                if L > -20 and log_target(L) > log_u:
                    L -= w
                else:
                    break
            for _ in range(max_expand):
                if R < 20 and log_target(R) > log_u:
                    R += w
                else:
                    break

            # 3. Shrinkage sampling within [L, R]
            while True:
                prop = L + rng.uniform() * (R - L)
                if log_target(prop) >= log_u:
                    gamma = prop
                    break
                if prop < gamma:
                    L = prop
                else:
                    R = prop

            samples[t] = np.exp(gamma)

        posterior = samples[mcmc_tune:]
        result[idx] = posterior.mean()

        if return_chain:
            chains[group] = posterior

        if return_stats:
            stats[group] = {
                'ess': float(ess(posterior)),
                'mean': float(posterior.mean()),
                'std': float(posterior.std()),
            }

    # Build return tuple with optional extras
    ret = (result,)
    if return_chain:
        ret += (chains,)
    if return_stats:
        ret += (stats,)
    return ret[0] if len(ret) == 1 else ret


# -------------------------------------------------------------------
# Method 3: Laplace approximation in log(σ) space
# -------------------------------------------------------------------

def _laplace_update(data, G, error, group_idx,
                    prior_scale=None, **kwargs):
    """
    Laplace approximation for σ in log-space (no MCMC needed).

    Models γ = log(σ) with likelihood N(G, exp(γ)²) and a
    HalfNormal(0, τ) prior on σ.  The posterior for γ is
    approximately Gaussian: N(γ*, -1/g''(γ*)).

    Returns σ_mode = exp(γ*).  This is faster than MCMC and
    includes prior regularisation that the residual method lacks.
    """
    result = np.empty(len(data))
    scales = _resolve_prior_scales(error, prior_scale)

    for group, idx in group_idx.items():
        y = data["Y"].values[idx]
        g = G[idx]
        tau = scales.get(group, 10.0)

        SS = np.sum((y - g) ** 2)
        n = len(y)

        # Starting value: residual std in log space
        gamma_0 = np.log(max(np.std(y - g), 1e-10))

        # Newton-Raphson for posterior mode of γ = log(σ)
        gamma = gamma_0
        for _ in range(20):
            e2g = np.exp(2 * gamma)
            gp = -(n - 1) + SS / e2g - e2g / tau ** 2      # first deriv
            gpp = -2 * SS / e2g - 2 * e2g / tau ** 2        # second deriv
            step = -gp / gpp
            gamma = gamma + step
            if abs(step) < 1e-8:
                break

        result[idx] = np.exp(gamma)
    return result


# -------------------------------------------------------------------
# Method 4: Grid approximation in log(σ) space
# -------------------------------------------------------------------

def _grid_update(data, G, error, group_idx,
                 prior_scale=None, grid_points=200, **kwargs):
    """
    Grid-based posterior for σ in log-space (full Bayesian, no MCMC).

    Evaluates the unnormalised log-posterior of γ = log(σ) on a
    fine grid spanning [log(1e-6), log(5τ)], normalises via softmax,
    and returns the posterior mean of σ = exp(γ).

    Gives the full posterior CDF without any sampling, and respects
    the positivity constraint naturally via the log parameterisation.
    """
    result = np.empty(len(data))
    scales = _resolve_prior_scales(error, prior_scale)

    for group, idx in group_idx.items():
        y = data["Y"].values[idx]
        g = G[idx]
        tau = scales.get(group, 10.0)

        SS = np.sum((y - g) ** 2)
        n = len(y)

        # Grid in log(σ) space — span from near-zero to 5×prior_scale
        gamma_0 = np.log(max(np.std(y - g), 1e-10))
        gamma_min = gamma_0 - 5.0
        gamma_max = np.log(5 * tau)
        gamma_grid = np.linspace(gamma_min, gamma_max, grid_points)

        # Unnormalised log-posterior for γ = log(σ)
        e2g = np.exp(2 * gamma_grid)
        log_post = (
            -(n - 1) * gamma_grid
            - SS / (2 * e2g)
            - e2g / (2 * tau ** 2)
        )

        # Numerically stable normalisation
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()

        result[idx] = np.exp(gamma_grid) @ post  # posterior mean of σ
    return result


# ===================================================================
# Helpers
# ===================================================================

def _resolve_prior_scales(error, override):
    """Extract per-group HalfNormal prior scales."""
    if override is not None:
        if isinstance(override, dict):
            return override
        # Single float applied to all groups
        return {group: float(override) for group in error}
    return error.copy() if isinstance(error, dict) else {"FLOW": 10.0}


def build_covariance_prior(Ne, data, error):
    """Initialise covariance prior from half-normal draws."""
    Nd = len(data)
    cov_prior = np.zeros([Nd, Ne])
    for i in range(Ne):
        for er in error:
            data.loc[data["group"] == er, "std"] = halfnorm.rvs(
                scale=error[er]
            )
        cov_prior[:, i] = data["std"].values
    return cov_prior


def get_group_list(the_dataframe):
    """Build index lookups for observation groups."""
    groups = the_dataframe.group.unique()
    long_ref_list = {}
    short_ref_list = {}
    for group in groups:
        short_ref_list[group] = np.min(
            the_dataframe.loc[the_dataframe["group"] == group].index
        )
        long_ref_list[group] = (
            the_dataframe.loc[the_dataframe["group"] == group].index
        )
    return short_ref_list, long_ref_list


# -------------------------------------------------------------------
# MCMC diagnostics
# -------------------------------------------------------------------

def ess(samples):
    """
    Effective sample size for a 1D MCMC chain via autocorrelation.

    Uses FFT-based autocorrelation truncated at the first negative
    lag, giving the standard Geyer (1992) initial positive sequence
    estimator.

    Parameters
    ----------
    samples : ndarray, shape (n,)
        MCMC samples (post-warmup).

    Returns
    -------
    float
        Effective sample size.  Clipped to [1, len(samples)].
    """
    n = len(samples)
    if n < 2:
        return float(n)

    # FFT autocorrelation
    x = samples - np.mean(samples)
    xcorr = correlate(x, x, mode='full', method='fft')
    acf = xcorr[n - 1:] / xcorr[n - 1]  # [ρ(0), ρ(1), ..., ρ(n-1)]

    # Geyer's initial positive sequence: sum ρ(k) until first negative
    rho = acf[1:]
    last_neg = np.where(rho < 0)[0]
    cutoff = last_neg[0] if len(last_neg) else len(rho)
    tau_int = 1 + 2 * np.sum(rho[:cutoff])
    return max(1.0, float(n) / tau_int)


# -------------------------------------------------------------------
# Backward-compatible alias for the original misleading name
# -------------------------------------------------------------------

def mcmc(odata, D, num_mcmc, alpha_j, phi_std, phi_0):
    """
    [DEPRECATED] Use covariance_matrix(D, ..., method='residual') instead.

    Previously named 'mcmc' but actually computes residual-based
    ``phi = std(Y - G)`` per group.  Retained for backward compatibility.
    """
    import warnings
    warnings.warn(
        "utils.mcmc() is deprecated. Use covariance_matrix(..., "
        "method='residual') or _residual_update() directly.",
        DeprecationWarning,
    )
    group_idx = {}
    for group in odata.group.unique():
        group_idx[group] = odata.loc[odata["group"] == group].index.values
    return _residual_update(odata, D, {}, group_idx)
