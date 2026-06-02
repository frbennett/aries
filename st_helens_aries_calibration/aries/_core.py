"""
ES-MDA (Ensemble Smoother with Multiple Data Assimilation) class.

Provides the main workflow: build prior ensemble, iterate over
fill-evaluate-update cycles, and report results.
"""

import os
import shutil
import time

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.stats import truncnorm

from ._linalg import dask_inverse, efast_inverse
from ._noise import (
    build_covariance_prior,
    build_prior,
    covariance_matrix,
    get_group_list,
    inv_trans,
)


# ===================================================================
# ESS-based tempering (CWIEKI / Botha style)
# ===================================================================


def _ess_weights(D, d_obs, phi_mean, delta_alpha):
    """
    Importance weights and ESS for a candidate tempering step Δα.

    Computes the incremental likelihood-power weights
        wₙ ∝ exp(−½ · Δα · Σᵢ (yᵢ − Gₙᵢ)² / φᵢ²)
    and the resulting effective sample size  ESS = 1 / Σ Wₙ².

    Parameters
    ----------
    D : ndarray (Nd, Ne)
        Ensemble predictions (transformed space).
    d_obs : ndarray (Nd,)
        Observations.
    phi_mean : ndarray (Nd,)
        Mean noise standard deviation per observation.
    delta_alpha : float
        Candidate tempering step size (Δα ≥ 0).

    Returns
    -------
    ess : float   Effective sample size (1 … Ne).
    W : ndarray (Ne,)  Normalised importance weights.
    """
    residuals = d_obs[:, None] - D                 # (Nd, Ne)
    weighted_ss = np.sum(residuals ** 2 / (phi_mean ** 2)[:, None], axis=0)  # (Ne,)
    log_w = -0.5 * delta_alpha * weighted_ss
    log_w -= log_w.max()  # numerical stability
    w = np.exp(log_w)
    W = w / w.sum()
    ess = 1.0 / np.sum(W ** 2)
    return ess, W


def _find_tempering_alpha(D, d_obs, phi_mean, alpha_prev, target_ess):
    """
    Binary search for the largest α ∈ [α_prev, 1] with ESS ≥ target_ess.

    ESS(α) is monotonic decreasing in Δα = α − α_prev, so the largest
    feasible step is the rightmost point where ESS still meets the
    threshold.
    """
    lo = alpha_prev
    hi = 1.0

    # Can we go all the way?
    ess_at_hi, _ = _ess_weights(D, d_obs, phi_mean, hi - alpha_prev)
    if ess_at_hi >= target_ess:
        return hi

    for _ in range(40):
        mid = (lo + hi) / 2
        ess_mid, _ = _ess_weights(D, d_obs, phi_mean, mid - alpha_prev)
        if ess_mid >= target_ess:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break

    return lo


class esmda:
    """Ensemble Smoother with Multiple Data Assimilation."""

    def __init__(self, **kwargs):
        # Set default parameters
        self.job_name = "esmda_job"
        self.parameter_file_name = "es_parameters.csv"
        self.observation_file_name = "es_data.csv"
        self.data_file_name = "es_data.csv"
        self.nEnsemble = 100
        self.maxIter = 10
        self.inversion_type = "svd"
        self.calculation_type = "ikea"
        self.phi_update = "laplace"
        self.inflation_schedule = "fixed"  # "fixed" for ES-MDA, "ess" for CWIEKI
        self.target_ess = 0.5              # ESS threshold (fraction of Ne)
        self.save_all_iterations = False
        # Student-t likelihood parameters
        self.likelihood = "gaussian"       # "gaussian" or "student_t"
        self.nu_init = 8.0                 # initial degrees of freedom
        self.nu_adapt = True               # adapt ν from residuals each iter
        self.nu_smooth = 0.7               # EMA smoothing (0=no memory, 1=frozen)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.parameter_data = pd.read_csv(self.parameter_file_name)
        self.observation_data = pd.read_csv(self.observation_file_name)

        # Initialise global arrays
        self.mLength = len(self.parameter_data)
        self.dLength = len(self.observation_data)
        self.mPrior = np.zeros([self.mLength, self.nEnsemble])
        self.mPrior_untransformed = np.zeros([self.mLength, self.nEnsemble])

        if os.path.exists(self.job_name):
            print("Deleting directory " + self.job_name)
            shutil.rmtree(self.job_name)
        os.makedirs(self.job_name, exist_ok=True)

    def report(self, iter, M, D, phi):
        """Save iteration results: parameters, data, and phi (if IKEA)."""
        base = os.path.basename(self.job_name)
        iteration_parameters = pd.DataFrame(
            M.T, columns=self.parameter_data.parameter.values
        )
        fname = f"{base}_{iter}_parameters.csv"
        iteration_parameters.T.to_csv(os.path.join(self.job_name, fname))

        iteration_data = pd.DataFrame(
            D.T, columns=self.observation_data.label.values
        )
        iteration_data = iteration_data.T
        fname = f"{base}_{iter}_data.csv"
        iteration_data.to_csv(os.path.join(self.job_name, fname))

        if self.calculation_type == "ikea":
            iteration_phi = pd.DataFrame(
                phi.T, columns=self.observation_data.label.values
            )
            fname = f"{base}_{iter}_phi.csv"
            iteration_phi.T.to_csv(os.path.join(self.job_name, fname))

        print(" ")
        print("Completed iteration ", iter)
        print("============================")
        print(" ")

    def run_esmda(self, fill_ensemble):
        """Run the ES-MDA iterative update loop."""
        alpha_j = 0.0
        Ne = self.nEnsemble
        Nd = self.dLength
        Nm = self.mLength
        Na = self.maxIter

        # Covariance initialisation
        if self.calculation_type == "ikea":
            phi = build_covariance_prior(Ne, self.observation_data, self.error)
        else:
            phi = self.observation_data.noise.values
        d_obs = self.observation_data["Y"]
        M = build_prior(self.parameter_data, self.nEnsemble)
        short_ref_list, long_ref_list = get_group_list(self.observation_data)

        # --- Tempering state ---
        alpha_temper = 0.0       # cumulative α ∈ [0, 1]  (ESS mode only)
        step_inflation = float(self.maxIter)  # used in Kalman equations

        for iter in range(self.maxIter):
            # Fill the ensemble
            M_invt = inv_trans(self.parameter_data, M)
            D = fill_ensemble(
                M_invt, self.nEnsemble, self.mLength, self.dLength
            )

            # ── Student-t: estimate ν and draw latent weights ────────────────
            if self.likelihood == "student_t":
                if iter == 0:
                    self._nu = self.nu_init

                if self.nu_adapt and iter > 0:
                    # Estimate ν from ensemble-mean residuals via kurtosis
                    D_mean_arr = D.mean(axis=1)
                    resid = np.asarray(d_obs) - D_mean_arr
                    phi_mean_arr = phi.mean(axis=1)
                    r_std = resid / np.maximum(phi_mean_arr, 1e-8)
                    # Clip extremes before computing kurtosis
                    r_clip = np.clip(r_std, -8, 8)
                    k = np.mean(r_clip ** 4) / (np.mean(r_clip ** 2) ** 2)
                    if k > 3.1:
                        nu_est = 4.0 + 6.0 / (k - 3.0)
                    else:
                        nu_est = 100.0  # effectively Gaussian
                    # EMA smoothing
                    self._nu = (self.nu_smooth * self._nu +
                                (1 - self.nu_smooth) * np.clip(nu_est, 3.0, 100.0))

                # Compute standardised residuals for λ draw
                D_mean_arr = D.mean(axis=1)
                resid = np.asarray(d_obs) - D_mean_arr
                phi_mean_arr = phi.mean(axis=1)
                r2 = (resid / np.maximum(phi_mean_arr, 1e-8)) ** 2

                nu = max(self._nu, 2.1)  # guard against degenerate values

                if iter == 0:
                    # Prior draw: λ_i ~ Gamma(ν/2, ν/2) — precision multiplier, mean=1
                    self._lam = np.random.gamma(
                        shape=nu / 2, scale=2.0 / nu, size=Nd
                    )
                else:
                    # Posterior draw: λ_i | r_i ~ Gamma((ν+1)/2, (ν + r_i²)/2)
                    self._lam = np.random.gamma(
                        shape=(nu + 1) / 2, scale=2.0 / (nu + r2), size=Nd
                    )

            # is_final depends on schedule
            if self.inflation_schedule == "ess":
                is_final = False  # determined by α reaching 1
            else:
                is_final = (iter == self.maxIter - 1)

            # --- Iteration header ---
            if self.inflation_schedule == "ess":
                print(f"\n═══ Iteration {iter + 1} ═══")
            else:
                print(f"\n═══ Iteration {iter + 1}/{self.maxIter} ═══")

            # --- Adaptive ESS-based step size (CWIEKI / Botha) ---
            if self.inflation_schedule == "ess":
                d_obs_arr = np.asarray(d_obs)
                phi_mean = (
                    np.mean(phi, axis=1)
                    if self.calculation_type == "ikea"
                    else phi
                )

                next_alpha = _find_tempering_alpha(
                    D, d_obs_arr, phi_mean, alpha_temper,
                    self.target_ess * Ne,
                )
                delta_alpha = next_alpha - alpha_temper
                alpha_temper = next_alpha
                step_inflation = 1.0 / max(delta_alpha, 1e-12)

                if alpha_temper >= 1.0 - 1e-8:
                    is_final = True
                elif iter == self.maxIter - 1:
                    print(f"  ⚠  Hit safety cap ({self.maxIter} iter) "
                          f"but α = {alpha_temper:.4f} < 1")
                    is_final = True

            # Report: only final iteration by default
            if self.save_all_iterations or is_final:
                self.report(iter, M_invt, D, phi)

            # Skip inversion on the final iteration — model already evaluated
            if is_final:
                break

            # Calculate del_D
            D_mean = D.mean(axis=1)
            del_D = np.zeros_like(D)
            for i in range(self.nEnsemble):
                del_D[:, i] = D[:, i] - D_mean

            # Calculate Cmd
            M_mean = M.mean(axis=1)
            del_M = np.zeros_like(M)
            for i in range(self.nEnsemble):
                del_M[:, i] = M[:, i] - M_mean

            Cmd = del_M @ del_D.T / (self.nEnsemble - 1)

            # Perturb observations
            Duc = np.zeros_like(D)
            # Student-t: scale perturbation by 1/√λ (precision multiplier)
            if self.likelihood == "student_t":
                wt_factor = 1.0 / np.sqrt(np.maximum(self._lam, 1e-6))
            else:
                wt_factor = 1.0

            for i in range(self.nEnsemble):
                if self.calculation_type == "ikea":
                    Duc[:, i] = (
                        np.sqrt(step_inflation) * phi[:, i] * wt_factor
                        * np.random.normal(0, 1, Nd)
                        + d_obs
                    )
                else:
                    Duc[:, i] = (
                        np.sqrt(step_inflation) * phi * wt_factor
                        * np.random.normal(0, 1, Nd)
                        + d_obs
                    )

            start = time.time()

            # Calculate M_update
            M_update = np.zeros_like(M)

            if self.inversion_type == "svd":
                Cdd = (del_D @ del_D.T) / (Ne - 1)
                for index in range(Ne):
                    Cd = np.zeros([self.dLength, self.dLength])
                    np.fill_diagonal(Cd, phi[:, index] ** 2)
                    K = Cdd + step_inflation * Cd
                    Kinv, svd_rank = sla.pinvh(K, return_rank=True)
                    M_update[:, index] = (
                        M[:, index] + Cmd @ Kinv @ (Duc[:, index] - D[:, index])
                    )

            if self.inversion_type == "subspace":
                Ud, Wd, Vd = np.linalg.svd(
                    del_D, full_matrices=False, compute_uv=True, hermitian=False
                )
                Binv = np.diag(Wd ** (-2))
                for index in range(Ne):
                    aCd = (Ne - 1) * step_inflation * phi[:, index] ** 2
                    Ainv = np.diag(aCd ** (-1))
                    bracket = Binv + Ud.T @ Ainv @ Ud
                    bracketinv = np.linalg.inv(bracket)
                    Kinv = (
                        (Ne - 1)
                        * (Ainv - Ainv @ Ud @ bracketinv @ Ud.T @ Ainv)
                    )
                    M_update[:, index] = (
                        M[:, index]
                        + Cmd @ Kinv @ (Duc[:, index] - D[:, index])
                    )

            if self.inversion_type == "fast_subspace":
                Ud, Wd, Vd = np.linalg.svd(
                    del_D, full_matrices=False, compute_uv=True, hermitian=False
                )
                Binv = np.diag(Wd ** (-2))
                for index in range(Ne):
                    aCd = (Ne - 1) * step_inflation * phi[:, index] ** 2
                    AinvUd = ((aCd ** (-1)) * Ud.T).T
                    bracket = Binv + Ud.T @ AinvUd
                    bracketinv = np.linalg.inv(bracket)
                    Kinv = (Ne - 1) * (
                        np.diag(aCd ** (-1)) - AinvUd @ bracketinv @ AinvUd.T
                    )
                    M_update[:, index] = (
                        M[:, index]
                        + Cmd @ Kinv @ (Duc[:, index] - D[:, index])
                    )

            if self.inversion_type == "efast_subspace":
                rand_phi = np.zeros(Nd)
                for i in range(Nd):
                    phi_mean_i = np.mean(phi[i, :])
                    phi_std_i = np.std(phi[i, :])
                    rand_phi[i] = truncnorm.rvs(-1, 1, phi_mean_i, phi_std_i)
                M_update = efast_inverse(
                    M, Cmd, Duc, D, del_D, rand_phi, step_inflation, Ne
                )

            if self.inversion_type == "dask":
                rand_phi = np.zeros(Nd)
                for i in range(Nd):
                    phi_mean_i = np.mean(phi[i, :])
                    phi_std_i = np.std(phi[i, :])
                    rand_phi[i] = truncnorm.rvs(-1, 1, phi_mean_i, phi_std_i)
                M_update = dask_inverse(
                    M, Cmd, Duc, D, del_D, rand_phi, step_inflation, Ne
                )

            if self.inversion_type == "esmda":
                Ud, Wd, Vd = np.linalg.svd(
                    del_D, full_matrices=False, compute_uv=True, hermitian=False
                )
                Binv = np.diag(Wd ** (-2))
                aCd = (Ne - 1) * step_inflation * phi ** 2
                AinvUd = ((aCd ** (-1)) * Ud.T).T
                bracket = Binv + Ud.T @ AinvUd
                bracketinv = np.linalg.inv(bracket)
                Kinv = (Ne - 1) * (
                    np.diag(aCd ** (-1)) - AinvUd @ bracketinv @ AinvUd.T
                )
                M_update = M + Cmd @ Kinv @ (Duc - D)

            if self.inversion_type == "esmda_dask":
                Ud, Wd, Vd = np.linalg.svd(
                    del_D, full_matrices=False, compute_uv=True, hermitian=False
                )
                Binv = np.diag(Wd ** (-2))
                aCd = (Ne - 1) * step_inflation * phi ** 2
                AinvUd = ((aCd ** (-1)) * Ud.T).T
                bracket = Binv + Ud.T @ AinvUd
                bracketinv = np.linalg.inv(bracket)
                Kinv = (Ne - 1) * (
                    np.diag(aCd ** (-1)) - AinvUd @ bracketinv @ AinvUd.T
                )
                M_update = M + Cmd @ Kinv @ (Duc - D)

            end = time.time()

            # Update phi
            if self.inflation_schedule == "ess":
                alpha_j = alpha_temper  # cumulative α for reporting
            else:
                alpha_j += 1 / Na  # fixed schedule progress

            if self.calculation_type == "ikea":
                phi_std = {}
                phi_val = {}
                for i in short_ref_list:
                    phi_std[i] = np.std(phi[short_ref_list[i], :])
                    phi_val[i] = phi[short_ref_list[i], :]

                if self.phi_update == "mcmc":
                    phi, phi_stats = covariance_matrix(
                        D, self.observation_data, self.error,
                        alpha_j, phi_std, phi_val,
                        method=self.phi_update,
                        return_stats=True,
                    )
                else:
                    phi = covariance_matrix(
                        D, self.observation_data, self.error,
                        alpha_j, phi_std, phi_val,
                        method=self.phi_update,
                    )

            # ── Diagnostics ──
            param_means = inv_trans(self.parameter_data, M).mean(axis=1)

            # Build diagnostic lines
            diag_parts = []
            diag_parts.append(f"  Inversion: {self.inversion_type}")
            if (end - start) > 0.01:
                diag_parts.append(f"  Time: {end - start:.2f}s")
            diag_parts.append(f"  φ = {phi.mean():.4f}")
            if self.likelihood == "student_t":
                diag_parts.append(f"  ν = {self._nu:.1f}")

            if self.inflation_schedule == "ess":
                diag_parts.append(
                    f"  α = {alpha_temper:.4f}  "
                    f"Δα = {delta_alpha:.6f}  "
                    f"Ensemble ESS(tgt) = {self.target_ess * Ne:.0f}"
                )

            if self.phi_update == "mcmc":
                diag_parts.append("  MCMC:")
                for grp, s in phi_stats.items():
                    diag_parts.append(
                        f"    {grp}: Chain ESS {s['ess_mean']:.0f} "
                        f"(min {s['ess_min']:.0f}), "
                        f"φ = {s['phi_mean']:.4f} ± {s['phi_std']:.4f}"
                    )

            # ── Parameter summary ──
            param_str = "  ".join(
                f"{v:.4f}" for v in param_means[:6]
            )
            if len(param_means) > 6:
                param_str += f"  … ({len(param_means)} params)"

            diag_parts.insert(0, f"  Params: {param_str}")

            print("\n".join(diag_parts))

            M = M_update

        if self.calculation_type == "ikea":
            base = os.path.basename(self.job_name)
            iteration_phi = pd.DataFrame(
                phi.T, columns=self.observation_data.label.values
            )
            fname = f"{base}_final_phi.csv"
            iteration_phi.T.to_csv(os.path.join(self.job_name, fname))

    def predictive_posterior(self, n):
        """Generate posterior predictive samples by adding observation noise."""
        base = os.path.basename(self.job_name)
        pred_post = pd.DataFrame()
        fname = f"{base}_{self.maxIter}_data.csv"
        posterior = pd.read_csv(os.path.join(self.job_name, fname))
        del posterior[posterior.columns[0]]
        noise = self.observation_data.noise.values
        for i in posterior.columns:
            samples = pd.DataFrame()
            data = posterior[i].values
            samples[i] = data
            for j in range(n):
                label = str(i) + "_" + str(j + 1)
                sample = (
                    np.random.normal(0, 1, self.dLength) * noise + data
                )
                samples[label] = sample
            pred_post = pd.concat([pred_post, samples], axis=1)
        pred_post.set_index(self.observation_data.name, inplace=True)
        pred_post.to_csv(
            os.path.join(self.job_name, "posterior_predictive.csv")
        )
