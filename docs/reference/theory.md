# Theory

## ES-MDA

The Ensemble Smoother with Multiple Data Assimilation
(Emerick & Reynolds, 2013) extends the Ensemble Kalman Filter to
inverse problems by iteratively updating an ensemble of parameter
estimates. At each iteration $k$:

1. Run the forward model for each ensemble member: $\mathbf{d}_k^{(i)} = g(\boldsymbol{\theta}_k^{(i)})$
2. Perturb observations: $\mathbf{d}_{\text{uc}}^{(i)} = \mathbf{d}_{\text{obs}} + \sqrt{K\alpha_k}\, \phi_k\, \boldsymbol{\varepsilon}^{(i)}$
3. Kalman update: $\mathbf{m}_k^{(i)} = \mathbf{m}_{k-1}^{(i)} + \mathbf{C}_{md}\,\mathbf{K}^{-1}\,(\mathbf{d}_{\text{uc}}^{(i)} - \mathbf{d}_k^{(i)})$

where $\alpha_k = 1/K$ is the inflation schedule over $K$ iterations.

## ARIES noise estimation

ARIES extends ES-MDA by adaptively estimating the observation noise
$\phi_k$ at each iteration. Instead of CWIEKI's MCMC approach
(Botha et al., 2023), ARIES uses a Laplace approximation:

$$ \sigma_k^* = \arg\max_\sigma \, p(\sigma \mid \mathbf{d}_{\text{obs}}, \bar{\mathbf{d}}_k) $$

where the posterior is:

$$ p(\sigma \mid \mathbf{d}_{\text{obs}}, \bar{\mathbf{d}}_k) \propto \sigma^{-n} \exp\!\left(-\frac{\text{SS}_k}{2\sigma^2} - \frac{\sigma^2}{2\tau^2}\right) $$

with $\text{SS}_k = \|\mathbf{d}_{\text{obs}} - \bar{\mathbf{d}}_k\|^2$ and $\tau$ the HalfNormal prior scale.
The mode is found via Newton–Raphson in $\gamma = \log\sigma$ space,
which is strictly concave and converges quadratically.

## CWIEKI tempering

The ESS-adaptive tempering schedule ($\texttt{inflation\_schedule="ess"}$)
uses ideas from likelihood tempering SMC (Del Moral et al., 2006).
At each iteration $j$, the tempering parameter $\alpha_j \in [0,1]$
controls how much of the likelihood is included:

$$ \pi_j(\boldsymbol{\theta}) \propto p(\mathbf{d}_{\text{obs}} \mid \boldsymbol{\theta})^{\alpha_j} \, p(\boldsymbol{\theta}) $$

The step $\Delta\alpha_j = \alpha_j - \alpha_{j-1}$ is chosen to maintain
a target effective sample size (ESS):

$$ \text{ESS}_j(\Delta\alpha) = \frac{1}{\sum_{n=1}^{N_e} (W_n)^2}, \quad
W_n \propto \exp\!\left(-\frac{\Delta\alpha}{2} \sum_i \frac{(y_i - G_{n,i})^2}{\phi_i^2}\right) $$

The algorithm stops when $\alpha_j = 1$. See Botha et al. (2023) for details.

## Sources of bias

Four sources of bias in ensemble smoothers (after Evensen, 2018):

1. **Linearisation**: The Kalman update replaces individual model gradients
   with an ensemble-averaged approximation.
2. **Finite ensemble**: Sample covariance underestimates true covariance
   in high dimensions; spurious correlations introduce noise.
3. **Gaussian assumption**: ES-MDA produces a Gaussian approximation to a
   non-Gaussian posterior for nonlinear models.
4. **Strong-constraint**: All prediction error is attributed to parameter
   uncertainty and observation noise; model structural error is not modelled.
