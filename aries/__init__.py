"""
ARIES — Adaptive Residual-based Iterative Ensemble Smoother.

Ensemble-based Bayesian inference for hydrological model calibration,
descended from CWIEKI (Botha et al., 2023).  Uses ES-MDA with
adaptively estimated noise covariance — no MCMC framework required.

Core workflow::

    from aries import esmda

    solver = esmda(
        parameter_file_name="es_parameters.csv",
        observation_file_name="es_data.csv",
        nEnsemble=1000,
        maxIter=12,
        inversion_type="efast_subspace",
        phi_update="laplace",           # or "mcmc", "grid", "residual"
        inflation_schedule="fixed",     # or "ess" for CWIEKI-style tempering
        calculation_type="ikea",        # adaptive noise estimation
    )
    solver.run_esmda(fill_ensemble)

Submodules:
    _core    : Main esmda class and ESS-based tempering.
    _noise   : Noise covariance estimation (phi update methods).
    _linalg  : Linear algebra (SVD, subspace inverses, Dask).
    _metrics : Prediction interval coverage metrics.
"""

__version__ = "0.1.0"

from ._core import esmda
from . import _noise as noise
from . import _linalg as linalg
from . import _metrics as metrics

__all__ = ["esmda", "noise", "linalg", "metrics"]
