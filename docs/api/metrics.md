# Metrics (`aries.metrics`)

Prediction interval coverage and scoring functions.

## PICP

Prediction Interval Coverage Probability — fraction of observations
within the specified confidence interval.

```python
from aries.metrics import PICP

coverage = PICP(posterior_csv, observation_csv, CI=95)
# returns percentage (e.g., 95.3 for 95.3% coverage)
```

## PICP2

Alternative PICP using CDF overlap between the predictive distribution
and the observation noise model. Useful when the observation noise
is known (e.g., from instrument specifications).

```python
from aries.metrics import PICP2

score = PICP2(posterior_csv, observation_csv, CI=95)
```

## Custom metrics

For CRPS, NSE, R-factor, and other probabilistic scores, use
[`properscoring`](https://pypi.org/project/properscoring/) or
[`scoringrules`](https://pypi.org/project/scoringrules/):

```python
from properscoring import crps_ensemble

crps = crps_ensemble(observations, ensemble_predictions)
```
