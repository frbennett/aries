# Core API: `esmda`

::: aries._core
    options:
      members: [esmda]
      show_root_heading: true

## Parameters

`esmda` accepts all parameters via `**kwargs`. Defaults are set before
keyword arguments are applied.

### Required

| Parameter | Type | Description |
|---|---|---|
| `parameter_file_name` | `str` | Path to CSV defining prior distributions |
| `observation_file_name` | `str` | Path to CSV of observation data |
| `error` | `dict` | HalfNormal prior scales per group, e.g. `{"FLOW": 10.0}` |
| `job_name` | `str` | Base name for output directory |

### Convergence and sampling

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nEnsemble` | `int` | `100` | Number of ensemble members |
| `maxIter` | `int` | `10` | Maximum iterations (fixed schedule) or safety cap (ESS schedule) |
| `inversion_type` | `str` | `"svd"` | Kalman inversion method |

### Noise estimation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `calculation_type` | `str` | `"ikea"` | `"ikea"` for adaptive noise, `"esmda"` for fixed noise |
| `phi_update` | `str` | `"laplace"` | Noise estimation method: `"laplace"`, `"mcmc"`, `"grid"`, `"residual"` |

### Tempering schedule

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inflation_schedule` | `str` | `"fixed"` | `"fixed"` for ES-MDA, `"ess"` for CWIEKI adaptive tempering |
| `target_ess` | `float` | `0.5` | ESS threshold as fraction of `nEnsemble` (ESS mode only) |

### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save_all_iterations` | `bool` | `False` | Save ensemble at every iteration |

## Parameter file format

**`es_parameters.csv`** — one row per model parameter:

| column | description |
|---|---|
| `parameter` | Parameter name |
| `lower` | Lower bound of truncated normal prior |
| `upper` | Upper bound |
| `mean` | Prior mean |
| `width` | Number of standard deviations from mean to bound (controls prior dispersion) |

## Observation file format

**`es_data.csv`** — one row per observation timestep:

| column | description |
|---|---|
| `label` | Observation identifier (e.g., date string) |
| `Y` | Observed value (transformed space) |
| `noise` | Measurement noise standard deviation |
| `group` | Group label (e.g., `"FLOW"`, `"STAGE"`) — used for per-group noise estimation |
