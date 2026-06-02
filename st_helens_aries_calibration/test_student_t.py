#!/usr/bin/env python3
"""Quick test of Student-t likelihood in ARIES."""
import sys, os, warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sacramento import sacsma
from aries import esmda

warnings.filterwarnings("ignore")

MODEL_DATE_RANGE = pd.date_range(start="1973-02-08", end="2018-12-31")
GAUGE_START, GAUGE_END = "1979-01-01", "2018-12-31"
CATCHMENT_AREA, LAM = 120_500_000.0, 0.5
forcing = pd.read_csv("data/st_helens_forcing_original.csv")
forcing["date"] = pd.to_datetime(forcing["date"], dayfirst=True)
forcing = forcing.set_index("date")
P, E = forcing["rainfall"].values, forcing["pet"].values

SPLIT_DATE = "2010-01-01"
all_obs_dates = pd.date_range(GAUGE_START, GAUGE_END)
train_mask = all_obs_dates < pd.Timestamp(SPLIT_DATE)

def run_model(ps):
    r = sacsma(P, E, np.asarray(ps, float)) * CATCHMENT_AREA / 1000 / (3600 * 24)
    df = pd.DataFrame({"data": r}, index=MODEL_DATE_RANGE)
    df = df[(df.index >= GAUGE_START) & (df.index <= GAUGE_END)]
    vals = np.clip(df["data"].values.copy(), 0, None) ** LAM
    return vals[train_mask]  # training period only

def fill_ensemble(m, nE, mL, dL):
    y = np.zeros([dL, nE])
    jobs = [delayed(run_model)(m[:, j]) for j in range(nE)]
    results = Parallel(n_jobs=4, verbose=0, backend="threading")(jobs)
    for j in range(nE):
        y[:, j] = results[j]
    return y

for ltype in ["gaussian", "student_t"]:
    print(f"\n{'='*60}")
    print(f"  Testing {ltype.upper()} likelihood")
    print(f"{'='*60}")
    solver = esmda(
        likelihood=ltype,
        nu_init=8.0, nu_adapt=True,
        parameter_file_name="data/es_parameters.csv",
        observation_file_name="data/es_data_train.csv",
        nEnsemble=100, maxIter=5,
        error={"FLOW": 10.0},
        job_name=f"/tmp/test_{ltype}",
        inversion_type="efast_subspace",
        calculation_type="ikea",
    )
    solver.run_esmda(fill_ensemble)

print("\n✅ Both tests passed")
