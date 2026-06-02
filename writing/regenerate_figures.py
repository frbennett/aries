#!/usr/bin/env python3
"""Regenerate all 4 figures with corrected Student-t results."""
import sys, os, glob, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJ = "/home/bennett/hermes_workspaces/ikea_paper/st_helens_aries_calibration"
sys.path.insert(0, PROJ)
from sacramento import sacsma

FIGDIR = "/home/bennett/hermes_workspaces/ikea_paper/writing/figures"
os.makedirs(FIGDIR, exist_ok=True)

SPLIT_DATE = "2010-01-01"
MODEL_DATE_RANGE = pd.date_range(start="1973-02-08", end="2018-12-31")
GAUGE_START, GAUGE_END = "1979-01-01", "2018-12-31"
CATCHMENT_AREA, LAM = 120_500_000.0, 0.5
all_obs_dates = pd.date_range(GAUGE_START, GAUGE_END)
train_mask = all_obs_dates < pd.Timestamp(SPLIT_DATE)
val_mask   = all_obs_dates >= pd.Timestamp(SPLIT_DATE)

forcing = pd.read_csv(f"{PROJ}/data/st_helens_forcing_original.csv")
forcing["date"] = pd.to_datetime(forcing["date"], dayfirst=True)
forcing = forcing.set_index("date")
forcing["in_gauge"] = (forcing.index >= GAUGE_START) & (forcing.index <= GAUGE_END)
gauge = forcing[forcing["in_gauge"]]
Y_all = gauge["Q_CUMEC"].values
Y_train, Y_val = Y_all[train_mask], Y_all[val_mask]
P, E = forcing["rainfall"].values, forcing["pet"].values

def run_model(ps):
    r = sacsma(P, E, np.asarray(ps, float)) * CATCHMENT_AREA / 1000 / (3600 * 24)
    df = pd.DataFrame({"data": r}, index=MODEL_DATE_RANGE)
    return df[(df.index >= GAUGE_START) & (df.index <= GAUGE_END)]["data"].values.copy()

# Load posteriors
gauss_file = sorted(glob.glob(f"{PROJ}/aries_output/*_parameters.csv"))[-1]
studt_file = sorted(glob.glob(f"{PROJ}/aries_student_t_output/*_parameters.csv"))[-1]

g = pd.read_csv(gauss_file, index_col=0).T.astype(float)
s = pd.read_csv(studt_file, index_col=0).T.astype(float)
g.columns = g.columns.str.strip()
s.columns = s.columns.str.strip()
pn = list(g.columns)
gp, sp = g[pn].values.astype(float), s[pn].values.astype(float)

C_GAUSS, C_STUDT, C_OBS = "#2980B9", "#E67E22", "#C0392B"

# Updated ν convergence data from new run
nu_vals = [8.0, 6.9, 6.1, 5.5, 5.1, 4.9, 4.7, 4.6, 4.5, 4.4, 4.4, 4.4]
iters = list(range(1, len(nu_vals) + 1))

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: ν and φ convergence
# ═══════════════════════════════════════════════════════════════════
print("Figure 1: ν/φ convergence")
phi_g = [0.7409, 0.6963, 0.6209, 0.5627, 0.5458, 0.5405, 0.5382, 0.5367, 0.5358, 0.5353, 0.5350, 0.5340]
phi_s = [0.7412, 0.7083, 0.6428, 0.5896, 0.5583, 0.5489, 0.5464, 0.5445, 0.5432, 0.5419, 0.5420, 0.5420]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(iters, nu_vals, "o-", color=C_STUDT, lw=2, markersize=6)
ax1.axhline(30, color="gray", ls="--", lw=1, alpha=0.5, label="Gaussian limit (ν→∞)")
ax1.set_xlabel("Iteration"); ax1.set_ylabel("ν (degrees of freedom)")
ax1.set_title("(a) ν convergence", fontsize=11, color=C_STUDT)
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 35)

ax2.plot(iters, phi_g, "s-", color=C_GAUSS, lw=1.5, markersize=5, label="Gaussian")
ax2.plot(iters, phi_s, "o-", color=C_STUDT, lw=1.5, markersize=5, label="Student-t")
ax2.set_xlabel("Iteration"); ax2.set_ylabel("φ (mean noise std, transformed space)")
ax2.set_title("(b) φ convergence", fontsize=11)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

fig.suptitle("Student-t ARIES: ν and φ Convergence (St Helens Creek)", fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_student_t_nu_convergence.png", dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Parameter posterior comparison
# ═══════════════════════════════════════════════════════════════════
print("Figure 2: Parameter posterior comparison")
n_params = len(pn)
n_cols = 5; n_rows = int(np.ceil(n_params / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows))
axes = axes.flatten()

for i, name in enumerate(pn):
    ax = axes[i]
    gm, gs = g[name].mean(), g[name].std()
    sm, ss = s[name].mean(), s[name].std()
    ax.hist(g[name], bins=45, density=True, alpha=0.5, color=C_GAUSS, edgecolor="white", linewidth=0.2, label="Gaussian")
    ax.hist(s[name], bins=45, density=True, alpha=0.5, color=C_STUDT, edgecolor="white", linewidth=0.2, label="Student-t")
    if abs(sm - gm) > 0.5 * min(gs, ss):
        ax.annotate("", xy=(sm, 0), xytext=(gm, 0), arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
    ax.set_title(name, fontsize=8, fontweight="bold"); ax.tick_params(labelsize=7)
    if i == 0: ax.legend(fontsize=7, framealpha=0.7)

for ax in axes[n_params:]: ax.set_visible(False)
fig.suptitle("Posterior Marginals: Gaussian vs Student-t ARIES (St Helens Creek)", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_student_t_marginals.png", dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: PPC comparison (recompute with correct noise)
# ═══════════════════════════════════════════════════════════════════
print("Figure 3: PPC comparison (computing posteriors...)")
for label, pp, nu_val, col in [("Gaussian", gp, None, C_GAUSS), ("Student-t", sp, 4.4, C_STUDT)]:
    all_raw = np.array(Parallel(n_jobs=4, verbose=0, backend="threading")(
        delayed(run_model)(pp[j]) for j in range(len(pp))))
    all_t = np.clip(all_raw, 0, None) ** LAM
    pt_t = all_t[:, train_mask]
    Yt_t = np.clip(Y_train, 0, None) ** LAM
    ns = np.std(Yt_t - pt_t.mean(axis=0))
    rng = np.random.default_rng(42)
    nppc = min(500, len(pp))
    idx = rng.choice(len(pp), size=nppc, replace=False)
    if nu_val is not None:
        # Student-t noise in transformed space: t_ν scaled by noise std
        noise_tr = rng.standard_t(df=nu_val, size=(nppc, pt_t.shape[1])) * ns
    else:
        noise_tr = rng.normal(0, ns, (nppc, pt_t.shape[1]))
    ppc_tr = (np.clip(np.clip(pt_t[idx], 0, None) + noise_tr, 0, None)) ** (1.0/LAM)
    if label == "Gaussian": g_ppc_tr = ppc_tr
    else: s_ppc_tr = ppc_tr

WINDOW = 365 * 3
d = all_obs_dates[:WINDOW]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharey=True)
for ax, ppc, color, title in [(ax1, g_ppc_tr, C_GAUSS, "Gaussian"), (ax2, s_ppc_tr, C_STUDT, "Student-t (ν = 4.4)")]:
    lo, hi = np.percentile(ppc, 2.5, 0), np.percentile(ppc, 97.5, 0)
    mu = ppc.mean(0)
    ax.fill_between(d, lo[:WINDOW], hi[:WINDOW], color=color, alpha=0.22, label="95% CI")
    ax.plot(d, mu[:WINDOW], color=color, lw=1.2, label="PPC mean")
    ax.scatter(d, Y_all[:WINDOW], c=C_OBS, s=2, alpha=0.6, linewidths=0, label="Observed", zorder=10)
    ax.set_ylabel("Discharge (m³/s)"); ax.set_title(title, fontsize=11, color=color)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.8); ax.grid(True, alpha=0.25)
ax2.set_xlabel("Date")
fig.suptitle("Posterior Predictive Check — St Helens Creek (first 3 years)", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_student_t_ppc.png", dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Stratified P-factor
# ═══════════════════════════════════════════════════════════════════
print("Figure 4: Stratified P-factor")
g_pf = {50: 91.8, 75: 84.2, 90: 65.6, 95: 49.6, 99: 25.4}
s_pf = {50: 93.5, 75: 88.3, 90: 74.1, 95: 61.2, 99: 42.0}
pcts = [50, 75, 90, 95, 99]; x = np.arange(len(pcts)); w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - w/2, [g_pf[p] for p in pcts], w, color=C_GAUSS, alpha=0.8, label="Gaussian")
bars2 = ax.bar(x + w/2, [s_pf[p] for p in pcts], w, color=C_STUDT, alpha=0.8, label="Student-t (ν = 4.4)")
ax.axhline(95, color="gray", ls="--", lw=1.5, alpha=0.5, label="Target (95%)")
for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.0f}%', ha='center', fontsize=7, color=C_GAUSS)
for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f'{bar.get_height():.0f}%', ha='center', fontsize=7, color=C_STUDT)
ax.set_xticks(x); ax.set_xticklabels([f">P{p}" for p in pcts])
ax.set_ylabel("P-factor (%)"); ax.set_xlabel("Observed flow percentile")
ax.set_title("Prediction Interval Coverage by Flow Regime", fontsize=12)
ax.legend(fontsize=9); ax.set_ylim(0, 105); ax.grid(True, alpha=0.2, axis="y")
fig.suptitle("Stratified P-factor: Gaussian vs Student-t ARIES", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_student_t_stratified_pf.png", dpi=200, bbox_inches="tight")
plt.close()

print("\n✅ All 4 figures regenerated")
